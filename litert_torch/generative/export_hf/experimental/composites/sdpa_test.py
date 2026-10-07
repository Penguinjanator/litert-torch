# Copyright 2026 The LiteRT Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for the transposed SDPA composite."""

import types
from unittest import mock

from absl.testing import parameterized
from litert_torch.generative.export_hf.core.sliding_window import attention_mask
from litert_torch.generative.export_hf.experimental.composites import sdpa
import torch
from absl.testing import absltest as googletest


# The GPU composite path always runs the runtime BMM, which expects the key
# time-major ([B, K, S, H]) and the value head-major ([B, K, H, S]).
_K_TS_IDX = 2
_V_TS_IDX = 3


def _make_inputs(num_query_heads, num_kv_heads, seq_len, kv_len, head_dim):
  """Builds a deterministic (query, key, value, mask, param_tensor) set."""
  generator = torch.Generator().manual_seed(1234)

  def randn(*shape):
    return torch.randn(*shape, generator=generator)

  query = randn(1, num_query_heads, seq_len, head_dim)
  key = randn(1, num_kv_heads, kv_len, head_dim)
  value = randn(1, num_kv_heads, head_dim, kv_len)
  # 0 means "attend"; the tail of the cache is masked out so that the mask
  # actually participates in the result.
  mask = torch.zeros(1, 1, seq_len, kv_len)
  mask[:, :, :, kv_len // 2 :] = sdpa.MASK_FILL_VALUE
  param_tensor = torch.ones((1, 1, 1, 7), dtype=torch.int32)
  return query, key, value, mask, param_tensor


def _run(query, key, value, mask, param_tensor, use_sdpa_composite):
  return sdpa.scaled_dot_product_attention_transposed(
      query=query,
      key=key,
      value=value,
      head_size=query.shape[-1],
      k_ts_idx=_K_TS_IDX,
      v_ts_idx=_V_TS_IDX,
      mask=mask,
      param_tensor=param_tensor,
      is_global=True,
      use_sdpa_composite=use_sdpa_composite,
  )


class ScaledDotProductAttentionTransposedTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # Decode: the composite output is flattened to [B, 1, N * H].
      ("gqa_decode", 8, 2, 1),
      ("mqa_decode", 8, 1, 1),
      ("mha_decode", 4, 4, 1),
      # Prefill: the composite output keeps the [B, T, N, H] layout.
      ("gqa_prefill", 8, 2, 6),
      ("mqa_prefill", 8, 1, 6),
      ("mha_prefill", 4, 4, 6),
  )
  def test_composite_matches_decomposed(
      self, num_query_heads, num_kv_heads, seq_len
  ):
    """Marking the composite region must not change the attention result."""
    kv_len, head_dim = 16, 8
    inputs = _make_inputs(
        num_query_heads, num_kv_heads, seq_len, kv_len, head_dim
    )

    reference = _run(*inputs, use_sdpa_composite=False)
    actual = _run(*inputs, use_sdpa_composite=True)

    # Decode folds the head dimension into the last axis; compare both in the
    # common [B, T, N, H] layout.
    shape = (1, seq_len, num_query_heads, head_dim)
    torch.testing.assert_close(actual.reshape(shape), reference.reshape(shape))

  @parameterized.named_parameters(
      ("decode", 1),
      ("prefill", 6),
  )
  def test_composite_boundary_shape_is_independent_of_gqa_ratio(self, seq_len):
    """The fused kernel matches on the boundary layout, so it must be stable."""
    kv_len, head_dim, num_query_heads = 16, 8, 8

    shapes = set()
    for num_kv_heads in (8, 2, 1):
      inputs = _make_inputs(
          num_query_heads, num_kv_heads, seq_len, kv_len, head_dim
      )
      shapes.add(tuple(_run(*inputs, use_sdpa_composite=True).shape))
    self.assertLen(shapes, 1)

  @parameterized.named_parameters(
      ("decode", 1),
      ("prefill", 6),
  )
  def test_kv_cache_is_not_broadcast(self, seq_len):
    """GQA must be expressed by packing queries, never by tiling the cache.

    Tiling the KV cache is invisible on GPU because the composite region is
    replaced by a fused kernel, but CPU backends execute the decomposition,
    where it becomes a per-layer, per-step copy of the whole cache that
    XNNPACK cannot delegate.
    """
    num_query_heads, num_kv_heads, kv_len, head_dim = 8, 2, 16, 8
    inputs = _make_inputs(
        num_query_heads, num_kv_heads, seq_len, kv_len, head_dim
    )

    class Wrapper(torch.nn.Module):

      def forward(self, query, key, value, mask, param_tensor):
        return _run(
            query, key, value, mask, param_tensor, use_sdpa_composite=True
        )

    exported = torch.export.export(Wrapper(), inputs)
    op_names = [
        str(node.target)
        for node in exported.graph.nodes
        if node.op == "call_function"
    ]
    self.assertNotIn(
        "aten.repeat_interleave.self_int",
        " ".join(op_names),
        f"Unexpected KV cache broadcast in {op_names}",
    )

  @parameterized.named_parameters(
      ("decode", 1),
      ("prefill", 6),
  )
  def test_composite_output_boundary_is_4d(self, seq_len):
    """The odml.sdpa_transposed composite output must stay 4D [B, N, T, H]."""
    num_query_heads, num_kv_heads, kv_len, head_dim = 8, 2, 16, 8
    inputs = _make_inputs(
        num_query_heads, num_kv_heads, seq_len, kv_len, head_dim
    )

    class Wrapper(torch.nn.Module):

      def forward(self, query, key, value, mask, param_tensor):
        return _run(
            query, key, value, mask, param_tensor, use_sdpa_composite=True
        )

    exported = torch.export.export(Wrapper(), inputs)
    sdpa_marks = [
        node
        for node in exported.graph.nodes
        if node.op == "call_function"
        and "mark_tensor" in str(node.target)
        and "odml.sdpa_transposed" in (str(node.args) + str(node.kwargs))
    ]
    self.assertNotEmpty(sdpa_marks)
    out_val = sdpa_marks[-1].meta["val"]
    self.assertEqual(
        tuple(out_val.shape), (1, num_query_heads, seq_len, head_dim)
    )

  @parameterized.named_parameters(
      ("prefill_default_global_non_causal", 6, True, None, False),
      ("prefill_default_sliding_non_causal", 6, False, None, False),
      ("prefill_explicit_non_causal", 6, True, False, False),
      ("prefill_explicit_causal", 6, True, True, True),
      ("decode_global_causal", 1, True, False, True),
      ("decode_non_ring_sliding_non_causal", 1, False, False, False),
  )
  def test_composite_emits_is_causal_attribute(
      self, seq_len, is_global, is_causal, expected_is_causal
  ):
    """The odml.sdpa_transposed composite must record `is_causal` in attrs."""
    inputs = _make_inputs(8, 2, seq_len, 16, 8)

    class Wrapper(torch.nn.Module):

      def forward(self, query, key, value, mask, param_tensor):
        return sdpa.scaled_dot_product_attention_transposed(
            query=query,
            key=key,
            value=value,
            head_size=query.shape[-1],
            k_ts_idx=_K_TS_IDX,
            v_ts_idx=_V_TS_IDX,
            mask=mask,
            param_tensor=param_tensor,
            is_global=is_global,
            use_sdpa_composite=True,
            **({} if is_causal is None else {"is_causal": is_causal}),
        )

    exported = torch.export.export(Wrapper(), inputs)
    sdpa_marks = [
        node
        for node in exported.graph.nodes
        if node.op == "call_function"
        and "mark_tensor" in str(node.target)
        and "odml.sdpa_transposed" in (str(node.args) + str(node.kwargs))
    ]
    self.assertNotEmpty(sdpa_marks)
    mark_str = str(sdpa_marks[-1].args) + str(sdpa_marks[-1].kwargs)
    self.assertIn(f"('is_causal', {expected_is_causal})", mark_str)


_WINDOW = 8
_PAST_LEN = 6
_NEW_LEN = 4  # _PAST_LEN + _NEW_LEN > _WINDOW, so the write wraps the ring.
_HEAD_DIM = 4


class _RingCacheLayer:
  """Minimal sliding-window cache layer as seen by `ring_buffer_sdpa`."""

  def __init__(self, keys, values):
    self.keys = keys
    self.values = values
    self.is_sliding = True
    self.max_cache_len = _WINDOW
    self.k_ts_idx = _K_TS_IDX
    self.v_ts_idx = _V_TS_IDX


_ORIGINAL_CACHE_UPDATE = sdpa.gpu_cache_update.cache_update


def _in_place_cache_update(*args, **kwargs):
  """`cache_update` that also writes into its cache inputs.

  The runtimes alias the KV cache input and output buffers, so the update is
  visible to every later reader of the *input* tensor. Eager PyTorch is
  functional and hides this; writing back in place reproduces it.

  Args:
    *args: Forwarded to `cache_update`.
    **kwargs: Forwarded to `cache_update`.

  Returns:
    The (aliased) key and value caches.
  """
  cache_k, cache_v = args[3], args[4]
  new_k, new_v = _ORIGINAL_CACHE_UPDATE(*args, **kwargs)
  cache_k.copy_(new_k)
  cache_v.copy_(new_v)
  return cache_k, cache_v


class SharedRingBufferWriteTest(parameterized.TestCase):
  """KV-shared layers must all read the ring buffer before it is written."""

  def setUp(self):
    super().setUp()
    generator = torch.Generator().manual_seed(42)

    def randn(*shape):
      return torch.randn(*shape, generator=generator)

    # Slots [0, _PAST_LEN) hold positions [0, _PAST_LEN); the rest is empty.
    self.cache_k = randn(1, 1, _WINDOW, _HEAD_DIM)
    self.cache_v = randn(1, 1, _HEAD_DIM, _WINDOW)
    # The donor's new K/V, which every sharing layer attends to.
    self.key_states = randn(1, 1, _NEW_LEN, _HEAD_DIM)
    self.value_states = randn(1, 1, _NEW_LEN, _HEAD_DIM)
    self.queries = [randn(1, 1, _NEW_LEN, _HEAD_DIM) for _ in range(3)]
    self.cache_position = torch.arange(
        _PAST_LEN, _PAST_LEN + _NEW_LEN, dtype=torch.int32
    )
    total = _PAST_LEN + _NEW_LEN
    self.param_tensor = torch.tensor(
        [[[[_PAST_LEN, total, total, _NEW_LEN, 0, 0, 0]]]], dtype=torch.int32
    )
    past_mask = torch.zeros(1, 1, _NEW_LEN, _WINDOW, dtype=torch.bool)
    past_mask[..., :_PAST_LEN] = True
    new_mask = torch.tril(torch.ones(_NEW_LEN, _NEW_LEN, dtype=torch.bool))
    self.mask = torch.cat([past_mask, new_mask.view(1, 1, _NEW_LEN, -1)], -1)

  def _attend(self, layer, query, skip_cache_update):
    return sdpa.ring_buffer_sdpa(
        query=query,
        key_past=layer.keys,
        value_past=layer.values,
        k_ts_idx=_K_TS_IDX,
        v_ts_idx=_V_TS_IDX,
        param_tensor=self.param_tensor,
        is_global=False,
        key_states=self.key_states,
        value_states=self.value_states,
        layer=layer,
        cache_position=self.cache_position,
        mask=self.mask,
        skip_cache_update=skip_cache_update,
    )

  def _run_donor_and_readers(self, skips):
    """Runs donor, reader, last reader over one shared, aliased cache.

    Args:
      skips: `skip_cache_update` for the donor, a reader and the last reader.

    Returns:
      The three attention outputs and the shared cache layer.
    """
    layer = _RingCacheLayer(self.cache_k.clone(), self.cache_v.clone())
    outputs = []
    with mock.patch.object(
        sdpa.gpu_cache_update, "cache_update", _in_place_cache_update
    ):
      for query, skip in zip(self.queries, skips):
        outputs.append(self._attend(layer, query, skip_cache_update=skip))
    return outputs, layer

  def _reference(self):
    """Every layer reads the pre-update cache; the cache is written once."""
    outputs = []
    for query in self.queries:
      layer = _RingCacheLayer(self.cache_k.clone(), self.cache_v.clone())
      outputs.append(self._attend(layer, query, skip_cache_update=True))
    layer = _RingCacheLayer(self.cache_k.clone(), self.cache_v.clone())
    self._attend(layer, self.queries[0], skip_cache_update=False)
    return outputs, layer

  def test_donor_writing_first_corrupts_later_readers(self):
    """Documents the hazard of the default ownership (the donor writes)."""
    expected, _ = self._reference()
    actual, _ = self._run_donor_and_readers([False, True, True])
    torch.testing.assert_close(actual[0], expected[0])
    self.assertFalse(torch.allclose(actual[1], expected[1]))
    self.assertFalse(torch.allclose(actual[2], expected[2]))

  def test_last_reader_writes_the_donor_cache(self):
    expected, expected_layer = self._reference()
    # The donor skips its write; the last reader writes the shared cache.
    actual, layer = self._run_donor_and_readers([True, True, False])
    for a, e in zip(actual, expected):
      torch.testing.assert_close(a, e)
    torch.testing.assert_close(layer.keys, expected_layer.keys)
    torch.testing.assert_close(layer.values, expected_layer.values)
    # The write wrapped the ring: slots 6, 7, 0, 1 hold the new tokens.
    self.assertFalse(torch.equal(layer.keys, self.cache_k))

  def test_rebind_updates_every_alias_of_the_donor_cache(self):
    layer = _RingCacheLayer(self.cache_k, self.cache_v)
    alias = _RingCacheLayer(self.cache_k, self.cache_v)
    other = _RingCacheLayer(self.cache_k.clone(), self.cache_v.clone())
    cache = types.SimpleNamespace(layers=[layer, alias, other])
    new_k = torch.zeros_like(self.cache_k)
    new_v = torch.zeros_like(self.cache_v)

    sdpa._rebind_layer_cache(cache, layer, new_k, new_v)

    self.assertIs(layer.keys, new_k)
    self.assertIs(alias.keys, new_k)
    self.assertIs(alias.values, new_v)
    self.assertIsNot(other.keys, new_k)


class RingBufferDecodeTest(parameterized.TestCase):
  """Sliding-window decode as `odml.sdpa_transposed` vs the runtime BMM path."""

  def _inputs(self, pos, window=6, groups=4):
    generator = torch.Generator().manual_seed(pos)

    def randn(*shape):
      return torch.randn(*shape, generator=generator)

    layer = _RingCacheLayer(
        randn(1, 1, _WINDOW, _HEAD_DIM), randn(1, 1, _HEAD_DIM, _WINDOW)
    )
    input_pos = torch.tensor([pos], dtype=torch.int32)
    return dict(
        query=randn(1, 1, groups, _HEAD_DIM),
        key_past=layer.keys,
        value_past=layer.values,
        param_tensor=torch.tensor(
            [[[[pos % _WINDOW, pos + 1, pos + 1, 1, 0, 0, 0]]]],
            dtype=torch.int32,
        ),
        key_states=randn(1, 1, 1, _HEAD_DIM),
        value_states=randn(1, 1, 1, _HEAD_DIM),
        layer=layer,
        cache_position=input_pos,
        mask=attention_mask.build_sliding_window_decode_mask(
            window, _WINDOW, input_pos, use_bool_mask=True
        ),
    )

  def _decode(self, inputs, use_sdpa_composite):
    return sdpa.ring_buffer_sdpa(
        k_ts_idx=_K_TS_IDX,
        v_ts_idx=_V_TS_IDX,
        is_global=False,
        use_sdpa_composite=use_sdpa_composite,
        **inputs,
    )

  @parameterized.product(pos=[0, 3, 7, 8, 13], groups=[1, 4])
  def test_composite_matches_runtime_bmm(self, pos, groups):
    expected = self._decode(
        self._inputs(pos, groups=groups), use_sdpa_composite=False
    )
    actual = self._decode(
        self._inputs(pos, groups=groups), use_sdpa_composite=True
    )
    self.assertEqual(actual.shape, expected.shape)
    torch.testing.assert_close(actual, expected)

  def test_emits_sdpa_transposed_with_param(self):
    test = self
    # Built outside `forward`: torch.export cannot trace a torch.Generator.
    inputs = self._inputs(9)

    class Decode(torch.nn.Module):

      def forward(self, x):
        return test._decode(inputs, use_sdpa_composite=True) + x

    exported = torch.export.export(Decode(), (torch.zeros(1),))
    marks = [
        str(n.args)
        for n in exported.graph.nodes
        if n.op == "call_function" and "mark_tensor" in str(n.target)
    ]
    sdpa_marks = [m for m in marks if "odml.sdpa_transposed" in m]
    # q, k, v, mask and param in, one output.
    self.assertLen(sdpa_marks, 6, sdpa_marks)
    self.assertFalse(any("odml.runtime_bmm" in m for m in marks))


if __name__ == "__main__":
  googletest.main()
