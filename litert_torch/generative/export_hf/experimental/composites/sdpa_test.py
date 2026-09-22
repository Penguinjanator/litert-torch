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

from absl.testing import parameterized
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


if __name__ == "__main__":
  googletest.main()
