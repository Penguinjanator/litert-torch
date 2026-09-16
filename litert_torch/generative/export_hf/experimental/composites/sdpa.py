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
"""Optimized GPU cache update operation for attention."""

import math
from typing import Any, Optional
from litert_torch.backend import composite
from litert_torch.generative.custom_ops import bmm_4d as bmm_lib
from litert_torch.generative.export_hf.core.cache import _get_slice_indices
from litert_torch.generative.export_hf.experimental.composites import cache_update as gpu_cache_update
from litert_torch.generative.export_hf.experimental.composites import runtime_batched_matmul
import torch
import torch.nn.functional as F

runtime_bmm = runtime_batched_matmul.runtime_bmm

# Fill value for attention mask. -10000.0 matches MLDrift convention.
MASK_FILL_VALUE = -10000.0


def _sync_and_rebind_cache(past_key_value, layer_idx, new_k, new_v):
  if past_key_value is None or layer_idx is None:
    return
  layer = past_key_value.layers[layer_idx]
  old_k = layer.keys
  old_v = layer.values

  for l in past_key_value.layers:
    if l.keys is old_k:
      l.keys = new_k
    if l.values is old_v:
      l.values = new_v


def ring_buffer_sdpa(
    query: torch.Tensor,
    key_past: torch.Tensor,
    value_past: torch.Tensor,
    k_ts_idx: int,
    v_ts_idx: int,
    param_tensor: torch.Tensor,
    is_global: bool,
    key_states: Optional[torch.Tensor] = None,
    value_states: Optional[torch.Tensor] = None,
    layer: Optional[Any] = None,
    cache_position: Optional[torch.Tensor] = None,
    past_key_value: Optional[Any] = None,
    layer_idx: Optional[int] = None,
    mask: Optional[torch.Tensor] = None,
    softcap: float | None = None,
):
  """Ring buffer SDPA."""
  if layer is None and past_key_value is not None and layer_idx is not None:
    layer = past_key_value.layers[layer_idx]
  assert layer is not None, "layer must be provided for ring buffer SDPA"
  assert mask is not None, "mask must be provided for ring buffer SDPA"

  if key_states is None and hasattr(layer, "pending_key_update"):
    key_states = layer.pending_key_update
    value_states = layer.pending_value_update
    cache_position = layer.pending_cache_position

  assert key_states is not None, "key_states must be provided"
  assert value_states is not None, "value_states must be provided"
  assert cache_position is not None, "cache_position must be provided"

  seq_len = cache_position.shape[0]

  if layer.is_sliding:
    S = layer.max_cache_len
    if seq_len > S:
      if layer.k_ts_idx == 3:
        key_states_for_update = key_states[..., -S:]
      else:
        key_states_for_update = key_states[..., -S:, :]
      if layer.v_ts_idx == 3:
        value_states_for_update = value_states[..., -S:]
      else:
        value_states_for_update = value_states[..., -S:, :]
      offset = (param_tensor[..., 0:1] + (seq_len - S)) % S
      start_position = cache_position[seq_len - S] % S
    else:
      key_states_for_update = key_states
      value_states_for_update = value_states
      offset = param_tensor[..., 0:1] % S
      start_position = cache_position[0] % S

    active = torch.clamp(param_tensor[..., 1:2], max=S)
    active_aligned = torch.clamp(param_tensor[..., 2:3], max=S)
    active_past = torch.clamp(param_tensor[..., 0:1], max=S)
    rest = param_tensor[..., 3:]
    if seq_len > S:
      update_len_tensor = torch.clamp(rest[..., 0:1], max=S)
      rest = torch.cat([update_len_tensor, rest[..., 1:]], dim=-1)
    modified_param_tensor = torch.cat(
        [offset, active, active_aligned, rest], dim=-1
    )
    modified_param_tensor_past = torch.cat(
        [offset, active_past, active_past, rest], dim=-1
    )
  else:
    key_states_for_update = key_states
    value_states_for_update = value_states
    modified_param_tensor = param_tensor
    modified_param_tensor_past = param_tensor
    start_position = cache_position[0]

  k_slice_indices = _get_slice_indices(
      start_position.clone(), 4, layer.k_ts_idx
  )
  v_slice_indices = _get_slice_indices(
      start_position.clone(), 4, layer.v_ts_idx
  )

  _, bk_size, v_dim2, v_dim3 = value_past.shape
  v_head_size = v_dim2 if layer.v_ts_idx == 3 else v_dim3
  cache_size = v_dim3 if layer.v_ts_idx == 3 else v_dim2

  param_for_bmm = (
      modified_param_tensor_past
      if (layer.is_sliding and seq_len > 1)
      else modified_param_tensor
  )
  bmm_fn_k = lambda x, y: runtime_batched_matmul.runtime_bmm(
      x, y, param_for_bmm, is_global=is_global, is_src=False
  )
  bmm_fn_v = lambda x, y: runtime_batched_matmul.runtime_bmm(
      x, y, param_for_bmm, is_global=is_global, is_src=True
  )

  gt = query.shape[2]
  is_split = layer.is_sliding and (seq_len > 1)

  if is_split:
    # 1. BMM with old cache (uses runtime_bmm)
    logits_past = bmm_fn_k(query, key_past)

    # 2. BMM with new tokens (regular BMM using full key_states)
    if k_ts_idx == 3:
      logits_current = torch.einsum("abth,abhs->abts", query, key_states)
    else:
      assert k_ts_idx == 2, "k_ts_idx must be 2 or 3."
      logits_current = bmm_lib.bmm_4d(query, key_states)

    S_past = key_past.size(2 if k_ts_idx == 2 else 3)
    t = mask.size(2)
    g = gt // t

    # Convert passed mask to bool if it's not
    if mask.dtype != torch.bool:
      mask: torch.Tensor = mask == 0

    # Slice mask for past and current BMM
    mask_past = mask[..., :S_past]
    mask_current = mask[..., S_past:]

    if g != 1:
      mask_past = torch.cat([mask_past] * g, dim=-2)
      mask_current = torch.cat([mask_current] * g, dim=-2)

    logits_all = torch.cat([logits_past, logits_current], dim=-1)
    mask_all = torch.cat([mask_past, mask_current], dim=-1)

    if softcap is not None:
      logits_all = torch.tanh(logits_all / softcap)
      logits_all = logits_all * softcap

    padded_logits_all = torch.where(
        mask_all,
        logits_all,
        torch.tensor(MASK_FILL_VALUE, dtype=logits_all.dtype),
    )

    attrs = {"axis": -1}
    builder = composite.StableHLOCompositeBuilder(
        name="odml.softmax", attr=attrs
    )
    padded_logits_all = builder.mark_inputs(padded_logits_all)
    probs_all = F.softmax(padded_logits_all, dim=-1)
    probs_all = builder.mark_outputs(probs_all)
    probs_all = probs_all.type_as(query)
    probs_past, probs_current = probs_all.split(S_past, dim=-1)

    # 3. V Matmuls
    # encoded_past uses old cache 'value_past' and runtime_bmm
    encoded_past = bmm_fn_v(probs_past, value_past)
    # encoded_current uses full new 'value_states'.
    if v_ts_idx == 3:
      bmm_fn_v_regular = bmm_lib.bmm_4d
    else:
      assert v_ts_idx == 2, "v_ts_idx must be 2 or 3."
      bmm_fn_v_regular = lambda x, y: torch.einsum("abts,absh->abth", x, y)
    encoded_current = bmm_fn_v_regular(probs_current, value_states)

    encoded = encoded_past + encoded_current

    # Force dependency: Make cache_update inputs depend on BMM2 output (encoded)
    # This ensures all reads of the old cache (both K and V) are completed
    # before cache_update overwrites them.
    dummy_dep = encoded.sum() * 0
    key_states_for_update_dep = key_states_for_update + dummy_dep
    value_states_for_update_dep = value_states_for_update + dummy_dep

    # 4. Update Cache (using old cache key_past/value_past as baseline)
    new_k, new_v = gpu_cache_update.cache_update(
        key_states_for_update_dep,
        value_states_for_update_dep.transpose(-2, -1),
        modified_param_tensor,
        key_past,
        value_past,
        indices_k=k_slice_indices,
        indices_v=v_slice_indices,
        kv_heads=bk_size,
        kv_batch_size=1,
        cache_len=cache_size,
        head_size=v_head_size,
        is_ring_buffer=layer.is_sliding,
    )
    layer.keys = new_k
    layer.values = new_v
    if past_key_value is not None and layer_idx is not None:
      _sync_and_rebind_cache(past_key_value, layer_idx, new_k, new_v)

    return encoded
  else:
    new_k, new_v = gpu_cache_update.cache_update(
        key_states_for_update,
        value_states_for_update.transpose(-2, -1),
        modified_param_tensor,
        key_past,
        value_past,
        indices_k=k_slice_indices,
        indices_v=v_slice_indices,
        kv_heads=bk_size,
        kv_batch_size=1,
        cache_len=cache_size,
        head_size=v_head_size,
        is_ring_buffer=layer.is_sliding,
    )
    layer.keys = new_k
    layer.values = new_v
    if past_key_value is not None and layer_idx is not None:
      _sync_and_rebind_cache(past_key_value, layer_idx, new_k, new_v)

    logits = bmm_fn_k(query, new_k)
    g = gt // mask.size(2)

    if softcap is not None:
      logits = torch.tanh(logits / softcap)
      logits = logits * softcap

    if mask.dtype != torch.bool:
      mask: torch.Tensor = mask == 0
    if g != 1:
      mask = torch.cat([mask] * g, dim=-2)
    padded_logits = torch.where(
        mask, logits, torch.tensor(MASK_FILL_VALUE, dtype=logits.dtype)
    )
    attrs = {"axis": -1}
    builder = composite.StableHLOCompositeBuilder(
        name="odml.softmax", attr=attrs
    )
    padded_logits = builder.mark_inputs(padded_logits)
    probs = F.softmax(padded_logits, dim=-1)
    probs = builder.mark_outputs(probs)
    probs = probs.type_as(query)
    encoded = bmm_fn_v(probs, new_v)
    return encoded


def scaled_dot_product_attention_transposed(
    query: torch.Tensor,
    key: torch.Tensor | tuple[Any, ...],
    value: torch.Tensor | tuple[Any, ...],
    head_size: int,
    k_ts_idx: int,
    v_ts_idx: int,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    softcap: Optional[float] = None,
    alibi_bias: Optional[torch.Tensor] = None,
    param_tensor: Optional[torch.Tensor] = None,
    is_global: bool = False,
    use_sdpa_composite: bool = True,
    enable_ring_buffer: bool = False,
    past_key_value: Optional[Any] = None,
    layer_idx: Optional[int] = None,
):
  """Scaled dot product attention with transposed key and value.

  Args:
    query: Query tensor, with shape [B, N, T, H] or [1, BK, GT, H].
    key: Key tensor, with shape [B, T, KV_LEN, H].
    value: Value tensor, with shape [B, T, H, KV_LEN].
    head_size (int): head dimension.
    k_ts_idx (int): the index of time step dimension in the key tensor.
    v_ts_idx (int): the index of time step dimension in the value tensor.
    mask (torch.Tensor): the optional mask tensor.
    scale (float): the optional scale factor.
    softcap (float): the optional softcap for the logits.
    alibi_bias (torch.Tensor): optional alibi bias tensor.
    param_tensor (torch.Tensor): optional param tensor for runtime bmm.
    is_global (bool): whether the attention is global.
    use_sdpa_composite (bool): whether to use composite for SDPA.
    enable_ring_buffer (bool): whether to enable ring buffer.
    past_key_value (Any): past key value cache.
    layer_idx (int): the index of the layer.

  Returns:
    The output tensor of scaled_dot_product_attention_transposed.
  """
  if isinstance(key, tuple):
    key_past, key_states, layer, cache_position, param_tensor_from_tuple = key
    value_past, value_states, _ = value
    if param_tensor is None:
      param_tensor = param_tensor_from_tuple
    is_sliding = layer.is_sliding
  else:
    key_past = key
    key_states = None
    value_past = value
    value_states = None
    layer = (
        past_key_value.layers[layer_idx]
        if past_key_value is not None and layer_idx is not None
        else None
    )
    cache_position = None
    is_sliding = (
        getattr(layer, "is_sliding", False)
        if layer is not None
        else not is_global
    )

  b, n, seq_len, h = query.shape
  is_decode_composite = use_sdpa_composite and seq_len == 1

  if isinstance(key, tuple) or (
      param_tensor is not None and enable_ring_buffer and is_sliding
  ):
    g = n // key_past.shape[1]
    num_query_groups = n // g
    query = query.reshape(1, b * num_query_groups, g * seq_len, h)
  elif not is_decode_composite:
    g = n // key_past.shape[1]
    num_query_groups = n // g
    query = query.reshape(1, b * num_query_groups, g * seq_len, h)

  if scale is None:
    scale = 1.0 / math.sqrt(head_size)

  if alibi_bias is not None:
    alibi_bias = alibi_bias * scale
    if mask is None:
      mask = alibi_bias
    else:
      mask = mask + alibi_bias

  query = query * scale

  assert mask is not None, "Mask should not be None!"

  if isinstance(key, tuple) or (
      param_tensor is not None and enable_ring_buffer and is_sliding
  ):
    encoded = ring_buffer_sdpa(
        query=query,
        key_past=key_past,
        value_past=value_past,
        k_ts_idx=k_ts_idx,
        v_ts_idx=v_ts_idx,
        param_tensor=param_tensor,
        is_global=is_global,
        key_states=key_states,
        value_states=value_states,
        layer=layer,
        cache_position=cache_position,
        past_key_value=past_key_value,
        layer_idx=layer_idx,
        mask=mask,
        softcap=softcap,
    )
    if is_decode_composite:
      encoded = encoded.permute(0, 2, 1, 3).reshape(b, 1, -1)
    else:
      encoded = encoded.reshape(b, -1, seq_len, h).permute(0, 2, 1, 3)
    return encoded

  assert isinstance(key_past, torch.Tensor)
  assert isinstance(value_past, torch.Tensor)
  key = key_past
  value = value_past
  t = mask.shape[2]
  gt = query.shape[2]
  g = gt // t

  # broadcasting mask
  if param_tensor is not None:
    if mask.dtype != torch.bool:
      mask: torch.Tensor = mask == 0
    if g != 1:
      mask = mask.to(torch.float32)  # pyrefly: ignore[missing-attribute]
      mask = torch.cat([mask] * g, dim=1)
      mask: torch.Tensor = mask != 0
      mask = mask.reshape(1, 1, gt, -1)  # pyrefly: ignore[missing-attribute]
  else:
    if g != 1:
      mask_to_bc = []
      for _ in range(g):
        mask_to_bc.append(mask)
      mask = torch.cat(mask_to_bc, dim=-2)  # 1, 1, gt, s

  attrs = {}
  attrs.update({
      "k_ts_idx": k_ts_idx,
      "v_ts_idx": v_ts_idx,
  })
  if softcap is not None:
    attrs["softcap"] = softcap
  if use_sdpa_composite:
    sdpa_builder = composite.StableHLOCompositeBuilder(
        name="odml.sdpa_transposed", attr=attrs
    )
    query, key_past, value_past, mask, param_tensor = sdpa_builder.mark_inputs(
        query, key_past, value_past, mask, param_tensor
    )
  else:
    sdpa_builder = None

  key_for_bmm = key_past
  value_for_bmm = value_past
  if is_decode_composite and query.shape[1] != key_past.shape[1]:
    g_ratio = query.shape[1] // key_past.shape[1]
    key_for_bmm = key_past.repeat_interleave(g_ratio, dim=1)
    value_for_bmm = value_past.repeat_interleave(g_ratio, dim=1)

  if param_tensor is not None:
    # Do not create nested odml.runtime_bmm composites inside odml.sdpa_transposed.
    # When use_sdpa_composite is False, enable composite on the individual BMMs instead.
    bmm_fn = lambda x, y: runtime_batched_matmul.runtime_bmm(
        x, y, param_tensor, is_global=is_global, is_src=False,
        use_composite=not use_sdpa_composite,
    )
  elif k_ts_idx == 2:
    bmm_fn = bmm_lib.bmm_4d
  else:
    assert k_ts_idx == 3, "k_ts_idx must be 2 or 3."
    bmm_fn = lambda x, y: torch.einsum("abth,abhs->abts", x, y)
  logits = bmm_fn(query, key_for_bmm)

  if softcap is not None:
    logits = torch.tanh(logits / softcap)
    logits = logits * softcap

  if mask.dtype == torch.bool:
    padded_logits = torch.where(
        mask, logits, torch.tensor(MASK_FILL_VALUE, dtype=logits.dtype)
    )
  else:
    padded_logits = logits + mask

  if padded_logits.dtype != torch.float32:
    attrs = {"axis": -1}
    builder = composite.StableHLOCompositeBuilder(
        name="odml.softmax", attr=attrs
    )
    padded_logits = builder.mark_inputs(padded_logits)
  else:
    builder = None
  probs = F.softmax(padded_logits, dim=-1)
  if builder is not None:
    probs = builder.mark_outputs(probs)
  probs = probs.type_as(key_past)
  if param_tensor is not None:
    # Do not create nested odml.runtime_bmm composites inside odml.sdpa_transposed.
    # When use_sdpa_composite is False, enable composite on the individual BMMs instead.
    bmm_fn = lambda x, y: runtime_batched_matmul.runtime_bmm(
        x, y, param_tensor, is_global=is_global, is_src=True,
        use_composite=not use_sdpa_composite,
    )
  elif v_ts_idx == 3:
    bmm_fn = bmm_lib.bmm_4d
  else:
    assert v_ts_idx == 2, "v_ts_idx must be 2 or 3."
    bmm_fn = lambda x, y: torch.einsum("abts,absh->abth", x, y)
  encoded = bmm_fn(probs, value_for_bmm)
  if is_decode_composite:
    encoded = encoded.permute(0, 2, 1, 3).reshape(b, 1, -1)
    if sdpa_builder is not None:
      encoded = sdpa_builder.mark_outputs(encoded)
  else:
    if sdpa_builder is not None:
      encoded = sdpa_builder.mark_outputs(encoded)
    encoded = encoded.reshape(b, -1, seq_len, h).permute(0, 2, 1, 3)

  return encoded
