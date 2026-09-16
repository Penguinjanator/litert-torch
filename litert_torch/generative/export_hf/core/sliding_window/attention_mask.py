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
"""Sliding Window attention mask."""

import torch


def generate_causal_right_with_valid_mask(
    valid_mask, W: int | None  # pylint: disable=invalid-name
) -> torch.Tensor:
  """Generates causal mask for right."""

  L = valid_mask.shape[1]  # pylint: disable=invalid-name
  row_indices = torch.arange(L, dtype=torch.int32).unsqueeze(
      1
  )  # [L, 1] (query indices)
  col_indices = torch.arange(L, dtype=torch.int32).unsqueeze(
      0
  )  # [1, L] (key indices)

  # Query at 'i' only attends to keys at 'j' that are at or before 'i'.
  causal_mask = col_indices <= row_indices  # [L, L]

  pads = valid_mask.squeeze(0)
  mask_rows = pads.unsqueeze(1)
  mask_cols = pads.unsqueeze(0)
  padding_mask = mask_rows & mask_cols
  global_mask = causal_mask & padding_mask

  if W is not None:
    # Key at 'j' is within the past window to 'i'.
    window_lower_bound_mask = col_indices >= (row_indices - W + 1)  # [L, L]

    local_mask = global_mask & window_lower_bound_mask
    return local_mask

  return global_mask


def generate_causal_left_with_ring_buffer(
    valid_mask, W: int | None, S: int, time_step: torch.Tensor  # pylint: disable=invalid-name
) -> torch.Tensor:
  """Generates causal mask for left."""
  # TODO: make f32 version for GPU only.
  # Convert inputs to float32 for safe execution on GPU accelerators
  time_step_f32 = time_step.clone().to(torch.float32).unsqueeze(0)
  L = valid_mask.shape[1]  # pylint: disable=invalid-name

  # row_indices: [L, 1] - query timestamps
  row_indices = (
      torch.arange(L, dtype=torch.float32, device=valid_mask.device).unsqueeze(1)
      + time_step_f32
  )

  # col_indices: [1, S] - logical timestamps of physical cache columns
  physical_cols = torch.arange(
      S, dtype=torch.float32, device=valid_mask.device
  ).unsqueeze(0)
  t_minus_one = time_step_f32 - 1.0

  curr_col = torch.remainder(t_minus_one, float(S))
  raw_diff = curr_col - physical_cols
  wrapped_diff = torch.where(raw_diff >= 0.0, raw_diff, raw_diff + float(S))
  col_indices = t_minus_one - wrapped_diff

  # Comparisons work fine on float32
  causal_mask = col_indices <= row_indices  # [L, S]
  mask_cols = (col_indices >= 0.0) & (col_indices < time_step_f32)

  causal_mask &= mask_cols

  if W is not None:
    # Key at 'j' is within the past window to 'i'.
    window_lower_bound_mask = col_indices >= (row_indices - float(W) + 1.0)  # [L, S]

    final_mask = causal_mask & window_lower_bound_mask
    return final_mask

  return causal_mask


def build_full_mask_with_valid_mask(
    valid_mask: torch.Tensor,
    W: int | None,  # pylint: disable=invalid-name
    S: int,  # pylint: disable=invalid-name
    time_step: torch.Tensor,
    use_bool_mask: bool = False,
):
  """Builds full attention mask."""
  left_mask = generate_causal_left_with_ring_buffer(valid_mask, W, S, time_step)
  right_mask = generate_causal_right_with_valid_mask(valid_mask, W)
  mask = torch.cat([left_mask, right_mask], dim=-1).unsqueeze(0).unsqueeze(0)
  if use_bool_mask:
    return mask
  else:
    return torch.logical_not(mask) * -1e4


def build_sliding_window_decode_mask(
    W: int,  # pylint: disable=invalid-name
    S: int,  # pylint: disable=invalid-name
    input_pos,
    use_bool_mask: bool = False,
):
  """Builds sliding window decode mask."""
  physical_index = torch.arange(S, dtype=torch.int32)
  diff = input_pos[0] - physical_index.view(1, 1, 1, S)
  distance = ((diff % S) + S) % S
  w_tensor = torch.tensor(W - 1, dtype=torch.int32).view(1, 1, 1, 1)
  input_pos_expanded = input_pos.view(1, 1, 1, 1)
  mask1 = distance <= w_tensor
  mask2 = distance <= input_pos_expanded
  bool_mask = mask1 & mask2
  if use_bool_mask:
    return bool_mask
  else:
    return torch.where(bool_mask, 0.0, -1e4)
