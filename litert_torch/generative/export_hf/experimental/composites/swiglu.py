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
"""Optimized GPU SwiGLU activation composite operation compatible with MLDrift."""

from litert_torch.backend import composite
import torch
import torch.nn.functional as F


def apply_swiglu(
    gate_up: torch.Tensor,
    gate_size: int | None = None,
) -> torch.Tensor:
  """Computes SwiGLU activation: silu(gate) * up.

  Args:
    gate_up: Input tensor containing concatenated gate and up projections.
    gate_size: Dimension size of the gate projection (defaults to half of the last dim).

  Returns:
    out: SwiGLU activated output tensor.
  """
  if gate_size is None:
    gate_size = gate_up.shape[-1] // 2

  attrs = {
      "gate_size": int(gate_size),
  }
  builder = composite.StableHLOCompositeBuilder(name="odml.swiglu", attr=attrs)
  gate_up = builder.mark_inputs(gate_up)

  # Fallback PyTorch execution during export tracing:
  gate, up = gate_up.split([gate_size, gate_up.shape[-1] - gate_size], dim=-1)
  out = F.silu(gate) * up
  out = builder.mark_outputs(out)
  return out
