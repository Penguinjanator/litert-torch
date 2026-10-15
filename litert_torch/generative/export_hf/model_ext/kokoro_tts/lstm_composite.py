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
"""Fused bidirectional LSTM composite.

Tracing a bidirectional LSTM through torch.export statically unrolls every time
step, so a single layer with T=128 expands into thousands of scalar TFLite
nodes. Wrapping the layer in a `litert_custom_op.` composite collapses it to one
`tfl.custom` node that is executed by `BidirectionalLstmCustomOpKernel` in the
LiteRT runtime.

Any composite whose name starts with `litert_custom_op.` is legalized to
`tfl.custom` by `LegalizeCompositeToCustomOpPass`, with the name carried through
verbatim as the `custom_code`. No converter change is needed, but the pass also
discards the decomposition region: a model exported with this composite will
only load in a runtime that has registered the matching kernel.
"""

from typing import Callable

from litert_torch.backend import composite
import torch

# Must match `BidirectionalLstmCustomOpKernel::OpName()` in the C++ runtime,
# `litert/experimental/custom_ops/bidirectional_lstm/`. The full dotted string
# (prefix included) is what lands in the flatbuffer as the `custom_code`, and is
# what the TFLite op resolver matches on.
COMPOSITE_NAME = "litert_custom_op.bidirectional_lstm"

# Gate ordering of the packed [4H, *] weight matrices. This is PyTorch's
# `nn.LSTM` convention: input, forget, cell, output.
GATE_ORDER = "ifgo"

# Signature of the decomposition body, which is also the eager-mode path:
#   (x, seq_lengths, mask, w_ih_fwd, w_hh_fwd, b_fwd, w_ih_bwd, w_hh_bwd, b_bwd)
ReferenceImpl = Callable[..., torch.Tensor]


def fused_bidirectional_lstm(
    x: torch.Tensor,
    seq_lengths: torch.Tensor,
    mask: torch.Tensor,
    w_ih_fwd: torch.Tensor,
    w_hh_fwd: torch.Tensor,
    b_fwd: torch.Tensor,
    w_ih_bwd: torch.Tensor,
    w_hh_bwd: torch.Tensor,
    b_bwd: torch.Tensor,
    hidden_size: int,
    reference_impl: ReferenceImpl,
) -> torch.Tensor:
  """Marks a bidirectional LSTM layer as a single fused custom op.

  The nine operands are marked in a fixed order that the C++ kernel relies on.
  Weights are passed as composite inputs rather than attributes because
  `LegalizeCompositeToCustomOpPass` can only serialize int, float and string
  attributes into the flexbuffer `custom_option`.

  Args:
    x: Layer input, `[B, T, D]` float32.
    seq_lengths: Unpadded length of each batch element, `[B]`. Cast to int32 for
      the op boundary; the GPU delegate rejects int64 casts and the kernel has
      no use for the extra range.
    mask: Padding mask, `[B, T]` float32, 1.0 on padding positions. Callers that
      have no mask must pass zeros rather than omitting the operand, so the op
      always has a fixed arity.
    w_ih_fwd: Forward input-to-hidden weights, `[4H, D]`, gates in `GATE_ORDER`.
    w_hh_fwd: Forward hidden-to-hidden weights, `[4H, H]`.
    b_fwd: Forward bias, `[4H]`. PyTorch's separate `bias_ih` and `bias_hh` are
      summed by the caller, since only their sum is observable.
    w_ih_bwd: Backward input-to-hidden weights, `[4H, D]`.
    w_hh_bwd: Backward hidden-to-hidden weights, `[4H, H]`.
    b_bwd: Backward bias, `[4H]`.
    hidden_size: Per-direction hidden size `H`.
    reference_impl: The unrolled implementation, traced as the composite's
      decomposition region. Keeping this a parameter means the fused and unfused
      paths share exactly one copy of the math.

  Returns:
    The concatenated forward and backward hidden states, `[B, T, 2H]`.
  """
  builder = composite.StableHLOCompositeBuilder(
      name=COMPOSITE_NAME,
      attr={"hidden_size": int(hidden_size), "gate_order": GATE_ORDER},
  )
  (
      x,
      seq_lengths,
      mask,
      w_ih_fwd,
      w_hh_fwd,
      b_fwd,
      w_ih_bwd,
      w_hh_bwd,
      b_bwd,
  ) = builder.mark_inputs(
      x,
      seq_lengths.to(torch.int32),
      mask,
      w_ih_fwd,
      w_hh_fwd,
      b_fwd,
      w_ih_bwd,
      w_hh_bwd,
      b_bwd,
  )
  # torch.gather inside the body needs int64 indices. This cast lives in the
  # discarded decomposition region, so it never reaches the flatbuffer.
  y = reference_impl(
      x,
      seq_lengths.to(torch.long),
      mask,
      w_ih_fwd,
      w_hh_fwd,
      b_fwd,
      w_ih_bwd,
      w_hh_bwd,
      b_bwd,
  )
  return builder.mark_outputs(y)
