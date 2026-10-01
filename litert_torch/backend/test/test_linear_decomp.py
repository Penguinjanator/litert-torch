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
"""Tests for the aten.linear pre-convert decomposition override."""

import torch

from litert_torch import fx_infra
from litert_torch.backend import export  # pylint: disable=unused-import

from absl.testing import absltest as googletest
from absl.testing import parameterized


def _decompose(model, args):
  exported = torch.export.export(model.eval(), args)
  return fx_infra.safe_run_decompositions(
      exported, fx_infra.decomp.pre_convert_decomp()
  )


def _call_targets(exported):
  return [
      n.target for n in exported.graph_module.graph.nodes
      if n.op == "call_function"
  ]


def _max_rank(exported, targets):
  ranks = [
      n.meta["val"].dim()
      for n in exported.graph_module.graph.nodes
      if n.op == "call_function" and n.target in targets
  ]
  return max(ranks, default=0)


class WindowedLinear(torch.nn.Module):
  """A linear applied after a windowing transpose (timm LocallyGroupedAttn)."""

  def __init__(self, bias):
    super().__init__()
    self.linear = torch.nn.Linear(16, 24, bias=bias)

  def forward(self, x):
    # (B, H/ws, ws, W/ws, ws, C) -> (B, H/ws, W/ws, ws, ws, C): non-contiguous.
    x = x.reshape(1, 2, 3, 2, 3, 16).transpose(2, 3)
    return self.linear(x)


class PlainLinear(torch.nn.Module):

  def __init__(self, bias):
    super().__init__()
    self.linear = torch.nn.Linear(16, 24, bias=bias)

  def forward(self, x):
    return self.linear(x)


_BATCHED = (
    torch.ops.aten.bmm.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.expand_copy.default,
)


class TestLinearDecomp(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)

  @parameterized.named_parameters(("bias", True), ("no_bias", False))
  def test_non_contiguous_high_rank_folds_to_2d(self, bias):
    model = WindowedLinear(bias)
    args = (torch.randn(1, 6, 6, 16),)
    exported = _decompose(model, args)

    targets = _call_targets(exported)
    self.assertIn(
        torch.ops.aten.addmm.default if bias else torch.ops.aten.mm.default,
        targets,
    )
    self.assertNotIn(torch.ops.aten.bmm.default, targets)
    self.assertLessEqual(_max_rank(exported, _BATCHED), 2)
    torch.testing.assert_close(exported.module()(*args), model(*args))

  @parameterized.named_parameters(
      ("rank2_bias", (5, 16), True),
      ("rank2_no_bias", (5, 16), False),
      ("rank3_bias", (2, 5, 16), True),
      ("rank3_no_bias", (2, 5, 16), False),
      ("rank4_bias", (2, 3, 5, 16), True),
      ("rank1_bias", (16,), True),
  )
  def test_contiguous_matches_eager(self, shape, bias):
    model = PlainLinear(bias)
    args = (torch.randn(*shape),)
    exported = _decompose(model, args)

    self.assertNotIn(torch.ops.aten.linear.default, _call_targets(exported))
    torch.testing.assert_close(exported.module()(*args), model(*args))


if __name__ == "__main__":
  googletest.main()
