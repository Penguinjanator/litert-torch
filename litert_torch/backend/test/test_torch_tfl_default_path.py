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
"""Tests for the LITERT_TORCH_FULL_TFL_DECOMPS opt-in flag.

Covers three things:
  1. The flag gates the full torch_tfl decomposition table on the default
     lowering path (`exported_program_to_mlir`), and is off by default.
  2. Binary-op type promotion matches eager PyTorch semantics.
  3. Ops guarded as unsupported (float64, rank > 5) fall back cleanly via
     `NotImplemented` instead of emitting an invalid tfl.* op.
"""

import re
from unittest import mock

import torch

from litert_torch import fx_infra
from litert_torch.backend import export
from litert_torch.backend.experimental import torch_tfl

from absl.testing import absltest as googletest
from absl.testing import parameterized


def _lower_to_mlir_text(model, args):
  """Export + lower via the default path; return the MLIR text."""
  exported = torch.export.export(model.eval(), args)
  exported = fx_infra.safe_run_decompositions(
      exported, fx_infra.decomp.pre_convert_decomp()
  )
  lowered = export.exported_program_to_mlir(exported)
  return lowered.get_text()


def _tfl_op_pattern(op_name):
  """Matches either spelling of a lowered tfl.* op in MLIR text."""
  escaped = re.escape(op_name)
  return rf'(@{escaped}\b|call_target_name = "{escaped}")'


class GeluModel(torch.nn.Module):
  """Minimal model whose only op has a torch_tfl decomposition."""

  def forward(self, x):
    return torch.nn.functional.gelu(x)


class TestTorchTflFullDecompsFlag(parameterized.TestCase):
  """The flag switches the default path between legacy and full decomps."""

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)

  def _enable_flag(self):
    self.enter_context(
        mock.patch.dict("os.environ", {"LITERT_TORCH_FULL_TFL_DECOMPS": "1"})
    )

  def test_gelu_lowered_via_torch_tfl(self):
    """With flag enabled, aten.gelu reaches tfl.gelu via torch_tfl decomp."""
    self._enable_flag()
    mlir_text = _lower_to_mlir_text(GeluModel(), (torch.randn(2, 8),))
    self.assertRegex(mlir_text, _tfl_op_pattern("tfl.gelu"))

  def test_mean_dim_lowered_via_torch_tfl(self):
    """With flag enabled, aten.mean.dim reaches tfl.mean."""

    class MeanModel(torch.nn.Module):

      def forward(self, x):
        return x.mean(dim=(2, 3))

    self._enable_flag()
    mlir_text = _lower_to_mlir_text(MeanModel(), (torch.randn(1, 4, 8, 8),))
    self.assertRegex(mlir_text, _tfl_op_pattern("tfl.mean"))

  def test_default_preserves_multinomial_only(self):
    """By default (flag off), gelu must NOT go through torch_tfl."""
    mlir_text = _lower_to_mlir_text(GeluModel(), (torch.randn(2, 8),))
    self.assertNotRegex(mlir_text, _tfl_op_pattern("tfl.gelu"))

  @parameterized.named_parameters(
      ("one", "1"),
      ("true", "true"),
      ("mixed_case_true", "TRUE"),
      ("padded", " 1 "),
  )
  def test_flag_accepts_truthy_spellings(self, value):
    """The flag parse is case-insensitive and tolerates surrounding space."""
    self.enter_context(
        mock.patch.dict(
            "os.environ", {"LITERT_TORCH_FULL_TFL_DECOMPS": value}
        )
    )
    mlir_text = _lower_to_mlir_text(GeluModel(), (torch.randn(2, 8),))
    self.assertRegex(mlir_text, _tfl_op_pattern("tfl.gelu"))

  @parameterized.named_parameters(
      ("zero", "0"),
      ("false", "false"),
      ("empty", ""),
  )
  def test_flag_rejects_falsy_spellings(self, value):
    """Anything not explicitly truthy leaves the legacy behavior in place."""
    self.enter_context(
        mock.patch.dict(
            "os.environ", {"LITERT_TORCH_FULL_TFL_DECOMPS": value}
        )
    )
    mlir_text = _lower_to_mlir_text(GeluModel(), (torch.randn(2, 8),))
    self.assertNotRegex(mlir_text, _tfl_op_pattern("tfl.gelu"))


class TestBinaryOpTypePromotion(parameterized.TestCase):
  """Decomposed binary ops must promote types the way eager PyTorch does.

  These call the registered decompositions directly so that a promotion
  regression is attributed to the decomp rather than to a downstream lowering.
  """

  @parameterized.named_parameters(
      # A float scalar must widen an integer tensor to float rather than being
      # truncated down into the tensor's integer dtype.
      ("int32_tensor_float_scalar", torch.int32, 2.5),
      ("int64_tensor_float_scalar", torch.int64, 0.5),
      # A float scalar must NOT widen a narrower float tensor to float32.
      ("float16_tensor_float_scalar", torch.float16, 2.5),
      ("bfloat16_tensor_float_scalar", torch.bfloat16, 2.5),
      ("float32_tensor_float_scalar", torch.float32, 2.5),
      # Integer scalars leave integer tensors alone.
      ("int32_tensor_int_scalar", torch.int32, 3),
  )
  def test_mul_scalar_matches_eager(self, dtype, scalar):
    """mul against a Python scalar must match eager dtype and value."""
    x = torch.tensor([1, 2, 3]).to(dtype)
    expected = x * scalar
    decomp = torch_tfl.decomps[torch.ops.aten.mul.Tensor]
    actual = decomp(x, scalar)

    self.assertEqual(actual.dtype, expected.dtype)
    torch.testing.assert_close(actual, expected)

  @parameterized.named_parameters(
      ("add", torch.ops.aten.add.Tensor, torch.add),
      ("sub", torch.ops.aten.sub.Tensor, torch.sub),
      ("mul", torch.ops.aten.mul.Tensor, torch.mul),
  )
  def test_int_tensor_float_scalar_matches_eager(self, op, ref):
    x = torch.tensor([1, 2, 3], dtype=torch.int32)
    expected = ref(x, 2.5)
    actual = torch_tfl.decomps[op](x, 2.5)

    self.assertEqual(actual.dtype, expected.dtype)
    torch.testing.assert_close(actual, expected)

  def test_mixed_dtype_tensors_promote(self):
    x = torch.tensor([1, 2, 3], dtype=torch.int32)
    y = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
    expected = x * y
    actual = torch_tfl.decomps[torch.ops.aten.mul.Tensor](x, y)

    self.assertEqual(actual.dtype, expected.dtype)
    torch.testing.assert_close(actual, expected)


class TestUnsupportedDtypeFallback(googletest.TestCase):
  """Guarded ops must fall back via NotImplemented, not emit bad tfl ops."""

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)
    self.enter_context(
        mock.patch.dict("os.environ", {"LITERT_TORCH_FULL_TFL_DECOMPS": "1"})
    )

  def test_float64_add_falls_back(self):
    """float64 is unsupported by TFLite, so add must not become tfl.add."""

    class AddModel(torch.nn.Module):

      def forward(self, x, y):
        return x + y

    args = (
        torch.randn(4, dtype=torch.float64),
        torch.randn(4, dtype=torch.float64),
    )
    mlir_text = _lower_to_mlir_text(AddModel(), args)
    self.assertNotRegex(mlir_text, _tfl_op_pattern("tfl.add"))

  def test_rank6_slice_falls_back(self):
    """Rank > 5 is unsupported by tfl.strided_slice, so it must fall back."""

    class SliceModel(torch.nn.Module):

      def forward(self, x):
        return x[:, :, :, :, :, 0:2]

    args = (torch.randn(1, 1, 1, 1, 2, 4),)
    mlir_text = _lower_to_mlir_text(SliceModel(), args)
    self.assertNotRegex(mlir_text, _tfl_op_pattern("tfl.strided_slice"))

  def test_float32_add_still_lowers(self):
    """Control for the float64 case: the guard must not over-trigger."""

    class AddModel(torch.nn.Module):

      def forward(self, x, y):
        return x + y

    args = (torch.randn(4), torch.randn(4))
    mlir_text = _lower_to_mlir_text(AddModel(), args)
    self.assertRegex(mlir_text, _tfl_op_pattern("tfl.add"))


if __name__ == "__main__":
  googletest.main()
