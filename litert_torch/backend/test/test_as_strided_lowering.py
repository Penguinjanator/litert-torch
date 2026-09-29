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
"""Tests that contiguous aten.as_strided views lower without a gather."""

import torch

from litert_torch.backend import export

from absl.testing import absltest as googletest
from absl.testing import parameterized


class AsStrided(torch.nn.Module):

  def __init__(self, sizes, strides, storage_offset=0):
    super().__init__()
    self.sizes = sizes
    self.strides = strides
    self.storage_offset = storage_offset

  def forward(self, x):
    return torch.ops.aten.as_strided(
        x, self.sizes, self.strides, self.storage_offset
    )


def _lower(model, args) -> str:
  exported = torch.export.export(model.eval(), args)
  return str(export.exported_program_to_mlir(exported).module.operation)


class TestAsStridedLowering(parameterized.TestCase):

  @parameterized.named_parameters(
      # swin_t's pooling tail: `mean` of a channels-last tensor gives sizes
      # [1, C, 1, 1] with strides [C, 1, C, C]. Only the size-1 dims disagree
      # with contiguous strides, so this is a plain reshape.
      ("size1_dims_have_arbitrary_strides", (1, 768), [1, 768, 1, 1],
       [768, 1, 768, 768], 0),
      ("contiguous_reshape", (4, 6), [2, 3, 4], [12, 4, 1], 0),
      ("contiguous_window_with_offset", (10, 10), [4, 5], [5, 1], 20),
  )
  def test_contiguous_view_lowers_without_gather(
      self, in_shape, sizes, strides, offset
  ):
    x = torch.randn(in_shape)
    mlir = _lower(AsStrided(sizes, strides, offset), (x,))
    self.assertNotIn("gather", mlir)

  @parameterized.named_parameters(
      ("gapped", (10, 10), [2, 2, 2], [8, 4, 1], 0),
      ("overlapping", (10, 10), [5, 5], [2, 2], 0),
  )
  def test_non_contiguous_view_still_gathers(
      self, in_shape, sizes, strides, offset
  ):
    x = torch.randn(in_shape)
    mlir = _lower(AsStrided(sizes, strides, offset), (x,))
    self.assertIn("gather", mlir)


if __name__ == "__main__":
  googletest.main()
