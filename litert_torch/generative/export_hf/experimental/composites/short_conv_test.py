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
"""Tests for short_conv composite op."""

from absl.testing import parameterized
from litert_torch.generative.export_hf.experimental.composites import short_conv
import torch

from absl.testing import absltest as googletest


class ShortConvTest(parameterized.TestCase):

  def test_short_conv_step_accuracy(self):
    batch_size = 1
    hidden_size = 64
    conv_L_cache = 3

    in_proj_out = torch.randn(batch_size, 1, 3 * hidden_size)
    conv_state = torch.randn(batch_size, hidden_size, conv_L_cache - 1)
    conv_weight = torch.randn(hidden_size, 1, conv_L_cache)
    conv_bias = torch.randn(hidden_size)

    # Reference computation:
    b, c, x_proj = in_proj_out.chunk(3, dim=-1)
    p = b * x_proj
    p_s = p.squeeze(1).unsqueeze(-1)
    padded = torch.cat([conv_state, p_s], dim=-1)
    expected_next_state = padded[:, :, -(conv_L_cache - 1) :]
    w = conv_weight.squeeze(1).unsqueeze(0)
    expected_out = (padded * w).sum(dim=-1) + conv_bias.unsqueeze(0)
    expected_y = c * expected_out.unsqueeze(1)

    actual_y, actual_next_state = short_conv.apply_short_conv_step(
        in_proj_out=in_proj_out,
        conv_state=conv_state,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        conv_L_cache=conv_L_cache,
    )

    self.assertTrue(torch.allclose(expected_y, actual_y, rtol=1e-5, atol=1e-5))
    self.assertTrue(
        torch.allclose(
            expected_next_state, actual_next_state, rtol=1e-5, atol=1e-5
        )
    )

  def test_short_conv_step_no_bias(self):
    batch_size = 1
    hidden_size = 64
    conv_L_cache = 3

    in_proj_out = torch.randn(batch_size, 1, 3 * hidden_size)
    conv_state = torch.randn(batch_size, hidden_size, conv_L_cache - 1)
    conv_weight = torch.randn(hidden_size, conv_L_cache)

    b, c, x_proj = in_proj_out.chunk(3, dim=-1)
    p = b * x_proj
    p_s = p.squeeze(1).unsqueeze(-1)
    padded = torch.cat([conv_state, p_s], dim=-1)
    expected_next_state = padded[:, :, -(conv_L_cache - 1) :]
    w = conv_weight.unsqueeze(0)
    expected_out = (padded * w).sum(dim=-1)
    expected_y = c * expected_out.unsqueeze(1)

    actual_y, actual_next_state = short_conv.apply_short_conv_step(
        in_proj_out=in_proj_out,
        conv_state=conv_state,
        conv_weight=conv_weight,
        conv_bias=None,
        conv_L_cache=conv_L_cache,
    )

    self.assertTrue(torch.allclose(expected_y, actual_y, rtol=1e-5, atol=1e-5))
    self.assertTrue(
        torch.allclose(
            expected_next_state, actual_next_state, rtol=1e-5, atol=1e-5
        )
    )


if __name__ == "__main__":
  googletest.main()
