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
"""Tests for SwiGLU composite op."""

from absl.testing import parameterized
from litert_torch.generative.export_hf.experimental.composites import swiglu
import torch
import torch.nn.functional as F

from absl.testing import absltest as googletest


class SwiGLUTest(parameterized.TestCase):

  def test_apply_swiglu(self):
    batch_size = 2
    seq_len = 4
    hidden_size = 32
    gate_up = torch.randn(batch_size, seq_len, hidden_size * 2)

    gate, up = gate_up.split([hidden_size, hidden_size], dim=-1)
    expected = F.silu(gate) * up
    actual = swiglu.apply_swiglu(gate_up, gate_size=hidden_size)

    self.assertTrue(torch.allclose(expected, actual, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
  googletest.main()
