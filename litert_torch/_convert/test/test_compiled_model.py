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
"""Tests for running converted PyTorch models via LiteRT CompiledModel API."""

import litert_torch
import numpy as np
import torch
from torch import nn

from absl.testing import absltest as googletest

try:
  from ai_edge_litert.litert_wrapper.compiled_model_wrapper.hardware_accelerator import HardwareAccelerator  # pylint: disable=g-import-not-at-top
except ImportError:
  from ai_edge_litert.compiled_model import HardwareAccelerator  # pylint: disable=g-import-not-at-top


class SimpleMLP(nn.Module):
  """Simple MLP module for testing CompiledModel execution."""

  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(16, 32)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(32, 8)

  def forward(self, x):
    return self.fc2(self.relu(self.fc1(x)))


class TestCompiledModelRunner(googletest.TestCase):
  """Test suite for LiteRTModel with CompiledModel runtime."""

  def test_compiled_runtime_cpu_parity(self):
    """Verifies LiteRTModel executes via CompiledModel in __call__ with exact PyTorch parity."""
    torch.manual_seed(42)
    model = SimpleMLP().eval()
    sample_input = (torch.randn(2, 16),)

    edge_model = litert_torch.convert(model, sample_input)
    edge_model.use_compiled_runtime(
        hardware_accel=HardwareAccelerator.CPU,
        require_fully_accelerated=True,
    )
    expected = model(*sample_input).detach().numpy()
    actual = edge_model(*sample_input)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    # Verify cached execution on subsequent calls
    actual_second = edge_model(*sample_input)
    np.testing.assert_allclose(actual_second, expected, rtol=1e-5, atol=1e-5)

  def test_runtime_switching(self):
    """Verifies switching between CompiledModel and Interpreter runtimes."""
    torch.manual_seed(42)
    model = SimpleMLP().eval()
    sample_input = (torch.randn(2, 16),)

    edge_model = litert_torch.convert(model, sample_input)
    expected = model(*sample_input).detach().numpy()

    # Default is Interpreter runtime
    actual_interp = edge_model(*sample_input)
    np.testing.assert_allclose(actual_interp, expected, rtol=1e-5, atol=1e-5)

    # Switch to CompiledModel runtime
    edge_model.use_compiled_runtime(hardware_accel=HardwareAccelerator.CPU)
    actual_compiled = edge_model(*sample_input)
    np.testing.assert_allclose(actual_compiled, expected, rtol=1e-5, atol=1e-5)

    # Switch back to Interpreter runtime
    edge_model.use_interpreter_runtime()
    actual_interp2 = edge_model(*sample_input)
    np.testing.assert_allclose(actual_interp2, expected, rtol=1e-5, atol=1e-5)

  def test_input_dtype_mismatch_is_rejected(self):
    """A dtype the model was not compiled for fails instead of being reinterpreted."""
    torch.manual_seed(42)
    model = SimpleMLP().eval()
    sample_input = (torch.randn(2, 16),)

    edge_model = litert_torch.convert(model, sample_input)
    edge_model.use_compiled_runtime(hardware_accel=HardwareAccelerator.CPU)

    # The model is float32; float64 must not be silently written to the buffer.
    wrong_dtype = np.zeros((2, 16), dtype=np.float64)
    with self.assertRaisesRegex(ValueError, "dtype"):
      edge_model(wrong_dtype)

  def test_static_shape_mismatch_is_rejected(self):
    """Resizing a static dimension fails loudly rather than being accepted."""
    torch.manual_seed(42)
    model = SimpleMLP().eval()
    sample_input = (torch.randn(2, 16),)

    edge_model = litert_torch.convert(model, sample_input)
    edge_model.use_compiled_runtime(hardware_accel=HardwareAccelerator.CPU)

    # This is the only shape coverage available today: a `dynamic_shapes`
    # conversion cannot be built here, because the default lowering path routes
    # ops through the JAX bridge, which asserts on dynamic dimensions. So this
    # asserts the negative half of the resize behavior -- a static dimension is
    # rejected rather than silently accepted -- which is what pins the
    # `strict=True` resize call in `_run_compiled` in place.
    with self.assertRaises(Exception):
      edge_model(torch.randn(4, 16))


if __name__ == "__main__":
  googletest.main()
