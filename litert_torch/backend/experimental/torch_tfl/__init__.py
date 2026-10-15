# Copyright 2025 The LiteRT Torch Authors.
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
"""Backward compatibility alias for litert_torch.backend.torch_tfl."""
from litert_torch.backend import torch_tfl as _torch_tfl
from litert_torch.backend.torch_tfl import *  # pylint: disable=wildcard-import
from litert_torch.backend.torch_tfl import _decomps
from litert_torch.backend.torch_tfl import _lowerings
from litert_torch.backend.torch_tfl import _ops
from litert_torch.backend.torch_tfl import torch_library_utils

decomps = _torch_tfl.decomps
