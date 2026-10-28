# Copyright 2024 The LiteRT Torch Authors.
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

import sys
import warnings

from litert_torch import converter_v2
from litert_torch._config import config
from litert_torch._convert.interface import convert
from litert_torch._convert.interface import Converter
from litert_torch._convert.interface import experimental_add_compilation_backend
from litert_torch._convert.interface import signature
from litert_torch._convert.to_channel_last_io import to_channel_last_io
from litert_torch.model import Model
from litert_torch.version import __version__

# Python 3.10 reached upstream end-of-life on 2026-10-01 (PEP 619). Support
# for it is deprecated and will be removed in the next release. FutureWarning
# (rather than DeprecationWarning) so end users see it even when this package
# is imported transitively (PEP 565 hides DeprecationWarning outside __main__).
if sys.version_info < (3, 11):
  warnings.warn(
      "litert-torch support for Python 3.10 is deprecated and will be removed"
      " in the next release, following Python 3.10's upstream end-of-life"
      " (https://peps.python.org/pep-0619/). Please upgrade to Python 3.11 or"
      " newer.",
      FutureWarning,
      stacklevel=2,
  )


def load(path: str) -> Model:
  """Imports an litert_torch model from disk.

  Args:
    path: The path to the serialized litert_torch model.

  Returns:
    An litert_torch.model.Model object.
  """
  return Model.load(path)
