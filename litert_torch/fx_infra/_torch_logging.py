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
"""Suppresses PyTorch internal logging spam during LiteRT-Torch operations."""

import logging
import os
import sys
from litert_torch._config import config

_TORCH_LOGGERS = (
    "torch",
    "torch._functorch",
    "torch._dynamo",
    "torch._inductor",
)


def _remove_stderr_handlers(logger: logging.Logger) -> None:
  for h in list(logger.handlers):
    if getattr(h, "stream", None) == sys.stderr or isinstance(
        h, logging.StreamHandler
    ):
      logger.removeHandler(h)


def silence_torch_logs() -> None:
  """Silences internal PyTorch loggers unless explicitly enabled via TORCH_LOGS."""
  if "TORCH_LOGS" in os.environ or not config.silence_torch_logs:
    return

  for name in _TORCH_LOGGERS:
    logger = logging.getLogger(name)
    logger.setLevel(logging.WARNING)
    # Keep propagate=True on non-artifact loggers so that legitimate WARNING
    # and ERROR records still reach the root absl handler.
    logger.propagate = True
    _remove_stderr_handlers(logger)

  # Silence any dynamically created artifact loggers
  for name, logger in list(logging.root.manager.loggerDict.items()):
    if not isinstance(logger, logging.Logger):
      continue
    if not name.startswith("torch."):
      continue
    if "__" in name:
      # Artifact loggers (e.g. __aot_graphs) are debug dumps; fully disable them.
      logger.setLevel(logging.WARNING)
      logger.propagate = False
      _remove_stderr_handlers(logger)
    elif logger.level != logging.NOTSET and logger.level < logging.WARNING:
      logger.setLevel(logging.WARNING)
      logger.propagate = True
      _remove_stderr_handlers(logger)
