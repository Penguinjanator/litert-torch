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
"""Exportable module and model-wrapper dispatchers for `export_hf`.

The core export pipeline (`core/export_lib.py`) calls the functions in this
module to look up architecture-specific `ExportableModule` subclasses, task
model wrappers (ASR, TTS, ImageGen, Any2Any), and checkpoint `model_type`
inference heuristics. Each function resolves the active `ModelExtension` via
`extension.resolve_model_extension` (combining built-in extensions from
`builtin_extensions` with any custom `--extension_modules` or
`model_extension=...` overrides) and applies standard fallbacks when an
extension returns `None`.
"""

from typing import Any

from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core.speech import exportables as speech_exportables
# Registers built-in extensions.
from litert_torch.generative.export_hf.model_ext import builtin_extensions as _
from litert_torch.generative.export_hf.model_ext import extension as extension_lib
import transformers


def get_prefill_decode_exportables(
    model_config: transformers.PretrainedConfig,
    export_config: exportable_module.ExportableModuleConfig,
):
  """Returns custom `(prefill_cls, decode_cls)` exportables or `None`.

  Delegates to `ModelExtension.get_prefill_decode_exportables`. When `None` is
  returned, `core/export_lib.py` falls back to the generic decoder-only
  prefill/decode exportable modules. See `model_ext/qwen3_5/extension.py`,
  `model_ext/gemma3n/extension.py`, and `model_ext/gemma4/extension.py` for
  built-in custom prefill/decode implementations.

  Args:
    model_config: Loaded `PretrainedConfig` for the checkpoint.
    export_config: Active `ExportableModuleConfig`.

  Returns:
    A 2-tuple `(prefill_cls, decode_cls)` of `ExportableModule` subclasses, or
    `None` if the generic core exportables should be used.
  """
  ext = extension_lib.resolve_model_extension(model_config, export_config)
  if ext is not None:
    return ext.get_prefill_decode_exportables(model_config, export_config)
  return None


def get_vision_exportables(
    model_config: transformers.PretrainedConfig,
    export_config: exportable_module.ExportableModuleConfig | None = None,
):
  """Returns `(encoder_cls, adapter_cls, eoi_cls)` for vision tower export.

  Delegates to `ModelExtension.get_vision_exportables`. See
  `model_ext/gemma3/extension.py`, `model_ext/gemma4/extension.py`, and
  `model_ext/lfm2_vl/extension.py` for built-in examples.

  Args:
    model_config: Loaded `PretrainedConfig` for the multimodal checkpoint.
    export_config: Optional `ExportableModuleConfig`.

  Returns:
    A 3-tuple `(encoder_cls, adapter_cls, eoi_cls)` where `adapter_cls` and
    `eoi_cls` may be `None`.

  Raises:
    ValueError: If no extension provides vision exportables for
      `model_config.model_type`.
  """
  ext = extension_lib.resolve_model_extension(model_config, export_config)
  if ext is not None:
    ext_exportables = ext.get_vision_exportables(model_config, export_config)
    if ext_exportables is not None:
      return tuple(ext_exportables)
  raise ValueError(f'Unsupported model type: {model_config.model_type}')


def get_audio_exportables(
    model_config: transformers.PretrainedConfig,
    export_config: exportable_module.ExportableModuleConfig | None = None,
):
  """Returns `(encoder_cls, adapter_cls, eoa_cls)` for audio tower export.

  Delegates to `ModelExtension.get_audio_exportables`. See
  `model_ext/gemma4/extension.py` for a built-in example.

  Args:
    model_config: Loaded `PretrainedConfig` for the multimodal checkpoint.
    export_config: Optional `ExportableModuleConfig`.

  Returns:
    A 3-tuple `(encoder_cls, adapter_cls, eoa_cls)` where `adapter_cls` and
    `eoa_cls` may be `None`.

  Raises:
    ValueError: If no extension provides audio exportables for
      `model_config.model_type`.
  """
  ext = extension_lib.resolve_model_extension(model_config, export_config)
  if ext is not None:
    ext_exportables = ext.get_audio_exportables(model_config, export_config)
    if ext_exportables is not None:
      return tuple(ext_exportables)
  raise ValueError(f'Unsupported model type: {model_config.model_type}')


def get_additional_exportables(
    model_config: transformers.PretrainedConfig,
    export_config: exportable_module.ExportableModuleConfig | None = None,
):
  """Returns `{submodel_name: exportable_cls}` for auxiliary submodels.

  Delegates to `ModelExtension.get_additional_exportables`. See
  `model_ext/gemma3n/extension.py` and `model_ext/gemma4/extension.py`
  (`'per_layer_embedder'`) for built-in examples.

  Args:
    model_config: Loaded `PretrainedConfig` for the checkpoint.
    export_config: Optional `ExportableModuleConfig`.

  Returns:
    Dictionary mapping submodel names to `ExportableModule` subclasses (empty
    `{}` if no auxiliary submodels apply).
  """
  ext = extension_lib.resolve_model_extension(model_config, export_config)
  if ext is not None:
    return dict(ext.get_additional_exportables(model_config, export_config))
  return {}


def get_speech_exportables(
    model_config: transformers.PretrainedConfig,
    export_config: exportable_module.ExportableModuleConfig | None = None,
):
  """Returns ASR exportable module classes, defaulting to encode + decode.

  Delegates to `ModelExtension.get_speech_exportables` and falls back to
  `(LiteRTExportableModuleForAsrEncode, LiteRTExportableModuleForAsrDecode)`
  when no custom speech exportables are specified. See `ParakeetCtcExtension`
  in `model_ext/parakeet/extension.py` for an encoder-only override example.

  Args:
    model_config: Loaded `PretrainedConfig` for the ASR checkpoint.
    export_config: Optional `ExportableModuleConfig`.

  Returns:
    Tuple of ASR `ExportableModule` subclasses to export.
  """
  ext = extension_lib.resolve_model_extension(model_config, export_config)
  if ext is not None:
    ext_exportables = ext.get_speech_exportables(model_config, export_config)
    if ext_exportables is not None:
      return ext_exportables
  return (
      speech_exportables.LiteRTExportableModuleForAsrEncode,
      speech_exportables.LiteRTExportableModuleForAsrDecode,
  )


def get_speech_model_cls(
    model_type: str,
    export_config: exportable_module.ExportableModuleConfig | None = None,
):
  """Returns the ASR model wrapper class registered for `model_type`.

  See `model_ext/whisper/extension.py`, `model_ext/parakeet/extension.py`,
  `model_ext/moonshine/extension.py`, and `model_ext/qwen3/extension.py` for
  built-in examples.

  Args:
    model_type: Speech `model_type` string from `PretrainedConfig`.
    export_config: Optional `ExportableModuleConfig`.

  Returns:
    The ASR model wrapper class for `model_type`.

  Raises:
    ValueError: If no extension provides a speech model class for `model_type`.
  """
  ext = extension_lib.resolve_model_extension(model_type, export_config)
  if ext is not None:
    cls = ext.get_speech_model_cls(model_type)
    if cls is not None:
      return cls
  raise ValueError(f'Unsupported speech model type: {model_type}')


def get_tts_model_cls(
    model_type: str,
    export_config: exportable_module.ExportableModuleConfig | None = None,
):
  """Returns the TTS model wrapper class registered for `model_type`.

  See `model_ext/kokoro_tts/extension.py` and `model_ext/qwen3_tts/extension.py`
  for built-in examples.

  Args:
    model_type: TTS `model_type` string.
    export_config: Optional `ExportableModuleConfig`.

  Returns:
    The TTS model wrapper class for `model_type`.

  Raises:
    ValueError: If no extension provides a TTS model class for `model_type`.
  """
  ext = extension_lib.resolve_model_extension(model_type, export_config)
  if ext is not None:
    cls = ext.get_tts_model_cls(model_type)
    if cls is not None:
      return cls
  raise ValueError(f'Unsupported TTS model type: {model_type}')


def infer_tts_model_type(
    config_dict: dict[str, Any],
    model_path: str,
    export_config: exportable_module.ExportableModuleConfig | None = None,
) -> str | None:
  """Infers `model_type` for a TTS checkpoint lacking `config.model_type`.

  See `model_ext/kokoro_tts/extension.py` for a built-in implementation of
  `ModelExtension.is_matching_tts_checkpoint`.

  Args:
    config_dict: Parsed `config.json` dictionary (or `{}` if missing).
    model_path: HuggingFace repo ID or local checkpoint directory path.
    export_config: Optional `ExportableModuleConfig`.

  Returns:
    Inferred `model_type` string, or `None` if no TTS extension matched.
  """
  return extension_lib.infer_model_type_from_extensions(
      model_path=model_path,
      task='text_to_speech',
      config_dict=config_dict,
      export_config=export_config,
  )


def infer_model_type(
    model_path: str,
    task: Any = None,
    config_dict: dict[str, Any] | None = None,
    export_config: exportable_module.ExportableModuleConfig | None = None,
) -> str | None:
  """Infers `model_type` for checkpoints without a standard HF `model_type`.

  See `model_ext/bonsai_flux2/extension.py` and
  `model_ext/flux2_klein/extension.py` for built-in `match_checkpoint` examples.

  Args:
    model_path: HuggingFace repo ID or local checkpoint directory path.
    task: Export task enum or string.
    config_dict: Optional parsed dictionary from `model_index.json` or
      `config.json`.
    export_config: Optional `ExportableModuleConfig`.

  Returns:
    Inferred `model_type` string, or `None` if no extension matched.
  """
  return extension_lib.infer_model_type_from_extensions(
      model_path=model_path,
      task=task,
      config_dict=config_dict,
      export_config=export_config,
  )


def get_image_gen_model_cls(
    model_type: str,
    export_config: exportable_module.ExportableModuleConfig | None = None,
):
  """Returns the ImageGen wrapper class by `model_type` or pipeline class name.

  See `model_ext/bonsai_flux2/extension.py` and
  `model_ext/flux2_klein/extension.py` for built-in examples.

  Args:
    model_type: ImageGen `model_type` or Diffusers `_class_name` string.
    export_config: Optional `ExportableModuleConfig`.

  Returns:
    The ImageGen model wrapper class for `model_type`.

  Raises:
    ValueError: If no extension provides an ImageGen model class for
      `model_type`.
  """
  ext = extension_lib.resolve_model_extension(model_type, export_config)
  if ext is not None:
    cls = ext.get_image_gen_model_cls(model_type)
    if cls is not None:
      return cls
  raise ValueError(f'Unsupported ImageGen model type: {model_type}')


def get_any2any_model_cls(
    model_type: str,
    export_config: exportable_module.ExportableModuleConfig | None = None,
):
  """Returns the Any2Any multimodal model wrapper class for `model_type`.

  See `model_ext/minicpm_o/extension.py` for a built-in example.

  Args:
    model_type: Any2Any `model_type` string.
    export_config: Optional `ExportableModuleConfig`.

  Returns:
    The Any2Any model wrapper class for `model_type`.

  Raises:
    ValueError: If no extension provides an Any2Any model class for
      `model_type`.
  """
  ext = extension_lib.resolve_model_extension(model_type, export_config)
  if ext is not None:
    cls = ext.get_any2any_model_cls(model_type)
    if cls is not None:
      return cls
  raise ValueError(f'Unsupported Any2Any model type: {model_type}')
