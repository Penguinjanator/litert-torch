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
"""Reusable contract-test and fast-inner-loop verification harness for `ModelExtension`.

Provides:
1. `ModelExtensionTestMixin`: A `unittest.TestCase`-compatible mixin that
   verifies a built-in or out-of-tree `ModelExtension` satisfies the core hook
   contracts expected by `export_hf` without requiring a multi-gigabyte
   checkpoint:
   - `model_types` contains non-empty string identifiers.
   - `update_export_config` returns an `ExportableModuleConfig`.
   - `patch_context` and `patch_model` enter and exit cleanly across repeated
     invocations (re-entrancy and state restoration).
   - `get_prefill_decode_exportables`, `get_vision_exportables`,
     `get_audio_exportables`, and `get_additional_exportables` return
     well-formed specs/containers.
   - `build_llm_metadata` returns a valid `LlmMetadata` protobuf message.
2. Fast-inner-loop verification helpers (available both as standalone functions
   and as methods on `ModelExtensionTestMixin`) that let coding agents and
   developers validate runtime patches and `torch.export.export` graph capture
   on tiny 2-layer synthetic models in under 2 seconds on CPU:
   - `assert_patch_numerical_parity`: Runs eager inference before and during
     `patch_model`, checks `torch.testing.assert_close` across all output
     tensors, and verifies submodule classes and forward functions are cleanly
     restored upon exiting `patch_model`.
   - `assert_tiny_prefill_decode_torch_export`: Wraps a tiny causal LM in the
     resolved prefill and decode `ExportableModule` classes (and `RoPEEmbedder`
     when `externalize_rope=True` / `split_cache=True`) and runs
     `torch.export.export` across all sample signatures.

See `model_ext/extension_test.py` and `experimental/users/weiyiw/kanana2_ext`
for concrete usage examples.
"""

from collections.abc import Mapping, Sequence
import contextlib
import copy
import dataclasses
import gc
import types
from typing import Any, cast

from litert_torch.generative.export_hf.core import attention as _
from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import patches as _
from litert_torch.generative.export_hf.core.external_emb import exportable_module as external_emb_module
from litert_torch.generative.export_hf.core.external_rope import exportable_module as external_rope_module
from litert_torch.generative.export_hf.core.external_rope import preprocess_model as external_rope_preprocess_model
from litert_torch.generative.export_hf.core.split_cache import attention as _
from litert_torch.generative.export_hf.core.split_cache import exportable_module as split_cache_module
from litert_torch.generative.export_hf.model_ext import exportables as model_ext_exportables
from litert_torch.generative.export_hf.model_ext import extension
from litert_torch.generative.export_hf.model_ext import patches as model_ext_patches
import torch
import torch.utils._pytree as pytree
import transformers

from litert_lm_builder.runtime.proto import llm_metadata_pb2


@contextlib.contextmanager
def _patch_builtin_tuple_for_export():
  """Temporarily injects a `.to()` method into Python's built-in `tuple`."""
  tuple_dict = gc.get_referents(tuple.__dict__)[0]
  had_original = 'to' in tuple_dict
  original_to = tuple_dict.get('to')

  def tuple_to(self, *args, **kwargs):
    return tuple(
        item.to(*args, **kwargs) if hasattr(item, 'to') else item
        for item in self
    )

  tuple_dict['to'] = tuple_to
  try:
    yield
  finally:
    if had_original:
      tuple_dict['to'] = original_to
    else:
      del tuple_dict['to']


def _extract_tensors(tree: Any) -> list[torch.Tensor]:
  """Extracts all `torch.Tensor` leaves from an arbitrary nested output structure."""
  if isinstance(tree, torch.Tensor):
    return [tree]
  if isinstance(tree, Mapping):
    out: list[torch.Tensor] = []
    for k in sorted(tree.keys(), key=str):
      out.extend(_extract_tensors(tree[k]))
    return out
  if isinstance(tree, (tuple, list)):
    out = []
    for item in tree:
      out.extend(_extract_tensors(item))
    return out
  leaves, _ = pytree.tree_flatten(tree)
  return [x for x in leaves if isinstance(x, torch.Tensor)]


def _snapshot_module_hierarchy(
    root: torch.nn.Module,
) -> dict[str, tuple[type[Any], Any]]:
  """Records `(type(module), type(module).forward)` for every named submodule."""
  return {
      name: (type(submod), getattr(type(submod), 'forward', None))
      for name, submod in root.named_modules(remove_duplicate=False)
  }


def assert_patch_numerical_parity(
    ext: extension.ModelExtension,
    model: torch.nn.Module,
    sample_args: Sequence[Any] = (),
    sample_kwargs: Mapping[str, Any] | None = None,
    export_config: (
        exportable_module_config.ExportableModuleConfig | None
    ) = None,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    check_restoration: bool = True,
) -> tuple[Any, Any]:
  """Verifies numerical parity before/during `patch_model` and state restoration.

  Runs `model(*sample_args, **sample_kwargs)` in `torch.no_grad()` mode before
  entering `ext.patch_model(model, export_config)`, runs it again inside the
  patch context, asserts all output tensors match within `(atol, rtol)`, and
  verifies that submodule classes and class `.forward` methods are restored
  after exiting the context.

  Args:
    ext: `ModelExtension` instance under test.
    model: Eager `torch.nn.Module` (e.g. a tiny 2-layer causal LM or block).
    sample_args: Positional arguments forwarded to `model`.
    sample_kwargs: Keyword arguments forwarded to `model`.
    export_config: Optional `ExportableModuleConfig` passed to `patch_model`.
    atol: Absolute tolerance for `torch.testing.assert_close`.
    rtol: Relative tolerance for `torch.testing.assert_close`.
    check_restoration: Whether to assert submodule types and `.forward`
      attributes are restored after exiting `patch_model`.

  Returns:
    A 2-tuple `(baseline_output, patched_output)`.

  Raises:
    AssertionError: If outputs diverge, no tensors are returned, or module
      hierarchy restoration is incomplete.
  """
  if export_config is None:
    export_config = exportable_module_config.ExportableModuleConfig(
        model='dummy',
        externalize_embedder=True,
        model_extension=ext,
    )
  elif getattr(export_config, 'model_extension', None) is None:
    export_config = dataclasses.replace(export_config, model_extension=ext)
  kwargs = dict(sample_kwargs or {})
  model.eval()
  model_type = getattr(getattr(model, 'config', None), 'model_type', None) or (
      ext.model_types[0] if ext.model_types else ''
  )
  before_snapshot = _snapshot_module_hierarchy(model)

  with torch.no_grad():
    baseline_out = model(*copy.deepcopy(sample_args), **copy.deepcopy(kwargs))
    with model_ext_patches.get_patch_context(model_type, export_config):
      with model_ext_patches.patch_model(model, model_type, export_config):
        patched_out = model(
            *copy.deepcopy(sample_args), **copy.deepcopy(kwargs)
        )

  if check_restoration:
    after_snapshot = _snapshot_module_hierarchy(model)
    if before_snapshot != after_snapshot:
      raise AssertionError(
          'Module hierarchy or class `.forward` methods were not cleanly'
          f' restored after exiting `patch_model`.\nBefore: {before_snapshot}\n'
          f'After:  {after_snapshot}'
      )

  baseline_tensors = _extract_tensors(baseline_out)
  patched_tensors = _extract_tensors(patched_out)
  if not baseline_tensors:
    raise AssertionError(
        'Model forward returned no `torch.Tensor` outputs to compare.'
    )
  if len(baseline_tensors) != len(patched_tensors):
    raise AssertionError(
        f'Baseline returned {len(baseline_tensors)} tensor(s), but patched'
        f' model returned {len(patched_tensors)} tensor(s).'
    )
  for idx, (expected, actual) in enumerate(
      zip(baseline_tensors, patched_tensors, strict=True)
  ):
    torch.testing.assert_close(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
        msg=lambda m, i=idx: f'Tensor output #{i} diverged after patch: {m}',
    )
  return baseline_out, patched_out


def _resolve_prefill_decode_cls(
    ext: extension.ModelExtension,
    model_config: transformers.PretrainedConfig,
    export_config: exportable_module_config.ExportableModuleConfig,
) -> tuple[Any, Any]:
  """Selects `(prefill_cls, decode_cls)` matching `core/export_lib.py`."""
  custom = ext.get_prefill_decode_exportables(model_config, export_config)
  if custom is not None:
    return custom
  from_registry = model_ext_exportables.get_prefill_decode_exportables(
      model_config, export_config
  )
  if from_registry is not None:
    return from_registry
  if export_config.split_cache:
    return (
        split_cache_module.LiteRTSplitCacheExportableModuleForDecoderOnlyLMPrefill,
        split_cache_module.LiteRTSplitCacheExportableModuleForDecoderOnlyLMGenerate,
    )
  if export_config.externalize_embedder:
    return (
        external_emb_module.LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder,
        external_emb_module.LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder,
    )
  return (
      exportable_module.LiteRTExportableModuleForDecoderOnlyLMPrefill,
      exportable_module.LiteRTExportableModuleForDecoderOnlyLMGenerate,
  )


def assert_tiny_prefill_decode_torch_export(
    ext: extension.ModelExtension,
    model: torch.nn.Module,
    model_config: transformers.PretrainedConfig | None = None,
    export_config: (
        exportable_module_config.ExportableModuleConfig | None
    ) = None,
) -> dict[str, torch.export.ExportedProgram]:
  """Runs `torch.export.export` on prefill, decode (and RoPE) for a tiny model.

  Mirrors the tracing setup in `export_lib.export_text_prefill_decode_model`
  and `export_lib.export_auxiliary_model` (`RoPEEmbedder`) so extensions can
  verify Dynamo / FX graph capture in `<2s` on CPU without loading a full
  checkpoint or running TFLite flatbuffer serialization.

  Args:
    ext: `ModelExtension` instance under test.
    model: Tiny causal LM (`PreTrainedModel` or compatible `nn.Module`). A
      `copy.deepcopy` is traced so the caller's `model` is not mutated by
      `inject_rotary_position_embedding` or `set_attn_implementation`.
    model_config: Optional `PretrainedConfig` (defaults to `model.config`).
    export_config: Optional `ExportableModuleConfig` (defaults to tiny
      `prefill_lengths=[8]`, `cache_length=16`, `externalize_embedder=True`).

  Returns:
    Dictionary mapping signature names (e.g. `'prefill_8'`, `'decode'`,
    `'prefill_rope_8'`, `'decode_rope'`) to `torch.export.ExportedProgram`s.
  """
  traced_model: Any = copy.deepcopy(model).eval()
  if hasattr(traced_model, 'get_text_model'):
    traced_model = traced_model.get_text_model()

  if model_config is None:
    model_config = getattr(traced_model, 'config', None)
  if model_config is None:
    raise ValueError(
        '`model_config` must be provided when `model` has no `.config`.'
    )

  if export_config is None:
    export_config = exportable_module_config.ExportableModuleConfig(
        model='dummy',
        prefill_lengths=[8],
        cache_length=16,
        cache_lengths=[16],
        externalize_embedder=True,
        model_extension=ext,
    )
  elif getattr(export_config, 'model_extension', None) is None:
    export_config = dataclasses.replace(export_config, model_extension=ext)

  if (
      export_config.split_cache
      and export_config.cache_implementation == 'LiteRTLMCache'
  ):
    export_config = dataclasses.replace(
        export_config, cache_implementation=None
    )

  export_config = ext.update_export_config(export_config, model_config)
  text_model_config = getattr(model_config, 'text_config', model_config)

  if hasattr(model_config, 'allow_global_per_layer_attribute_access'):
    setattr(model_config, 'allow_global_per_layer_attribute_access', True)
  if hasattr(text_model_config, 'allow_global_per_layer_attribute_access'):
    setattr(text_model_config, 'allow_global_per_layer_attribute_access', True)

  fake_source: Any = types.SimpleNamespace(
      model=traced_model,
      model_config=model_config,
      text_model_config=text_model_config,
      tokenizer=None,
      image_processor=None,
      feature_extractor=None,
      weights_loader=None,
  )

  model_type = (
      getattr(text_model_config, 'model_type', None)
      or getattr(model_config, 'model_type', None)
      or (ext.model_types[0] if ext.model_types else '')
  )

  exported_programs: dict[str, torch.export.ExportedProgram] = {}
  with (
      torch.no_grad(),
      _patch_builtin_tuple_for_export(),
      model_ext_patches.get_patch_context(model_type, export_config),
  ):
    with model_ext_patches.patch_model(traced_model, model_type, export_config):
      if export_config.externalize_rope:
        traced_model = (
            external_rope_preprocess_model.inject_rotary_position_embedding(
                traced_model
            )
        )
      if export_config.split_cache:
        if hasattr(traced_model, 'set_attn_implementation'):
          traced_model.set_attn_implementation('lrt_split_cache_attention')
        cast(Any, traced_model).config._attn_implementation = (  # pylint: disable=protected-access
            'lrt_split_cache_attention'
        )
      else:
        if hasattr(traced_model, 'set_attn_implementation'):
          traced_model.set_attn_implementation('lrt_transposed_attention')
        cast(Any, traced_model).config._attn_implementation = (  # pylint: disable=protected-access
            'lrt_transposed_attention'
        )

      cache_lengths = export_config.cache_lengths or [
          export_config.cache_length
      ]
      for cache_len in cache_lengths:
        iter_config = dataclasses.replace(export_config, cache_length=cache_len)
        prefill_cls, decode_cls = _resolve_prefill_decode_cls(
            ext, model_config, iter_config
        )
        prefill_module = prefill_cls(traced_model, iter_config, fake_source)
        decode_module = decode_cls(traced_model, iter_config, fake_source)
        for mod in (prefill_module, decode_module):
          mod.eval()
          for sig_name, (
              inputs,
              dynamic_shapes,
          ) in mod.get_sample_inputs(text_model_config).items():
            key = (
                f'{sig_name}_cache_{cache_len}'
                if len(cache_lengths) > 1
                else sig_name
            )
            exported_programs[key] = torch.export.export(
                mod,
                args=(),
                kwargs=inputs,
                dynamic_shapes=dynamic_shapes or None,
                strict=False,
            )

      if export_config.externalize_rope:
        rope_module = external_rope_module.RoPEEmbedder(traced_model).eval()
        for sig_name, (
            inputs,
            dynamic_shapes,
        ) in rope_module.get_sample_inputs(
            text_model_config, export_config
        ).items():
          exported_programs[sig_name] = torch.export.export(
              rope_module,
              args=(),
              kwargs=inputs,
              dynamic_shapes=dynamic_shapes or None,
              strict=False,
          )

  return exported_programs


class ModelExtensionTestMixin:
  """Mixin that runs standard contract tests against a `ModelExtension`.

  Subclass alongside `unittest.TestCase` (or `absltest.TestCase`), set
  `extension_cls` to the target `ModelExtension` class or instance, and
  optionally override `get_sample_model_config()`, `get_sample_export_config()`,
  or `get_sample_model()` when the extension requires specific config fields or
  module structure. See `model_ext/extension_test.py` for examples.

  Example:
    ```python
    class MyExtensionTest(ModelExtensionTestMixin, absltest.TestCase):
      extension_cls = MyExtension

      def get_sample_model_config(self):
        return transformers.PretrainedConfig(model_type='my_arch')
    ```

  Attributes:
    extension_cls: `ModelExtension` subclass or instance under test.
  """

  extension_cls: (
      type[extension.ModelExtension] | extension.ModelExtension | None
  ) = None

  def get_extension(self) -> extension.ModelExtension:
    """Returns an instantiated `ModelExtension` under test."""
    if self.extension_cls is None:
      raise NotImplementedError(
          'Subclasses of ModelExtensionTestMixin must set `extension_cls` or'
          ' override `get_extension()`.'
      )
    if isinstance(self.extension_cls, type) and issubclass(
        self.extension_cls, extension.ModelExtension
    ):
      return self.extension_cls()
    if isinstance(self.extension_cls, extension.ModelExtension):
      return self.extension_cls
    raise TypeError(f'Invalid extension_cls: {self.extension_cls!r}')

  def get_sample_model_config(self) -> transformers.PretrainedConfig:
    """Returns a tiny synthetic `PretrainedConfig` for exercising hooks."""
    ext = self.get_extension()
    model_type = ext.model_types[0] if ext.model_types else 'test_model'
    return transformers.PretrainedConfig.from_dict({
        'model_type': model_type,
        'num_hidden_layers': 2,
        'hidden_size': 16,
        'num_attention_heads': 2,
        'num_key_value_heads': 2,
        'head_dim': 8,
        'vocab_size': 64,
    })

  def get_sample_export_config(
      self,
  ) -> exportable_module_config.ExportableModuleConfig:
    """Returns a minimal `ExportableModuleConfig` for exercising hooks."""
    return exportable_module_config.ExportableModuleConfig(
        model='dummy',
        externalize_embedder=True,
    )

  def get_sample_model(self) -> torch.nn.Module:
    """Returns a tiny `torch.nn.Module` passed to `patch_model`."""
    return torch.nn.Linear(16, 16)

  def assert_patch_numerical_parity(
      self,
      model: torch.nn.Module | None = None,
      sample_args: Sequence[Any] = (),
      sample_kwargs: Mapping[str, Any] | None = None,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
      atol: float = 1e-4,
      rtol: float = 1e-4,
      check_restoration: bool = True,
  ) -> tuple[Any, Any]:
    """Verifies eager numerical parity before/during `patch_model` and state restoration."""
    target_model = model if model is not None else self.get_sample_model()
    target_export_cfg = (
        export_config
        if export_config is not None
        else self.get_sample_export_config()
    )
    return assert_patch_numerical_parity(
        ext=self.get_extension(),
        model=target_model,
        sample_args=sample_args,
        sample_kwargs=sample_kwargs,
        export_config=target_export_cfg,
        atol=atol,
        rtol=rtol,
        check_restoration=check_restoration,
    )

  def assert_tiny_prefill_decode_torch_export(
      self,
      model: torch.nn.Module | None = None,
      model_config: transformers.PretrainedConfig | None = None,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
  ) -> dict[str, torch.export.ExportedProgram]:
    """Verifies `torch.export.export` succeeds on prefill/decode for a tiny causal LM."""
    target_model = model if model is not None else self.get_sample_model()
    if model_config is not None:
      target_model_cfg = model_config
    else:
      target_model_cfg = (
          getattr(target_model, 'config', None)
          or self.get_sample_model_config()
      )
    return assert_tiny_prefill_decode_torch_export(
        ext=self.get_extension(),
        model=target_model,
        model_config=target_model_cfg,
        export_config=export_config,
    )

  def test_extension_instance_and_model_types_contract(self):
    """Verifies the extension instantiates and declares valid `model_types`."""
    ext = self.get_extension()
    self.assertIsInstance(ext, extension.ModelExtension)  # pyrefly: ignore[missing-attribute]
    for mt in ext.model_types:
      self.assertIsInstance(mt, str)  # pyrefly: ignore[missing-attribute]
      self.assertTrue(mt)  # pyrefly: ignore[missing-attribute]

  def test_extension_update_export_config_contract(self):
    """Verifies `update_export_config` returns an `ExportableModuleConfig`."""
    ext = self.get_extension()
    model_cfg = self.get_sample_model_config()
    export_cfg = self.get_sample_export_config()
    updated = ext.update_export_config(export_cfg, model_cfg)
    self.assertIsInstance(  # pyrefly: ignore[missing-attribute]
        updated, exportable_module_config.ExportableModuleConfig
    )

  def test_extension_patch_contexts_are_reentrant(self):
    """Verifies `patch_context` and `patch_model` support repeated entry."""
    ext = self.get_extension()
    model_cfg = self.get_sample_model_config()
    export_cfg = ext.update_export_config(
        self.get_sample_export_config(), model_cfg
    )
    sample_model = self.get_sample_model()
    with ext.patch_context():
      with ext.patch_model(sample_model, export_cfg):
        pass
    # Verify a second entry/exit cycle also succeeds cleanly.
    with ext.patch_context():
      with ext.patch_model(sample_model, export_cfg):
        pass

  def test_extension_exportables_contracts(self):
    """Verifies exportable selector hooks return well-shaped specs or dicts."""
    ext = self.get_extension()
    model_cfg = self.get_sample_model_config()
    export_cfg = ext.update_export_config(
        self.get_sample_export_config(), model_cfg
    )

    pd = ext.get_prefill_decode_exportables(model_cfg, export_cfg)
    if pd is not None:
      self.assertIsInstance(pd, tuple)  # pyrefly: ignore[missing-attribute]
      self.assertLen(pd, 2)  # pyrefly: ignore[missing-attribute]

    vis = ext.get_vision_exportables(model_cfg, export_config=export_cfg)
    if vis is not None:
      self.assertLen(tuple(vis), 3)  # pyrefly: ignore[missing-attribute]
      self.assertIsNotNone(tuple(vis)[0])  # pyrefly: ignore[missing-attribute]

    aud = ext.get_audio_exportables(model_cfg, export_config=export_cfg)
    if aud is not None:
      self.assertLen(tuple(aud), 3)  # pyrefly: ignore[missing-attribute]
      self.assertIsNotNone(tuple(aud)[0])  # pyrefly: ignore[missing-attribute]

    add = ext.get_additional_exportables(model_cfg, export_config=export_cfg)
    self.assertIsInstance(add, dict)  # pyrefly: ignore[missing-attribute]

  def test_extension_metadata_hooks_contract(self):
    """Verifies `build_llm_metadata` returns an `LlmMetadata` proto."""
    ext = self.get_extension()
    model_cfg = self.get_sample_model_config()
    export_cfg = ext.update_export_config(
        self.get_sample_export_config(), model_cfg
    )

    fake_source: Any = types.SimpleNamespace(
        model=self.get_sample_model(),
        model_config=model_cfg,
        text_model_config=model_cfg,
        tokenizer=None,
        image_processor=None,
        feature_extractor=None,
    )
    fake_exported: Any = types.SimpleNamespace(
        prefill_decode_model_path=None,
        embedder_model_path=None,
        vision_encoder_model_path=None,
        vision_adapter_model_path=None,
        eoi_model_path=None,
        audio_encoder_model_path=None,
        audio_adapter_model_path=None,
        eoa_model_path=None,
        auxiliary_model_path=None,
        tokenizer_model_path=None,
        additional_model_paths=None,
        litert_lm_model_path=None,
    )
    llm_meta = llm_metadata_pb2.LlmMetadata()
    updated_meta = ext.build_llm_metadata(
        fake_source, export_cfg, fake_exported, llm_meta
    )
    self.assertIsInstance(updated_meta, llm_metadata_pb2.LlmMetadata)  # pyrefly: ignore[missing-attribute]
