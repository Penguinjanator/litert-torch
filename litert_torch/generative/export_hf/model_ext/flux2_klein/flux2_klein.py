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
"""FLUX.2-klein sharded image generation model export for LiteRT."""

from __future__ import annotations

import os

import litert_torch
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.model_ext.bonsai_flux2 import bonsai_flux2
from litert_torch.generative.export_hf.model_ext.flux2_klein import modeling
import numpy as np
import torch
from torch import nn

try:
  from litert_lm_builder.runtime.proto import image_gen_metadata_pb2
  from litert_lm_builder.runtime.proto import image_gen_model_type_pb2
except ImportError:
  image_gen_metadata_pb2 = None
  image_gen_model_type_pb2 = None


def _split_into_shards(
    items: list[nn.Module], num_shards: int
) -> list[nn.ModuleList]:
  """Splits a list of modules into `num_shards` contiguous non-empty slices."""
  n = len(items)
  if n < num_shards:
    raise ValueError(
        f"Cannot split {n} blocks into {num_shards} shards (need >="
        f" {num_shards})."
    )
  shards: list[nn.ModuleList] = []
  for i in range(num_shards):
    start = (i * n) // num_shards
    end = ((i + 1) * n) // num_shards
    shards.append(nn.ModuleList(items[start:end]))
  return shards


class Flux2Klein(bonsai_flux2.BonsaiFlux2):
  """FLUX.2-klein multi-shard text-to-image model."""

  _TEXT_EMBED_FILENAME = "qwen_embed_fp16.bin"
  _NUM_TE_SHARDS = 3
  _NUM_DIT_DOUBLE_SHARDS = 2
  _NUM_DIT_SINGLE_SHARDS = 4

  def export_text_encoder_shards(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> dict[str, str]:
    """Exports the Qwen3 embedding table (fp16) and 3 text encoder shards."""
    print("Exporting FLUX.2-klein sharded text encoder...")
    qwen_model = self._load_qwen_model(export_config)
    setattr(qwen_model.config, "_attn_implementation", "eager")

    num_layers = getattr(qwen_model.config, "num_hidden_layers", 36)
    layers = tuple(k for k in self._text_encoder_layers if k <= num_layers)
    if len(layers) != self._NUM_TE_SHARDS:
      if num_layers < self._NUM_TE_SHARDS:
        raise ValueError(
            f"FLUX.2-klein requires at least {self._NUM_TE_SHARDS} Qwen3"
            f" layers, got {num_layers}."
        )
      step = max(1, num_layers // (self._NUM_TE_SHARDS + 1))
      layers = (step, 2 * step, 3 * step)
      if layers[-1] > num_layers:
        layers = (1, 2, 3)
    self._text_encoder_layers = layers

    # Export raw fp16 token embedding table [vocab_size, hidden_size].
    embed_weight = (
        qwen_model.embed_tokens.weight.detach()
        .cpu()
        .to(torch.float32)
        .numpy()
        .astype(np.float16)
    )
    embed_bin_path = os.path.join(output_dir, self._TEXT_EMBED_FILENAME)
    embed_weight.tofile(embed_bin_path)

    seq_len = self._get_max_seq_len(export_config)
    hidden_size = int(getattr(qwen_model.config, "hidden_size", 2560))
    num_heads = int(getattr(qwen_model.config, "num_attention_heads", 32))
    head_dim = int(
        getattr(qwen_model.config, "head_dim", hidden_size // num_heads)
    )

    sample_inputs = (
        torch.zeros(1, seq_len, hidden_size, dtype=torch.float32),
        torch.zeros(1, num_heads, seq_len, seq_len, dtype=torch.float32),
        torch.ones(1, seq_len, head_dim, dtype=torch.float32),
        torch.zeros(1, seq_len, head_dim, dtype=torch.float32),
    )

    all_layers = list(qwen_model.layers)
    boundaries = [0] + list(layers)
    artifacts: dict[str, str] = {self._TEXT_EMBED_FILENAME: embed_bin_path}

    for shard_idx in range(self._NUM_TE_SHARDS):
      start, end = boundaries[shard_idx], boundaries[shard_idx + 1]
      shard_layers = nn.ModuleList(all_layers[start:end])
      for layer in shard_layers:
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
          setattr(layer.self_attn.config, "_attn_implementation", "eager")
      norm = (
          getattr(qwen_model, "norm", None) if end == num_layers else None
      )
      shard_mod = modeling.KleinTextEncoderShard(
          shard_layers, norm=norm
      ).eval()
      fp32_path = os.path.join(
          output_dir, f"text_encoder_shard_{shard_idx}_fp32.tflite"
      )
      litert_torch.convert(shard_mod, sample_inputs).export(fp32_path)
      if self._should_quantize(export_config):
        int4_path = os.path.join(
            output_dir, f"text_encoder_shard_{shard_idx}_int4.tflite"
        )
        bonsai_flux2.quantize_text_encoder(fp32_path, int4_path)
        if not export_config.keep_temporary_files and os.path.exists(fp32_path):
          os.remove(fp32_path)
        artifacts[f"text_encoder_{shard_idx}"] = int4_path
      else:
        artifacts[f"text_encoder_{shard_idx}"] = fp32_path

    return artifacts

  def export_dit_shards(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> dict[str, str]:
    """Exports the 8 FLUX.2-klein DiT shards (initial, 2x double, 4x single, final)."""
    print("Exporting FLUX.2-klein sharded DiT transformer...")
    image_size = export_config.t2i_output_image_size
    if image_size <= 0 or image_size % 16 != 0:
      raise ValueError(
          "t2i_output_image_size must be a positive multiple of 16, got"
          f" {image_size}."
      )
    subfolder = (
        "transformer"
        if os.path.exists(os.path.join(self.model_dir, "transformer"))
        else None
    )
    dit = modeling.Flux2Transformer2DModel.from_pretrained(
        self.model_dir,
        subfolder=subfolder,
        torch_dtype=torch.float32,
        use_random_weights=export_config.use_random_weights,
    ).eval()
    self._dit_config = vars(dit.config)

    grid = image_size // 16
    img_tokens = grid * grid
    seq_len = self._get_max_seq_len(export_config)
    joint_tokens = seq_len + img_tokens
    in_channels = int(dit.config.in_channels)
    joint_attention_dim = int(dit.config.joint_attention_dim)
    head_dim = int(dit.config.attention_head_dim)
    num_heads = int(dit.config.num_attention_heads)
    inner_dim = num_heads * head_dim

    artifacts: dict[str, str] = {}

    def _convert_and_maybe_quantize(
        mod: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        stem: str,
        quantize: bool,
    ) -> str:
      fp32_path = os.path.join(output_dir, f"{stem}_fp32.tflite")
      litert_torch.convert(mod.eval(), inputs).export(fp32_path)
      if quantize and self._should_quantize(export_config):
        int4_path = os.path.join(output_dir, f"{stem}_int4b32.tflite")
        bonsai_flux2.quantize_dit(fp32_path, int4_path)
        if not export_config.keep_temporary_files and os.path.exists(fp32_path):
          os.remove(fp32_path)
        return int4_path
      return fp32_path

    # 1. Initial DiT stage (kept fp32 or channelwise int8 if quantized)
    init_inputs = (
        torch.zeros(1, img_tokens, in_channels, dtype=torch.float32),
        torch.zeros(1, seq_len, joint_attention_dim, dtype=torch.float32),
        torch.zeros(1, dtype=torch.float32),
    )
    artifacts["dit_initial"] = _convert_and_maybe_quantize(
        modeling.KleinDiTInitialWrapper(dit),
        init_inputs,
        "dit_initial",
        quantize=True,
    )

    # 2. Double-stream block shards (2 shards)
    double_shards = _split_into_shards(
        list(dit.transformer_blocks), self._NUM_DIT_DOUBLE_SHARDS
    )
    double_inputs = (
        torch.zeros(1, img_tokens, inner_dim, dtype=torch.float32),
        torch.zeros(1, seq_len, inner_dim, dtype=torch.float32),
        torch.ones(1, joint_tokens, 1, head_dim, dtype=torch.float32),
        torch.zeros(1, joint_tokens, 1, head_dim, dtype=torch.float32),
        torch.zeros(1, 6 * inner_dim, dtype=torch.float32),
        torch.zeros(1, 6 * inner_dim, dtype=torch.float32),
    )
    for idx, shard_blocks in enumerate(double_shards):
      artifacts[f"dit_double_block_{idx}"] = _convert_and_maybe_quantize(
          modeling.KleinDiTDoubleBlockWrapper(shard_blocks),
          double_inputs,
          f"dit_double_block_{idx}",
          quantize=True,
      )

    # 3. Single-stream block shards (4 shards)
    single_shards = _split_into_shards(
        list(dit.single_transformer_blocks), self._NUM_DIT_SINGLE_SHARDS
    )
    single_inputs = (
        torch.zeros(1, joint_tokens, inner_dim, dtype=torch.float32),
        torch.ones(1, joint_tokens, 1, head_dim, dtype=torch.float32),
        torch.zeros(1, joint_tokens, 1, head_dim, dtype=torch.float32),
        torch.zeros(1, 3 * inner_dim, dtype=torch.float32),
    )
    for idx, shard_blocks in enumerate(single_shards):
      artifacts[f"dit_single_block_{idx}"] = _convert_and_maybe_quantize(
          modeling.KleinDiTSingleBlockWrapper(shard_blocks),
          single_inputs,
          f"dit_single_block_{idx}",
          quantize=True,
      )

    # 4. Final DiT stage
    final_inputs = (
        torch.zeros(1, joint_tokens, inner_dim, dtype=torch.float32),
        torch.zeros(1, inner_dim, dtype=torch.float32),
    )
    artifacts["dit_final"] = _convert_and_maybe_quantize(
        modeling.KleinDiTFinalWrapper(dit, num_txt_tokens=seq_len),
        final_inputs,
        "dit_final",
        quantize=True,
    )

    return artifacts

  def export(
      self, export_config: exportable_module_config.ExportableModuleConfig
  ) -> dict[str, str]:
    """Exports all FLUX.2-klein submodules and shards to LiteRT files."""
    output_dir = (
        (
            export_config.work_dir
            if export_config.bundle_litert_lm
            else export_config.output_dir
        )
        or export_config.output_dir
        or export_config.work_dir
    )
    if not output_dir:
      raise ValueError("Either output_dir or work_dir must be specified.")
    os.makedirs(output_dir, exist_ok=True)
    artifacts: dict[str, str] = {}

    targets = set(export_config.extra_kwargs.get("targets", ["all"]))
    export_all = "all" in targets

    if export_all or "text_encoder" in targets:
      artifacts.update(
          self.export_text_encoder_shards(output_dir, export_config)
      )
    if export_all or "dit" in targets:
      artifacts.update(self.export_dit_shards(output_dir, export_config))
    if export_all or "vae_decoder" in targets:
      artifacts["vae_decoder"] = self.export_vae_decoder(
          output_dir, export_config
      )

    return artifacts

  def get_image_gen_metadata(
      self, export_config: exportable_module_config.ExportableModuleConfig
  ) -> image_gen_metadata_pb2.ImageGenMetadata:
    """Builds ImageGenMetadata proto for the exported FLUX.2-klein model."""
    if image_gen_metadata_pb2 is None or image_gen_model_type_pb2 is None:
      raise ImportError(
          "image_gen_metadata_pb2 and image_gen_model_type_pb2 are required to"
          " build ImageGenMetadata; please upgrade litert-lm-builder."
      )
    klein_proto = image_gen_model_type_pb2.Flux2Klein(
        flux2_params=self._build_flux2_params_proto(export_config),
        text_embed_table_key=self._TEXT_EMBED_FILENAME,
    )
    return image_gen_metadata_pb2.ImageGenMetadata(
        image_gen_model_type=image_gen_model_type_pb2.ImageGenModelType(
            flux2_klein=klein_proto
        ),
    )
