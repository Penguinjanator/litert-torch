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
"""PyTorch modeling classes and sharded wrappers for FLUX.2-klein."""

from __future__ import annotations

from litert_torch.generative.export_hf.model_ext.bonsai_flux2 import modeling_flux2
import torch
from torch import nn

Flux2Transformer2DModel = modeling_flux2.Flux2Transformer2DModel
AutoencoderKLFlux2 = modeling_flux2.AutoencoderKLFlux2


class KleinTextEncoderShard(nn.Module):
  """Wraps a contiguous slice of Qwen3 decoder layers for sharded export."""

  def __init__(self, layers: nn.ModuleList, norm: nn.Module | None = None):
    super().__init__()
    self.layers = layers
    self.norm = norm

  def forward(
      self,
      hidden_states: torch.Tensor,
      attention_mask: torch.Tensor,
      cos: torch.Tensor,
      sin: torch.Tensor,
  ) -> torch.Tensor:
    for layer in self.layers:
      layer_out = layer(
          hidden_states,
          attention_mask=attention_mask,
          position_embeddings=(cos, sin),
          use_cache=False,
      )
      hidden_states = (
          layer_out[0] if isinstance(layer_out, tuple) else layer_out
      )
    if self.norm is not None:
      hidden_states = self.norm(hidden_states)
    return hidden_states


class KleinDiTInitialWrapper(nn.Module):
  """Initial DiT stage: x_embedder, context_embedder, time_guidance_embed, modulation."""

  def __init__(self, dit: Flux2Transformer2DModel):
    super().__init__()
    self.x_embedder = dit.x_embedder
    self.context_embedder = dit.context_embedder
    self.time_guidance_embed = dit.time_guidance_embed
    self.double_stream_modulation_img = dit.double_stream_modulation_img
    self.double_stream_modulation_txt = dit.double_stream_modulation_txt
    self.single_stream_modulation = dit.single_stream_modulation

  def forward(
      self,
      hidden_states: torch.Tensor,
      encoder_hidden_states: torch.Tensor,
      timestep: torch.Tensor,
  ) -> tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
  ]:
    image = self.x_embedder(hidden_states)
    text = self.context_embedder(encoder_hidden_states)
    temb = self.time_guidance_embed(timestep * 1000.0, None)
    mod_img = self.double_stream_modulation_img(temb)
    mod_txt = self.double_stream_modulation_txt(temb)
    mod_single = self.single_stream_modulation(temb)
    return image, text, mod_img, mod_txt, mod_single, temb


class KleinDiTDoubleBlockWrapper(nn.Module):
  """Wraps a contiguous slice of Flux2TransformerBlock double-stream blocks."""

  def __init__(self, blocks: nn.ModuleList):
    super().__init__()
    self.blocks = blocks

  def forward(
      self,
      image: torch.Tensor,
      text: torch.Tensor,
      cos: torch.Tensor,
      sin: torch.Tensor,
      mod_img: torch.Tensor,
      mod_txt: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    for block in self.blocks:
      text, image = block(
          hidden_states=image,
          encoder_hidden_states=text,
          temb_mod_img=mod_img,
          temb_mod_txt=mod_txt,
          image_rotary_emb=(cos, sin),
      )
    return image, text


class KleinDiTSingleBlockWrapper(nn.Module):
  """Wraps a contiguous slice of Flux2SingleTransformerBlock single-stream blocks."""

  def __init__(self, blocks: nn.ModuleList):
    super().__init__()
    self.blocks = blocks

  def forward(
      self,
      joint: torch.Tensor,
      cos: torch.Tensor,
      sin: torch.Tensor,
      mod_single: torch.Tensor,
  ) -> torch.Tensor:
    for block in self.blocks:
      joint = block(
          hidden_states=joint,
          encoder_hidden_states=None,
          temb_mod=mod_single,
          image_rotary_emb=(cos, sin),
      )
    return joint


class KleinDiTFinalWrapper(nn.Module):
  """Final DiT stage: slices image tokens, applies norm_out and proj_out."""

  def __init__(self, dit: Flux2Transformer2DModel, num_txt_tokens: int):
    super().__init__()
    self.norm_out = dit.norm_out
    self.proj_out = dit.proj_out
    self.num_txt_tokens = num_txt_tokens

  def forward(
      self,
      joint: torch.Tensor,
      temb: torch.Tensor,
  ) -> torch.Tensor:
    image = joint[:, self.num_txt_tokens :, ...]
    image = self.norm_out(image, temb)
    return self.proj_out(image)
