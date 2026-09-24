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
"""Exportable modules for Gemma4 audio encoder and adapter."""

from litert_torch.generative.export_hf.core import exportable_module as exportable_module_base
import torch


class LiteRTExportableModuleForGemma4AudioEncoder(
    exportable_module_base.ExportableModuleBase
):
  """Exportable module for Gemma4 audio encoder."""

  def __init__(self, model: torch.nn.Module, export_config):
    super().__init__(export_config)
    self.model = model

  def forward(
      self,
      src_inputs,
      mask,
  ):
    input_features = src_inputs
    attention_mask = mask.squeeze(1)
    audio_model = self.model.model.audio_tower
    audio_output = audio_model(input_features, attention_mask, return_dict=True)
    ret = {
        'features': audio_output.last_hidden_state,
        'mask': audio_output.attention_mask,
    }
    return ret

  def get_sample_inputs(
      self, model_config, **kwargs
  ) -> dict[str, tuple[dict[str, torch.Tensor], dict[str, torch.export.Dim]]]:
    """Returns the sample inputs for the model."""
    del model_config
    feature_extractor = kwargs.get('feature_extractor', None)
    if feature_extractor is None:
      raise ValueError(
          'Feature extractor is required for Exporting Gemma4 audio encoder.'
      )
    input_features = torch.zeros((1, 816, 128), dtype=torch.float32)
    input_features_mask = torch.ones((1, 1, 816), dtype=torch.bool)
    inputs = {
        'src_inputs': input_features,
        'mask': input_features_mask,
    }
    return {'audio_204': (inputs, {})}


class LiteRTExportableModuleForGemma4AudioAdapter(
    exportable_module_base.ExportableModuleBase
):
  """Exportable module for Gemma4 audio adapter."""

  def __init__(self, model: torch.nn.Module, export_config, tokenizer):
    super().__init__(export_config)
    self.model = model
    self.tokenizer = tokenizer

  def forward(
      self,
      features,
      mask,
  ):
    pooler_output = self.model.model.embed_audio(inputs_embeds=features)
    mask = mask.unsqueeze(-1)
    pooler_output = pooler_output * mask
    return {'features': pooler_output}

  def get_sample_inputs(
      self, model_config, **kwargs
  ) -> dict[str, tuple[dict[str, torch.Tensor], dict[str, torch.export.Dim]]]:
    """Returns the sample inputs for the model."""
    del model_config, kwargs
    features = torch.zeros((1, 204, 1536), dtype=torch.float32)
    mask = torch.ones((1, 204), dtype=torch.bool)
    inputs = {'features': features, 'mask': mask}
    return {'audio_adapter_204': (inputs, {})}


class LiteRTExportableModuleForGemma4EndOfAudio(
    exportable_module_base.ExportableModuleBase
):
  """Exportable module for Gemma4 end of audio token."""

  def __init__(self, model: torch.nn.Module, export_config, tokenizer):
    super().__init__(export_config)
    self.model = model
    self.tokenizer = tokenizer

  def forward(self):
    return {
        'eoa_embedding': self.model.get_input_embeddings()(
            torch.tensor(
                [
                    self.tokenizer.encode(
                        self.tokenizer.special_tokens_map['eoa_token'],
                        add_special_tokens=False,
                    )
                ],
                dtype=torch.int32,
            )
        )
    }

  def get_sample_inputs(
      self, model_config, **kwargs
  ) -> dict[str, tuple[dict[str, torch.Tensor], dict[str, torch.export.Dim]]]:
    """Returns the sample inputs for the model."""
    del model_config, kwargs
    return {'eoa': (dict(), {})}
