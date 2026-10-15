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
"""Kokoro TTS model export to TFLite flatbuffers via LiteRT-Torch."""

import os
import shutil
from typing import Any

import huggingface_hub
import litert_torch
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import quantization
from litert_torch.generative.export_hf.core.speech import tts_model
from litert_torch.generative.export_hf.model_ext.kokoro_tts import istftnet
from litert_torch.generative.export_hf.model_ext.kokoro_tts import model as kokoro_model
from litert_torch.generative.export_hf.model_ext.kokoro_tts import modules
import numpy as np
import torch
from torch import nn

from ai_edge_litert.tools import flatbuffer_utils

# Hugging Face repo of the reference Kokoro-82M checkpoint and voices.
_DEFAULT_REPO_ID = "hexgrad/Kokoro-82M"

# Maximum number of phoneme tokens accepted by the acoustic model.
BUCKET_SIZE = 128

# Acoustic output frames reserved per input phoneme token. This sets
# `max_frames = FRAMES_PER_TOKEN * BUCKET_SIZE`, which is the hard ceiling on
# how much speech a single slice can emit. Measured English speech averages
# ~2.5 frames/token and reaches ~2.7 on long clauses. The runtime derives usable
# token capacity as `max_frames / kFramesPerTokenBudget`, so this must be >=
# that budget for the full bucket to be usable.
FRAMES_PER_TOKEN = 3

# Alignment the LiteRT `CustomOpDispatcher` requires for constant tensors passed
# to custom ops (`LITERT_HOST_MEMORY_BUFFER_ALIGNMENT`).
_CUSTOM_OP_BUFFER_ALIGNMENT = 64


def _align_buffers_64(model_path: str) -> None:
  """Stores every constant buffer 64-byte aligned, rewriting the file in place.

  LiteRT's CustomOpDispatcher wraps each custom-op operand in a host-memory
  tensor buffer that must be `LITERT_HOST_MEMORY_BUFFER_ALIGNMENT` (64) byte
  aligned, so the fused LSTM weights cannot stay where the converter (16-byte
  aligned inline data) or ai_edge_quantizer (16-byte packed offsets) put them.
  This moves all buffer data behind the flatbuffer at 64-byte aligned offsets.

  Args:
    model_path: Path of the .tflite file to rewrite in place.

  Raises:
    RuntimeError: If re-serializing the flatbuffer changes its size.
  """
  align = _CUSTOM_OP_BUFFER_ALIGNMENT

  def _round_up(n: int) -> int:
    return -(-n // align) * align

  with open(model_path, "rb") as f:
    raw = bytearray(f.read())
  model = flatbuffer_utils.read_model_from_bytearray(raw)
  blobs = []
  for buf in model.buffers:
    if buf.data is not None and len(buf.data):
      blobs.append((buf, np.asarray(buf.data, dtype=np.uint8).tobytes()))
      buf.data = None
      # Non-zero placeholders reserve the uint64 fields so the flatbuffer
      # layout (and size) is identical once the real offsets are filled in.
      buf.offset = 1 << 40
      buf.size = len(blobs[-1][1])
  if not blobs:
    return

  header_size = len(flatbuffer_utils.convert_object_to_bytearray(model))
  cur = _round_up(header_size)
  for buf, blob in blobs:
    buf.offset = cur
    cur = _round_up(cur + len(blob))
  header = flatbuffer_utils.convert_object_to_bytearray(model)
  if len(header) != header_size:
    raise RuntimeError("Flatbuffer size changed while aligning buffers.")

  out = bytearray(cur)
  out[:header_size] = header
  for buf, blob in blobs:
    out[buf.offset : buf.offset + len(blob)] = blob
  with open(model_path, "wb") as f:
    f.write(out)


class KokoroAcousticFusedModule(nn.Module):
  """Wraps Kokoro duration, time-alignment, and prosody predictor for FX export."""

  def __init__(self, kmodel: kokoro_model.KModel, max_frames: int = 512):
    super().__init__()
    self.bert = kmodel.bert
    self.bert_encoder = kmodel.bert_encoder
    self.predictor = kmodel.predictor
    self.duration_encoder = kmodel.predictor.text_encoder
    self.duration_proj = kmodel.predictor.duration_proj
    self.shared = kmodel.predictor.shared
    self.text_encoder = kmodel.text_encoder
    self.max_frames = max_frames

  def forward(
      self,
      phoneme_ids: torch.LongTensor,
      speaker_style: torch.FloatTensor,
      phoneme_length: torch.LongTensor,
  ) -> dict[str, torch.Tensor]:
    _, seq_len = phoneme_ids.shape
    time_arange_t = torch.arange(seq_len, device=phoneme_ids.device)
    text_mask = time_arange_t.unsqueeze(0) >= phoneme_length.unsqueeze(1)
    attn_mask = 1 - text_mask.int()

    bert_dur = self.bert(phoneme_ids, attention_mask=attn_mask)
    d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
    d = self.duration_encoder(d_en, speaker_style, phoneme_length, text_mask)

    x = self.predictor.lstm(d, phoneme_length, mask=text_mask)

    duration = torch.sigmoid(self.duration_proj(x)).sum(axis=-1)
    pred_dur = torch.round(duration).clamp(min=1.0).to(torch.float32)
    pred_dur = pred_dur * (1.0 - text_mask.float())

    cumsum = torch.cumsum(pred_dur, dim=-1)
    start_frames = cumsum - pred_dur
    speech_frame_length = torch.clamp(
        cumsum[:, -1:], max=self.max_frames
    ).long()

    time_arange_f = torch.arange(
        self.max_frames, device=phoneme_ids.device
    ).unsqueeze(0)
    time_grid = time_arange_f.unsqueeze(1)
    start_expanded = start_frames.unsqueeze(-1)
    end_expanded = cumsum.unsqueeze(-1)

    aln = ((time_grid >= start_expanded) & (time_grid < end_expanded)).float()
    frame_mask = time_arange_f >= speech_frame_length
    valid_aln_mask = (1.0 - text_mask.float().unsqueeze(-1)) * (
        1.0 - frame_mask.float().unsqueeze(1)
    )
    aln_masked = aln * valid_aln_mask

    en = d.transpose(-1, -2) @ aln_masked
    en_masked = en * (1.0 - frame_mask.float().unsqueeze(1))

    x_in = en_masked.transpose(-1, -2)
    x_shared = self.shared(x_in, speech_frame_length, mask=frame_mask)

    f_mask = (1.0 - frame_mask.float()).unsqueeze(1)
    f0_pred = x_shared.transpose(-1, -2)
    for b in self.predictor.F0:
      f0_pred = b(f0_pred, speaker_style, f_mask)
    f0_pred = self.predictor.F0_proj(f0_pred).squeeze(1)

    n_pred = x_shared.transpose(-1, -2)
    for b in self.predictor.N:
      n_pred = b(n_pred, speaker_style, f_mask)
    n_pred = self.predictor.N_proj(n_pred).squeeze(1)

    t_en = self.text_encoder(phoneme_ids, phoneme_length, text_mask)
    asr = t_en @ aln_masked

    f0_mask = torch.arange(
        2 * self.max_frames, device=f0_pred.device
    ).unsqueeze(0) >= (2 * speech_frame_length)
    f0_pred = f0_pred * (1.0 - f0_mask.float())
    n_pred = n_pred * (1.0 - f0_mask.float())
    asr = asr * (1.0 - frame_mask.float().unsqueeze(1))

    return {
        "acoustic_features": asr,
        "pitch_contour": f0_pred,
        "energy_contour": n_pred,
        "speech_frame_length": speech_frame_length,
    }


class KokoroVocoderModule(nn.Module):
  """Wraps Kokoro iSTFTNet vocoder for FX export."""

  def __init__(self, kmodel: kokoro_model.KModel):
    super().__init__()
    self.decoder = kmodel.decoder
    self.decoder.generator.return_spec_phase = True

  def forward(
      self,
      acoustic_features: torch.FloatTensor,
      pitch_contour: torch.FloatTensor,
      energy_contour: torch.FloatTensor,
      speaker_style: torch.FloatTensor,
      speech_frame_length: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    sfl_f = speech_frame_length.float()
    t_frame = acoustic_features.shape[-1]
    time_arange_f = torch.arange(
        t_frame, device=acoustic_features.device, dtype=torch.float32
    ).unsqueeze(0)
    frame_mask = time_arange_f >= sfl_f.reshape(-1, 1)
    t_f0 = pitch_contour.shape[-1]
    f0_mask = torch.arange(
        t_f0, device=pitch_contour.device, dtype=torch.float32
    ).unsqueeze(0) >= (2.0 * sfl_f).reshape(-1, 1)

    asr_masked = acoustic_features * (1.0 - frame_mask.float().unsqueeze(1))
    f0_masked = pitch_contour * (1.0 - f0_mask.float())
    n_masked = energy_contour * (1.0 - f0_mask.float())

    f0 = self.decoder.generator.f0_upsamp(f0_masked[:, None]).transpose(1, 2)
    har_source, _, _ = self.decoder.generator.m_source(f0)
    har_source = har_source.transpose(1, 2).squeeze(1)
    har_spec, har_phase = self.decoder.generator.stft.transform(har_source)
    har = torch.cat([har_spec, har_phase], dim=1)

    spec, phase = self.decoder(
        asr_masked,
        f0_masked,
        n_masked,
        speaker_style,
        harmonic_features=har,
        l_speech=speech_frame_length,
    )
    return {
        "magnitude_spectrogram": spec,
        "phase_spectrogram": phase,
    }


def prepare_kmodel_for_export(
    kmodel: kokoro_model.KModel,
    fuse_lstm: bool = True,
) -> kokoro_model.KModel:
  """Rewrites a KModel into its exportable form, in place.

  Removes weight_norm, replaces ConvTranspose1d with InterleavedConvTranspose1d,
  and fuses each bidirectional LSTM into a single custom op.

  Args:
    kmodel: The Kokoro model to rewrite.
    fuse_lstm: Whether to emit bidirectional LSTMs as
      `litert_custom_op.bidirectional_lstm` custom ops. Always True for export;
      tests disable it to run the converted graph in the stock TFLite
      interpreter, which has no kernel for the custom op.

  Returns:
    The same `kmodel`, for chaining.
  """
  kmodel.eval()
  for m in kmodel.modules():
    try:
      torch.nn.utils.remove_weight_norm(m)
    except ValueError:
      pass
    try:
      torch.nn.utils.parametrize.remove_parametrizations(m, "weight")
    except ValueError:
      pass

  for module in list(kmodel.modules()):
    if isinstance(module, istftnet.AdainResBlk1d) and isinstance(
        module.pool, nn.ConvTranspose1d
    ):
      module.pool = istftnet.InterleavedConvTranspose1d.from_conv_transpose(
          module.pool
      )

  ups = kmodel.decoder.generator.ups
  for i, up in enumerate(ups):
    if isinstance(up, nn.ConvTranspose1d):
      ups[i] = istftnet.InterleavedConvTranspose1d.from_conv_transpose(up)

  # Collapse each bidirectional LSTM into a single
  # `litert_custom_op.bidirectional_lstm` node instead of letting torch.export
  # unroll every time step. This removes ~96% of the acoustic graph and most of
  # its runtime memory; the LiteRT-LM Kokoro runtime registers the kernel.
  for module in kmodel.modules():
    if isinstance(module, modules.BidirectionalFusedLSTM):
      module.fuse_as_custom_op = fuse_lstm

  return kmodel


class KokoroTts(tts_model.TtsModel):
  """Kokoro TTS exportable model class."""

  @staticmethod
  def is_kokoro_checkpoint(
      config_dict: dict[str, Any], model_path: str
  ) -> bool:
    """Returns whether a checkpoint without `model_type` is a Kokoro model.

    Kokoro's `config.json` has no `model_type`, so it is recognized by its
    iSTFTNet / token-vocabulary keys or, when there is no config, by its path.

    Args:
      config_dict: The raw `config.json` contents, or `{}` if there is none.
      model_path: HF repo id or local checkpoint directory.
    """
    return (
        "istftnet" in config_dict
        or "n_token" in config_dict
        or "kokoro" in str(model_path).lower()
    )

  def __init__(
      self,
      model_path: str,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
      override_transformers: bool = False,
  ):
    super().__init__(model_path, export_config)
    del override_transformers
    use_random_weights = bool(
        export_config and export_config.use_random_weights
    )
    if os.path.exists(model_path):
      self.model_dir = model_path
      self.repo_id = _DEFAULT_REPO_ID
    elif use_random_weights:
      self.repo_id = model_path
      self.model_dir = ""
    else:
      self.repo_id = model_path
      self.model_dir = huggingface_hub.snapshot_download(
          repo_id=model_path,
          allow_patterns=["config.json", "*.pth", "voices/*.pt"],
          revision=kokoro_model.KModel.PINNED_REVISIONS.get(model_path),
      )

  def _apply_quantization(
      self, model_path: str, quant_recipe: str | None
  ) -> str:
    """Quantizes an exported .tflite in place with the shared export_hf helper.

    Args:
      model_path: Path of the float .tflite model; overwritten on success.
      quant_recipe: Recipe accepted by `quantization.maybe_quantize_model`.

    Returns:
      `model_path`.

    Raises:
      ValueError: If the recipe is unknown or cannot be loaded.
    """
    quantized_path = quantization.maybe_quantize_model(model_path, quant_recipe)
    if quantized_path == model_path:
      return model_path
    shutil.move(quantized_path, model_path)
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Quantized {model_path} with {quant_recipe} ({file_size_mb:.2f} MB)")
    return model_path

  def export(
      self, export_config: exportable_module_config.ExportableModuleConfig
  ) -> dict[str, str]:
    """Exports Kokoro acoustic, vocoder, and voice packs."""
    output_dir = export_config.output_dir or export_config.work_dir
    if not output_dir:
      raise ValueError("Either output_dir or work_dir must be specified.")
    os.makedirs(output_dir, exist_ok=True)
    artifacts = {}

    # Disable default torch.ops.aten.lstm.input decomposition to preserve Fused
    # LSTM.
    try:
      litert_torch.fx_infra.decomp.remove_pre_convert_decomp(
          torch.ops.aten.lstm.input
      )
    except Exception:  # pylint: disable=broad-exception-caught
      pass

    config_path = (
        os.path.join(self.model_dir, "config.json") if self.model_dir else None
    )
    pth_candidates = [
        "kokoro-v1_0.pth",
        "kokoro-v1_1-zh.pth",
        "kokoro.pth",
    ]
    pth_path = None
    if self.model_dir:
      for cand in pth_candidates:
        cand_path = os.path.join(self.model_dir, cand)
        if os.path.exists(cand_path):
          pth_path = cand_path
          break

    use_random_weights = bool(export_config.use_random_weights)
    if self.model_dir and pth_path is None and not use_random_weights:
      # Without this check KModel would silently download hexgrad/Kokoro-82M
      # weights and pair them with the local config.
      raise FileNotFoundError(
          f"No Kokoro checkpoint found in {self.model_dir}; expected one of"
          f" {pth_candidates}."
      )
    print(
        f"Initializing Kokoro KModel from {self.model_dir or self.repo_id}"
        f" (config: {config_path}, pth: {pth_path},"
        f" use_random_weights: {use_random_weights})..."
    )
    kmodel = kokoro_model.KModel(
        repo_id=self.repo_id,
        config=(
            config_path if config_path and os.path.exists(config_path) else None
        ),
        model=pth_path,
        disable_complex=True,
        use_random_weights=use_random_weights,
    ).eval()
    prepare_kmodel_for_export(kmodel)

    quant_recipe = export_config.quantization_recipe
    bucket_size = BUCKET_SIZE
    max_frames = FRAMES_PER_TOKEN * bucket_size

    # 1. Export Unified Acoustic Model (kokoro_acoustic.tflite)
    ac_path = os.path.join(output_dir, "kokoro_acoustic.tflite")
    print(
        f"Exporting acoustic model (bucket={bucket_size},"
        f" max_frames={max_frames}) to {ac_path}..."
    )
    mod_ac = KokoroAcousticFusedModule(kmodel, max_frames=max_frames).eval()
    # Keep int64 `phoneme_ids` / `phoneme_length` I/O: the LiteRT-LM
    # `KokoroAcousticStage` requires int64 phoneme ids.
    conv_ac = litert_torch.convert(
        mod_ac,
        sample_kwargs={
            "phoneme_ids": torch.zeros((1, bucket_size), dtype=torch.int64),
            "speaker_style": torch.zeros((1, 128), dtype=torch.float32),
            "phoneme_length": torch.tensor([bucket_size], dtype=torch.int64),
        },
        enable_x64=True,
    )
    conv_ac.export(ac_path)
    self._apply_quantization(ac_path, quant_recipe)
    _align_buffers_64(ac_path)
    artifacts["kokoro_acoustic"] = ac_path

    # 2. Export Vocoder Model (kokoro_vocoder.tflite)
    voc_path = os.path.join(output_dir, "kokoro_vocoder.tflite")
    print(f"Exporting vocoder model (max_frames={max_frames}) to {voc_path}...")
    mod_voc = KokoroVocoderModule(kmodel).eval()
    conv_voc = litert_torch.convert(
        mod_voc,
        sample_kwargs={
            "acoustic_features": torch.zeros(
                (1, 512, max_frames), dtype=torch.float32
            ),
            "pitch_contour": torch.zeros(
                (1, 2 * max_frames), dtype=torch.float32
            ),
            "energy_contour": torch.zeros(
                (1, 2 * max_frames), dtype=torch.float32
            ),
            "speaker_style": torch.zeros((1, 128), dtype=torch.float32),
            "speech_frame_length": torch.tensor(
                [max_frames], dtype=torch.int32
            ),
        },
    )
    conv_voc.export(voc_path)
    # Vocoder is preserved in FP32 for audio fidelity and full GPU OpenCL
    # delegation.
    artifacts["kokoro_vocoder"] = voc_path

    # 3. Export Voice Packs (.pt to binary .bin float32 format)
    voices_out_dir = os.path.join(output_dir, "voices")
    os.makedirs(voices_out_dir, exist_ok=True)
    voices_src_dir = (
        os.path.join(self.model_dir, "voices") if self.model_dir else None
    )
    voice_files = []
    if voices_src_dir and os.path.exists(voices_src_dir):
      voice_files = [
          os.path.join(voices_src_dir, f)
          for f in os.listdir(voices_src_dir)
          if f.endswith(".pt")
      ]
    if (
        not any(f.endswith("af_heart.pt") for f in voice_files)
        and not use_random_weights
    ):
      try:
        af_heart_pt = huggingface_hub.hf_hub_download(
            repo_id=_DEFAULT_REPO_ID,
            filename="voices/af_heart.pt",
            revision=kokoro_model.KModel.PINNED_REVISIONS.get(_DEFAULT_REPO_ID),
        )
        voice_files.append(af_heart_pt)
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Warning: could not download default voice af_heart.pt: {e}")

    for vf in voice_files:
      v_name = os.path.splitext(os.path.basename(vf))[0]
      v_tensor = torch.load(vf, map_location="cpu", weights_only=True)
      if isinstance(v_tensor, torch.Tensor):
        bin_path = os.path.join(voices_out_dir, f"{v_name}.bin")
        v_tensor.detach().numpy().astype(np.float32).tofile(bin_path)
        artifacts[f"voice_{v_name}"] = bin_path

    if not voice_files and use_random_weights:
      bin_path = os.path.join(voices_out_dir, "af_heart.bin")
      np.zeros((510, 1, 256), dtype=np.float32).tofile(bin_path)
      artifacts["voice_af_heart"] = bin_path

    # 4. Copy espeak-ng-data from the source model directory. It is GPL-3.0
    # licensed, so the exporter never downloads it; callers that need it in the
    # output must place it next to the checkpoint.
    espeak_out_dir = os.path.join(output_dir, "espeak-ng-data")
    src_espeak = (
        os.path.join(self.model_dir, "espeak-ng-data")
        if self.model_dir
        else None
    )
    if os.path.exists(espeak_out_dir):
      artifacts["espeak-ng-data"] = espeak_out_dir
    elif src_espeak and os.path.isdir(src_espeak):
      print(
          f"Copying espeak-ng-data from source model directory {src_espeak}"
          f" to {espeak_out_dir}..."
      )
      shutil.copytree(src_espeak, espeak_out_dir, dirs_exist_ok=True)
      artifacts["espeak-ng-data"] = espeak_out_dir
    else:
      print(
          "Warning: espeak-ng-data not found in the source model directory;"
          " the Kokoro runtime needs it for phonemization. Copy it into"
          f" {output_dir} or the source model directory."
      )

    return artifacts
