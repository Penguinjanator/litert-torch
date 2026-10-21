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
"""Kokoro acoustic, prosody, and duration predictor modules."""

# https://github.com/yl4579/StyleTTS2/blob/main/models.py
from litert_torch.generative.export_hf.model_ext.kokoro_tts import istftnet
from litert_torch.generative.export_hf.model_ext.kokoro_tts import lstm_composite
from litert_torch.generative.layers import normalization
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
import transformers


class LinearNorm(nn.Module):
  """Linear layer with Xavier uniform initialization."""

  def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
    super().__init__()
    self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
    nn.init.xavier_uniform_(
        self.linear_layer.weight,
        gain=nn.init.calculate_gain(w_init_gain),  # pyrefly: ignore[bad-argument-type]
    )

  def forward(self, x):
    return self.linear_layer(x)


class LayerNorm(normalization.LayerNorm):
  """Channel-first LayerNorm: normalizes each frame of a [B, C, T] input over C.

  Reuses the generative `LayerNorm` without its HLFB composite: that composite
  lowers to `odml.group_norm` with a single group, which normalizes over all
  non-batch axes rather than per frame.
  """

  def __init__(self, channels, eps=1e-5):
    super().__init__(channels, eps=eps)
    nn.init.ones_(self.weight)
    if self.bias is not None:
      nn.init.zeros_(self.bias)

  def _load_from_state_dict(
      self,
      state_dict,
      prefix,
      local_metadata,
      strict,
      missing_keys,
      unexpected_keys,
      error_msgs,
  ):
    # Kokoro checkpoints name the affine parameters `gamma` / `beta`.
    for old, new in (("gamma", "weight"), ("beta", "bias")):
      if prefix + old in state_dict:
        state_dict[prefix + new] = state_dict.pop(prefix + old)
    super()._load_from_state_dict(
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    )

  def forward(self, x):
    return super().forward(x.transpose(1, -1)).transpose(1, -1)


def unidirectional_lstm(
    x: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias: torch.Tensor,
    hidden_size: int,
) -> torch.Tensor:
  """Runs a unidirectional LSTM over the time axis of x.

  Args:
    x: Input sequence, [B, T, D].
    weight_ih: Input-to-hidden weights, [4H, D], gates ordered (i, f, g, o).
    weight_hh: Hidden-to-hidden weights, [4H, H].
    bias: Combined bias, [4H]. Only the sum of PyTorch's bias_ih and bias_hh is
      observable, so callers pass it pre-summed.
    hidden_size: Hidden size H.

  Returns:
    The hidden state at every step, [B, T, H].
  """
  b, t_steps, _ = x.shape
  # 1 single BMM for all T steps:
  x_proj = F.linear(x, weight_ih, bias)  # [B, T, 4*H]

  h = torch.zeros((b, hidden_size), dtype=x.dtype, device=x.device)
  c = torch.zeros((b, hidden_size), dtype=x.dtype, device=x.device)

  x_proj_steps = torch.unbind(x_proj, dim=1)
  h_list = []
  for t in range(t_steps):
    # 1 single recurrent linear transformation per step:
    gates = x_proj_steps[t] + F.linear(h, weight_hh)  # [B, 4*H]
    i, f, g, o = gates.chunk(4, dim=-1)
    i = torch.sigmoid(i)
    f = torch.sigmoid(f)
    g = torch.tanh(g)
    o = torch.sigmoid(o)
    c = f * c + i * g
    h = o * torch.tanh(c)
    h_list.append(h)

  return torch.stack(h_list, dim=1)


class CustomFusedLSTM(nn.Module):
  """Single-layer unidirectional fused LSTM with precomputed input projection."""

  def __init__(self, input_size: int, hidden_size: int):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
    self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
    self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size))
    self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size))
    self.reset_parameters()

  def reset_parameters(self):
    stdv = 1.0 / (self.hidden_size**0.5) if self.hidden_size > 0 else 0.0
    for weight in self.parameters():
      nn.init.uniform_(weight, -stdv, stdv)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return unidirectional_lstm(
        x,
        self.weight_ih,
        self.weight_hh,
        self.bias_ih + self.bias_hh,
        self.hidden_size,
    )


class BidirectionalFusedLSTM(nn.Module):
  """Bidirectional wrapper around CustomFusedLSTM."""

  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.lstm_fwd = CustomFusedLSTM(input_size, hidden_size)
    self.lstm_bwd = CustomFusedLSTM(input_size, hidden_size)
    self.hidden_size = hidden_size
    # When True, `forward` emits the layer as a single
    # `litert_custom_op.bidirectional_lstm` composite instead of letting
    # torch.export unroll every time step. Set by `prepare_kmodel_for_export`.
    self.fuse_as_custom_op = False

  def _load_from_state_dict(
      self,
      state_dict,
      prefix,
      local_metadata,
      strict,
      missing_keys,
      unexpected_keys,
      error_msgs,
  ):
    for name in ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"]:
      fwd_key = prefix + name
      rev_key = prefix + name + "_reverse"
      mapped_name = name.removesuffix("_l0")
      if fwd_key in state_dict:
        state_dict[prefix + "lstm_fwd." + mapped_name] = state_dict.pop(fwd_key)
      if rev_key in state_dict:
        state_dict[prefix + "lstm_bwd." + mapped_name] = state_dict.pop(rev_key)
    super()._load_from_state_dict(
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    )

  def forward(self, x, seq_lengths, mask=None):
    weights = (
        self.lstm_fwd.weight_ih,
        self.lstm_fwd.weight_hh,
        self.lstm_fwd.bias_ih + self.lstm_fwd.bias_hh,
        self.lstm_bwd.weight_ih,
        self.lstm_bwd.weight_hh,
        self.lstm_bwd.bias_ih + self.lstm_bwd.bias_hh,
    )
    if not self.fuse_as_custom_op:
      return self._unrolled_forward(x, seq_lengths, mask, *weights)

    # The custom op has fixed arity, so callers without a mask supply an
    # all-zero one. That is the identity for the arithmetic masking below.
    mask_f = (
        torch.zeros(x.shape[:2], dtype=x.dtype, device=x.device)
        if mask is None
        else mask.float()
    )
    return lstm_composite.fused_bidirectional_lstm(
        x,
        seq_lengths,
        mask_f,
        *weights,
        hidden_size=self.hidden_size,
        reference_impl=self._unrolled_forward,
    )

  def _unrolled_forward(
      self,
      x,
      seq_lengths,
      mask,
      weight_ih_fwd,
      weight_hh_fwd,
      bias_fwd,
      weight_ih_bwd,
      weight_hh_bwd,
      bias_bwd,
  ):
    """Statically unrolled bidirectional pass.

    Also serves as the decomposition region of the fused custom op, which is
    why the weights arrive as arguments rather than being read off the child
    modules: the composite builder has to see them as marked operands.

    Args:
      x: Input sequence, [B, T, D].
      seq_lengths: Unpadded length of each batch element, [B].
      mask: Optional padding mask, [B, T], true (or 1.0) on padding positions.
      weight_ih_fwd: Forward input-to-hidden weights, [4H, D].
      weight_hh_fwd: Forward hidden-to-hidden weights, [4H, H].
      bias_fwd: Forward combined bias, [4H].
      weight_ih_bwd: Backward input-to-hidden weights, [4H, D].
      weight_hh_bwd: Backward hidden-to-hidden weights, [4H, H].
      bias_bwd: Backward combined bias, [4H].

    Returns:
      Concatenated forward and backward hidden states, [B, T, 2H].
    """
    t_steps = x.shape[1]
    time_arange = torch.arange(t_steps, device=x.device)
    idx_b = (seq_lengths.reshape(-1, 1) - 1 - time_arange.unsqueeze(0)).clamp(
        min=0, max=t_steps - 1
    )

    x_fwd = unidirectional_lstm(
        x, weight_ih_fwd, weight_hh_fwd, bias_fwd, self.hidden_size
    )

    x_b = torch.gather(x, 1, idx_b.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    if mask is not None:
      x_b = x_b * (1.0 - mask.float().unsqueeze(-1))

    x_bwd_b = unidirectional_lstm(
        x_b, weight_ih_bwd, weight_hh_bwd, bias_bwd, self.hidden_size
    )

    x_bwd = torch.gather(
        x_bwd_b, 1, idx_b.unsqueeze(-1).expand(-1, -1, self.hidden_size)
    )
    if mask is not None:
      keep = 1.0 - mask.float().unsqueeze(-1)
      x_fwd = x_fwd * keep
      x_bwd = x_bwd * keep

    x_out = torch.cat([x_fwd, x_bwd], dim=-1)
    return x_out


class TextEncoder(nn.Module):
  """Kokoro CNN + BiLSTM text encoder."""

  def __init__(
      self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)
  ):
    super().__init__()
    self.embedding = nn.Embedding(n_symbols, channels)
    padding = (kernel_size - 1) // 2
    self.cnn = nn.ModuleList()
    for _ in range(depth):
      self.cnn.append(
          nn.Sequential(
              weight_norm(
                  nn.Conv1d(
                      channels,
                      channels,
                      kernel_size=kernel_size,
                      padding=padding,
                  )
              ),
              LayerNorm(channels),
              actv,
              nn.Dropout(0.2),
          )
      )
    self.lstm = BidirectionalFusedLSTM(channels, channels // 2)

  def forward(self, x, input_lengths, m):
    x = self.embedding(x)  # [B, T, emb]
    x = x.transpose(1, 2)  # [B, emb, T]
    m_inv = (1.0 - m.float()).unsqueeze(1)
    x = x * m_inv
    for c in self.cnn:
      x = c(x)
      x = x * m_inv
    x = x.transpose(1, 2)  # [B, T, chn]
    x = self.lstm(x, input_lengths, mask=m)
    x = x.transpose(-1, -2)
    x = x * m_inv
    return x


class AdaLayerNorm(nn.Module):
  """Adaptive Layer Normalization conditioned on speaker style."""

  def __init__(self, style_dim, channels, eps=1e-5):
    super().__init__()
    self.channels = channels
    self.eps = eps
    self.fc = nn.Linear(style_dim, channels * 2)

  def forward(self, x, s):
    x = x.transpose(-1, -2)
    x = x.transpose(1, -1)
    h = self.fc(s)
    h = h.view(h.size(0), h.size(1), 1)
    gamma, beta = torch.chunk(h, chunks=2, dim=1)
    gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
    x = F.layer_norm(x, (self.channels,), eps=self.eps)
    x = (1 + gamma) * x + beta
    return x.transpose(1, -1).transpose(-1, -2)


class ProsodyPredictor(nn.Module):
  """Duration, F0, and energy prosody predictor."""

  # pylint: disable=invalid-name
  def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
    super().__init__()
    self.text_encoder = DurationEncoder(
        sty_dim=style_dim, d_model=d_hid, nlayers=nlayers, dropout=dropout
    )
    self.lstm = BidirectionalFusedLSTM(d_hid + style_dim, d_hid // 2)
    self.duration_proj = LinearNorm(d_hid, max_dur)
    self.shared = BidirectionalFusedLSTM(d_hid + style_dim, d_hid // 2)
    self.F0 = nn.ModuleList()
    self.F0.append(
        istftnet.AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout)
    )
    self.F0.append(
        istftnet.AdainResBlk1d(
            d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout
        )
    )
    self.F0.append(
        istftnet.AdainResBlk1d(
            d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout
        )
    )
    self.N = nn.ModuleList()
    self.N.append(
        istftnet.AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout)
    )
    self.N.append(
        istftnet.AdainResBlk1d(
            d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout
        )
    )
    self.N.append(
        istftnet.AdainResBlk1d(
            d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout
        )
    )
    self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
    self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

  def forward(self, texts, style, text_lengths, alignment, m):
    d = self.text_encoder(texts, style, text_lengths, m)
    x = self.lstm(d, text_lengths, mask=m)
    duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=False))
    en = d.transpose(-1, -2) @ alignment
    return duration.squeeze(-1), en

  def F0Ntrain(self, x, s):
    x = self.shared(
        x.transpose(-1, -2),
        torch.full(
            (x.shape[0],), x.shape[-1], device=x.device, dtype=torch.long
        ),
    )

    f0 = x.transpose(-1, -2)
    for block in self.F0:
      f0 = block(f0, s)
    f0 = self.F0_proj(f0)
    n_pred = x.transpose(-1, -2)
    for block in self.N:
      n_pred = block(n_pred, s)
    n_pred = self.N_proj(n_pred)
    return f0.squeeze(1), n_pred.squeeze(1)

  # pylint: enable=invalid-name


class DurationEncoder(nn.Module):
  """Duration encoder stacking BidirectionalFusedLSTM and AdaLayerNorm."""

  def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
    super().__init__()
    self.lstms = nn.ModuleList()
    for _ in range(nlayers):
      self.lstms.append(BidirectionalFusedLSTM(d_model + sty_dim, d_model // 2))
      self.lstms.append(AdaLayerNorm(sty_dim, d_model))
    self.dropout = dropout
    self.d_model = d_model
    self.sty_dim = sty_dim

  def forward(self, x, style, text_lengths, m):
    m_inv_01 = (1.0 - m.float()).unsqueeze(-1).transpose(0, 1)
    m_inv_12 = (1.0 - m.float()).unsqueeze(-1).transpose(-1, -2)
    x = x.permute(2, 0, 1)
    s = style.expand(x.shape[0], x.shape[1], -1)
    x = torch.cat([x, s], dim=-1)
    x = x * m_inv_01
    x = x.transpose(0, 1)
    x = x.transpose(-1, -2)
    for block in self.lstms:
      if isinstance(block, AdaLayerNorm):
        x = block(x.transpose(-1, -2), style).transpose(-1, -2)
        x = torch.cat([x, s.permute(1, 2, 0)], dim=1)
        x = x * m_inv_12
      else:
        x = x.transpose(-1, -2)
        x = block(x, text_lengths, mask=m)
        x = F.dropout(x, p=self.dropout, training=False)
        x = x.transpose(-1, -2)

    return x.transpose(-1, -2)


# https://github.com/yl4579/StyleTTS2/blob/main/Utils/PLBERT/util.py
class CustomAlbert(transformers.AlbertModel):
  """Custom ALBERT wrapper returning last_hidden_state."""

  def forward(self, *args, **kwargs):
    outputs = super().forward(*args, **kwargs)
    return outputs.last_hidden_state
