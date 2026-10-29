---
name: export-hf-model-extension
description: >-
  Onboards and exports new or custom Hugging Face (or Diffusers/PyTorch) model
  architectures in `litert_torch.generative.export_hf` using pluggable
  `ModelExtension` classes (built-in under `model_ext/<arch>/` or out-of-tree
  via `--extension_modules`). Use when supporting a new `model_type` in
  `export_hf` or `litert_lm`, inspecting a checkpoint with
  `model_ext/inspector.py`, patching RMSNorm/SwiGLU/ShortConv/Attention/RoPE
  for `torch.export`, configuring fused weights or `LlmMetadata`, or verifying
  extensions with `model_ext/testing.py`. Don't use for generic non-generative
  PyTorch-to-TFLite conversion or standalone post-export equivalence evaluation.
---

# Onboarding New Model Architectures in `export_hf` (`ModelExtension`)

Use the pluggable `ModelExtension` framework
(`litert_torch.generative.export_hf.model_ext`) to support new or custom Hugging
Face (`trust_remote_code`) checkpoints in `export_hf` and `litert_lm` without
modifying core `litert_torch` or checkpoint files.

---

## 1. Standard 4-Step Workflow

### Step 1: Run the Architecture Inspector (`model_ext/inspector.py`)

Always inspect the checkpoint or Hugging Face repo ID first. The inspector
shrinks the config to a tiny CPU model, checks `.safetensors` prefixes,
RMSNorm/MLP/Attention/ShortConv/RoPE modules, runs a `<5s` `torch.export`
dry-run, and prints a starter `ModelExtension` skeleton:

```bash
bazel run \
  //third_party/py/litert_torch/generative/export_hf/model_ext:inspector -- \
  --model=/path/to/checkpoint \
  --trust_remote_code
```

Or via Python:

```bash
python -m litert_torch.generative.export_hf.model_ext.inspector \
  --model=/path/to/checkpoint \
  --trust_remote_code
```

### Step 2: Author the `ModelExtension`

Subclass `extension.ModelExtension` (either in an out-of-tree `.py` file passed
via `--extension_modules=/path/to/my_ext.py` or built-in under
`model_ext/<arch>/extension.py`) and set `model_types = ('<model_type>',)`.

### Step 3: Fast Inner-Loop Verification (`<2s` on CPU)

Before running a multi-minute full checkpoint export, verify your extension on a
tiny model (2–3 layers, `vocab_size=256`, `dtype=torch.float32`) using
`model_ext/testing.py`:

1. **`testing.assert_patch_numerical_parity(ext, tiny_model, ...)`**: Verifies
   eager numerical parity before vs. during `patch_model` and asserts that all
   submodule classes and `.forward` methods are cleanly restored on exit.
2. **`testing.assert_tiny_prefill_decode_torch_export(ext, tiny_model, ...)`**:
   Verifies `torch.export.export` graph capture across `prefill_*` and `decode`
   (and `RoPEEmbedder` when `split_cache=True`).

### Step 4: Full Export & `litert_lm` Verification

* **From source (`bazel run`)**:
  ```bash
  bazel run \
    //third_party/py/litert_torch/generative/export_hf:export_main -- \
    --model=/path/to/checkpoint \
    --output_dir="$HOME/export_hf_test_models/outputs/my_model" \
    --extension_modules=/path/to/my_extension.py \
    --trust_remote_code=True \
    --bundle_litert_lm=True
  ```
* **Python / Installed Package CLI**: Run `litert-torch export_hf ...` or
  `python -m litert_torch.generative.export_hf.export_main ...` (do **not** run
  `python -m litert_torch.generative.export_hf.export`, which has no `__main__`
  block).

---

## 2. LiteRT-Torch Composite Ops & Import Cheat-Sheet

Do not guess import paths or function signatures for LiteRT composites:

| Operator | Import Path | Exact Call Signature & Contract |
| :--- | :--- | :--- |
| **RMSNorm (`odml.rms_norm`)** | `from litert_torch.generative.layers import normalization` | `normalization.rms_norm_with_hlfb(hidden_states, weight, eps, torch.ones((hidden_size,), dtype=torch.float32, device=hidden_states.device))` — requires **4 positional arguments** (note the 4th `ones` tensor). Lives in `generative.layers`, **not** `experimental.composites`. |
| **SwiGLU (`odml.swiglu`)** | `from litert_torch.generative.export_hf.experimental.composites import swiglu` | `swiglu.apply_swiglu(gate_up)` — takes concatenated `[gate, up]` tensor (`[B, T, 2 * D_ff]`) from `gate_up_proj(x)` and returns `[B, T, D_ff]` **before** `down_proj` / `w2`. |
| **ShortConv (`odml.short_conv`)** | `from litert_torch.generative.export_hf.experimental.composites import short_conv` | `short_conv.apply_short_conv_step(in_proj_out, conv_state, conv_weight, conv_bias, num_valid_tokens, conv_L_cache)` (composite) or `short_conv.short_conv(...)` (eager fallback) — takes `in_proj_out = self.in_proj(hidden_states)` (`[B, T, 3 * D]`), **not** `hidden_states` (`[B, T, D]`), and returns `(y, new_conv_state)` **before** `self.out_proj(y)`. |
| **Attention (`ALL_ATTENTION_FUNCTIONS`)** | `from litert_torch.generative.export_hf.core import attention as _` | Dispatch via `modeling_utils.ALL_ATTENTION_FUNCTIONS[config._attn_implementation]`. Pass `past_key_value=past_key_values` and `sliding_window=self.sliding_window`. |

---

## 3. Critical Architecture Gotchas (From Real Failure Modes)

### A. Post-SDPA `attn_output` Shape Contract (`[B, T, N, H]` vs. `[B, 1, N*H]`)
`lrt_transposed_attention` returns `attn_output` with shape
`[B, T, num_heads, head_dim]` on CPU/prefill, whereas the `gpu_sdpa` decode
composite returns `[B, 1, num_heads * head_dim]`.
* If an architecture applies per-head operations after SDPA (such as `LFM3`'s
  head-wise sigmoid gate `torch.sigmoid(self.gate_proj(hidden_states)).unsqueeze(-1)`),
  always normalize the shape first:
  ```python
  attn_output = attn_output.reshape(*input_shape, self.num_heads, self.head_dim)
  attn_output = (attn_output * gate).reshape(*input_shape, -1).contiguous()
  ```

### B. Hybrid Architectures (`"conv"` + `"sliding_attention"` + `"full_attention"`)
1. **Mask dictionary keys**: `exportable_module.adapt_inputs` passes
   `attention_mask = {'full_attention': ..., 'sliding_attention': ...}` when
   `sliding_window` is set. If `config.layer_types` also contains `'conv'` (or
   `'linear_attention'`), use `causal_mask_mapping.get(layer_type)` or
   `causal_mask_mapping.setdefault('conv', None)` in the model's `forward` so
   indexing `causal_mask_mapping[layer_type]` does not raise `KeyError: 'conv'`.
2. **Forwarding `valid_mask` (`**kwargs`) to `ShortConv`**:
   `LiteRTExportableModuleForDecoderOnlyLMPrefill` passes `valid_mask` via
   `**kwargs` so `ShortConv` layers mask out right-padded prefill tokens before
   updating `past_key_values.layers[layer_idx].conv_states`. Ensure
   `DecoderLayer.forward` forwards `**kwargs` to `self.conv(...)`.
3. **Shrinking hybrid configs for tiny tests**: Do not blindly slice
   `config.layer_types[:2]` if the first two layers have the same type (e.g.,
   `LFM3-1.7B` has `['conv', 'conv']` at layers 0–1). Include at least one of
   each distinct layer type (e.g.
   `['conv', 'sliding_attention', 'full_attention']`), and clamp `pad_token_id`
   / `bos_token_id` / `eos_token_id` to `< vocab_size` when shrinking
   `vocab_size` so `nn.Embedding(..., padding_idx=pad_token_id)` does not fail.

### C. `trust_remote_code` Dynamic Module Lookup & Converter V2 (`--use_v2`)
When Hugging Face loads a checkpoint with `trust_remote_code=True`, the modeling
module is registered in `sys.modules` under
`transformers_modules.<repo>.modeling_<arch>`, not `transformers.models.<arch>`.
* Scan `sys.modules` to locate the module for `patch_context`, `patch_model`,
  and `get_hf_model_cls`:
  ```python
  def _find_modeling_module() -> Any | None:
    for mod_name, mod in list(sys.modules.items()):
      if mod_name.endswith('modeling_my_arch') and hasattr(mod, 'MyArchForCausalLM'):
        return mod
    return None
  ```
* Return `mod.MyArchForCausalLM` from `get_hf_model_cls` so Converter V2
  (`use_v2=True` meta-device `from_config`) can instantiate the model directly.
* If the checkpoint uses custom MLP weight names (e.g., `w1` and `w3` instead of
  `gate_proj` and `up_proj`), declare `fused_weight_specs` with
  `extension.FusedWeightSpec('gate_up_proj', ('w1', 'w3'), dim=0)`.

### D. `sliding_window_ring_buffer_size` Flag Coupling
When using `LiteRTLMCache` (`split_cache=False`), setting
`sliding_window_ring_buffer_size=<W>` requires `apply_gpu_composites=True` and
`use_bool_mask=True` (`extra_kwargs={'use_bool_mask': True}`). Without
`apply_gpu_composites=True`, `_update_kv_sliding_impl` returns a
`(cache_past, cache_slice)` tuple of length `W + T` while `local_mask` has
length `W`, causing a dimension broadcast error (`W` vs `W + 1`) in decode.

### E. `bfloat16` Config Default in `assert_patch_numerical_parity`
Many `config.json` files specify `"dtype": "bfloat16"`. Calling
`AutoModelForCausalLM.from_config(cfg)` initializes weights in `bfloat16`, where
multi-layer rounding differences are `~1e-2` vs. `atol=1e-4`. Always test parity
in `torch.float32` (`assert_patch_numerical_parity` automatically casts
`bfloat16`/`float16` models to `float32` when `cast_to_float32=True`).
