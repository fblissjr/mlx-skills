last updated: 2026-04-09

# mlx-vlm Reference

mlx-vlm is a third-party library that extends mlx-lm patterns for
vision-language models. It shares mlx-lm's sampling, logits processing, and
KV cache infrastructure while adding vision and audio encoding stages.

## Architecture Overview

Two-stage pipeline:

1. **Encode**: Vision/audio encoder produces embedding sequences
2. **Decode**: Language model (mlx-lm architecture) consumes interleaved
   text tokens and encoder embeddings

Key design points:

- **Processor-centric**: Depends on HuggingFace Transformers processors for
  image/audio preprocessing. Each model has a custom processor class.
- **Shared utilities**: Uses mlx-lm's `make_sampler`, `make_logits_processors`,
  and cache infrastructure directly.
- **Multi-modal inputs**: Text tokens and image/audio embeddings are
  interleaved via special boundary tokens (e.g., `<boi>/<eoi>` for images,
  `<boa>/<eoa>` for audio).

## Supported Models (55+ architectures)

### Vision-Language

Aya Vision, DeepSeek VL v2, DeepSeek OCR (1/2), DOTS OCR, ERNIE 4.5 MoE VL,
Falcon-OCR, Falcon-Perception, FastVLM, Florence 2, Gemma 3/3n/4, GLM OCR,
GLM-4V (dense/MoE), Granite Vision (3.2/4.0), Hunyuan VL, Idefics 2/3,
InternVL Chat, Jina VLM, Kimi VL, LFM2 VL, Llama 4, LLaVA/LLaVA-Bunny/
LLaVA-Next, MiniCPM-O, Mistral 3/4, MLlama, Molmo/Molmo2/MolMo Point,
Moondream3, PaddleOCR VL, PaLIGemma, Phi-3V/Phi-4 SigLIP/Phi-4 MM, Pixtral,
Qwen 2/2.5/3/3.5 VL (dense/MoE), SmolVLM

### Vision + Audio

Gemma 4 (image + audio), Qwen3-Omni-MoE

### Detection and Segmentation

RF-DETR, SAM 3, SAM 3.1

## Gemma 4 Multimodal

Gemma 4 in mlx-vlm is a full multimodal model with vision, audio, and sparse
MoE language backbone. The language model reuses mlx-lm's `gemma4_text`
patterns (for text-side architecture details including mixed attention, shared
KV, dual RoPE, and MoE, see `patterns.md`).

### Vision Encoder

- Vision transformer with patch size 16
- `ClippableLinear`: Linear layer with optional input/output clamping via
  `mx.clip`. Clip bounds stored as checkpoint buffers (default +/-inf = no-op).
- `VisionRMSNorm` (with learned scale) and `VisionRMSNormNoScale` (parameter-free)
- Bidirectional attention (no causal mask) for full image context
- Default output: 280 tokens per image
- Supported soft token lengths: 70, 140, 280, 560, 1120

```python
class ClippableLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, use_clipping=True):
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if use_clipping:
            self.input_min = mx.array(float("-inf"))
            self.input_max = mx.array(float("inf"))
            # ... output_min, output_max similarly

    def __call__(self, x):
        if self.use_clipping:
            x = mx.clip(x, self.input_min, self.input_max)
        x = self.linear(x)
        if self.use_clipping:
            x = mx.clip(x, self.output_min, self.output_max)
        return x
```

### Audio Encoder

Conformer architecture (SSCP + 12 Conformer blocks):

- **SSCP** (SubSample Conv Projection): 2x Conv2d blocks + Linear. Downsamples
  time dimension by 4x via stride-2 convolutions.
- **Conformer blocks**: FFW -> Attention -> LightConv1d -> FFW -> Clamp -> RMSNorm
- Chunked local attention with relative position embeddings and logit
  softcapping (cap=50.0)
- 8 attention heads, attention chunk size 12, context left 13
- FFT-based audio resampling (replaces linear interpolation)
- Output: 750 tokens at 40ms per token, projected to text hidden size (1536)

### Processor

`Gemma4Processor` handles combined image + text + audio inputs:

- Image: aspect-ratio-preserving resize, rescale to [0,1], 280 tokens per image
- Audio: mel spectrogram preprocessing with correct feature extraction pipeline,
  convolutional subsampling, 750 tokens
- Multi-image support with shape normalization via `group_images_by_shape()`
- Special tokens: `image_token_id`/`boi_token`/`eoi_token`,
  `audio_token_id`/`boa_token`/`eoa_token`

### Known Fixes (Apr 2026)

- Audio preprocessing: corrected mel feature extraction, weight loading paths,
  and feature extractor initialization
- Quantized models: fixed per-layer projection loading for quantized Gemma 4
- Batched caches: `cache.offset` is now snapshotted to prevent alias mutation
  when caches are shared across batched operations

## TurboQuant KV Cache Quantization

TurboQuant provides fractional-bit KV cache quantization with fused Metal
kernels. It achieves 89% KV cache memory savings with minimal quality loss.

### How It Works

Splits bit budget between keys and values. For example, `--kv-bits 3.5` means
3-bit keys + 4-bit values. Integer bit values fall back to uniform quantization.

Three codec families, each with fused Metal kernel implementations:

| Codec | Method | Best For |
|-------|--------|----------|
| MSE | Mean squared error minimization | General use, default |
| Polar/Polar-Prod | Polar coordinate encoding with multi-level angle quantization | High-dim KV heads |
| Prod | Product quantization | Compact representations |

### Usage

```bash
# Fractional bits auto-select TurboQuant
python -m mlx_vlm.generate \
    --model mlx-community/gemma-4-4b-it-4bit \
    --kv-bits 3.5 \
    --image photo.jpg \
    --prompt "Describe this image"

# Explicit scheme selection
python -m mlx_vlm.generate \
    --kv-bits 4 \
    --kv-quant-scheme turboquant \
    --quantized-kv-start 5000
```

### CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--kv-bits` | float | None | Bits for KV quantization. Fractional values auto-enable TurboQuant |
| `--kv-quant-scheme` | str | `"uniform"` | `"uniform"` or `"turboquant"` |
| `--kv-group-size` | int | 64 | Group size for uniform quantization |
| `--quantized-kv-start` | int | 5000 | Start layer index for quantization |

### Key Classes

```python
class TurboQuantKVCache(_BaseCache):
    """Fractional-bit KV cache with fused Metal kernels."""

    def update_and_fetch(self, keys, values):
        # Quantize and store new KV, return attention scores
        ...

    def quantize(self, vectors):
        # Compress to packed quantized form
        ...

    def dequantize(self, state):
        # Reconstruct from quantized state
        ...
```

State types (NamedTuples): `TurboQuantMSEState`, `TurboQuantProdState`,
`TurboQuantPolarState`, `TurboQuantSplitState` -- each stores
`(keys_norms, keys_packed, values_norms, values_packed)`.

### Performance

- Fused Metal kernels reduce dispatch count from 7 to 1
- Tiled MSE score kernel preloads queries in registers
- Two-pass fused decode kernels, single-tile value weighted sum (2x speedup)
- Hadamard transform integration (RHT forward/inverse)
- Fused KV quantization: single dispatch for both keys + values

### Race Condition Fix (Apr 2026)

The fused fast-quantize Metal kernels had a race condition in threadgroup
memory: non-atomic `|=` operations on `packed_shared` caused silent data
corruption when multiple threads wrote to the same 32-bit packed word. Only
one thread's contribution survived per word, with zeros elsewhere. This
produced inverted MSE behavior (4-bit cache was MORE corrupted than 2-bit).

Fix: `threadgroup atomic_uint` with `atomic_store_explicit` /
`atomic_fetch_or_explicit` / `atomic_load_explicit` (relaxed ordering). After
the fix, fast and slow quantize paths produce byte-identical packed indices
for all tested bit-widths (2/3/4/5/6, dim 64 and 128), and MSE is monotone
in bits as expected.

## VisionFeatureCache

`VisionFeatureCache` caches vision encoder outputs across conversation turns,
avoiding redundant reprocessing when the same image appears in multi-turn
conversations. This is especially important for VLMs where vision encoding
is the most expensive step in the first turn.

## Trust Level

mlx-vlm is third-party code. When using its patterns:

- Verify attention implementations match mlx-lm conventions
- Check that KV cache usage follows the `_BaseCache` interface
- Confirm `mx.fast` ops are used where appropriate (SDPA, RMS norm)
- Watch for type promotion issues in vision/audio encoders (float32
  computation in norms is intentional for numerical stability)
- Audio encoder uses manual padding + Conv2d rather than MLX's built-in
  padding -- verify shapes match expectations
