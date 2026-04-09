last updated: 2026-04-09

# mlx-lm CLI Reference

All commands are invoked as `mlx_lm.<command>` or `python -m mlx_lm <command>`.

## generate

Generate text from a prompt.

```bash
mlx_lm.generate --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --prompt "Hello" \
    --max-tokens 100 \
    --temp 0.7
```

Key flags: `--model`, `--prompt`, `--max-tokens`, `--temp`, `--top-p`,
`--min-p`, `--repetition-penalty`, `--kv-bits`, `--max-kv-size`,
`--prefill-step-size`, `--adapter-path`, `--draft-model`, `--num-draft-tokens`.

## chat

Interactive chat session with a model.

```bash
mlx_lm.chat --model mlx-community/Llama-3.2-3B-Instruct-4bit
```

Same generation flags as `generate`, plus `--system-prompt`.

## lora

Fine-tune with LoRA or DoRA adapters.

```bash
mlx_lm.lora --model MODEL --data DATA_DIR --train --iters 1000
```

Key flags: `--model`, `--data`, `--train`, `--test`, `--iters`, `--batch-size`,
`--lora-layers`, `--adapter-path`, `--val-batches` (0 to skip validation),
`--config` (path to training config JSON).

## convert

Convert HuggingFace models to MLX format.

```bash
mlx_lm.convert --hf-path MODEL --mlx-path OUTPUT --quantize --q-bits 4
```

Key flags: `--hf-path`, `--mlx-path`, `--quantize`, `--q-bits`, `--q-group-size`.

## fuse

Fuse LoRA/DoRA adapter weights into the base model.

```bash
mlx_lm.fuse --model MODEL --adapter-path ADAPTER --save-path OUTPUT
```

## server

OpenAI-compatible HTTP server. For comprehensive serving documentation
including both mlx-lm and mlx-vlm servers, architecture comparison, batching,
and deployment patterns, see [references/serving.md](references/serving.md).

```bash
mlx_lm.server --model MODEL --port 8080 --host 0.0.0.0
```

Key flags: `--model`, `--port`, `--host`, `--adapter-path`, `--log-level`,
`--chat-template`, `--pipeline`, `--decode-concurrency`, `--prompt-cache-size`.

Exposes `/v1/chat/completions` and `/v1/completions`.

## benchmark

Benchmark model inference speed.

```bash
mlx_lm.benchmark --model MODEL --prompt "Test prompt"
```

Key flags: `--model`, `--prompt`, `--prompt-tokens`, `--generation-tokens`,
`--batch-size`, `--num-trials`, `--kv-bits`, `--max-kv-size`.

Reports tokens/second for prefill and generation.

## cache_prompt

Pre-compute and save a prompt's KV cache.

```bash
mlx_lm.cache_prompt --model MODEL --prompt "System prompt..." \
    --prompt-cache-file cache.safetensors
```

Key flags: `--model`, `--prompt` (`-` for stdin), `--prompt-cache-file`,
`--kv-bits`, `--kv-group-size`, `--quantized-kv-start`, `--max-kv-size`.

## evaluate

Evaluate model on datasets.

```bash
mlx_lm.evaluate --model MODEL --data DATA_DIR
```

## perplexity

Compute perplexity on text.

```bash
mlx_lm.perplexity --model MODEL --text "Sample text..."
```

## manage

Manage downloaded models.

```bash
mlx_lm.manage --scan     # List downloaded models
mlx_lm.manage --delete MODEL  # Delete a model
```

## upload / share

Upload or share models on HuggingFace Hub.

```bash
mlx_lm.upload --model MODEL --repo-id USER/REPO
mlx_lm.share --model MODEL --repo-id USER/REPO
```

## Quantization Commands

### awq (Activation-Aware Weight Quantization)

```bash
mlx_lm.awq --model MODEL --mlx-path OUTPUT --q-bits 4
```

Uses activation statistics from calibration data for optimal scale computation.

### gptq

```bash
mlx_lm.gptq --model MODEL --mlx-path OUTPUT --q-bits 4
```

Post-training quantization using calibration data and Hessian approximation.

### dwq (Data-Aware Weight Quantization)

```bash
mlx_lm.dwq --model MODEL --mlx-path OUTPUT
```

### dynamic_quant

```bash
mlx_lm.dynamic_quant --model MODEL --mlx-path OUTPUT
```

Per-token dynamic quantization for variable precision.

---

# mlx-vlm CLI Reference

All commands are invoked as `mlx_vlm.<command>` or `python -m mlx_vlm.<command>`.
For server details, see [references/serving.md](references/serving.md).

## generate

Generate text from image/audio inputs with a vision-language model.

```bash
mlx_vlm.generate --model mlx-community/gemma-4-4b-it-4bit \
    --image photo.jpg --prompt "Describe this image" \
    --max-tokens 200
```

Key flags: `--model`, `--image` (repeatable), `--audio` (repeatable),
`--prompt`, `--system`, `--max-tokens` (default: 100), `--temperature`
(default: 0.7), `--chat` (multi-turn mode), `--resize-shape`,
`--prefill-step-size` (default: 512), `--kv-bits`, `--kv-quant-scheme`,
`--enable-thinking`, `--thinking-budget`, `--adapter-path`.

## chat

Interactive multi-turn chat with vision-language models.

```bash
mlx_vlm.chat --model mlx-community/idefics2-8b-chatty-4bit
```

Same generation flags as `generate`, plus `--eos-tokens`,
`--skip-special-tokens`, `--kv-group-size` (default: 64),
`--quantized-kv-start` (default: 128).

## convert

Convert HuggingFace vision-language models to MLX format.

```bash
mlx_vlm.convert --hf-path MODEL --mlx-path OUTPUT --quantize --q-bits 4
```

Key flags: `--hf-path` / `--model`, `--mlx-path` (default: `mlx_model`),
`-q` / `--quantize`, `--q-bits`, `--q-group-size`, `--q-mode` (choices:
affine, mxfp4, nvfp4, mxfp8), `--dtype` (float32/float16/bfloat16),
`--quant-predicate` (mixed-bit recipes: mixed_2_6, mixed_3_4, etc.),
`-d` / `--dequantize`, `--upload-repo`.

## lora

Fine-tune vision-language models with LoRA or full weight tuning.

```bash
mlx_vlm.lora --model-path MODEL --dataset DATASET --iters 1000
```

Key flags: `--model-path`, `--dataset`, `--learning-rate` (default: 2e-5),
`--batch-size` (default: 4), `--iters` (default: 1000), `--epochs`,
`--max-seq-length` (default: 2048), `--lora-rank` (default: 8),
`--lora-alpha` (default: 16), `--full-finetune` (all weights),
`--train-vision` (unfreeze vision encoder), `--train-mode` (sft or orpo),
`--output-path` (default: `adapters.safetensors`), `--adapter-path` (resume).

## chat_ui

Web-based Gradio chat interface.

```bash
mlx_vlm.chat_ui --model mlx-community/gemma-4-4b-it-4bit
```

Key flags: `--model`. Temperature, max tokens, top-p, and repetition penalty
are configured via the Gradio UI controls.
