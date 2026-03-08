last updated: 2026-03-08

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

OpenAI-compatible HTTP server.

```bash
mlx_lm.server --model MODEL --port 8080 --host 0.0.0.0
```

Key flags: `--model`, `--port`, `--host`, `--adapter-path`, `--log-level`,
`--chat-template`, `--pipeline`.

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
