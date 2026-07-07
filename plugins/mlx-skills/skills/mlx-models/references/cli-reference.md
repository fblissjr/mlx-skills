last updated: 2026-07-07

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
`--min-p`, `--top-k`, `--kv-bits`, `--max-kv-size`, `--quantized-kv-start`,
`--adapter-path`, `--draft-model`, `--num-draft-tokens`. (No
`--repetition-penalty` or `--prefill-step-size` flag -- those aren't part of
this CLI's argparser.)

## chat

Interactive chat session with a model.

```bash
mlx_lm.chat --model mlx-community/Llama-3.2-3B-Instruct-4bit
```

`chat` has its own reduced argparser, not the full `generate` flag set. Key
flags: `--model`, `--temp`, `--top-p`, `--xtc-probability`,
`--xtc-threshold`, `--seed`, `--max-kv-size`, `--max-tokens`,
`--system-prompt`, `--adapter-path`, `--pipeline`. Notably missing `--top-k`,
`--min-p`, and `--kv-bits`, which `generate` has.

## lora

Fine-tune with LoRA or DoRA adapters.

```bash
mlx_lm.lora --model MODEL --data DATA_DIR --train --iters 1000
```

Key flags: `--model`, `--data`, `--train`, `--test`, `--iters`, `--batch-size`,
`--num-layers` (layers to fine-tune, -1 for all), `--fine-tune-type`
(lora/dora/full), `--adapter-path`, `--val-batches` (0 to skip validation),
`--config` (path to a YAML training config file).

## convert

Convert HuggingFace models to MLX format.

```bash
mlx_lm.convert --hf-path MODEL --mlx-path OUTPUT --quantize --q-bits 4
```

Key flags: `--hf-path` / `--model`, `--mlx-path`, `-q` / `--quantize`,
`--q-bits`, `--q-group-size`, `--q-mode` (choices: affine, mxfp4, nvfp4,
mxfp8), `--dtype` (float16/bfloat16/float32), `--quant-predicate`
(mixed-bit recipes), `-d` / `--dequantize`, `--upload-repo`,
`--trust-remote-code`.

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

Benchmark model inference speed using randomly generated prompt tokens (no
`--prompt` flag -- length is controlled by `--prompt-tokens`).

```bash
mlx_lm.benchmark --model MODEL --prompt-tokens 512 --generation-tokens 1024
```

Key flags: `--model`, `--prompt-tokens`, `--generation-tokens`,
`--batch-size`, `--num-trials`, `--pipeline`, `--quantize-activations`,
`--prefill-step-size`, `--delay`.

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

Evaluate a model on lm-evaluation-harness tasks.

```bash
mlx_lm.evaluate --model MODEL --tasks hellaswag arc_easy
```

Key flags: `--model`, `--tasks` (required, one or more lm-evaluation-harness
task names), `--batch-size`, `--num-shots`, `--limit`,
`--fewshot-as-multiturn`, `--apply-chat-template`.

## perplexity

Compute perplexity on a Hugging Face dataset (mlx-lm dataset format).

```bash
mlx_lm.perplexity --model MODEL --data-path allenai/tulu-3-sft-mixture
```

Key flags: `--model`, `--data-path`, `--num-samples`, `--sequence-length`,
`--batch-size`.

## manage

Manage downloaded models.

```bash
mlx_lm.manage --scan                    # List downloaded models
mlx_lm.manage --delete --pattern MODEL  # Delete models matching pattern
```

## upload

Upload a converted model to HuggingFace Hub.

```bash
mlx_lm.upload --path MODEL_DIR --upload-repo USER/REPO
```

## share

Distribute a model's files to other nodes in an MLX distributed cluster --
this is not a HuggingFace Hub operation.

```bash
mlx_lm.share --model MODEL --hostfile hosts.json
```

Key flags: `--path` / `--model` (one required), `--hostfile`, `--dst`,
`--tmpdir`.

## Quantization Commands

### awq (Activation-Aware Weight Quantization)

```bash
mlx_lm.awq --model MODEL --mlx-path OUTPUT --bits 4
```

Uses activation statistics from calibration data for optimal scale computation.

### gptq

```bash
mlx_lm.gptq --model MODEL --mlx-path OUTPUT --bits 4
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

All commands are invoked as `mlx_vlm.<command>` or `python -m mlx_vlm <command>`
(space-separated subcommand; the dotted `python -m mlx_vlm.<command>` form
still works but is deprecated in favor of these two). `lora` is an exception
-- see below. For server details, see
[references/serving.md](references/serving.md).

## generate

Generate text from image/audio inputs with a vision-language model.

```bash
mlx_vlm.generate --model mlx-community/gemma-4-4b-it-4bit \
    --image photo.jpg --prompt "Describe this image" \
    --max-tokens 200
```

Key flags: `--model`, `--image` (repeatable), `--audio` (repeatable),
`--prompt`, `--system`, `--max-tokens` (default: 2048), `--temperature`
(default: 0.0), `--chat` (multi-turn mode), `--resize-shape`,
`--prefill-step-size` (default: 2048), `--kv-bits`, `--kv-quant-scheme`,
`--enable-thinking`, `--thinking-budget`, `--adapter-path`.

## chat

Interactive multi-turn chat with vision-language models.

```bash
mlx_vlm.chat --model mlx-community/idefics2-8b-chatty-4bit
```

`chat` has its own reduced argparser (not the full `generate` flag set --
no `--image`/`--audio`/`--prompt`/`--system`, since those are supplied
interactively). Key flags: `--model`, `--temperature`, `--max-tokens`,
`--resize-shape`, `--prefill-step-size`, `--max-kv-size`, `--kv-bits`,
`--kv-group-size` (default: 64), `--kv-quant-scheme`, `--quantized-kv-start`
(default: 5000), `--eos-tokens`, `--skip-special-tokens`,
`--enable-thinking`, `--thinking-budget`.

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

Fine-tune vision-language models with LoRA or full weight tuning. Unlike the
other `mlx_vlm` commands, `lora` has no console-script entry point and is not
wired into the `mlx_vlm` subcommand dispatcher -- it can only be invoked as
a module.

```bash
python -m mlx_vlm.lora --model-path MODEL --dataset DATASET --iters 1000
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
