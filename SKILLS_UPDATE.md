# MLX Skill Update Report

Changes since: 30days

## mlx

**54 commits**, **872 files changed** (116 Python)

### Watched File Changes

  - `python/mlx/nn/layers/linear.py`
  - `python/mlx/nn/layers/normalization.py`
  - `python/mlx/nn/layers/transformer.py`
  - `python/mlx/nn/layers/activations.py`
  - `python/mlx/nn/layers/convolution.py`
  - `python/mlx/nn/layers/recurrent.py`
  - `python/mlx/nn/layers/positional_encoding.py`
  - `python/mlx/nn/layers/quantized.py`
  - `python/mlx/nn/layers/embedding.py`
  - `python/mlx/nn/losses.py`
  - `python/mlx/nn/init.py`
  - `python/mlx/nn/utils.py`
  - `python/mlx/optimizers/__init__.py`
  - `python/mlx/optimizers/optimizers.py`
  - `python/mlx/optimizers/schedulers.py`
  - `python/mlx/utils.py`
  - `docs/src/usage/lazy_evaluation.rst`
  - `docs/src/usage/compile.rst`
  - `docs/src/usage/unified_memory.rst`

### Recent Commits

  - `d4c81062ad0d` [Metal] Fix 32-bit integer overflow in conv3d unfold kernel (#3143) (Kellen Sun, 2026-02-19)
  - `f2f2d1645156` Export: preserve Dtype state values in export callback arguments (#3145) (Alex Skryl, 2026-02-19)
  - `daf18e76ca14` Fix fence synchronization accross command buffers (#3144) (Awni Hannun, 2026-02-18)
  - `06305022abac` Tensor scale nvfp4 (#3022) (Anastasiia Filippova, 2026-02-18)
  - `360639c2dfc1` Add the hamming window function (#3135) (willem adnet, 2026-02-17)
  - `3bbe87e6dcc7` Add hanning window function (#3124) (willem adnet, 2026-02-16)
  - `e226af720e15` Propagate quantization mode in quantized layers (#3133) (vskiwi, 2026-02-16)
  - `43f4a7482652` Manage stream placement in import function (#3127) (Awni Hannun, 2026-02-15)
  - `c184262d29a8` Fix donation in sdpa vector (#3121) (Angelos Katharopoulos, 2026-02-12)
  - `72e94c81e168` [CUDA] Attention sinks in cuDNN SDPA (#3118) (Cheng, 2026-02-11)
  - `4c86c1e55adc` Fix precision in Metal fused attention (#3119) (Awni Hannun, 2026-02-10)
  - `be52cf660b5b` register pressure (#3116) (Anastasiia Filippova, 2026-02-10)
  - `54bb3eea424b` [CUDA] Use cuDNN SDPA for decoding when using fixed-size KV cache (#3113) (Cheng, 2026-02-10)
  - `5e018de4e5f1` Quantize module to QQLinear (#3106) (Anastasiia Filippova, 2026-02-09)
  - `9cd4b9be91d3` [CUDA] Set current device before allocating memory (#3110) (Cheng, 2026-02-08)
  - ... and 39 more

### Other Python Files Changed

  - `benchmarks/numpy/single_ops.py`
  - `benchmarks/numpy/time_utils.py`
  - `benchmarks/python/batch_matmul_bench.py`
  - `benchmarks/python/blas/bench_gemm.py`
  - `benchmarks/python/blas/bench_gemv.py`
  - `benchmarks/python/comparative/bench_mlx.py`
  - `benchmarks/python/comparative/bench_torch.py`
  - `benchmarks/python/comparative/compare.py`
  - `benchmarks/python/compile_bench.py`
  - `benchmarks/python/conv1d_bench.py`
  - `benchmarks/python/conv2d_bench_cpu.py`
  - `benchmarks/python/conv2d_train_bench_cpu.py`
  - `benchmarks/python/conv2d_transpose_bench_cpu.py`
  - `benchmarks/python/conv3d_bench_cpu.py`
  - `benchmarks/python/conv3d_train_bench_cpu.py`
  - ... and 101 more

## mlx-lm

**46 commits**, **209 files changed** (189 Python)

### Watched File Changes

  - `mlx_lm/generate.py`
  - `mlx_lm/models/cache.py`
  - `mlx_lm/models/base.py`
  - `mlx_lm/utils.py`
  - `mlx_lm/tuner/lora.py`
  - `mlx_lm/tuner/trainer.py`
  - `mlx_lm/models/llama.py`
  - `mlx_lm/sample_utils.py`

### Recent Commits

  - `321e764e0ab6` Make the cache limits more friendly (#910) (Angelos Katharopoulos, 2026-02-19)
  - `83ff9c96d571` Improve the cache size limits (#906) (Angelos Katharopoulos, 2026-02-19)
  - `9c113f701968` Allow reading LFM2 models nested rope params (#908) (Yuri Khrustalev, 2026-02-18)
  - `7d6c5e4af7aa` Add tie_word_embeddings modulars in mistral and qwen3 moe (#889) (Gökdeniz Gülmez, 2026-02-18)
  - `ad067ea627c7` bump for next version (#904) (Awni Hannun, 2026-02-17)
  - `d7b91e80f073` Fix sharded rms norm in MiniMax M2.5 (#898) (Angelos Katharopoulos, 2026-02-16)
  - `1fd521c3c79e` fix qwen3.5 casting to fp32 (#902) (Awni Hannun, 2026-02-16)
  - `572ada278c39` server: add usage.prompt_tokens_details.cached_tokens to json response (#849) (Ryan Goulden, 2026-02-16)
  - `fb47f8fb9944` Add the trust remote code option to mlx_lm perplexity  (#896) (Ivan Fioravanti, 2026-02-16)
  - `7a720882a7d4` Add JoyAI LLM Flash (#894) (Tarjei Mandt, 2026-02-16)
  - `014ebc6a4614` Fix mixed quant predicates for MLA models (#892) (spicyneuron, 2026-02-15)
  - `c6d9d3c9f58b` Share model (#871) (Angelos Katharopoulos, 2026-02-13)
  - `bcf630614ffb` Fix save/load of CacheList (#886) (Angelos Katharopoulos, 2026-02-12)
  - `1974376d704a` Add GLM5 (#867) (Gökdeniz Gülmez, 2026-02-12)
  - `7e67225e1d7b` Faster DSV32 generation (#885) (Tarjei Mandt, 2026-02-13)
  - ... and 31 more

### Model Files Changed

  - `mlx_lm/models/Klear.py`
  - `mlx_lm/models/__init__.py`
  - `mlx_lm/models/activations.py`
  - `mlx_lm/models/afm7.py`
  - `mlx_lm/models/afmoe.py`
  - `mlx_lm/models/apertus.py`
  - `mlx_lm/models/baichuan_m1.py`
  - `mlx_lm/models/bailing_moe.py`
  - `mlx_lm/models/bailing_moe_linear.py`
  - `mlx_lm/models/base.py`
  - `mlx_lm/models/bitlinear_layers.py`
  - `mlx_lm/models/bitnet.py`
  - `mlx_lm/models/cache.py`
  - `mlx_lm/models/cohere.py`
  - `mlx_lm/models/cohere2.py`
  - `mlx_lm/models/dbrx.py`
  - `mlx_lm/models/deepseek.py`
  - `mlx_lm/models/deepseek_v2.py`
  - `mlx_lm/models/deepseek_v3.py`
  - `mlx_lm/models/deepseek_v32.py`
  - ... and 97 more

### Tuner Files Changed

  - `mlx_lm/tuner/__init__.py`
  - `mlx_lm/tuner/callbacks.py`
  - `mlx_lm/tuner/datasets.py`
  - `mlx_lm/tuner/dora.py`
  - `mlx_lm/tuner/lora.py`
  - `mlx_lm/tuner/losses.py`
  - `mlx_lm/tuner/trainer.py`
  - `mlx_lm/tuner/utils.py`

### Other Python Files Changed

  - `benchmarks/server_benchmark.py`
  - `mlx_lm/__init__.py`
  - `mlx_lm/__main__.py`
  - `mlx_lm/_version.py`
  - `mlx_lm/benchmark.py`
  - `mlx_lm/cache_prompt.py`
  - `mlx_lm/chat.py`
  - `mlx_lm/chat_templates/__init__.py`
  - `mlx_lm/chat_templates/deepseek_v32.py`
  - `mlx_lm/cli.py`
  - `mlx_lm/convert.py`
  - `mlx_lm/evaluate.py`
  - `mlx_lm/examples/batch_generate_response.py`
  - `mlx_lm/examples/chat.py`
  - `mlx_lm/examples/generate_response.py`
  - ... and 49 more

## mlx-vlm

**49 commits**, **367 files changed** (285 Python)

### Watched File Changes

  - `mlx_vlm/utils.py`

### Recent Commits

  - `72d53ecc52bb` Fix batch inference to use InputEmbeddingsFeatures (#760) (Prince Canuma, 2026-02-23)
  - `8d16d655955c` Initialize qwen3_5_moe.LanguageModel with _position_ids (#753) (will-lms, 2026-02-19)
  - `3dd075d43a5c` Adding full weight finetuning (#499) (Gökdeniz Gülmez, 2026-02-19)
  - `a947fc76ebba` fix gemma3n short context (#751) (Prince Canuma, 2026-02-17)
  - `739f4ab7d4b6` Add dots-ocr (#749) (Prince Canuma, 2026-02-17)
  - `05296a595ad5` [Phiv3] Fix dtype cast (#748) (Prince Canuma, 2026-02-16)
  - `ac2f06803b75` bump mlx-lm (Prince Canuma, 2026-02-16)
  - `dafc569eb0ca` Update package versions in uv.lock: bump mlx-lm to 0.30.7, transformers to 5.2.0, and adjust related sdist and wheel URLs. (Prince Canuma, 2026-02-16)
  - `90e6c9d5731b` [Ministral3] Fix multi-image gen (Prince Canuma, 2026-02-16)
  - `0e769cf40e53` fix qwen2 (Prince Canuma, 2026-02-16)
  - `529d5698824e` fix qwen2.5 vision leak (Prince Canuma, 2026-02-16)
  - `3c7588329a13` format (Prince Canuma, 2026-02-16)
  - `97f13ca5710c` Add dtype fallback from text_config in convert.py; introduce quant_predicate and cast_predicate properties in language.py and qwen3_5.py (Prince Canuma, 2026-02-16)
  - `eb81dc0176a9` format (Prince Canuma, 2026-02-16)
  - `0b5791c38891` normalize inputs (Prince Canuma, 2026-02-16)
  - ... and 34 more

### Model Files Changed

  - `mlx_vlm/models/__init__.py`
  - `mlx_vlm/models/aya_vision/__init__.py`
  - `mlx_vlm/models/aya_vision/aya_vision.py`
  - `mlx_vlm/models/aya_vision/config.py`
  - `mlx_vlm/models/aya_vision/language.py`
  - `mlx_vlm/models/aya_vision/vision.py`
  - `mlx_vlm/models/base.py`
  - `mlx_vlm/models/cache.py`
  - `mlx_vlm/models/deepseek_vl_v2/__init__.py`
  - `mlx_vlm/models/deepseek_vl_v2/config.py`
  - `mlx_vlm/models/deepseek_vl_v2/conversation.py`
  - `mlx_vlm/models/deepseek_vl_v2/deepseek_vl_v2.py`
  - `mlx_vlm/models/deepseek_vl_v2/language.py`
  - `mlx_vlm/models/deepseek_vl_v2/processing_deepsek_vl_v2.py`
  - `mlx_vlm/models/deepseek_vl_v2/vision.py`
  - `mlx_vlm/models/deepseekocr/__init__.py`
  - `mlx_vlm/models/deepseekocr/config.py`
  - `mlx_vlm/models/deepseekocr/conversation.py`
  - `mlx_vlm/models/deepseekocr/deepseekocr.py`
  - `mlx_vlm/models/deepseekocr/language.py`
  - ... and 217 more

### Other Python Files Changed

  - `computer_use/autonomous_gui_agent.py`
  - `computer_use/autonomous_gui_agent_voice.py`
  - `computer_use/gui_agent.py`
  - `computer_use/gui_agent_voice.py`
  - `computer_use/utils.py`
  - `dev/load_q.py`
  - `examples/omni.py`
  - `examples/qwen3_omni_demo.py`
  - `examples/structured_outputs.py`
  - `examples/utils.py`
  - `mlx_vlm/__init__.py`
  - `mlx_vlm/__main__.py`
  - `mlx_vlm/chat.py`
  - `mlx_vlm/chat_ui.py`
  - `mlx_vlm/convert.py`
  - ... and 33 more

---

## Suggested Actions

Review the watched file changes above and update the following
skill reference files as needed:

- `mlx/references/nn-and-training.md` -- if nn layers, losses, optimizers, or schedulers changed
- `mlx/references/fundamentals.md` -- if core MLX APIs changed
- `mlx/references/anti-patterns.md` -- if new footguns were discovered
- `mlx-lm/references/patterns.md` -- if model patterns changed (cache, attention, generation)
- `mlx-lm/references/architecture.md` -- if loading, generation flow, or model registration changed
- `fast-mlx/references/*.md` -- if optimization techniques changed
