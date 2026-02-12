# MLX Skill Update Report

Changes since: 30days

## mlx

**80 commits**, **872 files changed** (116 Python)

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

### Potential Breaking Changes

  - [a8197795f5d6] replace MLX_IBV_COORDINATOR with MLX_JACCL_COORDINATOR (#2986)

### Recent Commits

  - `72e94c81e168` [CUDA] Attention sinks in cuDNN SDPA (#3118) (Cheng, 2026-02-11)
  - `4c86c1e55adc` Fix precision in Metal fused attention (#3119) (Awni Hannun, 2026-02-10)
  - `be52cf660b5b` register pressure (#3116) (Anastasiia Filippova, 2026-02-10)
  - `54bb3eea424b` [CUDA] Use cuDNN SDPA for decoding when using fixed-size KV cache (#3113) (Cheng, 2026-02-10)
  - `5e018de4e5f1` Quantize module to QQLinear (#3106) (Anastasiia Filippova, 2026-02-09)
  - `9cd4b9be91d3` [CUDA] Set current device before allocating memory (#3110) (Cheng, 2026-02-08)
  - `566bc16b7c37` Cleanup test_fast_sdpa.py (#3112) (Cheng, 2026-02-08)
  - `8fe1d09207c6` Fix residency set with user provided buffer (#3108) (Awni Hannun, 2026-02-06)
  - `ef3fbc60a3ad` is_available() should check the device index too (#3107) (Ronan Collobert, 2026-02-06)
  - `69fd3fa9b1e2` Patch bump (#3102) (Angelos Katharopoulos, 2026-02-06)
  - `185b06d9efc1` Patch for multi device CUDA (#3100) (Awni Hannun, 2026-02-05)
  - `90e38f7b931e` Fix qmv_impl for small N (#3096) (Manuel Candales, 2026-02-05)
  - `ceea57149007` JACCL update (#3094) (Angelos Katharopoulos, 2026-02-05)
  - `99ca62c4d337` Fix 2pass sdpa on < M2 (#3099) (Awni Hannun, 2026-02-05)
  - `206cf07e5b6b` Fix non simd f16 build (#3097) (Awni Hannun, 2026-02-05)
  - ... and 65 more

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

**55 commits**, **207 files changed** (187 Python)

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

  - `7e67225e1d7b` Faster DSV32 generation (#885) (Tarjei Mandt, 2026-02-13)
  - `0fd312649627` [MODEL] support qwen3.5 series w/o vision (#869) (JJJYmmm, 2026-02-12)
  - `ca0d1c9630d9` LongCat MLA (#868) (Tarjei Mandt, 2026-02-13)
  - `82edd51a1e60` Devstral tool parser (#874) (Awni Hannun, 2026-02-11)
  - `aca4c149a189` Make validation set optional in training process (#857) (Gökdeniz Gülmez, 2026-02-11)
  - `8f1c56ec8303` Fix DeepSeek V3.2 indexer and weight loading (#866) (Tarjei Mandt, 2026-02-11)
  - `84ae19e675fb` Pythonic tool calling for LFM2 models (#864) (viktike, 2026-02-10)
  - `645a326a2e35` Bump version for next release (#865) (Awni Hannun, 2026-02-09)
  - `fd6959dca7a7` Fix Kimi Linear (#853) (Tarjei Mandt, 2026-02-07)
  - `f18526f8d66f` DSV3 MLA (#839) (Awni Hannun, 2026-02-04)
  - `25a4c8369e3c` Fix sliding window mask during generation (#843) (Tarjei Mandt, 2026-02-05)
  - `e08ec15b7201` Fix batch mamba (#842) (Awni Hannun, 2026-02-03)
  - `b77ec6b951e8` Fix Step 3.5 Flash model conversion (#840) (Tarjei Mandt, 2026-02-04)
  - `ab050d1fac2e` Deepseek V3.2 implementation fixes (#838) (Sebastian Jug, 2026-02-03)
  - `942b3ed4b61b` fix: handle GLM 4.7 tool call fallbacks (#792) (Josh Lehman, 2026-02-03)
  - ... and 40 more

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
  - ... and 96 more

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
  - ... and 48 more

## mlx-vlm

**31 commits**, **359 files changed** (278 Python)

### Watched File Changes

  - `mlx_vlm/utils.py`

### Recent Commits

  - `dd7cef17b029` Honor quantization_config (#692) (Pedro Cuenca, 2026-02-11)
  - `4fca3b488440` video_generate.py: handle capitalized extensions in is_video_file (#719) (Nixuge, 2026-02-11)
  - `2ee061823281` Fix KeyError: image_token_index for Qwen2-VL models (#720) (Jesse Rodriguez, 2026-02-11)
  - `6ce2ee35fd3f` Use mlx-lm provided logits processors and samplers (#724) (Sunny He, 2026-02-11)
  - `eadc25f6a682` docs: clarify that some models need [torch] extra for torchvision (#716) (Harikrishna KP, 2026-02-11)
  - `ebe2bc47c4e2` qwen3_omni_moe: Fix destructuring error after get_input_embeddings's result changed from a tuple to an object (#723) (Nixuge, 2026-02-11)
  - `1028599f8f8d` [MODEL] support qwen3.5 series (#722) (JJJYmmm, 2026-02-09)
  - `21f9b458765e` Refactor Attention class to calculate n_kv_heads dynamically based on head_dim, improving clarity and maintainability of the code. (#713) (Prince Canuma, 2026-02-04)
  - `f0b4f32327ee` [PaddleOCR] Fix hardcoded processor config (#712) (Prince Canuma, 2026-02-04)
  - `a2542774e545` Fix wired limit and add prefill step size argument (#699) (Prince Canuma, 2026-02-04)
  - `61b411d462ce` fix: enhance model import error messages. (#710) (Anton Vice, 2026-02-03)
  - `d54b53c579a8` Add GLM-OCR (#706) (Prince Canuma, 2026-02-03)
  - `79fddfc58a0c` fix to attention and mask for qwen2_vl (#704) (hturbe, 2026-02-02)
  - `aaab6e53014b` Refactor input embedding handling in batch gen (#694) (Prince Canuma, 2026-01-29)
  - `48fc189b3c3e` TFMS v5 RC3 + Fix processor registry (#693) (Prince Canuma, 2026-01-28)
  - ... and 16 more

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
  - ... and 211 more

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
  - ... and 32 more

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
