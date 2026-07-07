---
name: mlx-cuda
description: >-
  CUDA backend support for running MLX on NVIDIA GPUs. Use when writing,
  debugging, or porting MLX code that targets CUDA instead of Metal. Triggers
  on "cuda", "nvidia", "mx.cuda", "cuda_kernel", "precompiled_cuda_kernel",
  "NVIDIA GPU", "run mlx on cuda", "mlx cuda backend", or any project using
  MLX with CUDA. Covers backend detection, custom CUDA kernels, and
  Metal-to-CUDA kernel migration.
  Invoke with /mlx-skills:mlx-cuda.
license: MIT
compatibility: "Requires NVIDIA GPU with CUDA support and Python 3.9+"
allowed-tools: "Read, Glob, Grep"
metadata:
  author: Fred Bliss
  version: 0.5.11
  last_verified: "2026-07-07"
---

# MLX CUDA Backend

> **This skill covers MLX's CUDA backend for NVIDIA GPUs. For core MLX concepts
> (lazy evaluation, unified memory, compilation), load the `mlx` skill first.
> Do not search the web unless you have exhausted the reference files and
> confirmed the information is not here.**

MLX supports NVIDIA GPUs via CUDA as an alternative backend to Metal on Apple
silicon. The API is identical -- the same MLX code runs on either backend
transparently. The differences are in backend detection, custom kernels, and
device-specific tuning.

## Backend Detection

```python
import mlx.core as mx

mx.cuda.is_available()   # True if CUDA backend is available
mx.metal.is_available()  # Apple silicon only -- False on CUDA-only systems

# mx.gpu resolves to whichever backend is available
mx.default_device()      # Shows the active device
```

When both backends are available, MLX uses the platform default. All standard
MLX operations, compilation, and function transformations work identically
regardless of backend.

## Custom CUDA Kernels

Use `mx.fast.cuda_kernel` to write custom CUDA kernels. The API mirrors
`mx.fast.metal_kernel` with CUDA-specific differences.

```python
kernel = mx.fast.cuda_kernel(
    name="my_kernel",
    input_names=["inp"],
    output_names=["out"],
    source=source,           # CUDA kernel body
    header="",               # Optional header
    ensure_row_contiguous=True,
    shared_memory=0,         # Dynamic shared memory in bytes
)

outputs = kernel(
    inputs=[a],
    output_shapes=[a.shape],
    output_dtypes=[a.dtype],
    grid=(a.size, 1, 1),    # Total threads (not blocks)
    threadgroup=(256, 1, 1), # Threads per block
    template=[("T", mx.float32)],
    init_value=None,
    verbose=False,
)
```

### Simple Example

```python
source = '''
    auto grid = cooperative_groups::this_grid();
    uint elem = grid.thread_rank();
    T tmp = inp[elem];
    out[elem] = exp(tmp);
'''

kernel = mx.fast.cuda_kernel(
    name="myexp",
    input_names=["inp"],
    output_names=["out"],
    source=source,
)

def exp_elementwise(a):
    return kernel(
        inputs=[a],
        template=[("T", mx.float32)],
        grid=(a.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
    )[0]
```

### Precompiled CUDA Kernels

For production deployment, precompile kernels to avoid JIT overhead:

```python
kernel = mx.fast.precompiled_cuda_kernel(
    name="my_kernel",
    compiled_source=compiled_bytes,  # Pre-compiled kernel binary
    input_names=["inp"],
    output_names=["out"],
    ...
)
```

### Differentiable Kernels

Custom CUDA kernels work with `@mx.custom_function` identically to Metal
kernels. See `references/custom-kernels.md` in the `mlx` skill for the full
pattern with `@my_op.vjp`.

## Porting Metal Kernels to CUDA

| Metal | CUDA |
|-------|------|
| `thread_position_in_grid.x` | `cooperative_groups::this_grid().thread_rank()` |
| `threads_per_simdgroup` | `warpSize` |
| `simd_sum(x)` | Use `__shfl_down_sync` warp primitives |
| `metal::exp(x)` | `exp(x)` |
| `metal::fast::exp(x)` | `__expf(x)` |
| `atomic_fetch_add_explicit` | `atomicAdd` |
| `atomic_outputs=True` | `shared_memory=N` (dynamic shared memory in bytes) |
| `threadgroup T* shared` | `extern __shared__ T shared[]` (via `shared_memory` param) |

Grid is specified in total threads (not blocks), matching Metal convention.

For general custom kernel performance guidance (build once/call many, templates,
verbose mode), see `references/custom-kernels.md` in the `mlx` skill.

## CUDA-Specific Operations

### SegmentedMM for MoE Routing

`mx.segmented_mm` performs batched variable-length matrix multiplications, useful
for Mixture of Experts routing where each expert processes a different number of
tokens:

```python
# a: (total_tokens, D), b: (num_experts, D, E)
# segments: 1D int array of length num_experts with cumulative token counts
# e.g., segments=[5, 12, 20] means expert 0 gets tokens 0-4, expert 1 gets 5-11, etc.
result = mx.segmented_mm(a, b, segments)  # -> (total_tokens, E)
```

### QMV Kernel Support

The CUDA backend supports quantized matrix-vector (QMV) kernels with expanded
floating-point quantization modes including 3-bit, 5-bit, and 6-bit support in
addition to the standard 4-bit and 8-bit modes. These are used internally by
`nn.QuantizedLinear` -- specify the bit width via the `bits` parameter:

```python
layer = nn.QuantizedLinear(in_dims, out_dims, bits=6, group_size=64)
```

### Windows Build Support

MLX's CUDA backend now supports building on Windows with cuDNN integration,
enabling NVIDIA GPU acceleration on Windows systems.

## References

- [references/cuda_ops.md](references/cuda_ops.md) -- CUDA quantized matmul
  operations: gather_qmm (MoE routing), segmented_mm, bit width support per
  hardware, split-K optimization

## Related Skills

- **`mlx`** -- Core MLX framework (lazy evaluation, compile, nn.Module, etc.)
- **`fast-mlx`** -- Performance optimization (profiling, memory, compilation)
