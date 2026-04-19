last updated: 2026-04-07

# CUDA Quantized Operations

Operations specific to or optimized for the CUDA backend. For core MLX
quantization concepts and `nn.QuantizedLinear`, load the `mlx` skill.

## gather_qmm -- Gather Quantized Matrix Multiplication

Batched quantized matmul with index-based gathering, primarily used for MoE
expert routing where different inputs route to different weight matrices.

```python
mx.gather_qmm(
    x,                    # Input array (>= 2D)
    w,                    # Quantized weight matrix (packed uint32)
    scales,               # Per-group scales
    biases=None,          # Per-group biases (required for affine mode)
    lhs_indices=None,     # Indices into x batch dims
    rhs_indices=None,     # Indices into w batch dims
    transpose=True,       # x @ w.T (True) or x @ w (False)
    group_size=None,      # Quantization group size (mode-dependent default)
    bits=None,            # Bits per element (mode-dependent default)
    mode="affine",        # Quantization mode
    sorted_indices=False, # Faster path if indices are pre-sorted
)
```

### Quantization Modes

| Mode | Default Group Size | Default Bits | Biases | Notes |
|------|-------------------|--------------|--------|-------|
| `affine` | 64 | 4 | Required | Standard, supports 3/4/5/6/8-bit |
| `nvfp4` | 16 | 4 | No | NVIDIA FP4, uint8 scales, global scale |
| `mxfp4` | 32 | 4 | No | Microscaling FP4, uint8 scales |
| `mxfp8` | 32 | 8 | No | Microscaling FP8, uint8 scales |

### Bit Width Support (affine mode)

| Bits | CUDA SM 90 (Hopper) | CUDA SM 80 (Ampere) | CUDA Naive | Metal |
|------|---------------------|---------------------|------------|-------|
| 3 | No | Yes | Yes | Yes |
| 4 | Yes | Yes | Yes | Yes |
| 5 | No | Yes | Yes | Yes |
| 6 | No | Yes | Yes | Yes |
| 8 | Yes | Yes | Yes | Yes |

SM 90 kernels use CUTLASS with cluster shapes for maximum throughput but only
support even bit widths. The SM 80 and naive fallbacks handle all widths via
`NumericArrayConverter` specializations.

### MoE Routing Example

```python
# Route tokens to quantized expert weights
# x: (batch, tokens, hidden)
# expert_weights: (num_experts, hidden, intermediate) quantized
# expert_scales: matching scales
# indices: (batch, tokens) -> expert assignments

output = mx.gather_qmm(
    x, expert_weights, expert_scales,
    biases=expert_biases,
    rhs_indices=indices,
    sorted_indices=True,  # Enable fast path if indices are sorted
    group_size=64,
    bits=4,
)
```

### Hardware Dispatch

On CUDA, the kernel is selected by problem shape and compute capability:

- **QMV path** (M=1, small N/K): Optimized matrix-vector kernels
- **SM 90 path** (Hopper): CUTLASS with tile shapes up to 128x256, cluster
  shapes for multi-SM execution. Requires affine mode, transpose=True,
  even bit widths, group_size >= K
- **SM 80 path** (Ampere): CUTLASS with full bit width support
- **Naive fallback**: CuTe-based generic kernel for all configurations

## segmented_mm -- Segmented Matrix Multiplication

Variable-length batched matrix multiplication for MoE. Each segment defines
a contiguous slice of the K dimension, enabling per-expert computation without
padding.

```python
mx.segmented_mm(
    a,         # (M, K) input matrix -- must be 2D
    b,         # (K, N) weight matrix -- must be 2D
    segments,  # (..., 2) uint32 segment boundaries [start, end)
)
# Returns: (num_segments, M, N)
```

### MoE Usage

```python
# Each expert processes a different token subset
# segments define which rows of K each expert handles
segments = mx.array([[0, 64], [64, 192], [192, 256]], dtype=mx.uint32)
result = mx.segmented_mm(tokens, weights, segments)
# result[0] = tokens @ weights[0:64, :]
# result[1] = tokens @ weights[64:192, :]
# result[2] = tokens @ weights[192:256, :]
```

### Constraints

- Both inputs must be exactly 2D (no batch dimensions)
- Segments must have `uint32` dtype with last dimension = 2
- Returns real floating-point types only

## Split-K for Quantized Matmul (Metal)

On Metal, quantized matmul for small M values uses a split-K decomposition
that divides the K dimension across multiple threadgroups, each computing a
partial result. The partials are then reduced. This improves GPU utilization
when M is too small to fill all threadgroups with the standard tiling.

This is transparent -- no API change. The backend selects split-K
automatically based on the problem shape.

## When to Use Each Op

| Scenario | Op | Why |
|----------|-----|-----|
| Standard quantized inference | `nn.QuantizedLinear` | Wraps `mx.quantized_matmul` |
| MoE with quantized experts | `mx.gather_qmm` | Index-based routing to quantized weights |
| MoE with dense experts | `mx.segmented_mm` | Variable-length segments without padding |
| Single-token inference | (auto) | QMV kernels selected automatically for M=1 |
| Small-M quantized matmul on Metal | (auto) | Split-K selected automatically |
