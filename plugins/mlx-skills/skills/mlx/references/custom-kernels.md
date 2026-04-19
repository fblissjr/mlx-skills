last updated: 2026-04-09

# Custom Metal Kernels

Write custom GPU kernels in MLX using `mx.fast.metal_kernel` for Apple silicon.
Use when no built-in op exists for your computation or when fusing many ops into
a single kernel improves performance.

For CUDA kernel support on NVIDIA GPUs, load the `mlx-cuda` skill.

## metal_kernel API

```python
kernel = mx.fast.metal_kernel(
    name="my_kernel",
    input_names=["inp"],
    output_names=["out"],
    source=source,           # Metal kernel body (not full function)
    header="",               # Optional header for helpers/includes
    ensure_row_contiguous=True,  # Copy inputs to row-contiguous if needed
    atomic_outputs=False,    # Use atomic outputs for concurrent writes
)

outputs = kernel(
    inputs=[a],
    output_shapes=[a.shape],
    output_dtypes=[a.dtype],
    grid=(a.size, 1, 1),    # Total threads (not threadgroups)
    threadgroup=(256, 1, 1), # Threads per threadgroup
    template=[("T", mx.float32)],  # Optional type templates
    init_value=None,         # Optional: initialize outputs to this value
    verbose=False,           # Print generated source for debugging
)
```

### Auto-generated function signature

The kernel body is wrapped in a full Metal function. The signature is generated
from:

- **Inputs**: `const device T* inp` for each input name + dtype
- **Outputs**: `device T* out` (or `device atomic<T>* out` if `atomic_outputs=True`)
- **Templates**: `template <typename T>` when template params are provided
- **Metal attributes**: `thread_position_in_grid`, `threads_per_simdgroup`, etc.
  are auto-detected from `source` and added as function arguments
- **Shape/strides**: `inp_shape`, `inp_strides`, `inp_ndim` are available if
  referenced in `source` (useful with `ensure_row_contiguous=False`)

### Simple example

```python
source = '''
    uint elem = thread_position_in_grid.x;
    T tmp = inp[elem];
    out[elem] = metal::exp(tmp);
'''

kernel = mx.fast.metal_kernel(
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

### Strided input support

Set `ensure_row_contiguous=False` and use the auto-provided shape/strides:

```python
source = '''
    uint elem = thread_position_in_grid.x;
    uint loc = elem_to_loc(elem, inp_shape, inp_strides, inp_ndim);
    T tmp = inp[loc];
    out[elem] = metal::exp(tmp);
'''

kernel = mx.fast.metal_kernel(
    name="myexp_strided",
    input_names=["inp"],
    output_names=["out"],
    source=source,
    ensure_row_contiguous=False,
)
```

`elem_to_loc` comes from MLX's built-in `utils.h`, automatically included.

### Atomic outputs for backward passes

For custom VJP kernels that accumulate gradients from multiple threads:

```python
kernel = mx.fast.metal_kernel(
    name="my_grad",
    input_names=["x", "cotangent"],
    output_names=["x_grad"],
    source=source,
    atomic_outputs=True,  # Enables atomic_fetch_add_explicit
)

outputs = kernel(
    inputs=[x, cotangent],
    output_shapes=[x.shape],
    output_dtypes=[x.dtype],
    grid=(grid_size, 1, 1),
    threadgroup=(256, 1, 1),
    init_value=0,  # Zero-initialize before atomic accumulation
)
```

### Pairing with mx.custom_function

Use `@mx.custom_function` to make custom kernels differentiable:

```python
@mx.custom_function
def my_op(x):
    return forward_kernel(inputs=[x], ...)[0]

@my_op.vjp
def my_op_vjp(primals, cotangent, output):
    x = primals[0]
    grad = backward_kernel(inputs=[x, cotangent], ...)[0]
    return (grad,)
```

This enables `mx.grad(my_op)` to work correctly with your custom kernel.

## Performance notes

- Build the kernel once, call many times -- avoid repeated JIT compilation
- Use `template` params for type-generic kernels instead of multiple kernel objects
- Set `verbose=True` to inspect the generated function signature for debugging
- For backward passes, use `simd_sum` (Metal) to reduce within simdgroups before
  atomic writes for much better performance
