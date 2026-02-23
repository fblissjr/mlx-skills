last updated: 2026-02-23

# nn Module and Training

Comprehensive reference for MLX's neural network module system, layers, losses,
optimizers, schedulers, and training patterns.

## nn.Module System

MLX's `nn.Module` subclasses `dict`. Parameters are dictionary entries, not
registered via a separate `nn.Parameter` wrapper like PyTorch.

Key design differences from PyTorch:

- `__call__` IS the forward pass. There is no separate `forward` method.
- Any `mx.array` attribute set on the module is a parameter.
- Module children are any attributes that are also `nn.Module` instances.
- The module tree is a nested dict structure, enabling JAX-style tree operations.

### Core Methods

| Method | Purpose |
|--------|---------|
| `parameters()` | Returns nested dict of all `mx.array` parameters |
| `trainable_parameters()` | Returns only unfrozen parameters (respects `freeze`) |
| `freeze()` | Marks all parameters as non-trainable |
| `unfreeze()` | Marks all parameters as trainable |
| `load_weights(path)` | Load weights from safetensors/npz file |
| `update(params)` | Update parameters from a nested dict |
| `apply_to_modules(fn)` | Apply a function to all submodules by name |
| `children()` | Returns immediate child modules |
| `leaf_modules()` | Returns all leaf modules (no children) |
| `state` | Property returning the full state dict (parameters + non-array state) |
| `set_dtype(dtype)` | Cast all parameters to the given dtype |

### Parameter Freezing

```python
model.freeze()  # Freeze everything
# Selectively unfreeze specific modules
model.apply_to_modules(lambda k, m: m.unfreeze() if "lora" in k else None)

# Check what is trainable
trainable = model.trainable_parameters()
```

## Building Custom Layers

```python
import mlx.core as mx
import mlx.nn as nn

class MyLayer(nn.Module):
    def __init__(self, in_dims: int, out_dims: int):
        super().__init__()
        self.linear = nn.Linear(in_dims, out_dims)
        self.norm = nn.RMSNorm(out_dims)

    def __call__(self, x: mx.array) -> mx.array:
        return self.norm(self.linear(x))
```

Composing layers:

```python
# Sequential composition
encoder = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
)

# Nesting modules
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [MyLayer(784, 256), MyLayer(256, 64)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

Lists of modules stored as attributes are automatically tracked as children.

## Available Layers

### Linear

| Layer | Description |
|-------|-------------|
| `nn.Linear(in, out, bias=True)` | Standard linear transform. Weight shape: `(out, in)` |
| `nn.Bilinear(in1, in2, out, bias=True)` | Bilinear transform on two inputs |

### Convolution

| Layer | Description |
|-------|-------------|
| `nn.Conv1d(in_ch, out_ch, kernel, stride=1, padding=0)` | 1D convolution |
| `nn.Conv2d(in_ch, out_ch, kernel, stride=1, padding=0)` | 2D convolution |
| `nn.Conv3d(in_ch, out_ch, kernel, stride=1, padding=0)` | 3D convolution |
| `nn.ConvTranspose1d(...)` | 1D transposed convolution |
| `nn.ConvTranspose2d(...)` | 2D transposed convolution |
| `nn.ConvTranspose3d(...)` | 3D transposed convolution |

### Normalization

| Layer | Description |
|-------|-------------|
| `nn.LayerNorm(dims, eps=1e-5)` | Layer normalization |
| `nn.RMSNorm(dims, eps=1e-5)` | Root mean square normalization |
| `nn.BatchNorm(num_features, eps=1e-5, momentum=0.1)` | Batch normalization |
| `nn.GroupNorm(num_groups, dims, eps=1e-5)` | Group normalization |
| `nn.InstanceNorm(dims, eps=1e-5)` | Instance normalization |

Prefer `mx.fast.rms_norm` and `mx.fast.layer_norm` for performance-critical
paths -- they accumulate in higher precision internally.

### Activation Functions

All available as both `nn.*` layers and `nn.*` functions:

| Function | Notes |
|----------|-------|
| `nn.ReLU` | Standard ReLU |
| `nn.LeakyReLU(negative_slope=0.01)` | Leaky ReLU |
| `nn.PReLU(num_parameters=1)` | Parametric ReLU (learnable slope) |
| `nn.ELU(alpha=1.0)` | Exponential linear unit |
| `nn.CELU(alpha=1.0)` | Continuously differentiable ELU |
| `nn.SELU` | Scaled ELU (self-normalizing) |
| `nn.ReLU6` | ReLU clamped to [0, 6] |
| `nn.GELU(approx="none")` | Gaussian error linear unit |
| `nn.SiLU` | Sigmoid linear unit (swish) |
| `nn.Mish` | Mish activation |
| `nn.Sigmoid` | Sigmoid |
| `nn.Tanh` | Hyperbolic tangent |
| `nn.Hardswish` | Hard approximation of swish |
| `nn.Hardshrink(lambd=0.5)` | Hard shrinkage |
| `nn.Softshrink(lambd=0.5)` | Soft shrinkage |
| `nn.Softplus(beta=1, threshold=20)` | Smooth approximation of ReLU |
| `nn.Softsign` | Softsign activation |
| `nn.Softmax` | Softmax (use `mx.softmax(x, precise=True)` for half-precision) |
| `nn.Softmin` | Softmin |
| `nn.LogSoftmax` | Log softmax |
| `nn.LogSigmoid` | Log sigmoid |
| `nn.GLU(axis=-1)` | Gated linear unit |
| `nn.Step(threshold=0)` | Step function |
| `nn.Hardtanh(min_val=-1, max_val=1)` | Clamped linear |

### Pooling

| Layer | Description |
|-------|-------------|
| `nn.MaxPool1d(kernel, stride=None, padding=0)` | 1D max pooling |
| `nn.MaxPool2d(kernel, stride=None, padding=0)` | 2D max pooling |
| `nn.MaxPool3d(kernel, stride=None, padding=0)` | 3D max pooling |
| `nn.AvgPool1d(kernel, stride=None, padding=0)` | 1D average pooling |
| `nn.AvgPool2d(kernel, stride=None, padding=0)` | 2D average pooling |
| `nn.AvgPool3d(kernel, stride=None, padding=0)` | 3D average pooling |

### Dropout

| Layer | Description |
|-------|-------------|
| `nn.Dropout(p=0.5)` | Standard dropout (only active during training) |
| `nn.Dropout2d(p=0.5)` | Channel-wise dropout for 2D inputs |
| `nn.Dropout3d(p=0.5)` | Channel-wise dropout for 3D inputs |

### Recurrent

| Layer | Description |
|-------|-------------|
| `nn.RNN(input_size, hidden_size, bias=True, nonlinearity="tanh")` | Vanilla RNN |
| `nn.GRU(input_size, hidden_size, bias=True)` | Gated recurrent unit |
| `nn.LSTM(input_size, hidden_size, bias=True)` | Long short-term memory |

### Transformer

| Layer | Description |
|-------|-------------|
| `nn.MultiHeadAttention(dims, num_heads, ...)` | Multi-head attention |
| `nn.Transformer(dims, num_heads, ...)` | Full encoder-decoder transformer |
| `nn.TransformerEncoder(num_layers, dims, num_heads)` | Encoder stack |
| `nn.TransformerDecoder(num_layers, dims, num_heads)` | Decoder stack |
| `nn.TransformerEncoderLayer(dims, num_heads, ...)` | Single encoder layer |
| `nn.TransformerDecoderLayer(dims, num_heads, ...)` | Single decoder layer |

Note: mlx-lm implements custom attention and transformer blocks rather than
using `nn.MultiHeadAttention`/`nn.Transformer`, because the custom versions
integrate better with KV caching, RoPE variants, and quantization. For mlx-lm's
implementation patterns, load the `mlx-lm` skill and see `references/patterns.md`.

### Embedding

| Layer | Description |
|-------|-------------|
| `nn.Embedding(num_embeddings, dims)` | Standard embedding lookup |

`Embedding` also supports `as_linear(x)` which computes `x @ weight.T` for
weight-tied models.

### Positional Encoding

| Layer | Description |
|-------|-------------|
| `nn.RoPE(dims, traditional=False, base=10000)` | Rotary position embeddings |
| `nn.SinusoidalPositionalEncoding(dims, ...)` | Sinusoidal position encoding |
| `nn.ALiBi(...)` | Attention with linear biases |

For RoPE, prefer `mx.fast.rope` for the computation and `nn.RoPE` for the
module wrapper.

### Quantized

| Layer | Description |
|-------|-------------|
| `nn.QuantizedLinear(in, out, bias=True, group_size=None, bits=None, mode="affine")` | Quantized linear; stores packed uint32 weights with per-group scales. Modes: `"affine"` (default, g=64/b=4), `"mxfp4"` (g=32/b=4), `"nvfp4"` (g=16/b=4), `"mxfp8"` (g=32/b=8). |
| `nn.QQLinear(in, out, mode="nvfp4")` | Trainable quantized linear; quantizes both weights and inputs via `mx.qqmm`. Supports `"nvfp4"` and `"mxfp8"` modes. No bias support. |
| `nn.QuantizedEmbedding(num, dims, group_size=None, bits=None, mode="affine")` | Quantized embedding; same mode options as QuantizedLinear |

`nn.QQLinear` vs `nn.QuantizedLinear`: QQLinear weights are trainable -- they
dequantize on `.train()` and quantize in eval mode. Use
`nn.QQLinear.from_linear(layer, mode="nvfp4")` to convert. Prefer
`QuantizedLinear` when weights do not need training.

### Other

| Layer | Description |
|-------|-------------|
| `nn.Sequential(*layers)` | Sequential container |
| `nn.Identity` | Identity pass-through |
| `nn.Upsample(scale_factor, mode="nearest")` | Spatial upsampling |

## Loss Functions

All in `nn.losses`:

| Loss | Signature | Notes |
|------|-----------|-------|
| `cross_entropy(logits, targets, ...)` | Classification; supports label smoothing, class weights |
| `binary_cross_entropy(logits, targets, ...)` | Binary classification; supports `with_logits` mode |
| `l1_loss(predictions, targets, reduction)` | Mean absolute error |
| `mse_loss(predictions, targets, reduction)` | Mean squared error |
| `nll_loss(inputs, targets, ...)` | Negative log-likelihood |
| `gaussian_nll_loss(inputs, targets, vars)` | Gaussian negative log-likelihood |
| `kl_div_loss(inputs, targets, ...)` | KL divergence |
| `smooth_l1_loss(predictions, targets, beta)` | Smooth L1 (Huber-like) |
| `triplet_loss(anchors, positives, negatives, ...)` | Triplet margin loss |
| `hinge_loss(inputs, targets, reduction)` | SVM-style hinge loss |
| `huber_loss(inputs, targets, delta)` | Huber loss |
| `log_cosh_loss(inputs, targets, reduction)` | Log-cosh loss |
| `cosine_similarity_loss(x1, x2, ...)` | Cosine similarity loss |
| `margin_ranking_loss(inputs1, inputs2, targets, ...)` | Margin ranking loss |

All loss functions accept a `reduction` parameter: `"none"`, `"mean"`, or
`"sum"`. Default is `"none"` (returns per-element loss). You typically want
`.mean()` for training.

## Parameter Initialization

All in `nn.init`:

| Initializer | Description |
|-------------|-------------|
| `nn.init.constant(value, dtype)` | Fill with a constant value |
| `nn.init.normal(mean=0, std=1, dtype)` | Normal distribution |
| `nn.init.uniform(low=0, high=1, dtype)` | Uniform distribution |
| `nn.init.identity(dtype)` | Identity matrix |
| `nn.init.glorot_normal(dtype)` | Xavier/Glorot normal |
| `nn.init.glorot_uniform(dtype)` | Xavier/Glorot uniform |
| `nn.init.he_normal(dtype)` | Kaiming/He normal |
| `nn.init.he_uniform(dtype)` | Kaiming/He uniform |
| `nn.init.sparse(sparsity, mean=0, std=1, dtype)` | Sparse matrix initialization (2D only) |
| `nn.init.orthogonal(gain=1.0, dtype)` | Orthogonal via QR decomposition (2D only) |

Initializers return callables. Apply them using `model.apply`:

```python
def init_weights(key, module):
    # key is the module path string, e.g. "layers.0.self_attn.q_proj"
    if isinstance(module, nn.Linear):
        # Each initializer returns a function that takes a shape
        init_fn = nn.init.he_normal()
        module.weight = init_fn(module.weight.shape)

model.apply_to_modules(init_weights)
```

## Optimizers

All in `mlx.optimizers`:

| Optimizer | Key Args |
|-----------|----------|
| `SGD(learning_rate, momentum=0, weight_decay=0)` | Stochastic gradient descent |
| `Adam(learning_rate=0.001, betas=(0.9, 0.999), eps=1e-8)` | Adam |
| `AdamW(learning_rate=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)` | Adam with decoupled weight decay |
| `Adamax(learning_rate=0.001, betas=(0.9, 0.999), eps=1e-8)` | Adamax (infinity norm variant) |
| `Adagrad(learning_rate=0.01, eps=1e-8)` | Adagrad |
| `AdaDelta(learning_rate=1.0, rho=0.9, eps=1e-6)` | AdaDelta |
| `RMSprop(learning_rate=0.01, alpha=0.99, eps=1e-8)` | RMSprop |
| `Lion(learning_rate=1e-4, betas=(0.9, 0.99), weight_decay=0)` | Lion (EvoLved Sign Momentum) |
| `Adafactor(learning_rate=None, ...)` | Adafactor (memory-efficient) |
| `Muon(learning_rate, momentum=0.95, weight_decay=0.01, nesterov=True, ns_steps=5)` | MomentUm Orthogonalized by Newton-schulz |

Muon is sub-optimal for embedding layers, final fully connected layers, and
0D/1D parameters. Pair with AdamW via `MultiOptimizer` for those parameter groups.

### Usage

```python
import mlx.optimizers as optim

optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)
optimizer.update(model, grads)
```

### MultiOptimizer

For per-parameter-group optimization:

```python
optimizer = optim.MultiOptimizer(
    {"encoder": optim.Adam(learning_rate=1e-4),
     "decoder": optim.Adam(learning_rate=1e-3)},
    model
)
```

### Gradient Clipping

```python
grads = optim.clip_grad_norm(grads, max_norm=1.0)
```

## Learning Rate Schedulers

All in `mlx.optimizers`:

| Scheduler | Description |
|-----------|-------------|
| `exponential_decay(init, decay_rate, decay_steps)` | Exponential decay |
| `step_decay(init, decay_rate, step_size)` | Step-wise decay |
| `cosine_decay(init, decay_steps, end=0)` | Cosine annealing |
| `linear_schedule(init, end, steps)` | Linear interpolation |
| `join_schedules(schedules, boundaries)` | Combine multiple schedules |

Usage:

```python
schedule = optim.cosine_decay(1e-3, decay_steps=10000)
optimizer = optim.AdamW(learning_rate=schedule)
```

Warmup + cosine decay:

```python
warmup = optim.linear_schedule(0, 1e-3, steps=1000)
cosine = optim.cosine_decay(1e-3, decay_steps=9000)
schedule = optim.join_schedules([warmup, cosine], [1000])
optimizer = optim.AdamW(learning_rate=schedule)
```

## Training Loop Patterns

### Basic Training Loop

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

def loss_fn(model, x, y):
    return nn.losses.cross_entropy(model(x), y).mean()

optimizer = optim.AdamW(learning_rate=1e-4)

def step(x, y):
    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
    optimizer.update(model, grads)
    return loss

for x, y in dataset:
    loss = step(x, y)
    mx.eval(model.parameters(), optimizer.state)
```

Key details:
- `nn.value_and_grad(model, loss_fn)` differentiates w.r.t. model parameters
  (first argument to `loss_fn`)
- Wrapping in a function ensures `grads` is released before evaluation, reducing
  peak memory
- Evaluate both model parameters and optimizer state together for efficiency

### nn.value_and_grad vs mx.value_and_grad

- `nn.value_and_grad(model, fn)` -- differentiates w.r.t. `model.trainable_parameters()`. Use for model training.
- `mx.value_and_grad(fn)` -- differentiates w.r.t. the first argument (an `mx.array`). Use for arbitrary differentiable functions.

### Compiled Training

For maximum performance, compile the training step:

```python
from functools import partial

state = [model.state, optimizer.state]

@partial(mx.compile, inputs=state, outputs=state)
def compiled_step(x, y):
    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
    optimizer.update(model, grads)
    return loss

for x, y in dataset:
    loss = compiled_step(x, y)
    mx.eval(loss)
```

The `inputs` and `outputs` parameters tell the compiler which state is read
and mutated. This avoids the need to pass and return state explicitly.

### Gradient Checkpointing

Trade compute for memory by recomputing activations during the backward pass:

```python
# Checkpoint specific layers
def apply_checkpointing(layer):
    original_call = type(layer).__call__

    def checkpointed_call(self, *args, **kwargs):
        def inner(params, *args, **kwargs):
            self.update(params)
            return original_call(self, *args, **kwargs)
        return mx.checkpoint(inner)(self.trainable_parameters(), *args, **kwargs)

    type(layer).__call__ = checkpointed_call

for layer in model.layers:
    apply_checkpointing(layer)
```

### Gradient Accumulation

```python
accumulated_grads = None

for i, (x, y) in enumerate(dataset):
    _, grads = nn.value_and_grad(model, loss_fn)(model, x, y)

    if accumulated_grads is None:
        accumulated_grads = grads
    else:
        accumulated_grads = tree_map(
            lambda a, g: a + g, accumulated_grads, grads
        )

    if (i + 1) % accumulation_steps == 0:
        # Scale gradients
        accumulated_grads = tree_map(
            lambda g: g * (1.0 / accumulation_steps), accumulated_grads
        )
        optimizer.update(model, accumulated_grads)
        mx.eval(model.parameters(), optimizer.state)
        accumulated_grads = None
```

### Distributed Training

For multi-device training, use `mx.distributed` with averaged gradients:

```python
group = mx.distributed.init()

def step(x, y):
    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
    # Average gradients across devices
    grads = tree_map(lambda g: mx.distributed.all_sum(g) / group.size(), grads)
    optimizer.update(model, grads)
    return loss
```

### Distributed Layers

For model parallelism, MLX provides sharded linear layers that handle
distributed communication automatically:

| Layer | Description |
|-------|-------------|
| `nn.AllToShardedLinear` | Input replicated, output split across devices |
| `nn.ShardedToAllLinear` | Input split, output all-reduced across devices |
| `nn.QuantizedAllToShardedLinear` | Quantized variant of AllToShardedLinear |
| `nn.QuantizedShardedToAllLinear` | Quantized variant of ShardedToAllLinear |

Factory function:

```python
group = mx.distributed.init()
sharded_layer = nn.shard_linear(linear_layer, "all-to-sharded", group=group)
```

`nn.shard_linear` automatically selects the quantized variant when given a
`QuantizedLinear` input. See mlx-lm's `shard()` method for the full sharding
pattern applied to transformer models.
