last updated: 2026-02-28

# Porting from PyTorch to MLX

Step-by-step guide for converting PyTorch models and training code to MLX.
MLX's API is intentionally similar to PyTorch, so most ports are mechanical --
but the differences that exist are fundamental and will silently break your code
if you miss them.

## Before You Start

Read the comparison table in SKILL.md and skim `references/anti-patterns.md`.
The most common porting failures come from three assumptions:

1. **Eager execution** -- PyTorch evaluates immediately; MLX is lazy
2. **Device management** -- PyTorch requires explicit `.cuda()` / `.to()`; MLX
   does not (unified memory)
3. **Gradient mechanics** -- PyTorch uses `.backward()` on tensors; MLX uses
   functional transformations on functions

## Step 1: Model Definition

### nn.Module differences

```python
# PyTorch
class MyModel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)
        self.norm = torch.nn.LayerNorm(dim)
        self.register_buffer("mask", torch.ones(dim))

    def forward(self, x):
        return self.norm(self.linear(x)) * self.mask
```

```python
# MLX
class MyModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.mask = mx.ones(dim)  # No register_buffer -- just assign

    def __call__(self, x):  # __call__, not forward
        return self.norm(self.linear(x)) * self.mask
```

Key differences:

| PyTorch | MLX |
|---------|-----|
| `forward(self, x)` | `__call__(self, x)` |
| `nn.Parameter(tensor)` | Just assign `mx.array` -- all arrays are parameters |
| `register_buffer("name", tensor)` | `self.name = mx_array` -- tracked automatically |
| `model.named_parameters()` | `model.parameters()` (returns nested dict) |
| `model.state_dict()` | `model.state` property |
| `model.load_state_dict(d)` | `model.load_weights(path)` or `model.update(d)` |

### Layer mapping

| PyTorch | MLX |
|---------|-----|
| `nn.Linear` | `nn.Linear` |
| `nn.Conv1d/2d/3d` | `nn.Conv1d/2d/3d` |
| `nn.LayerNorm` | `nn.LayerNorm` (or `mx.fast.layer_norm` for speed) |
| `nn.RMSNorm` | `nn.RMSNorm` (or `mx.fast.rms_norm` for speed) |
| `nn.BatchNorm1d/2d` | `nn.BatchNorm` |
| `nn.GroupNorm` | `nn.GroupNorm` |
| `nn.Embedding` | `nn.Embedding` |
| `nn.Dropout` | `nn.Dropout` |
| `nn.MultiheadAttention` | `nn.MultiHeadAttention` (note the capital H) |
| `nn.TransformerEncoder` | `nn.TransformerEncoder` |
| `nn.Sequential` | `nn.Sequential` |
| `nn.GELU` | `nn.GELU` |
| `nn.SiLU` | `nn.SiLU` |
| `nn.ReLU` | `nn.ReLU` |
| `nn.MaxPool1d/2d/3d` | `nn.MaxPool1d/2d/3d` |

Layers that need manual replacement:

| PyTorch | MLX Equivalent |
|---------|----------------|
| `nn.ModuleList` | Python `list` (MLX tracks lists of modules automatically) |
| `nn.ModuleDict` | Python `dict` (MLX tracks dicts of modules automatically) |
| `nn.ParameterList` | Python `list` of `mx.array` |
| `nn.DataParallel` / `nn.DistributedDataParallel` | `mx.distributed` (see nn-and-training.md) |

### Activation functions

Most are identical. Watch for:

```python
# PyTorch
F.gelu(x, approximate="tanh")

# MLX
nn.gelu_approx(x)    # Or nn.gelu_fast_approx(x)
# Or use the module: nn.GELU(approx="tanh")
```

## Step 2: Forward Pass

### Remove device management

Delete all of these:

```python
# Delete these lines entirely:
model.cuda()
model.to(device)
model.to("mps")
x = x.to(device)
x = x.cuda()
x = x.cpu()
torch.device("cuda")
with torch.cuda.device(0):
```

MLX uses unified memory. Arrays are always accessible by both CPU and GPU.

### Replace tensor operations

| PyTorch | MLX |
|---------|-----|
| `torch.tensor(data)` | `mx.array(data)` |
| `torch.zeros(shape)` | `mx.zeros(shape)` |
| `torch.ones(shape)` | `mx.ones(shape)` |
| `torch.randn(shape)` | `mx.random.normal(shape=shape)` |
| `torch.rand(shape)` | `mx.random.uniform(shape=shape)` |
| `torch.arange(n)` | `mx.arange(n)` |
| `torch.cat(tensors, dim)` | `mx.concatenate(arrays, axis)` |
| `torch.stack(tensors, dim)` | `mx.stack(arrays, axis)` |
| `x.view(shape)` | `x.reshape(shape)` |
| `x.permute(dims)` | `x.transpose(*dims)` |
| `x.contiguous()` | Delete (not needed -- MLX handles layout) |
| `x.detach()` | `mx.stop_gradient(x)` |
| `x.clone()` | `mx.array(x)` (copy) |
| `x.unsqueeze(dim)` | `mx.expand_dims(x, axis=dim)` |
| `x.squeeze(dim)` | `mx.squeeze(x, axis=dim)` |
| `x.float()` | `x.astype(mx.float32)` |
| `x.half()` | `x.astype(mx.float16)` |
| `torch.where(cond, a, b)` | `mx.where(cond, a, b)` |
| `torch.einsum(eq, *tensors)` | `mx.einsum(eq, *arrays)` |
| `torch.triu(x, diagonal=1)` | `mx.triu(x, k=1)` |
| `torch.inf` | `float("inf")` |
| `x.masked_fill(mask, val)` | `mx.where(mask, val, x)` (note: args reversed) |
| `F.softmax(x, dim=-1)` | `mx.softmax(x, axis=-1)` |

### Handle in-place operations

MLX arrays are immutable graph nodes. "In-place" syntax works but creates new
nodes:

```python
# PyTorch: true in-place mutation
x[0] = 1.0  # Mutates memory

# MLX: creates a new computation node (no memory saving)
x[0] = 1.0  # Works but is NOT in-place
x = x.at[0].add(1.0)  # Also works -- returns new array
```

### Replace context managers

| PyTorch | MLX |
|---------|-----|
| `with torch.no_grad():` | Not needed (gradients are explicit) |
| `with torch.cuda.amp.autocast():` | Not needed (use explicit dtypes) |
| `torch.inference_mode()` | Not needed |

### Attention pattern

```python
# PyTorch
attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
attn = attn.masked_fill(mask == 0, float("-inf"))
attn = F.softmax(attn, dim=-1)
out = torch.matmul(attn, v)
```

```python
# MLX -- use the fused op instead of manual implementation
out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
```

Always prefer `mx.fast.scaled_dot_product_attention` over manual attention.

### Normalization

```python
# PyTorch -- manual RMS norm
def rms_norm(x, weight, eps):
    x_float = x.float()
    norm = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + eps)
    return (norm * weight).to(x.dtype)
```

```python
# MLX -- use the fused op (handles precision internally)
x = mx.fast.rms_norm(x, weight, eps=eps)
```

### RoPE

```python
# PyTorch -- manual rotary embedding application
cos = freqs_cos[position_ids]
sin = freqs_sin[position_ids]
q = (q * cos) + (rotate_half(q) * sin)
k = (k * cos) + (rotate_half(k) * sin)
```

```python
# MLX -- use the fused op
q = mx.fast.rope(q, dims=head_dim, traditional=False, base=base, offset=offset)
k = mx.fast.rope(k, dims=head_dim, traditional=False, base=base, offset=offset)
```

## Step 3: Training Loop

This is where the biggest conceptual change happens. PyTorch mutates tensors
in-place via `.backward()`; MLX uses functional transformations.

```python
# PyTorch
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for x, y in dataloader:
    x, y = x.cuda(), y.cuda()
    optimizer.zero_grad()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    print(f"Loss: {loss.item()}")
```

```python
# MLX
import mlx.optimizers as optim

optimizer = optim.AdamW(learning_rate=1e-4)

def loss_fn(model, x, y):
    return nn.losses.cross_entropy(model(x), y).mean()

def step(x, y):
    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
    grads = optim.clip_grad_norm(grads, max_norm=1.0)
    optimizer.update(model, grads)
    return loss

for x, y in dataset:
    loss = step(x, y)
    mx.eval(model.parameters(), optimizer.state)
    # loss.item() is fine here -- outside the step function
```

Key differences:

| PyTorch | MLX |
|---------|-----|
| `optimizer.zero_grad()` | Not needed (functional -- no gradient accumulation) |
| `loss.backward()` | `nn.value_and_grad(model, loss_fn)(model, x, y)` |
| `optimizer.step()` | `optimizer.update(model, grads)` |
| `clip_grad_norm_` | `optim.clip_grad_norm(grads, max_norm)` (returns new grads) |
| Implicit evaluation | `mx.eval(model.parameters(), optimizer.state)` |
| `x.cuda()` | Delete |

### Learning rate schedulers

```python
# PyTorch
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)
# ... in loop:
scheduler.step()
```

```python
# MLX -- scheduler is passed directly to the optimizer
schedule = optim.cosine_decay(1e-3, decay_steps=10000)
optimizer = optim.AdamW(learning_rate=schedule)
# No scheduler.step() needed -- optimizer handles it internally
```

### Data loading

MLX does not have a built-in DataLoader. Common patterns:

```python
# Simple batching
def batch_iterate(dataset, batch_size):
    perm = mx.array(np.random.permutation(len(dataset)))
    for i in range(0, len(dataset), batch_size):
        ids = perm[i:i + batch_size]
        yield dataset[ids]
```

For large datasets, use NumPy or a generator to load batches and convert to
`mx.array` at the boundary.

## Step 4: Inference

```python
# PyTorch
model.eval()
with torch.no_grad():
    output = model(input_tensor.cuda())
    predictions = output.cpu().numpy()
```

```python
# MLX
output = model(input_array)
mx.eval(output)
predictions = np.array(output)  # Converts to NumPy (triggers eval if needed)
```

No `.eval()` mode switch needed unless using Dropout or BatchNorm. No device
management. No gradient context manager.

### Weight saving and loading

```python
# PyTorch
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt"))
```

```python
# MLX -- use safetensors format
from mlx.utils import tree_flatten
mx.save_safetensors("model.safetensors", dict(tree_flatten(model.parameters())))
model.load_weights("model.safetensors")
```

## Step 5: Add Evaluation Points

After porting, the most common bug is forgetting evaluation. MLX is lazy -- if
you never call `mx.eval()`, the computation graph grows until you run out of
memory.

Add `mx.eval()` at:

1. **End of each training step**: `mx.eval(model.parameters(), optimizer.state)`
2. **End of each inference call**: `mx.eval(output)` (before using the result)
3. **After loading and transforming weights**: `mx.eval(model.parameters())`
4. **Before converting to NumPy or Python**: `mx.eval(x)` (or let `.item()` /
   `np.array()` handle it implicitly)

Do NOT add `mx.eval()` inside:

- The step function (creates sync points that hurt throughput)
- Compiled functions (breaks tracing)
- The model's `__call__` method (prevents graph optimization)

## Step 6: Use mx.fast Ops

After porting, search for manual implementations that should use fused ops:

| Pattern | Replace With |
|---------|-------------|
| Manual attention (q @ k.T, softmax, @ v) | `mx.fast.scaled_dot_product_attention` |
| Manual RMS norm (x * rsqrt(mean(x^2) + eps)) | `mx.fast.rms_norm` |
| Manual layer norm | `mx.fast.layer_norm` |
| Manual RoPE (cos/sin rotation) | `mx.fast.rope` |
| Manual softmax with upcast | `mx.softmax(x, precise=True)` |

## Step 7: Fix Type Promotion

Check every constant and scalar operation:

```python
# These break half-precision in MLX:
x * mx.array(2.0)         # Promotes to float32!
mx.zeros(shape)            # Creates float32 by default!
mx.array(1.0, mx.bfloat16) * mx.array(1.0, mx.float16)  # Promotes to float32!

# These preserve dtype:
x * 2.0                    # Python scalar adapts to x's dtype
mx.zeros(shape, dtype=x.dtype)  # Match dtype explicitly
```

Rule: use Python scalars (`2.0`, `0.5`) for constants, not `mx.array(2.0)`.

## Common Porting Mistakes

### float64

```python
# PyTorch allows float64 on GPU
x = torch.tensor(1.0, dtype=torch.float64).cuda()  # Works

# MLX does NOT support float64 on GPU
x = mx.array(1.0, mx.float64)  # Will fail or fall back to CPU
# Fix: use float32 as the highest precision
```

### Random number generation

```python
# PyTorch -- global state
torch.manual_seed(42)
x = torch.randn(3, 4)

# MLX -- explicit seeds (like JAX)
mx.random.seed(42)
x = mx.random.normal(shape=(3, 4))
# Or with explicit keys for reproducibility:
key = mx.random.key(42)
x = mx.random.normal(shape=(3, 4), key=key)
```

### Fancy indexing

```python
# PyTorch -- fast fancy indexing
result = x[indices]

# MLX -- prefer explicit gather
result = mx.take_along_axis(x, indices[:, None], axis=0)
```

### torch.compile vs mx.compile

They serve the same purpose but work differently:

- `torch.compile` is opt-in and can handle dynamic shapes well
- `mx.compile` traces with symbolic inputs; shape changes cause recompilation
- `mx.compile` requires pure functions (no `mx.eval`, no print, no side effects)
- Varying Python scalar inputs to `mx.compile` cause recompilation -- wrap in
  `mx.array` to avoid this

## Checklist

Use this to verify your port is complete:

- [ ] All `forward()` methods renamed to `__call__()`
- [ ] All `.cuda()`, `.to(device)`, `.cpu()` removed
- [ ] All `.backward()` replaced with `nn.value_and_grad`
- [ ] All `optimizer.zero_grad()` removed
- [ ] All `optimizer.step()` replaced with `optimizer.update(model, grads)`
- [ ] `mx.eval()` added at iteration boundaries
- [ ] No `mx.eval()` inside step functions or compiled functions
- [ ] Manual attention replaced with `mx.fast.scaled_dot_product_attention`
- [ ] Manual norms replaced with `mx.fast.rms_norm` / `mx.fast.layer_norm`
- [ ] Manual RoPE replaced with `mx.fast.rope`
- [ ] Constants use Python scalars, not `mx.array`
- [ ] No float64 usage
- [ ] `torch.no_grad()` contexts removed (not needed)
- [ ] `contiguous()` calls removed (not needed)
- [ ] `ModuleList` / `ModuleDict` replaced with plain Python lists/dicts
- [ ] Weight save/load uses safetensors format
