last updated: 2026-02-23

# DiT / Diffusion Optimization Guide

Performance optimization techniques for diffusion models / DiT models on MLX.

## Denoising Step Compilation

### Compiling the Denoising Function

The denoising function is called many times (20-50+ steps) with the same shapes.
This makes it an excellent candidate for compilation:

```python
@mx.compile
def denoise_step(noisy_latent, timestep, text_embeddings):
    noise_pred = model(noisy_latent, timestep, text_embeddings)
    # Scheduler step
    return scheduler.step(noise_pred, timestep, noisy_latent)
```

Since latent shapes are fixed for a given resolution, shape-dependent
recompilation only happens once per resolution.

### Avoiding Recompilation

Common pitfalls in diffusion model compilation:

```python
# BAD: timestep as Python int causes recompilation each step
for t in timesteps:
    denoise_step(latent, t, embeddings)  # Recompiles 50+ times!

# GOOD: timestep as mx.array
for t in timesteps:
    denoise_step(latent, mx.array(t), embeddings)  # Compiles once
```

```python
# BAD: guidance scale as Python float
denoise_step(latent, t, embeddings, guidance_scale=7.5)  # Constant!

# GOOD: pass as mx.array
denoise_step(latent, t, embeddings, mx.array(7.5))
```

### What NOT to Compile

- The VAE decoder (runs once, complex shapes)
- The text encoder (runs once per prompt)
- Functions with dynamic control flow based on array values

## CFG (Classifier-Free Guidance) Optimization

### Batched CFG

Instead of two separate forward passes (conditional + unconditional), batch them:

```python
# SLOW: two separate passes
uncond_pred = model(latent, t, uncond_embeddings)
cond_pred = model(latent, t, cond_embeddings)
noise_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
```

```python
# FAST: single batched pass
latent_input = mx.concatenate([latent, latent], axis=0)
embeddings = mx.concatenate([uncond_embeddings, cond_embeddings], axis=0)
noise_pred_both = model(latent_input, t, embeddings)

uncond_pred, cond_pred = mx.split(noise_pred_both, 2, axis=0)
noise_pred = uncond_pred + 7.5 * (cond_pred - uncond_pred)
```

The batched version is significantly faster because it amortizes kernel launch
overhead and enables better GPU utilization.

### Guidance Scale Precision

```python
# BAD: guidance_scale as mx.array defaults to float32
noise_pred = uncond + mx.array(7.5) * (cond - uncond)  # Promotes to fp32!

# GOOD: use Python scalar
noise_pred = uncond + 7.5 * (cond - uncond)  # Stays in model precision
```

## Vision Attention Optimization

### Use mx.fast.scaled_dot_product_attention

Always use the fused attention kernel instead of manual computation:

```python
# SLOW: manual attention
scores = queries @ keys.transpose(0, 1, 3, 2) * scale
scores = mx.softmax(scores, axis=-1, precise=True)
output = scores @ values

# FAST: fused kernel
output = mx.fast.scaled_dot_product_attention(
    queries, keys, values, scale=scale, mask=mask
)
```

### Patch Embedding Optimization

For vision transformers, the initial patch embedding is typically a strided
convolution. Ensure the conv weights are the right dtype:

```python
# Watch for dtype mismatch in patch embeddings
patches = self.patch_embed(pixel_values)  # Check pixel_values dtype
```

## Memory Management for Diffusion

### Latent vs Pixel Space

Diffusion models often generate in latent space (e.g., 64x64) and decode to
pixel space (e.g., 512x512). The VAE decode step has the highest peak memory:

```python
# Evaluate model outputs before VAE decode
mx.eval(latents)

# Clear diffusion model intermediates
mx.clear_cache()

# Now decode -- peak memory is just VAE + output image
image = vae_decoder(latents)
mx.eval(image)
```

### Evaluation Strategy for Denoising

```python
for i, t in enumerate(timesteps):
    latent = denoise_step(latent, mx.array(t), embeddings)

    # Evaluate each step to prevent graph accumulation
    mx.eval(latent)

    # Clear cache periodically (especially with variable guidance)
    if i % 10 == 0:
        mx.clear_cache()
```

Do NOT skip the per-step evaluation -- the denoising graph accumulates
rapidly and will cause OOM.

### Multi-Resolution Support

If supporting multiple resolutions, be aware that each unique resolution
triggers a recompilation of compiled functions. Cache compiled graphs by
sticking to standard resolutions (512x512, 768x768, 1024x1024).

## Scheduler Optimization

### Precompute Scheduler Constants

Many schedulers use timestep-dependent constants. Precompute them as MLX arrays:

```python
# Precompute all alpha/sigma values
alphas = mx.array(scheduler.alphas_cumprod)
sigmas = mx.array(scheduler.sigmas)

# Index into precomputed arrays rather than computing per-step
alpha_t = alphas[t]
sigma_t = sigmas[t]
```

### Avoid Python Loops in Schedulers

```python
# BAD: Python loop with per-element operations
for i in range(num_steps):
    noise_pred = model(...)
    latent = alpha[i] * latent + sigma[i] * noise_pred
    mx.eval(latent)

# BETTER: keep the loop but ensure clean evaluation
for i in range(num_steps):
    with mx.stream(generation_stream):
        noise_pred = model(...)
        latent = alphas[i] * latent + sigmas[i] * noise_pred
    mx.eval(latent)
```

## Profiling Diffusion Models

### Identifying Bottlenecks

1. **Text encoding**: Should be fast (one-time cost). If slow, check for
   unnecessary evaluations in the encoder.
2. **Denoising loop**: Should be the main cost. Check:
   - Is the compiled function being recompiled each step?
   - Is CFG properly batched?
   - Are evaluations happening at the right granularity?
3. **VAE decode**: One-time cost. If slow, check for dtype issues.

### Timing Denoising Steps

```python
import time

mx.synchronize()
times = []

for t in timesteps:
    tic = time.perf_counter()
    latent = denoise_step(latent, mx.array(t), embeddings)
    mx.eval(latent)
    times.append(time.perf_counter() - tic)

# First step includes compilation -- measure from second onwards
avg_step = sum(times[1:]) / len(times[1:])
print(f"Average step: {avg_step * 1000:.1f}ms")
print(f"First step (includes compile): {times[0] * 1000:.1f}ms")
```
