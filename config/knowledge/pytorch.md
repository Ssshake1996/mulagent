# PyTorch Knowledge Base

## Common Runtime Errors

### Shape Mismatch
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x768 and 512x256)
```
**Diagnosis**: Print shapes at each layer boundary.
```python
print(f"input: {x.shape}, weight: {self.fc.weight.shape}")
```
**Fix**: Align dimensions — `nn.Linear(in_features, out_features)` where `in_features` must match input's last dim.

### Device Mismatch
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu
```
**Fix**: Ensure model and data on same device:
```python
model = model.to(device)
inputs = inputs.to(device)
labels = labels.to(device)
```

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```
**Fix hierarchy** (try in order):
1. Reduce `batch_size`
2. Use `torch.cuda.amp.autocast()` (mixed precision)
3. Use `torch.utils.checkpoint.checkpoint()` (gradient checkpointing)
4. Use `model.gradient_checkpointing_enable()` (HuggingFace models)
5. Move to multi-GPU: `DataParallel` or `DistributedDataParallel`
6. If OOM at batch_size=1 → model too large for GPU, need model parallelism or smaller model

### Detached Tensor in Loss
```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```
**Cause**: `.detach()`, `.data`, or `.numpy()` called before loss computation breaks gradient chain.
**Fix**: Don't detach tensors that need gradients for backpropagation.

### In-Place Operation Breaking Autograd
```
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
```
**Cause**: `x += 1`, `x[0] = 5`, `tensor.add_(other)` modifies tensor in-place.
**Fix**: Use out-of-place: `x = x + 1`, `x = x.clone(); x[0] = 5`.

### Backward Through Graph Twice
```
RuntimeError: Trying to backward through the graph a second time
```
**Fix**: Use `loss.backward(retain_graph=True)` if intentional, or restructure to avoid double backward.

## 10 Most Common Error Patterns

| # | Error | Quick Fix |
|---|---|---|
| 1 | Shape mismatch (mat1/mat2) | Print shapes, align Linear in_features |
| 2 | Device mismatch (CPU/GPU) | `.to(device)` on model + all inputs |
| 3 | CUDA OOM | Reduce batch_size → AMP → gradient checkpointing |
| 4 | Detached tensor (no grad_fn) | Remove `.detach()` / `.data` before loss |
| 5 | Batch size inconsistency | Check DataLoader `drop_last=True` for last batch |
| 6 | In-place op breaks autograd | Replace `+=` with `x = x + ...` |
| 7 | DataLoader `collate_fn` error | Check inconsistent tensor sizes, pad sequences |
| 8 | cuDNN error | Set `torch.backends.cudnn.enabled = False` to isolate, then fix |
| 9 | Embedding index out of range | Check `num_embeddings` ≥ max index + 1 |
| 10 | Backward through graph twice | `retain_graph=True` or restructure |

## Shape Debugging

```python
# Inject shape prints at strategic points
class DebugModel(nn.Module):
    def forward(self, x):
        print(f"[input] {x.shape}")
        x = self.encoder(x)
        print(f"[encoder] {x.shape}")
        x = self.decoder(x)
        print(f"[decoder] {x.shape}")
        return x

# Use torchsummary for full model shape trace
from torchinfo import summary
summary(model, input_size=(batch_size, channels, height, width))
```

## Memory Debugging

```python
# Track GPU memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Find memory leaks: check for growing tensors
import gc
gc.collect()
torch.cuda.empty_cache()

# Profile memory by layer
with torch.cuda.memory_stats() as stats:
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
```

## Mixed Precision (AMP)

```python
# Automatic Mixed Precision — saves ~30-50% GPU memory
scaler = torch.amp.GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    with torch.amp.autocast(device_type='cuda'):
        output = model(batch['input'].to(device))
        loss = criterion(output, batch['target'].to(device))

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### AMP Common Issues
| Issue | Fix |
|---|---|
| `GradScaler found inf/nan` | Normal — scaler adjusts automatically. If persistent, check data for NaN |
| Loss becomes NaN with AMP | Disable AMP for loss computation: compute loss in float32 |
| Custom autograd function fails | Add `@torch.amp.custom_fwd` / `@torch.amp.custom_bwd` |

## Gradient Checkpointing

```python
# Trades compute for memory — recomputes activations during backward
from torch.utils.checkpoint import checkpoint

class BigModel(nn.Module):
    def forward(self, x):
        # Checkpoint expensive layers
        x = checkpoint(self.encoder, x, use_reentrant=False)
        x = checkpoint(self.decoder, x, use_reentrant=False)
        return self.head(x)

# HuggingFace models:
model.gradient_checkpointing_enable()
```

## DataLoader Best Practices

```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,           # Shuffle training data
    num_workers=4,          # Parallel data loading (set to CPU count)
    pin_memory=True,        # Faster CPU→GPU transfer
    drop_last=True,         # Avoid batch size inconsistency in last batch
    persistent_workers=True, # Don't restart workers each epoch
)
```

| Issue | Fix |
|---|---|
| Slow data loading | Increase `num_workers`, add `pin_memory=True` |
| `RuntimeError: DataLoader worker exited unexpectedly` | Reduce `num_workers`, check shared memory: `df -h /dev/shm` |
| Inconsistent batch shapes | Use custom `collate_fn` with padding |
| Memory grows over epochs | Check for list accumulation in dataset, use `persistent_workers` |

## Training Loop Checklist

- [ ] `model.train()` before training, `model.eval()` before validation
- [ ] `optimizer.zero_grad()` at start of each step (or `set_to_none=True`)
- [ ] `torch.no_grad()` during validation/inference
- [ ] Learning rate scheduler `.step()` called (per-epoch or per-step as appropriate)
- [ ] Best model checkpoint saved based on validation metric
- [ ] Gradient clipping if training is unstable: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- [ ] Random seeds set for reproducibility: `torch.manual_seed(42)`

## Stop Conditions for Build Resolver

- Same error after 3 fix attempts → report error and stop
- Requires fundamental architecture change → report, don't attempt
- Hardware/driver incompatibility (CUDA version, GPU compute capability) → report
- OOM at batch_size=1 → model too large for available GPU, report alternatives
