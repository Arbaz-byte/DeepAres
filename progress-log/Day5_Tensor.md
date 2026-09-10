# Day 5 – Sep, 2026

## Topic
PyTorch Tensor Fundamentals

## Covered
- Tensor creation: `empty`, `zeros`, `ones`, `rand`, `manual_seed`, `tensor`, `arange`, `linspace`, `eye`, `full`
- Shapes/dtypes: `shape`, `*_like`, `int8`/`float64` conversion
- Operations: scalar, element-wise, unary, reduction, matrix, comparison, in-place
- GPU: MPS detected; moved tensors to `mps`; CPU vs GPU matmul speedup ≈ **476×**
- Reshaping: `reshape`, `flatten`, `permute`, `unsqueeze`, `squeeze`
- NumPy interop: `.numpy()`, `torch.from_numpy()`

## Key Takeaway
Tensors are NumPy-like but GPU-accelerated. Explicit device placement is required. In-place ops mutate the original tensor; use `clone()` for an independent copy.

## Challenge
- Remembering axis/dim conventions for reductions.
- Managing CPU/GPU transfers and in-place side effects.
