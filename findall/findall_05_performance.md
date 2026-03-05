# findall — Performance Analysis

## Bandwidth Model

`findall` has two GPU passes:

| Pass | Operation | Traffic | Bandwidth |
|------|-----------|---------|-----------|
| Step 1: cumsum | 4 passes (Blelloch up+down sweep) | 4×4n bytes | 360 GB/s GPU |
| Step 2: scatter | 1 read (bools) + 1 write (ys) | 2×4n bytes | 360 GB/s GPU |
| **Total GPU** | | **6×4n bytes** | **360 GB/s** |

CPU scalar fallback (PCIe round-trip):

| Pass | Traffic | Bandwidth |
|------|---------|-----------|
| Transfer bools to host + sequential scan + transfer back | 3×4n bytes | 12 GB/s PCIe |

---

## Projected Timings: RTX 3060, Float32, ~50% true

| Array size | CPU fallback (ms) | KA kernel (ms) | Speedup |
|:---:|:---:|:---:|:---:|
| 10K | 0.8 | 0.07 | 11× |
| 100K | 2.7 | 0.18 | 15× |
| 1M | 26.7 | 1.8 | 15× |
| 10M | 267.0 | 17.8 | 15× |
| 100M | 2670.0 | 178.0 | 15× |

Model: CPU = 3×4n / 12e9 seconds · GPU = 6×4n / 360e9 seconds.  
Speedup saturates at ~15× (lower than `reverse` 30× and `accumulate!` 22× due to 6 passes vs 2 and 4).

Real-hardware validation on RTX 3060 planned before PR submission.

---

## Why Lower Speedup Than reverse?

`reverse` does 2 GPU passes (read + write). `findall` does 6 passes (cumsum 4 + scatter 2).  
The speedup ceiling = (GPU_passes × 4n / GPU_BW) / (3 × 4n / PCIe_BW) = (3/GPU_passes) × (GPU_BW/PCIe_BW).

```
reverse:    (3/2)  × (360/12) = 1.5 × 30 = 45×  → realistically ~30× (overhead)
accumulate: (3/4)  × (360/12) = 0.75 × 30 = 22×
findall:    (3/6)  × (360/12) = 0.5  × 30 = 15×
```

The tradeoff is worthwhile: we exchange some bandwidth efficiency for correct parallel semantics and deterministic output ordering.

---

## The Real Cost of the CPU Fallback

For `n = 10^7` booleans with `allowscalar(true)`:

- 10,000,000 individual `bools[i]` reads → 10,000,000 separate HIP/CUDA API calls
- Each call synchronises the PCIe bus
- Wall time: minutes, not milliseconds
- **Numerically correct. No warning.**

For a model that filters activations by mask at every training step, this silently dominates total runtime.
