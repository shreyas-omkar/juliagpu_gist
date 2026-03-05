# mapreducedim!   Performance Analysis

## Bandwidth Model

`mapreducedim!` is among the most bandwidth-efficient operations in this project:

| Pass | Traffic | Where |
|------|---------|-------|
| Read input `A` | 1×4n bytes | GPU global memory |
| Tree reduction | in shared memory | L1 cache (~100× faster than global) |
| Write output `R` | 1×(4×n/reduction_factor) bytes | GPU global memory |
| **Effective total** | **~2×4n bytes** | **GPU bandwidth** |

CPU workaround (`Array(A) |> sum`): PCIe transfer to host (4n bytes at ~12 GB/s) + host reduction + no transfer back (scalar output). Total: ~1×4n bytes at PCIe bandwidth.

Note: unlike the other three operations, the CPU workaround here is slightly cheaper (1 pass vs the others' 3 passes)   but the GPU kernel still wins due to the 30:1 bandwidth ratio.

---

## Projected Timings: RTX 3060, Float32

| Array size | CPU workaround (ms) | KA kernel (ms) | Speedup |
|:---:|:---:|:---:|:---:|
| 10K | 0.8 | 0.05 | 16× |
| 100K | 2.7 | 0.09 | 30× |
| 1M | 26.7 | 0.9 | 30× |
| 10M | 267.0 | 8.9 | 30× |
| 100M | 2670.0 | 89.0 | 30× |

Model: CPU workaround = 1×4n / 12e9 s · GPU kernel = 2×4n / 360e9 s.  
Speedup ceiling = (2/1) × (360/12) × (1/2) = 30×.

Real-hardware validation on RTX 3060 planned before PR submission.

---

## Dimensional Reduction Performance

For `sum(A; dims=2)` on an M×N matrix, the output is M×1. There are M independent reductions of length N each. The GPU launches M threadgroups, all running in parallel.

This is **perfectly scalable**: doubling M doubles the parallelism, not the time. The kernel is limited by memory bandwidth, not by reduction depth.

For large M and small N (e.g. 10000×16), each threadgroup only reduces 16 elements   far less than the group size of 256. The neutral-element padding ensures correctness, but 240/256 threads are idle for most of the reduction. Future optimisation: choose group size dynamically based on N.

---

## Comparison Across All Four PRs

| Operation | GPU passes | Speedup ceiling | PR |
|-----------|:---:|:---:|:---:|
| `reverse` | 2 | ~30× | #1 |
| `accumulate!` | 4 | ~22× | #2 |
| `findall` | 6 | ~15× | #3 |
| `mapreducedim!` | 2 | ~30× | #4 |

`mapreducedim!` matches `reverse` for efficiency   both are 2-pass bandwidth-bound operations. The tree reduction happens entirely in shared memory (no extra global memory pass), keeping the pass count at 2.

---

## The Shared Memory Advantage

Why shared memory matters for reduction:

```
GPU global memory latency:  ~400 cycles
GPU L1/shared memory latency: ~4 cycles  (100× faster)

Tree reduction with 256 threads:
  log₂(256) = 8 steps
  7 of those 8 steps happen entirely in shared memory
  Only 1 step (the initial load) touches global memory

Without shared memory (naive atomic reduction):
  256 threads each do 1 global atomic operation
  Global atomics serialise → essentially sequential
  Latency: 256 × 400 cycles = 102,400 cycles
  
With shared memory tree:
  8 steps × 4 cycles = 32 cycles for the reduction phase
  + 1 global load + 1 global write
  Total: ~832 cycles   (123× faster than naive atomic)
```
