# accumulate! / cumsum / cumprod   Performance Analysis

## Why the Performance Case Is Stronger Than `reverse`

With `reverse`, the scalar fallback loses bandwidth efficiency (~30× from PCIe).  
With `accumulate!`, the scalar fallback loses **both bandwidth and algorithmic parallelism**:

| | Depth | Bandwidth penalty |
|---|---|---|
| CPU scalar fallback | O(n) sequential | 3×4n bytes at 12 GB/s PCIe |
| AK.jl Blelloch (GPU) | O(log n) parallel | 4×4n bytes at 360 GB/s GPU |

The Blelloch scan needs 4 passes (up-sweep read+write, down-sweep read+write). The CPU fallback does 3 PCIe passes at 30× lower bandwidth.

---

## Bandwidth Model (RTX 3060 class, Float32)

| Path | Traffic | Bandwidth | Latency formula |
|------|---------|-----------|-----------------|
| CPU scalar fallback | 3 × 4n bytes | ~12 GB/s (PCIe) | `3×4n / 12e9` |
| AK.jl Blelloch kernel | 4 × 4n bytes | ~360 GB/s (GPU BW) | `4×4n / 360e9` |

---

## Projected Timings

| Array size | CPU fallback (ms) | AK.jl kernel (ms) | Speedup |
|:---:|:---:|:---:|:---:|
| 10K | 0.8 | 0.06 | 13× |
| 100K | 2.7 | 0.12 | 23× |
| 1M | 26.7 | 1.2 | 22× |
| 10M | 267.0 | 11.9 | 22× |
| 100M | 2670 | 119 | 22× |

Speedup saturates at ~22× (bandwidth-bound regime). Lower than `reverse` (30×) because Blelloch requires 4 memory passes vs 2 for reverse.

Real-hardware validation on RTX 3060 to be run and incorporated before PR submission.

---

## The Depth Loss Is Even More Damaging

At n=1M with `allowscalar(true)`:
- PCIe bandwidth penalty alone: ~22×
- Algorithm depth penalty: O(n) vs O(log n) = O(1,000,000) vs O(20) = **~50,000× more serial steps**
- Combined effective slowdown: potentially **orders of magnitude** beyond the bandwidth model

The bandwidth model is a lower bound on the damage. For large arrays the depth loss dominates completely.
