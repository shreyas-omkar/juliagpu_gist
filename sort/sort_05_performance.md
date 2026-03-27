# `sort!` / `sortperm` Performance Analysis

## Architectural Strategy
Our implementation delegates all sorting operations to the iterative **Bottom-Up Merge Sort** in `AcceleratedKernels.jl`. 
* **Phase 1 (Local):** Data is tiled into blocks and sorted within the GPU's shared memory.
* **Phase 2 (Global Merge):** The kernel executes $\lceil \log_2(N / \text{BlockSize}) \rceil$ global passes. In each pass, sorted sub-lists are merged until the entire array is ordered.

This approach ensures **Stability** (equal elements maintain relative order), matching `Base.sort` behavior across all backends.

---

## Performance Derivation

The "Speed-of-Light" (SoL) for sorting is dictated by the massive volume of global memory traffic across the logarithmic merge passes.

### 1. Data Traffic Volume
For $N$ elements of `Float32` (4 bytes) and a block-size of 256:
* **Initial Local Pass:** 1 Read + 1 Write ($2 \times 4N$ bytes).
* **Global Merge Passes:** Approximately 18 passes for $50\text{M}$ elements.
* **Total Traffic:** $2 \times (\text{Passes} + 1) \times N \times 4$ bytes.

**For 100 Million elements:**
$$2 \times 20 \times 100,000,000 \times 4 \text{ bytes} \approx \mathbf{16.0 \text{ GB total traffic}}$$

### 2. The Speed Limit (SoL)
On a Tesla T4 (280 GB/s), the physical minimum time ($T_{min}$) for $100\text{M}$ elements is:
$$T_{min} = \frac{16.0 \text{ GB}}{280 \text{ GB/s}} \approx \mathbf{57.14 \text{ ms}}$$

---

## Observed vs. Forecasted Performance (Tesla T4)

| Size (N) | Total Data Moved | Ideal Limit (SoL) | Native CUDA (CUB) | AK Target Forecast |
| :--- | :--- | :--- | :--- | :--- |
| 1 M   | 160.0 MB | 0.57 ms  | 18.669 ms   | **~22.0 – 25.0 ms** |
| 10 M  | 1.60 GB  | 5.71 ms  | 161.506 ms  | **~190.0 – 210.0 ms** |
| 50 M  | 8.00 GB  | 28.57 ms | 959.289 ms  | **~1.10 – 1.25 s** |
| 100 M | 16.00 GB | 57.14 ms | 2170.757 ms | **~2.50 – 2.80 s** |

---

## Technical Constraints & Bottlenecks

### 1. Instruction-Bound Merge Logic
Unlike a simple `reverse`, a merge pass is **instruction-bound**. Each thread performs multiple comparisons and address calculations per data load. This high instruction density, combined with non-coalesced memory access as the merge windows grow, prevents the kernel from fully saturating the 280 GB/s memory bus.

### 2. Stable Merge Sort vs. Radix Sort
While Radix Sort is faster for numeric types, it is generally non-stable. Our implementation uses **Merge Sort** to guarantee stability and support for arbitrary types and predicates across all `GPUArrays` backends. This provides a reliable, linearly-scaling fallback that fills functional gaps in current vendor packages.

### 3. Filling the `sortperm` Gap
Because our implementation lives at the `AnyGPUArray` level, it provides `sortperm` and `sortperm!` to `CUDA.jl` users for the first time. The data traffic for `sortperm` is significantly higher (as it moves both 4-byte values and 8-byte indices), but it eliminates the need for expensive CPU-side sorting of GPU data.
