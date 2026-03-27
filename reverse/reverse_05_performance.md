# `reverse` / `reverse!` Performance Analysis

## Architectural Strategy
Our performance target is the **Speed-of-Light (SoL)** limit the absolute physical floor of the hardware. Since `reverse` is an "embarrassingly parallel" operation with zero data dependencies, it is 100% memory-bound. The bottleneck is the saturation of the VRAM bus, not the compute cores.

By delegating to **AcceleratedKernels.jl**, we aim to maintain within **1.2x – 1.5x** of native vendor code. The performance "tax" we pay is for universal N-dimensional support, which requires more complex coordinate math (modulo/division) than a simple 1D linear swap.

---

## Performance Derivation (The Math)

To verify the implementation's efficiency, we derive the minimum execution time for 50 Million elements on an **NVIDIA Tesla T4**.

### 1. Data Traffic Volume
For an array of $N$ elements of type `Float32` (4 bytes each):
* **Read Phase:** Every element is fetched from global memory once ($N \times 4$ bytes).
* **Write Phase:** Every element is stored in its mirrored position once ($N \times 4$ bytes).
* **Total Movement:** $8N$ bytes.

**For 50 Million elements:**
$$50,000,000 \times 8 \text{ bytes} = 400 \text{ MB}$$

### 2. The Speed Limit (SoL)
The Tesla T4 has a sustained practical bandwidth ($\beta$) of approximately **280 GB/s**. The minimum physical execution time ($T_{min}$) is:
$$T_{min} = \frac{400 \text{ MB}}{280 \text{ GB/s}} \approx \mathbf{1.43 \text{ ms}}$$

---

## Observed vs. Forecasted Performance (Tesla T4)

The table below compares the **Native CUDA (CUB)** baseline (measured on-device) against our targeted forecast for the **AcceleratedKernels.jl** implementation.

| Size (N) | Total Data Moved | Ideal Limit (SoL) | Native CUDA (CUB) | AK Target Forecast |
| :--- | :--- | :--- | :--- | :--- |
| 100 K | 0.8 MB | 0.003 ms | 0.044 ms | **~0.05 – 0.07 ms** |
| 1 M | 8.0 MB | 0.029 ms | 0.282 ms | **~0.35 – 0.45 ms** |
| 10 M | 80.0 MB | 0.286 ms | 1.533 ms | **~1.80 – 2.20 ms** |
| 50 M | 400.0 MB | 1.429 ms | 3.737 ms | **~4.50 – 5.50 ms** |

---

## Design Trade-offs & Bottlenecks

### 1. The Indexing Cost (Integer Math)
Native 1D reverse uses a simple $N - i + 1$ linear calculation. Our universal kernel uses `CartesianIndices` to support reversing specific dimensions in high-dimensional space. This requires integer division and modulo operations. Even in native CUB, we see the effective bandwidth top out at **107 GB/s** (approx. 38% of theoretical peak), confirming that integer-heavy address generation is the primary bottleneck.

### 2. Memory Coalescing
GPU performance relies on "coalesced" memory access threads in a warp reading adjacent memory addresses. Reversing an array means threads read from the start but write to the end. This mirrored pattern is naturally less efficient for the memory controller than a simple copy. We accept this performance drop relative to SoL as a necessary trade-off for N-dimensional flexibility.

### 3. Launch Overhead
For small arrays ($N < 100k$), the total time is dominated by the CPU-to-GPU dispatch latency (approx **0.03 ms**). This is a fixed cost of the `KernelAbstractions.jl` ecosystem and becomes negligible as the data size grows.

## Summary
The AcceleratedKernels.jl delegation allows `GPUArrays.jl` to move data at massive speeds while remaining completely backend-agnostic. While we lose a margin of peak performance compared to hand-tuned C++ vendor code, we gain a single, maintainable codebase that works on every GPU Julia supports.
