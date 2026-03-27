# `reverse` / `reverse!` Performance Analysis

## Architectural Strategy
Our performance target is the **Speed of Light (SoL)** limit of the hardware. Since `reverse` has no data dependencies, it is 100% memory-bound. The bottleneck is not the GPU’s compute throughput, but the saturation of the VRAM bus.

By delegating to **AcceleratedKernels.jl**, we aim for **30–40% efficiency** relative to the hardware peak. While a native 1D CUDA kernel can hit 90%, we face "N-dimensional tax" to support universal array shapes, which requires more complex coordinate arithmetic than a simple 1D linear swap.

---

## Performance Derivation (The Math)

To verify the integrity of the implementation, we derive the minimum execution time for 50 Million elements on an **NVIDIA Tesla T4**.

### 1. Data Traffic Volume
For an array of $N$ elements of type `Float32` (4 bytes each):
* **Read Phase:** Every element is fetched from global memory once ($N \times 4$ bytes).
* **Write Phase:** Every element is stored in its new mirrored position ($N \times 4$ bytes).
* **Total Movement:** $8 \times N$ bytes.

**For 50 Million elements:**
$$50,000,000 \times 8 \text{ bytes} = 400 \text{ MB}$$

### 2. The Speed Limit (SoL)
The Tesla T4 has a sustained practical bandwidth ($\beta$) of approximately **280 GB/s**. The minimum physical execution time ($T_{min}$) is:
$$T_{min} = \frac{400 \text{ MB}}{280 \text{ GB/s}} \approx \mathbf{1.43 \text{ ms}}$$

Any result under **5 ms** for 50M elements indicates that the kernel is effectively saturating the hardware bus despite the overhead of Julia's N-dimensional indexing logic.

---

## Design Trade-offs & Bottlenecks

### 1. The Indexing Cost (Integer Math)
A native 1D reverse uses a simple $N - i + 1$ calculation. Our universal kernel uses `CartesianIndices` to support reversing specific dimensions in high-dimensional space. This requires integer division and modulo operations. Because GPUs are optimized for floating-point math, these integer instructions create a slight "instruction-bound" delay that prevents 100% bandwidth saturation.

### 2. Memory Coalescing
GPU performance relies on "coalesced" memory access threads in a warp reading adjacent memory addresses. Reversing an array means threads read from the start but write to the end. This mirrored pattern is naturally less efficient for the memory controller than a simple copy. We accept this performance drop as a trade-off for N-dimensional flexibility.

### 3. Launch Overhead
For small arrays ($N < 100k$), the total time is dominated by the CPU-to-GPU dispatch latency (approx **0.03 ms**). This is a fixed cost of the `KernelAbstractions.jl` ecosystem and becomes negligible as the data size grows.

---

## Forecasted Performance (Tesla T4)

| Size (N) | Total Data Moved | Ideal Limit (SoL) | AK Forecast (Target) |
| :--- | :--- | :--- | :--- |
| 100 K | 0.8 MB | 0.003 ms | 0.04 – 0.06 ms |
| 1 M | 8.0 MB | 0.029 ms | 0.12 – 0.18 ms |
| 10 M | 80.0 MB | 0.286 ms | 0.70 – 0.90 ms |
| 50 M | 400.0 MB | 1.429 ms | **3.50 – 4.50 ms** |

---

## Summary
The AcceleratedKernels.jl delegation allows `GPUArrays.jl` to move data at massive speeds roughly **100+ GB/s** while remaining completely backend-agnostic. We lose a small margin of peak performance compared to hand-tuned C++ vendor code, but we gain a single, maintainable codebase that works on every GPU Julia supports.
