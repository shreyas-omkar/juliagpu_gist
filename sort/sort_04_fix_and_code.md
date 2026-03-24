# `sort!` / `sortperm` / `sortperm!` Implementation Details & Code

## Architectural Strategy
This PR (PR #5) implements GPU sorting operations. Following the established architectural pattern, we delegate the heavy algorithmic lifting to `AcceleratedKernels.jl`, which contains a highly optimized, fully portable iterative merge sort. `GPUArrays.jl` provides the thin delegation wrappers to catch missing vendor implementations.

---

## Design Decisions

### 1. Delegate to AcceleratedKernels.jl, do not write a new kernel
`AMDGPU.jl` and `oneAPI.jl` both delegate to AK.jl in ~6 total lines. AK.jl's merge sort is:
* Thoroughly tested across CUDA, AMDGPU, oneAPI, and Metal
* Stable (preserves equal-element order)
* Full-featured: accepts `lt`, `by`, `rev`, `order` kwargs
* Natively supports `sortperm!` (co-sorts the index array)
* O(log n) passes, each fully parallel

Writing a new KernelAbstractions merge sort kernel from scratch in `GPUArrays.jl` would duplicate hundreds of lines of complex logic. Delegation is the correct engineering choice.

### 2. Dispatch at `AnyGPUArray`, not specific array types
The fallback must live at the `AnyGPUArray` level so that:
* AMDGPU (`AnyROCArray`) keeps its own method (more specific, wins dispatch)
* oneAPI (`oneArray`) keeps its own method (more specific, wins dispatch)
* CUDA (`AnyCuArray`) keeps its own `sort!` method, but **now automatically inherits `sortperm!` and `sortperm` from this fallback**, seamlessly filling the massive gap in CUDA.jl without touching the vendor package.
* Metal, JLArray, and all future backends get all three methods instantly.

### 3. `sortperm` initialization on the GPU device
`sortperm!` requires an index array `ix` initialized to `1:length(A)`. Doing this via a CPU-side `collect(1:n)` and copying to the GPU would incur a severe PCIe transfer penalty. Instead, we use a device-side broadcast:
```julia
ix .= 1:length(A)   # GPU broadcast, allocates and fills entirely on device
```

## Complete Implementation
File: `GPUArrays.jl/src/host/sort.jl` **(NEW FILE)**
```Julia
import AcceleratedKernels as AK

# Delegates to AK.sort! which implements bottom-up GPU merge sort.
function Base.sort!(x::AnyGPUArray; kwargs...)
    AK.sort!(x; kwargs...)
    KernelAbstractions.synchronize(get_backend(x))
    return x
end

# Co-sorts ix alongside x: ix[i] gives the original position of the i-th
# smallest element. AK.sortperm! is stable — equal values preserve index order.
function Base.sortperm!(ix::AnyGPUArray{<:Integer}, x::AnyGPUArray; kwargs...)
    AK.sortperm!(ix, x; kwargs...)
    KernelAbstractions.synchronize(get_backend(x))
    return ix
end

# Out-of-place: allocate ix on the same device as x, then call sortperm!.
# Uses Int (Julia default for indices) to match CPU sortperm behaviour.
function Base.sortperm(x::AnyGPUArray; kwargs...)
    ix = similar(x, Int, length(x))
    ix .= 1:length(x)   # Initialize on GPU, no PCIe transfer
    return sortperm!(ix, x; kwargs...)
end
```
Addition to `src/GPUArrays.jl`
```Julia
# Alongside other host includes:
include("host/sort.jl")
```
## Dispatch Table After This PR
```Plaintext
BEFORE:
sort!(A::AnyCuArray)           →  CUDA.jl quicksort          ✓
sort!(A::AnyROCArray)          →  AMDGPU → AK merge sort     ✓
sort!(A::oneArray)             →  oneAPI → AK merge sort     ✓
sort!(A::MtlArray)             →  Base.sort! → scalar ERROR  ✗ (Crash)
sort!(A::JLArray)              →  Base.sort! → scalar ERROR  ✗ (Crash)

sortperm(A::AnyCuArray)        →  Base.sortperm → scalar     ✗ (CUDA gap!)
sortperm(A::AnyROCArray)       →  AMDGPU → AK merge sort     ✓
sortperm(A::oneArray)          →  oneAPI → AK merge sort     ✓
sortperm(A::MtlArray)          →  Base.sortperm → scalar     ✗ (Crash)
sortperm(A::JLArray)           →  Base.sortperm → scalar     ✗ (Crash)


AFTER:
sort!(A::AnyCuArray)           →  CUDA.jl quicksort          ✓ (unchanged)
sort!(A::AnyROCArray)          →  AMDGPU → AK merge sort     ✓ (unchanged)
sort!(A::oneArray)             →  oneAPI → AK merge sort     ✓ (unchanged)
sort!(A::MtlArray)             →  GPUArrays → AK merge sort  ✓ (fixed)
sort!(A::JLArray)              →  GPUArrays → AK merge sort  ✓ (fixed)

sortperm(A::AnyCuArray)        →  GPUArrays → AK merge sort  ✓ (CUDA gap filled!)
sortperm(A::AnyROCArray)       →  AMDGPU → AK merge sort     ✓ (unchanged)
sortperm(A::oneArray)          →  oneAPI → AK merge sort     ✓ (unchanged)
sortperm(A::MtlArray)          →  GPUArrays → AK merge sort  ✓ (fixed)
sortperm(A::JLArray)           →  GPUArrays → AK merge sort  ✓ (fixed)
```
**Note**: For `sortperm(A::AnyCuArray)`, because `AnyCuArray <: AnyGPUArray` and `CUDA.jl` provides no specific `sortperm` override, the new `GPUArrays` fallback natively assumes command. This fills the known `CUDA.jl` `sortperm` gap without touching a single line of `CUDA.jl` code.
