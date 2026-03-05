# sort! / sortperm / sortperm!   The Fix: Design and Complete Implementation

## Design Decisions

### 1. Delegate to AcceleratedKernels.jl, not write a new kernel

AMDGPU.jl and oneAPI.jl both delegate to AK.jl in ~6 total lines. AK.jl's merge sort is:
- Thoroughly tested across CUDA, AMDGPU, oneAPI, Metal
- Stable (preserves equal-element order)
- Full-featured: `lt`, `by`, `rev`, `order` kwargs
- natively supports `sortperm!` (co-sorts index array)
- O(log n) passes, each fully parallel

Writing a new KA merge sort kernel from scratch would be ~300 lines and would duplicate AK.jl's work. The right engineering choice is delegation.

### 2. Add AcceleratedKernels to GPUArrays.jl Project.toml

GPUArrays.jl does not currently depend on AcceleratedKernels.jl. For `accumulate!` (PR #2), the same dependency was added. This PR is the second consumer of that dependency   the `[deps]` entry is already there after PR #2 lands.

If landing before PR #2: add to `[deps]` and `[compat]`:
```toml
[deps]
AcceleratedKernels = "1e1770d6-e3a8-4f41-8603-0a2fe7d08f8c"

[compat]
AcceleratedKernels = "0.2"
```

### 3. Dispatch at AnyGPUArray   not at specific array types

The fallback must live at `AnyGPUArray` so that:
- AMDGPU (`AnyROCArray`) keeps its own method (more specific, wins)
- oneAPI (`oneArray`) keeps its own method (more specific, wins)
- CUDA (`AnyCuArray`) keeps its own `sort!` method, but NOW also gets `sortperm!` and `sortperm` as fallback   filling the CUDA gap
- Metal, JLArray, and all future backends get all three for free

### 4. sortperm initialization on GPU device

`sortperm!` requires `ix` initialized to `1:length(A)`. This must happen on the GPU   a CPU-side `collect(1:n)` copied to GPU would be correct but slow. Instead we use:
```julia
ix .= 1:length(A)   # GPU broadcast, allocates on device
```
This matches what AMDGPU does: `ROCArray(1:length(x))`.

### 5. File location: new src/host/sort.jl

Sort is not in any existing `src/host/` file. Create `src/host/sort.jl` and add `include("host/sort.jl")` to `src/GPUArrays.jl`. This matches the structure: each operation family has its own file (`mapreduce.jl`, `indexing.jl`, `base.jl`).

---

## Complete Implementation

### File: `GPUArrays.jl/src/host/sort.jl` (NEW FILE)

```julia
# GPUArrays.jl/src/host/sort.jl
# Portable sort!, sortperm!, sortperm for all GPU array backends
# via delegation to AcceleratedKernels.jl (same pattern as AMDGPU.jl, oneAPI.jl)

import AcceleratedKernels as AK

# ── sort! ─────────────────────────────────────────────────────────────────────
# Delegates to AK.sort! which implements bottom-up GPU merge sort.
# AMDGPU/oneAPI have their own more-specific methods; this is the fallback
# for Metal, JLArray, and any future backend.
# CUDA.jl also lacks sortperm   the sortperm methods below fill that gap.
function Base.sort!(x::AnyGPUArray; kwargs...)
    AK.sort!(x; kwargs...)
    return x
end

# ── sortperm! ─────────────────────────────────────────────────────────────────
# Co-sorts ix alongside x: ix[i] gives the original position of the i-th
# smallest element. AK.sortperm! is stable   equal values preserve index order.
function Base.sortperm!(ix::AnyGPUArray{<:Integer}, x::AnyGPUArray; kwargs...)
    AK.sortperm!(ix, x; kwargs...)
    return ix
end

# ── sortperm ──────────────────────────────────────────────────────────────────
# Out-of-place: allocate ix on the same device as x, then call sortperm!.
# Uses Int (Julia default for indices) to match CPU sortperm behaviour.
function Base.sortperm(x::AnyGPUArray; kwargs...)
    ix = similar(x, Int, length(x))
    ix .= 1:length(x)   # initialize on GPU   no PCIe transfer
    return sortperm!(ix, x; kwargs...)
end
```

### Addition to `src/GPUArrays.jl`

```julia
# In src/GPUArrays.jl, alongside other host includes:
include("host/sort.jl")
```

### Addition to `Project.toml`

```toml
[deps]
AcceleratedKernels = "1e1770d6-e3a8-4f41-8603-0a2fe7d08f8c"
# (already present after accumulate! PR #2)

[compat]
AcceleratedKernels = "0.2"
```

---

## Dispatch Table After This PR

```
BEFORE:
sort!(A::AnyCuArray)           →  CUDA.jl quicksort          ✓
sort!(A::AnyROCArray)          →  AMDGPU → AK merge sort      ✓
sort!(A::oneArray)             →  oneAPI → AK merge sort       ✓
sort!(A::MtlArray)             →  Base.sort! → scalar ERROR    ✗
sort!(A::JLArray)              →  Base.sort! → scalar ERROR    ✗
sort!(A::<future>)             →  Base.sort! → scalar ERROR    ✗

sortperm(A::AnyCuArray)        →  Base.sortperm → scalar ERROR ✗  ← CUDA gap!
sortperm(A::AnyROCArray)       →  AMDGPU → AK merge sort       ✓
sortperm(A::oneArray)          →  oneAPI → AK merge sort        ✓
sortperm(A::MtlArray)          →  Base.sortperm → scalar ERROR  ✗
sortperm(A::JLArray)           →  Base.sortperm → scalar ERROR  ✗

AFTER:
sort!(A::AnyCuArray)           →  CUDA.jl quicksort          ✓  (unchanged)
sort!(A::AnyROCArray)          →  AMDGPU → AK merge sort      ✓  (unchanged)
sort!(A::oneArray)             →  oneAPI → AK merge sort       ✓  (unchanged)
sort!(A::MtlArray)             →  GPUArrays → AK merge sort   ✓  (fixed)
sort!(A::JLArray)              →  GPUArrays → AK merge sort   ✓  (fixed)
sort!(A::<future>)             →  GPUArrays → AK merge sort   ✓  (fixed)

sortperm(A::AnyCuArray)        →  GPUArrays → AK merge sort   ✓  (CUDA gap filled!)
sortperm(A::AnyROCArray)       →  AMDGPU → AK merge sort       ✓  (unchanged)
sortperm(A::oneArray)          →  oneAPI → AK merge sort        ✓  (unchanged)
sortperm(A::MtlArray)          →  GPUArrays → AK merge sort   ✓  (fixed)
sortperm(A::JLArray)           →  GPUArrays → AK merge sort   ✓  (fixed)
```

Note: For `sortperm(A::AnyCuArray)`, `AnyCuArray <: AnyGPUArray`, so the new GPUArrays method applies. This is intentional   it fills the known CUDA.jl sortperm gap without touching CUDA.jl itself.

---

## Why sortperm is Harder Than sort!

For sort!, you only move values. For sortperm!, you must track provenance   where did each value originally come from? The co-sort approach (carry ix alongside vals through every merge step) is the standard stable solution. AK.jl implements it in `merge_sortperm.jl`.

The memory cost is higher: merge sort already needs 1 temporary buffer. sortperm! needs 2 (one for values, one for indices). Total memory for sortperm on an array of n Float32 elements:

```
Input array:      n × 4 bytes
Output ix array:  n × 8 bytes (Int64)
Temp val buffer:  n × 4 bytes
Temp ix buffer:   n × 8 bytes
─────────────────────────────
Total:            n × 24 bytes
```

For n = 10^7: 240 MB. RTX 3060 has 12 GB VRAM   comfortably fits. For very large n, users should be aware of this 6× memory overhead vs the input alone.
