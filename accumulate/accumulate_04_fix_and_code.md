# `accumulate!` / `cumsum` / `cumprod` Implementation Details & Code

## Design Decisions

### 1. Delegate to AcceleratedKernels.jl don't reimplement
`AcceleratedKernels.jl` is an official JuliaGPU package with a well-tested, optimized Blelloch scan and DecoupledLookback scan for all KA backends. `AMDGPU.jl` and `oneAPI.jl` already use it for exactly this purpose. Reimplementing the same algorithm in `GPUArrays.jl` would duplicate maintenance with zero benefit.

### 2. Native Backend Resolution
The `GPUArrays.jl` fallback passes the arrays directly to `AK.accumulate!`, which internally uses `get_backend(A)` to resolve the correct backend at runtime for any array type `JLArray`, future backends, or any user-defined `AnyGPUArray` subtype.

### 3. Do not pass `block_size`
The `oneAPI.jl` `block_size=64` workaround is Intel-specific. Since `oneAPI.jl`'s method is more specific and always wins dispatch for `oneArray`, the fallback never runs on Intel hardware. Hardcoding that workaround in the universal fallback would be an architectural error.

### 4. Dispatch at `AnyGPUArray`
All four vendor methods dispatch on more specific types (`AnyCuArray`, `MtlArray`, etc.). The `AnyGPUArray` fallback is never invoked for any existing vendor only `JLArray`, future backends, and user-defined subtypes.

### 5. One `Project.toml` change the only one in the entire project
`AcceleratedKernels.jl` must be added as an explicit dependency of `GPUArrays.jl`. This is the only `Project.toml` change across all six PRs in this GSoC proposal.

---

## Complete Implementation

### New file: `GPUArrays.jl/src/host/accumulate.jl`

```julia
import AcceleratedKernels as AK

# Universal fallback for prefix scans
function Base.accumulate!(op, B::AnyGPUArray, A::AnyGPUArray; kwargs...)
    AK.accumulate!(op, B, A; kwargs...)
    return B
end

# Specialized in-place wrappers
function Base.cumsum!(B::AnyGPUArray, A::AnyGPUArray; kwargs...)
    return Base.accumulate!(+, B, A; kwargs...)
end

function Base.cumprod!(B::AnyGPUArray, A::AnyGPUArray; kwargs...)
    return Base.accumulate!(*, B, A; kwargs...)
end

# Out-of-place allocation wrappers
function Base.cumsum(A::AnyGPUArray; kwargs...)
    return Base.cumsum!(similar(A), A; kwargs...)
end

function Base.cumprod(A::AnyGPUArray; kwargs...)
    return Base.cumprod!(similar(A), A; kwargs...)
end
```
15 lines of pure routing. All algorithmic complexity and performance optimization lives upstream inside `AcceleratedKernels.jl`.

---

### Add to `Project.toml`
```TOML
[deps]
AcceleratedKernels = "..."   # UUID

[compat]
AcceleratedKernels = "0.2"
```

### Add to `src/GPUArrays.jl`
```Julia
include("host/accumulate.jl")
```
---
### Dispatch Table After PR
```Plaintext
BEFORE:
accumulate!(op, B::AnyCuArray,  A)  →  CUDA.jl @cuda kernel     ✓
accumulate!(op, B::AnyROCArray, A)  →  AMDGPU → AK.jl           ✓
accumulate!(op, B::oneArray,    A)  →  oneAPI → AK.jl (bs=64)   ✓
accumulate!(op, B::MtlArray,    A)  →  Metal scratch kernel     ✓
accumulate!(op, B::JLArray,     A)  →  Base sequential loop     ✗ (ERROR / silent)
accumulate!(op, B::<future>,    A)  →  Base sequential loop     ✗ (ERROR / silent)

AFTER:
accumulate!(op, B::AnyCuArray,  A)  →  CUDA.jl @cuda kernel     ✓ (unchanged)
accumulate!(op, B::AnyROCArray, A)  →  AMDGPU → AK.jl           ✓ (unchanged)
accumulate!(op, B::oneArray,    A)  →  oneAPI → AK.jl (bs=64)   ✓ (unchanged)
accumulate!(op, B::MtlArray,    A)  →  Metal scratch kernel     ✓ (unchanged)
accumulate!(op, B::JLArray,     A)  →  GPUArrays → AK.jl        ✓ (fixed)
accumulate!(op, B::<future>,    A)  →  GPUArrays → AK.jl        ✓ (automatic)
```
