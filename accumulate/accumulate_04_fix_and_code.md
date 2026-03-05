# accumulate! / cumsum / cumprod   The Fix

## Design Decisions

### 1. Delegate to AcceleratedKernels.jl   don't reimplement
AK.jl is an official JuliaGPU package with a well-tested, optimised Blelloch scan and DecoupledLookback scan for all KA backends. AMDGPU.jl and oneAPI.jl already use it for exactly this purpose. Reimplementing the same algorithm in GPUArrays.jl would duplicate maintenance with zero benefit.

### 2. Use `get_backend(A)`   not a hardcoded backend
AMDGPU.jl passes `ROCBackend()` explicitly. The GPUArrays.jl fallback uses `get_backend(A)` to resolve the correct backend at runtime for any array type   JLArray, future backends, or any user-defined `AnyGPUArray` subtype.

### 3. Do not pass `block_size`
The oneAPI.jl `block_size=64` workaround is Intel-specific. Since `oneAPI.jl`'s method is more specific and always wins dispatch for `oneArray`, the fallback never runs on Intel hardware. Hardcoding that workaround here would be wrong.

### 4. Dispatch at `AnyGPUArray`
All four vendor methods dispatch on more specific types. The `AnyGPUArray` fallback is never invoked for any existing vendor   only JLArray, future backends, and user-defined subtypes.

### 5. One `Project.toml` change   the only one in the entire project
`AcceleratedKernels.jl` must be added as an explicit dependency of `GPUArrays.jl`. This is the only `Project.toml` change across all four PRs.

---

## Complete Implementation

### New file: `GPUArrays.jl/src/host/accumulate.jl`

```julia
import AcceleratedKernels as AK

function Base.accumulate!(op, B::AnyGPUArray, A::AnyGPUArray;
                          init=zero(eltype(A)), kwargs...)
    AK.accumulate!(op, B, A, get_backend(A); init, kwargs...)
end

function Base.cumsum(A::AnyGPUArray; kwargs...)
    accumulate!(+, similar(A), A; init=zero(eltype(A)), kwargs...)
end

function Base.cumprod(A::AnyGPUArray; kwargs...)
    accumulate!(*, similar(A), A; init=one(eltype(A)), kwargs...)
end
```

10 lines. All complexity lives inside AcceleratedKernels.jl.

### Add to `Project.toml`

```toml
[deps]
AcceleratedKernels = "..."   # UUID

[compat]
AcceleratedKernels = "0.2"
```

### Add to `src/GPUArrays.jl`

```julia
include("host/accumulate.jl")
```

---

## Dispatch Table After PR

```
BEFORE:
accumulate!(op, B::AnyCuArray,  A)  →  CUDA.jl @cuda kernel     ✓
accumulate!(op, B::AnyROCArray, A)  →  AMDGPU → AK.jl           ✓
accumulate!(op, B::oneArray,    A)  →  oneAPI → AK.jl (bs=64)   ✓
accumulate!(op, B::MtlArray,    A)  →  Metal scratch kernel      ✓
accumulate!(op, B::JLArray,     A)  →  Base sequential loop      ✗
accumulate!(op, B::<future>,    A)  →  Base sequential loop      ✗

AFTER:
accumulate!(op, B::AnyCuArray,  A)  →  CUDA.jl @cuda kernel     ✓  (unchanged)
accumulate!(op, B::AnyROCArray, A)  →  AMDGPU → AK.jl           ✓  (unchanged)
accumulate!(op, B::oneArray,    A)  →  oneAPI → AK.jl (bs=64)   ✓  (unchanged)
accumulate!(op, B::MtlArray,    A)  →  Metal scratch kernel      ✓  (unchanged)
accumulate!(op, B::JLArray,     A)  →  GPUArrays → AK.jl        ✓  (fixed)
accumulate!(op, B::<future>,    A)  →  GPUArrays → AK.jl        ✓  (fixed)
```

All four vendor methods remain untouched. More specific types always win Julia dispatch.
