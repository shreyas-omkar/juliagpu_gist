# accumulate! / cumsum / cumprod   Backend Implementations

Unlike `reverse`, **all four vendor backends already have `accumulate!`**.  
The gap is at the `GPUArrays.jl` level   no fallback exists for non-vendor backends.

---

## CUDA.jl   [`src/accumulate.jl`](https://github.com/JuliaGPU/CUDA.jl/blob/master/src/accumulate.jl)

Implements from scratch using `@cuda`. Two-pass strategy:
1. Local scan within each thread block (shared memory)
2. Inter-block correction pass via a shared auxiliary array

Exploits warp-level primitives not available through KernelAbstractions.jl for optimal NVIDIA performance.

---

## AMDGPU.jl   [`src/kernels/accumulate.jl`](https://github.com/JuliaGPU/AMDGPU.jl/blob/master/src/kernels/accumulate.jl)

Seven lines. Delegates entirely to AcceleratedKernels.jl, passing `ROCBackend()`:

```julia
Base.accumulate!(op, B::AnyROCArray, A::AnyROCArray;
                 init=zero(eltype(A)), kwargs...) =
    AK.accumulate!(op, B, A, ROCBackend(); init, kwargs...)

Base.cumsum(src::AnyROCArray; kwargs...)  = AK.cumsum(src,  ROCBackend(); kwargs...)
Base.cumprod(src::AnyROCArray; kwargs...) = AK.cumprod(src, ROCBackend(); kwargs...)
```

---

## oneAPI.jl   [`src/accumulate.jl`](https://github.com/JuliaGPU/oneAPI.jl/blob/master/src/accumulate.jl)

Also delegates to AcceleratedKernels.jl, but with a **critical workaround**: `block_size` capped at 64.

Intel GPU hardware has a **verified bug** in the Blelloch scan at `block_size ≥ 128`   produces incorrect results. Fix is a hardcoded constant:

```julia
const _ACCUMULATE_BLOCK_SIZE = 64  # Intel Blelloch bug at >= 128

Base.accumulate!(op, B::oneArray, A::oneArray;
                 init=zero(eltype(A)),
                 block_size=_ACCUMULATE_BLOCK_SIZE, kwargs...) =
    AK.accumulate!(op, B, A, oneAPIBackend(); init, block_size, kwargs...)
```

> **Note:** This workaround is NOT needed in the GPUArrays.jl fallback   `oneAPI.jl`'s more specific method always wins dispatch for `oneArray`. The fallback never executes on Intel hardware.

---

## Metal.jl   [`src/accumulate.jl`](https://github.com/JuliaGPU/Metal.jl/blob/master/src/accumulate.jl)

**Cannot use AcceleratedKernels.jl.** AK's `DecoupledLookback` algorithm requires `memory_order_acq_rel` atomics   Metal's shader model does not expose these to compute kernels.

Instead, Metal.jl implements a full Blelloch scan from scratch using Metal-specific intrinsics:

```julia
# MtlThreadGroupArray = Metal shared memory
# threadgroup_barrier(MemoryFlagThreadGroup) = Metal sync primitive

function partial_scan(op, output, input, ::Val{maxthreads}) where maxthreads
    shared = MtlThreadGroupArray(eltype(output), 2 * maxthreads)
    threadgroup_barrier(MemoryFlagThreadGroup)
    # full up-sweep and down-sweep in Metal intrinsics
end

function aggregate_partial_scan(op, output, aggregates, ::Val{maxthreads})
    # second pass: inter-block corrections
end
```

Metal.jl also overrides `Base._accumulate!` and `accumulate_pairwise!` to ensure all code paths route through GPU. Most complete and complex of the four vendor implementations.

---

## Summary

| Backend | Approach | Delegates to AK.jl? |
|---------|----------|:-:|
| CUDA.jl | Own `@cuda` two-pass kernel | No |
| AMDGPU.jl | 7-line delegation | Yes |
| oneAPI.jl | Delegation + `block_size=64` workaround | Yes |
| Metal.jl | Full Blelloch scratch (no atomics available) | No |
| **GPUArrays.jl (PR #2)** | **Delegation via `get_backend(A)`** | **Yes** |
