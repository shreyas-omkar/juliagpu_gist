# `reverse` / `reverse!` Implementation Details & Code

## Architectural Strategy
This PR follows the delegation pattern established by `AMDGPU.jl` and `oneAPI.jl` for `accumulate!` and `sort!`: heavy kernel logic lives upstream in `AcceleratedKernels.jl`, and `GPUArrays.jl` adds only a thin, 3-line wrapper dispatching on `AnyGPUArray`.

**Two-phase split:**
1. **AcceleratedKernels.jl** Adds `src/reverse.jl` containing the universal ND kernels.
2. **GPUArrays.jl** Adds the 3-line delegation wrappers in `src/host/base.jl`.

---

## Design Decisions

### 1. `Val` wrap tuple arguments for GPU compiler inference
Passing runtime Julia tuples as kernel arguments fails on several backends (especially older CUDA and ROCm targets) because the compiler cannot infer their size at specialisation time, leading to dynamic allocation on the device. All per-kernel tuple arguments (`rev_dims`, `ref_sz`, `reduced_sz`) are `Val`-wrapped so they compile down to zero-cost constants:
`kernel!(dst, src, Val(rev_dims), Val(ref_sz); ndrange=length(src))`

### 2. Out-of-place: One thread per element
A thread at linear index `i` reads `CartesianIndices(src)[i]`, mirrors the coordinate along each reversed dimension via `ref_sz[d] - nd[d]` (equivalent to $N - i + 1$ in 1D), and writes to `dst`. Zero data dependencies, zero synchronisation.

### 3. In-place: Single-dimension Halving & Linear Guard
Launching a full thread domain for an in-place reversal would double-swap elements, undoing the work. Instead, we launch exactly half the required threads by halving **exactly one** of the reversed dimensions (the last one found). 
* Halving *one* dimension yields exactly $\lceil N / 2 \rceil$ total threads. (Halving *all* reversed dimensions would erroneously shrink the thread domain to a fraction of what is needed).
* The guard `if lin_in < lin_out` handles the rest. It mathematically skips:
  1. The exact middle element of odd-length dimensions (self-swap).
  2. Any overlapping threads caused by odd-length boundary conditions, preventing un-swapping.

### 4. GPUArrays Synchronization Convention
The `GPUArrays.jl` wrapper explicitly delegates without calling `KernelAbstractions.synchronize()`. This matches the established `GPUArrays` convention where async stream semantics are the caller's responsibility, allowing sequential array operations to pipeline efficiently on the device without blocking the host.

---

## Phase 1: AcceleratedKernels.jl Implementation

**File:** `AcceleratedKernels.jl/src/reverse.jl` *(New File)*

```julia
# ── Out-of-place kernel ───────────────────────────────────────────────────────

@kernel function _reverse_kernel!(dst, @Const(src), ::Val{rev_dims}, ::Val{ref_sz}) where {rev_dims, ref_sz}
    i = @index(Global, Linear)
    @inbounds begin
        nd = CartesianIndices(src)[i]
        nd_out = CartesianIndex(
            ntuple(d -> rev_dims[d] ? ref_sz[d] - nd[d] : nd[d], ndims(src))
        )
        dst[nd_out] = src[nd]
    end
end

function reverse(src::AbstractArray{T, N}, backend::Backend; dims=:) where {T, N}
    dims_iter = dims isa Colon ? (1:N) : (dims isa Integer ? (dims,) : dims)
    for d in dims_iter
        1 <= d <= N || throw(ArgumentError("dimension $d out of range for $(N)D array"))
    end

    rev_dims = ntuple(d -> d in dims_iter && size(src, d) > 1, N)
    ref_sz   = ntuple(d -> size(src, d) + 1, N) # pre-shifted: ref[d] - nd[d] = N - i + 1
    
    # Use similar() to preserve specific array wrapper types (e.g., CuArray)
    dst = similar(src)

    kernel! = _reverse_kernel!(backend)
    kernel!(dst, src, Val(rev_dims), Val(ref_sz); ndrange=length(src))

    return dst
end

# ── In-place kernel ───────────────────────────────────────────────────────────

@kernel function _reverse_inplace_kernel!(A, ::Val{rev_dims}, ::Val{ref_sz}, ::Val{reduced_sz}) where {rev_dims, ref_sz, reduced_sz}
    i = @index(Global, Linear)
    @inbounds begin
        # Map linear thread index into the halved index space
        idx_in  = CartesianIndices(reduced_sz)[i]
        lin_in  = LinearIndices(A)[idx_in]

        # Calculate the mirrored coordinate in the full index space
        idx_out = CartesianIndex(
            ntuple(d -> rev_dims[d] ? ref_sz[d] - idx_in[d] : idx_in[d], ndims(A))
        )
        lin_out = LinearIndices(A)[idx_out]

        # Guard: skip middle element of odd-length dims and prevent double-swap
        if lin_in < lin_out
            tmp        = A[lin_in]
            A[lin_in]  = A[lin_out]
            A[lin_out] = tmp
        end
    end
end

function reverse!(A::AbstractArray{T, N}, backend::Backend; dims=:) where {T, N}
    dims_iter = dims isa Colon ? (1:N) : (dims isa Integer ? (dims,) : dims)
    for d in dims_iter
        1 <= d <= N || throw(ArgumentError("dimension $d out of range for $(N)D array"))
    end

    rev_dims = ntuple(d -> d in dims_iter && size(A, d) > 1, N)
    half_dim = findlast(rev_dims)
    
    # If no dimensions are larger than 1, reversing is a no-op
    isnothing(half_dim) && return A

    # Halve EXACTLY ONE reversed dimension to yield ⌈N/2⌉ total threads
    reduced_sz = ntuple(d -> d == half_dim ? cld(size(A, d), 2) : size(A, d), N)
    ref_sz     = ntuple(d -> size(A, d) + 1, N)

    kernel! = _reverse_inplace_kernel!(backend)
    kernel!(A, Val(rev_dims), Val(ref_sz), Val(reduced_sz); ndrange=prod(reduced_sz))

    return A
end
```

(Add include("reverse.jl") to src/AcceleratedKernels.jl)

--- 

## Phase 2: GPUArrays.jl Delegation
**File**: `GPUArrays.jl/src/host/base.jl`


```
import AcceleratedKernels as AK

function Base.reverse(A::AnyGPUArray; dims=:)
    return AK.reverse(A, get_backend(A); dims=dims)
end

function Base.reverse!(A::AnyGPUArray; dims=:)
    return AK.reverse!(A, get_backend(A); dims=dims)
end
```
---

## Dispatch Table After This PR
Julia's multiple dispatch guarantees that `AnyCuArray <: AnyGPUArray` and `AnyROCArray <: AnyGPUArray`. Because the vendor methods are more specific, they always win. This ensures zero changes to any vendor package while seamlessly fixing the missing backends.

```
BEFORE:
reverse(A::AnyCuArray)  →  CUDA.jl @cuda kernel        ✓
reverse(A::AnyROCArray) →  AMDGPU.jl @roc kernel       ✓
reverse(A::oneArray)    →  Base scalar loop            ✗ (ERROR / silent)
reverse(A::MtlArray)    →  Base scalar loop            ✗ (ERROR / silent)
reverse(A::JLArray)     →  Base scalar loop            ✗ (ERROR / silent)
reverse(A::<future>)    →  Base scalar loop            ✗ (ERROR / silent)

AFTER:
reverse(A::AnyCuArray)  →  CUDA.jl @cuda kernel        ✓ (unchanged)
reverse(A::AnyROCArray) →  AMDGPU.jl @roc kernel       ✓ (unchanged)
reverse(A::oneArray)    →  GPUArrays → AK.jl kernel    ✓ (fixed)
reverse(A::MtlArray)    →  GPUArrays → AK.jl kernel    ✓ (fixed)
reverse(A::JLArray)     →  GPUArrays → AK.jl kernel    ✓ (fixed)
reverse(A::<future>)    →  GPUArrays → AK.jl kernel    ✓ (automatic)
```
