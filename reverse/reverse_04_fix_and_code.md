# reverse / reverse! — The Fix

## Design Decisions

### 1. One thread per element (out-of-place)
No data dependencies → embarrassingly parallel. No synchronisation needed.  
Each thread reads one element, writes to its mirror. `ndrange = length(A)`.

### 2. Half-thread launch (in-place)
Launch ⌈n/2⌉ threads, not n. Each thread owns one swap pair.  
Guard `lin_in < lin_out` skips the middle element of odd-length dimensions.  
Matches CUDA.jl's and AMDGPU.jl's proven strategy exactly.

### 3. `@index(Global, Linear)` instead of vendor intrinsics
Replaces `blockDim().x * (blockIdx().x-1) + threadIdx().x` (CUDA)  
and `workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x` (AMDGPU).  
KernelAbstractions.jl generates the correct intrinsic per backend at compile time.  
**One source file, all backends.**

### 4. Dispatch at `AnyGPUArray`, not a specific type
CUDA.jl's `AnyCuArray` and AMDGPU.jl's `AnyROCArray` are more specific — they always win.  
The new method is only ever reached by: oneAPI, Metal, JLArray, and any future backend.  
**Zero changes to any vendor package.**

---

## Complete Implementation

File: `GPUArrays.jl/src/host/base.jl` — add after existing `issorted_kernel!`

```julia
# ── Out-of-place ───────────────────────────────────────────────────────────────
@kernel function reverse_kernel!(dst, src, rev_dims, ref)
    i = @index(Global, Linear)
    @inbounds begin
        nd     = CartesianIndices(src)[i]
        nd_out = CartesianIndex(
            ntuple(d -> rev_dims[d] ? ref[d] - nd[d] : nd[d], ndims(src))
        )
        dst[nd_out] = src[nd]
    end
end

function Base.reverse(A::AnyGPUArray{T,N}; dims=:) where {T,N}
    dims_iter = dims isa Colon ? (1:N) : dims
    !all(1 .≤ dims_iter .≤ N) && throw(ArgumentError("dimension out of range"))
    rev_dims = ntuple(d -> d in dims_iter && size(A,d) > 1, N)
    out      = similar(A)
    reverse_kernel!(get_backend(A))(out, A, rev_dims, size(A) .+ 1;
                                    ndrange = length(A))
    return out
end

# ── In-place ───────────────────────────────────────────────────────────────────
@kernel function reverse_inplace_kernel!(A, rev_dims, ref, nd_reduced)
    i = @index(Global, Linear)
    @inbounds begin
        idx_in  = CartesianIndices(nd_reduced)[i]
        lin_in  = LinearIndices(A)[idx_in]
        idx_out = CartesianIndex(
            ntuple(d -> rev_dims[d] ? ref[d] - idx_in[d] : idx_in[d], ndims(A))
        )
        lin_out = LinearIndices(A)[idx_out]
        if lin_in < lin_out
            A[lin_in], A[lin_out] = A[lin_out], A[lin_in]
        end
    end
end

function Base.reverse!(A::AnyGPUArray{T,N}; dims=:) where {T,N}
    dims_iter = dims isa Colon ? (1:N) : dims
    !all(1 .≤ dims_iter .≤ N) && throw(ArgumentError("dimension out of range"))
    rev_dims   = ntuple(d -> d in dims_iter && size(A,d) > 1, N)
    half_dim   = findlast(rev_dims)
    isnothing(half_dim) && return A
    reduced_sz = ntuple(d -> d == half_dim ? cld(size(A,d), 2) : size(A,d), N)
    reverse_inplace_kernel!(get_backend(A))(
        A, rev_dims, size(A) .+ 1, reduced_sz;
        ndrange = prod(reduced_sz)
    )
    return A
end
```

---

## Dispatch Table After PR

```
BEFORE:
reverse(A::AnyCuArray)   →  CUDA.jl kernel      ✓
reverse(A::AnyROCArray)  →  AMDGPU.jl kernel    ✓
reverse(A::oneArray)     →  Base → ERROR         ✗
reverse(A::MtlArray)     →  Base → ERROR         ✗
reverse(A::JLArray)      →  Base → ERROR         ✗
reverse(A::<future>)     →  Base → ERROR         ✗

AFTER:
reverse(A::AnyCuArray)   →  CUDA.jl kernel      ✓  (unchanged)
reverse(A::AnyROCArray)  →  AMDGPU.jl kernel    ✓  (unchanged)
reverse(A::oneArray)     →  GPUArrays KA kernel  ✓  (fixed)
reverse(A::MtlArray)     →  GPUArrays KA kernel  ✓  (fixed)
reverse(A::JLArray)      →  GPUArrays KA kernel  ✓  (fixed)
reverse(A::<future>)     →  GPUArrays KA kernel  ✓  (fixed)
```

Julia's dispatch guarantees this automatically — more specific types always win. No vendor package is touched.
