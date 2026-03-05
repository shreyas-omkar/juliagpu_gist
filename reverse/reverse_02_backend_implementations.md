# reverse / reverse! — Backend Implementations

The algorithm is identical across every backend that has one. What differs is **only the kernel launch syntax and thread index intrinsic**.

---

## CUDA.jl — [`src/reverse.jl`](https://github.com/JuliaGPU/CUDA.jl/blob/master/src/reverse.jl)

Defines `_reverse` (out-of-place) and `_reverse!` (in-place) as internal functions, each containing a `@cuda`-launched closure.

- Thread index: `blockDim().x * (blockIdx().x - 1i32) + threadIdx().x`
- Mirror formula: `ifelse.(rev_dims, ref .- idx, idx)`
- In-place: `cld(size(x,d), 2)` halving + `index_in < index_out` guard

```julia
function _reverse(input::AnyCuArray{T,N}, output::AnyCuArray{T,N}; dims=1:ndims(input)) where {T,N}
    rev_dims = ntuple(d -> d in dims && size(input,d) > 1, N)
    ref      = size(input) .+ 1
    function kernel(input, output)
        i = blockDim().x * (blockIdx().x - 1i32) + threadIdx().x
        @inbounds if i <= length(input)
            idx = ifelse.(rev_dims, ref .- Tuple(CartesianIndices(input)[i]),
                                        Tuple(CartesianIndices(input)[i]))
            output[LinearIndices(input)[idx...]] = input[i]
        end
        return
    end
    @cuda threads=256 blocks=cld(length(input), 256) kernel(input, output)
end
```

---

## AMDGPU.jl — [`src/kernels/reverse.jl`](https://github.com/JuliaGPU/AMDGPU.jl/blob/master/src/kernels/reverse.jl)

Source file explicitly comments *"Adapted from CUDA.jl"* — algorithm is identical. Only the intrinsics change:

- Thread index: `workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x`
- Launch: `@roc groupsize=256 gridsize=cld(length(x), 256)`
- Same mirror formula, same `cld` halving, same `idx_in < idx_out` guard

```julia
function _kernel!(y, x)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    if i <= length(x)
        idx = ifelse.(rev_dims, ref .- Tuple(nd_ids[i]), Tuple(nd_ids[i]))
        y[lin_ids[idx...]] = x[i]
    end
    return
end
@roc groupsize=256 gridsize=cld(length(x), 256) _kernel!(y, x)
```

---

## oneAPI.jl — No Implementation

Confirmed by exhaustive audit of [`JuliaGPU/oneAPI.jl`](https://github.com/JuliaGPU/oneAPI.jl). No `reverse` or `reverse!` method exists. Dispatch falls through to `Base.reverse(::AbstractArray)`.

**Result with `allowscalar(false)`:** hard error at first `getindex`.  
**Result with `allowscalar(true)`:** silent scalar CPU loop — one PCIe transfer per element, no warning.

---

## Metal.jl — No Implementation

Confirmed by exhaustive audit of [`JuliaGPU/Metal.jl`](https://github.com/JuliaGPU/Metal.jl). Same failure mode as oneAPI.jl.

---

## Summary

| Backend | Has `reverse`? | Approach |
|---------|:-:|---|
| CUDA.jl | ✅ | Own `@cuda` kernel |
| AMDGPU.jl | ✅ | Own `@roc` kernel (adapted from CUDA.jl) |
| oneAPI.jl | ❌ | Falls to `Base` → ERROR / silent CPU |
| Metal.jl | ❌ | Falls to `Base` → ERROR / silent CPU |
| **GPUArrays.jl (PR #1)** | ✅ | **KA kernel at `AnyGPUArray` — fixes all** |
