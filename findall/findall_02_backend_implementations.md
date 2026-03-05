# findall — Backend Implementations

All four vendor backends use the **same two-step stream compaction algorithm**. What differs is only the kernel launch syntax and, in Metal's case, a memory storage qualifier.

---

## CUDA.jl — [`src/indexing.jl`](https://github.com/JuliaGPU/CUDA.jl/blob/master/src/indexing.jl)

- Step 1: own `cumsum` implementation
- Step 2: `@cuda`-launched scatter kernel
- Thread index: `threadIdx().x + (blockIdx().x - 1i32) * blockDim().x`
- Output size: `@allowscalar indices[end]`
- Memory: `unsafe_free!(indices)` to reclaim GPU memory immediately

```julia
function Base.findall(bools::AnyCuArray{Bool})
    I       = keytype(bools)
    indices = cumsum(reshape(bools, prod(size(bools))))
    n       = isempty(indices) ? 0 : @allowscalar indices[end]
    ys      = CuArray{I}(undef, n)
    if n > 0
        function kernel(ys, bools, indices)
            i = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
            @inbounds if i <= length(bools) && bools[i]
                ys[indices[i]] = CartesianIndices(bools)[i]
            end
            return
        end
        kernel  = @cuda name="findall" launch=false kernel(ys, bools, indices)
        config  = launch_configuration(kernel.fun)
        threads = min(length(indices), config.threads)
        kernel(ys, bools, indices; threads, blocks=cld(length(indices), threads))
    end
    unsafe_free!(indices)
    return ys
end
```

---

## AMDGPU.jl — [`src/indexing.jl`](https://github.com/JuliaGPU/AMDGPU.jl/blob/master/src/indexing.jl)

Structure is identical to CUDA.jl. Only the intrinsics change:

- Thread index: `workitemIdx().x + (workgroupIdx().x - Int32(1)) * workgroupDim().x`
- Launch: `@roc launch=false` + `launch_configuration`
- Same `@allowscalar indices[end]` + `unsafe_free!(indices)`

```julia
function _ker!(ys, bools, indices)
    i = workitemIdx().x + (workgroupIdx().x - Int32(1)) * workgroupDim().x
    @inbounds if i ≤ length(bools) && bools[i]
        ys[indices[i]] = CartesianIndices(bools)[i]
    end
    return
end
# launched with @roc groupsize=... gridsize=...
```

---

## oneAPI.jl — [`src/indexing.jl`](https://github.com/JuliaGPU/oneAPI.jl/blob/master/src/indexing.jl)

Kernel must be defined **outside** the function body (Intel compiler requirement).  
Thread index intrinsic: `get_global_id()`.  
Launch keywords: `items=group_size, groups=cld(...)`.

```julia
function _ker!(ys, bools, indices)     # top-level, not a closure
    i = get_global_id()
    @inbounds if i ≤ length(bools) && bools[i]
        ys[indices[i]] = CartesianIndices(bools)[i]
    end
    return
end

function Base.findall(bools::oneArray{Bool})
    indices    = cumsum(reshape(bools, prod(size(bools))))
    n          = isempty(indices) ? 0 : @allowscalar indices[end]
    ys         = oneArray{keytype(bools)}(undef, n)
    kernel     = @oneapi launch=false _ker!(ys, bools, indices)
    group_size = launch_configuration(kernel)
    n > 0 && kernel(ys, bools, indices; items=group_size, groups=cld(length(bools), group_size))
    return ys
end
```

---

## Metal.jl — [`src/indexing.jl`](https://github.com/JuliaGPU/Metal.jl/blob/master/src/indexing.jl)

**Notable difference:** uses `Metal.SharedStorage` (unified CPU+GPU memory) for the indices array.  
This means `indices[end]` can be read directly — **no `@allowscalar` needed**.

```julia
function Base.findall(bools::WrappedMtlArray{Bool})
    boolslen = prod(size(bools))
    # SharedStorage = unified memory, CPU-readable without PCIe transfer
    indices  = MtlVector{Int64, Metal.SharedStorage}(undef, boolslen)
    cumsum!(indices, reshape(bools, boolslen))
    n        = isempty(indices) ? 0 : indices[end]   # ← no @allowscalar

    ys = similar(bools, keytype(bools), n)
    if n > 0
        function kernel(ys::MtlDeviceArray, bools, indices)
            i = (threadgroup_position_in_grid().x - Int32(1)) *
                threads_per_threadgroup().x + thread_position_in_threadgroup().x
            @inbounds if i <= length(bools) && bools[i]
                ys[indices[i]] = CartesianIndices(bools)[i]
            end
            return
        end
        kernel  = @metal name="findall" launch=false kernel(ys, bools, indices)
        threads = Int(kernel.pipeline.maxTotalThreadsPerThreadgroup)
        kernel(ys, bools, indices; groups=cld(length(indices), threads), threads)
    end
    unsafe_free!(indices)
    return ys
end
```

**Why Metal can do this:** Apple Silicon has unified memory — CPU and GPU share the same physical DRAM, no PCIe bus. `SharedStorage` allocates into this shared region. The other three backends (CUDA, AMDGPU, oneAPI) run on discrete GPUs with separate memory, so they must use `@allowscalar` to do a single device-to-host scalar read.

The `GPUArrays.jl` fallback uses `@allowscalar` (matching CUDA/AMDGPU/oneAPI), since it must be correct on discrete-GPU architectures.

---

## Summary

| Backend | cumsum source | scatter launch | indices[end] read | `unsafe_free!` |
|---------|:---:|:---:|:---:|:---:|
| CUDA.jl | own impl | `@cuda` | `@allowscalar` | ✅ |
| AMDGPU.jl | own impl | `@roc` | `@allowscalar` | ✅ |
| oneAPI.jl | own impl | `@oneapi` | `@allowscalar` | ❌ |
| Metal.jl | own impl | `@metal` | direct (SharedStorage) | ✅ |
| **GPUArrays PR #3** | **PR #2 cumsum** | **KA `@kernel`** | **`@allowscalar`** | **✅** |
