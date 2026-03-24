## Architectural Strategy
This PR (PR #4) implements stream compaction (prefix sum + scatter) to solve boolean indexing. Following the project's architectural pivot, the heavy scatter kernel is upstreamed to `AcceleratedKernels.jl`, while `GPUArrays.jl` handles the orchestration, the `Base.to_index` overrides, and delegates the prefix sum to `AK.cumsum`.

**Two-phase split:**
1. **AcceleratedKernels.jl** Adds the missing `stream_compact_scatter!` kernel.
2. **GPUArrays.jl** Adds the `findall` wrapper and logical indexing fallbacks.

---

## Design Decisions

### 1. Upstream the Scatter Kernel to AcceleratedKernels.jl
The scatter phase is embarrassingly parallel: one boolean read, one conditional write. By upstreaming this to AK.jl, we use `@index(Global, Linear)` which compiles to the correct vendor intrinsic per backend, keeping `GPUArrays.jl` purely as a delegation layer.

### 2. Delegate Step 1 to `AK.cumsum`
Step 1 of stream compaction is a prefix sum. Rather than relying on `Base.cumsum` (which could trigger circular dispatch issues), the wrapper directly calls `AcceleratedKernels.cumsum(bools)` (implemented in PR #3) to get the exact output offsets.

### 3. `@allowscalar` for the Output Size Read
Reading `indices[end]` to allocate the final output array requires one unavoidable scalar access. All non-Metal vendor backends do exactly this. The fallback safely wraps this single read in `@allowscalar`.

### 4. No `unsafe_free!`
Vendor-specific memory management like `unsafe_free!` cannot be used in a generic `AnyGPUArray` fallback, as it is not guaranteed to exist on future or minimal backends. We rely on standard GC for the temporary `indices` array.

### 5. `Base.to_index` and `Base.to_indices` Overrides
These are critical. Without these overrides, logical indexing like `A[mask]` routes through `Base.to_index` and completely ignores our new `findall` method, falling back to a CPU scalar loop.

---

## Phase 1: AcceleratedKernels.jl Implementation

**File:** `AcceleratedKernels.jl/src/indexing.jl` *(New File)*


```julia

@kernel function stream_compact_scatter!(out, @Const(mask), @Const(offsets))
    i = @index(Global, Linear)
    @inbounds if i <= length(mask) && mask[i]
        out[offsets[i]] = CartesianIndices(mask)[i]   # Write to unique slot
    end
end
```
(Add `include("indexing.jl")` to `src/AcceleratedKernels.jl`)

---

## Phase 2: GPUArrays.jl Delegation
File: `GPUArrays.jl/src/host/indexing.jl`

```Julia
import AcceleratedKernels as AK

function Base.findall(mask::AnyGPUArray{Bool})
    # Step 1: Prefix sum to get write offsets
    offsets = AK.cumsum(mask)
    
    # Step 2: Read total true count
    n_out = isempty(offsets) ? 0 : @allowscalar offsets[end]
    
    # Step 3: Allocate exact-sized output
    out = similar(mask, CartesianIndex{ndims(mask)}, n_out)
    
    # Step 4: Scatter
    if n_out > 0
        backend = get_backend(mask)
        kernel! = AK.stream_compact_scatter!(backend)
        kernel!(out, mask, offsets; ndrange=length(mask))
        KernelAbstractions.synchronize(backend)
    end
    
    return out
end

function Base.findall(f::Function, A::AnyGPUArray)
    mask = map(f, A)
    return findall(mask)
end

Base.to_index(::AnyGPUArray, I::AbstractArray{Bool}) = findall(I)

if VERSION >= v"1.11.0-DEV.1157"
    Base.to_indices(A::AnyGPUArray, I::Tuple{AbstractArray{Bool}}) =
        (Base.to_index(A, I[1]),)
else
    Base.to_indices(A::AnyGPUArray, inds,
                    I::Tuple{Union{Array{Bool,N}, BitArray{N}}}) where {N} =
        (Base.to_index(A, I[1]),)
end
```

## Dispatch Table After PR #4

```Plaintext
BEFORE:
findall(mask::JLArray{Bool})      →  Base scalar push! loop        ✗ (ERROR / silent)
findall(mask::<future>{Bool})     →  Base scalar push! loop        ✗ (ERROR / silent)
A[mask] on JLArray                →  Base.to_index → scalar loop   ✗

AFTER:
findall(mask::AnyCuArray{Bool})   →  CUDA.jl @cuda scatter         ✓ (unchanged)
findall(mask::AnyROCArray{Bool})  →  AMDGPU @roc scatter           ✓ (unchanged)
findall(mask::oneArray{Bool})     →  oneAPI @oneapi scatter        ✓ (unchanged)
findall(mask::MtlArray{Bool})     →  Metal @metal scatter          ✓ (unchanged)
findall(mask::JLArray{Bool})      →  GPUArrays → AK.jl scatter     ✓ (fixed)
findall(mask::<future>{Bool})     →  GPUArrays → AK.jl scatter     ✓ (fixed)
A[mask] on JLArray                →  GPUArrays.to_index → GPU      ✓ (fixed)
```
