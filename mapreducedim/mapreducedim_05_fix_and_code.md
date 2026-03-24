# `mapreducedim!` Implementation Details & Code

## Architectural Strategy
This is **PR #1**. Rather than building a custom `KernelAbstractions` reduction kernel directly inside `GPUArrays.jl` (which duplicates effort), this proposal upstreams the generalized multidimensional reduction to `AcceleratedKernels.jl`. `GPUArrays.jl` is then established as a thin delegation layer, proving the architecture for the remainder of the project.

---

## Design Decisions

### 1. Delegate the Tiled Shared-Memory Tree to AK.jl
While a native two-kernel strategy (utilizing `@localmem`, `@synchronize`, and a threadgroup tree reduction) is mathematically sound, writing it directly in `GPUArrays.jl` duplicates existing efforts. To minimize redundancy, this implementation utilizes `AcceleratedKernels.jl`, which already implements this exact pattern for parallel scans and reductions. 

### 2. Valentin Churavy's Prototype & The Insertion Fix
Building directly on Valentin Churavy's prototypes and AK.jl PR #77, this implementation integrates a robust fix for handling the neutral element. Specifically, for thread blocks that are not perfectly full, the neutral element (`init`) is safely **inserted** into the remaining shared memory slots. This pads the buffer, allowing the tree reduction to execute safely without requiring expensive, warp-divergent boundary checks inside the inner loop.

### 3. Generalizing AK.mapreduce_nd for Dims Tuples
The current prototype in `AcceleratedKernels.jl` is limited to single-integer dimensions (`dims::Int`). To support full `GPUArrays.jl` functionality (e.g., `sum(A, dims=(1,3))`), the upstream kernel must handle arbitrary dimension tuples. The coordinate mapping inside the AK.jl kernel is updated to use `CartesianIndices` and `ntuple`, clamping the reduced dimensions to 1 while preserving the unreduced dimensions.

### 4. The Recursive Loop Fix (Preventing Circular Dependencies)
When a reduction dimension exceeds the block size, the GPU must perform a partial reduction into a temporary buffer, which is then reduced again. The original AK.jl prototype incorrectly called `GPUArrays.mapreducedim!` for this second pass, creating an infinite circular dependency back to the caller. The fix ensures multi-pass execution stays entirely within the `AcceleratedKernels.jl` namespace.

### 5. Preserving the GPUArrays Orchestration Layer
`GPUArrays.jl` already has a robust ~200-line orchestration layer in `src/host/mapreduce.jl` that handles `neutral_element(op, T)` initialization, output array allocation, and type inference. **This orchestration layer remains entirely untouched.** The fix is surgical, replacing only the final `error("Not implemented")` stub.

---

## Phase 1: AcceleratedKernels.jl Implementation

**File:** `AcceleratedKernels.jl/src/reduce/mapreduce_nd.jl` *(Updates to existing file)*

~~~julia
@kernel function mapreduce_nd_kernel!(f, op, neutral, R, @Const(A), ::Val{dims}) where {dims}
    idx = @index(Global, Linear)
    tid = @index(Local, Linear)
    
    # Allocate shared memory
    shared = @localmem eltype(R) (@groupsize()[1],)
    
    @inbounds begin
        # --- Valentin's Insertion Fix ---
        # Insert the neutral element into shared memory for padding
        # This protects the reduction tree from garbage data in out-of-bounds threads
        shared[tid] = neutral
        
        if idx <= length(A)
            src_idx = CartesianIndices(A)[idx]
            
            # Generalised coordinate mapping for arbitrary Dims tuples
            dst_idx = CartesianIndex(
                ntuple(i -> i in dims ? 1 : src_idx.I[i], ndims(A))
            )
            
            # Load and apply function
            shared[tid] = f(A[src_idx])
        end
        
        @synchronize()
        
        # ... [Existing shared memory tree reduction logic] ...
    end
end

function mapreduce_nd!(f, op, R::AbstractArray, A::AbstractArray, backend::Backend; dims=:)
    # ... [Existing setup and block size calculations] ...
    
    if items > 1
        # CRITICAL FIX: Do not call GPUArrays.mapreducedim! here. 
        # Stay within AcceleratedKernels for the partial reduction pass.
        mapreduce_nd!(identity, op, R, partial, backend; dims=dims)
    else
        # Final pass writes directly to R
        # ...
    end
    
    return R
end
~~~

---

## Phase 2: GPUArrays.jl Delegation

**File:** `GPUArrays.jl/src/host/mapreduce.jl`

Find the existing fallback stub:
~~~julia
Base.mapreducedim!(f, op, R::AnyGPUArray, A::AnyGPUArray) = error("Not implemented")
~~~

Replace it entirely with the delegation wrapper:
~~~julia
import AcceleratedKernels as AK

function Base.mapreducedim!(f, op, R::AnyGPUArray, A::AnyGPUArray)
    # Extract the dimensions being reduced by comparing the sizes of R and A
    dims = Tuple(findall(size(R) .!= size(A)))
    
    # Handle the case where all dimensions are reduced (returns a 1-element array)
    if isempty(dims) && length(R) == 1
        dims = ntuple(identity, ndims(A))
    end

    # Delegate to AK.jl
    AK.mapreduce_nd!(f, op, R, A, get_backend(A); dims=dims)
    KernelAbstractions.synchronize(get_backend(A))
    
    return R
end
~~~

---

## Dispatch Table After PR #1

Because `sum`, `prod`, `maximum`, `minimum`, `any`, and `all` (and their `dims=` variants) all route through `mapreducedim!` internally, this single 15-line wrapper fixes the entire high-level reduction API for all missing backends simultaneously.

~~~text
BEFORE:
sum(A::AnyCuArray)  →  CUDA.jl warp shuffle       ✓
sum(A::AnyROCArray) →  AMDGPU.jl wavefront ops    ✓
sum(A::oneArray)    →  oneAPI.jl subgroup ops     ✓
sum(A::MtlArray)    →  Metal.jl shared memory     ✓
sum(A::JLArray)     →  GPUArrays stub → ERROR     ✗ (Crash)
sum(A::<future>)    →  GPUArrays stub → ERROR     ✗ (Crash)

AFTER:
sum(A::AnyCuArray)  →  CUDA.jl warp shuffle       ✓ (unchanged)
sum(A::AnyROCArray) →  AMDGPU.jl wavefront ops    ✓ (unchanged)
sum(A::oneArray)    →  oneAPI.jl subgroup ops     ✓ (unchanged)
sum(A::MtlArray)    →  Metal.jl shared memory     ✓ (unchanged)
sum(A::JLArray)     →  GPUArrays → AK.jl kernel   ✓ (fixed)
sum(A::<future>)    →  GPUArrays → AK.jl kernel   ✓ (automatic)
~~~
