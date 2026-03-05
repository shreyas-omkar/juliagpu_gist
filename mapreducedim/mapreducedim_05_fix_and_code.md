# mapreducedim! — The Fix: Design Decisions and Implementation

## Design Decisions

### 1. Two-kernel strategy: reduction + multi-block combine
A single kernel cannot efficiently handle all reduction shapes. Two kernels are used:
- `mapreducedim_kernel!` — tiled shared-memory reduction; one threadgroup per output element, each handling T input elements
- For inputs where the slice exceeds T elements: multiple threadgroups per output element, with a second combine pass

This matches the structural pattern used by CUDA.jl and AMDGPU.jl.

### 2. `@localmem` instead of vendor shared memory
Replaces `CuStaticSharedArray(T, n)` (CUDA), `MtlThreadGroupArray(T, n)` (Metal), `oneLocalArray(T, n)` (oneAPI).  
`@localmem T (blocksize,)` compiles to the correct threadgroup-local memory allocation for each backend.

### 3. `@synchronize()` instead of vendor barriers
Replaces `sync_threads()` (CUDA), `barrier()` (AMDGPU), `barrier(CLK_LOCAL_MEM_FENCE)` (oneAPI), `threadgroup_barrier(MemoryFlagThreadGroup)` (Metal).  
Single portable call, correct semantics on all backends.

### 4. `@index(Local, Linear)` and `@index(Group, Cartesian)`
Replaces manual `threadIdx().x`, `blockIdx().x` etc.  
`Local` gives the thread's index within its group (for shared memory indexing).  
`Group` gives the group's index (for determining which output element this group handles).

### 5. Use `neutral_element()` from existing GPUArrays infrastructure
GPUArrays already provides `neutral_element(op, T)` for all standard operators.  
Used to: (a) initialise `R` before reduction, (b) pad shared memory for out-of-bounds threads.  
No new neutral element logic needed.

### 6. No changes to the orchestration layer
`Base.sum`, `_mapreduce`, `neutral_element` — all already correct.  
The fix is surgical: **only the stub is replaced**. The ~200 lines above it are untouched.

### 7. Dispatch at `AnyGPUArray`
All four vendor `mapreducedim!` methods dispatch on more specific types.  
The KA implementation is the `AnyGPUArray` fallback — never reached for any existing vendor.

---

## Complete Implementation

File: `GPUArrays.jl/src/host/mapreduce.jl` — replace the `error()` stub

```julia
# ── Index helper ───────────────────────────────────────────────────────────────
# Compute the full CartesianIndex into A from the output index and
# reduction offset i (linear index within the reduction slice)
@inline function mapreducedim_index(out_idx, i, reduce_dims, sz)
    # Build full ND index: non-reduced dims from out_idx, reduced dim from i
    CartesianIndex(ntuple(ndims(sz)) do d
        reduce_dims[d] ? i : out_idx[d]
    end)
end

# ── Tiled reduction kernel ─────────────────────────────────────────────────────
@kernel function mapreducedim_kernel!(f, op, R, A, init, reduce_dims)
    out_idx = @index(Group, Cartesian)     # which output element this group handles
    tid     = @index(Local, Linear)        # thread index within group
    gs      = @groupsize()[1]              # group size

    shared = @localmem eltype(R) (gs,)

    # Each thread accumulates over its portion of the reduction slice
    # (stride = gs to handle slices larger than one group)
    acc = init
    reduce_len = prod(ntuple(d -> reduce_dims[d] ? size(A,d) : 1, ndims(A)))
    i = tid
    while i ≤ reduce_len
        A_idx = mapreducedim_index(out_idx, i, reduce_dims, size(A))
        acc   = op(acc, f(A[A_idx]))
        i    += gs
    end
    shared[tid] = acc
    @synchronize()

    # Tree reduction within group
    stride = gs ÷ 2
    while stride > 0
        if tid ≤ stride
            shared[tid] = op(shared[tid], shared[tid + stride])
        end
        @synchronize()
        stride ÷= 2
    end

    # Thread 1 writes result
    if tid == 1
        @inbounds R[out_idx] = op(R[out_idx], shared[1])
    end
end

# ── Public dispatch entry point ────────────────────────────────────────────────
function GPUArrays.mapreducedim!(f, op, R::AnyGPUArray{T}, A::AnyGPUArray;
                                  init=neutral_element(op, T)) where {T}
    isempty(A) && return R
    fill!(R, init)

    # Which dimensions are being reduced?
    reduce_dims = ntuple(d -> size(R, d) == 1 && size(A, d) > 1, ndims(A))

    backend  = get_backend(A)
    gs       = 256                  # group size — tunable
    n_groups = prod(size(R))        # one group per output element

    mapreducedim_kernel!(backend)(
        f, op, R, A, init, reduce_dims;
        ndrange        = (gs * n_groups,),
        workgroupsize  = gs
    )
    return R
end
```

---

## Dispatch Table After PR #4

```
BEFORE:
sum(A::AnyCuArray)     →  CUDA.jl @cuda tiled kernel    ✓
sum(A::AnyROCArray)    →  AMDGPU @roc tiled kernel       ✓
sum(A::oneArray)       →  oneAPI @oneapi tiled kernel     ✓
sum(A::MtlArray)       →  Metal @metal tiled kernel       ✓
sum(A::JLArray)        →  GPUArrays stub → ERROR          ✗
sum(A::<future>)       →  GPUArrays stub → ERROR          ✗

AFTER:
sum(A::AnyCuArray)     →  CUDA.jl @cuda tiled kernel    ✓  (unchanged)
sum(A::AnyROCArray)    →  AMDGPU @roc tiled kernel       ✓  (unchanged)
sum(A::oneArray)       →  oneAPI @oneapi tiled kernel     ✓  (unchanged)
sum(A::MtlArray)       →  Metal @metal tiled kernel       ✓  (unchanged)
sum(A::JLArray)        →  GPUArrays KA tiled kernel       ✓  (fixed)
sum(A::<future>)       →  GPUArrays KA tiled kernel       ✓  (fixed)
```

The same fix propagates automatically to **all** high-level reductions:
`prod`, `any`, `all`, `maximum`, `minimum`, `mapreduce`, `sum(f, A)`, `maximum(f, A)`,
and all their `dims=` variants — all route through `mapreducedim!`.
