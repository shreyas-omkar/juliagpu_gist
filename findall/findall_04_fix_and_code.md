# findall   The Fix: Design Decisions and Implementation

## Design Decisions

### 1. Use PR #2's `cumsum` for Step 1
Step 1 is exactly `cumsum(bools)`. PR #2 already implements this at `AnyGPUArray`.  
No need to call `AcceleratedKernels.jl` directly   the delegation is already one level up.

### 2. KA scatter kernel for Step 2
The scatter is simpler than `reverse_kernel!`: one boolean read, one conditional write.  
Embarrassingly parallel, no data dependencies, no synchronisation.  
`@index(Global, Linear)` compiles to the correct vendor intrinsic per backend.

### 3. `@allowscalar` for the single scalar read
Reading `indices[end]` for output size requires one unavoidable scalar access.  
All non-Metal vendor backends use `@allowscalar` for exactly this one read.  
The fallback matches this pattern   one explicit, intentional scalar access.

### 4. `unsafe_free!` the temporary
The `indices` prefix sum array is temporary. Call `unsafe_free!` immediately to reclaim GPU memory without waiting for GC. Matches CUDA, AMDGPU, Metal pattern.

### 5. Add `Base.to_index` and `Base.to_indices` overrides
Required for `A[mask]` to route through GPU `findall` rather than `Base.to_index` → `Base.findall`.  
Without these overrides, logical indexing remains broken even after `findall` is fixed.

### 6. Dispatch at `AnyGPUArray`
All four vendor methods dispatch on more specific types   they always win.  
The fallback only catches: JLArray, future backends, user-defined GPU array types.

---

## Complete Implementation

File: `GPUArrays.jl/src/host/indexing.jl`

```julia
# ── Logical indexing routing ───────────────────────────────────────────────────
Base.to_index(::AnyGPUArray, I::AbstractArray{Bool}) = findall(I)

if VERSION >= v"1.11.0-DEV.1157"
    Base.to_indices(A::AnyGPUArray, I::Tuple{AbstractArray{Bool}}) =
        (Base.to_index(A, I[1]),)
else
    Base.to_indices(A::AnyGPUArray, inds,
                    I::Tuple{Union{Array{Bool,N}, BitArray{N}}}) where {N} =
        (Base.to_index(A, I[1]),)
end

# ── Scatter kernel ─────────────────────────────────────────────────────────────
@kernel function findall_kernel!(ys, bools, indices)
    i = @index(Global, Linear)
    @inbounds if i <= length(bools) && bools[i]
        ys[indices[i]] = CartesianIndices(bools)[i]   # unique write slot
    end
end

# ── findall fallback ───────────────────────────────────────────────────────────
function Base.findall(bools::AnyGPUArray{Bool})
    I       = keytype(bools)
    flat    = reshape(bools, prod(size(bools)))
    indices = cumsum(flat)                         # uses PR #2 AnyGPUArray fallback
    n       = isempty(indices) ? 0 : @allowscalar indices[end]   # single scalar read
    ys      = similar(bools, I, n)
    if n > 0
        findall_kernel!(get_backend(bools))(ys, bools, indices; ndrange=length(bools))
    end
    unsafe_free!(indices)
    return ys
end

# ── findall(f, A) ──────────────────────────────────────────────────────────────
function Base.findall(f::Function, A::AnyGPUArray)
    bools = map(f, A)
    ys    = findall(bools)
    unsafe_free!(bools)
    return ys
end
```

---

## Dispatch Table After PR #3

```
BEFORE:
findall(bools::JLArray{Bool})     →  Base scalar push! loop    ✗ ERROR / silent
findall(bools::<future>{Bool})    →  Base scalar push! loop    ✗ ERROR / silent
A[mask] on JLArray                →  Base.to_index → Base.findall  ✗

AFTER:
findall(bools::AnyCuArray{Bool})  →  CUDA.jl @cuda scatter     ✓  (unchanged)
findall(bools::AnyROCArray{Bool}) →  AMDGPU @roc scatter       ✓  (unchanged)
findall(bools::oneArray{Bool})    →  oneAPI @oneapi scatter     ✓  (unchanged)
findall(bools::MtlArray{Bool})    →  Metal @metal scatter       ✓  (unchanged)
findall(bools::JLArray{Bool})     →  GPUArrays KA scatter       ✓  (fixed)
findall(bools::<future>{Bool})    →  GPUArrays KA scatter       ✓  (fixed)
A[mask] on JLArray                →  GPUArrays.to_index → GPUArrays.findall  ✓  (fixed)
```

Zero changes to any vendor package. Julia's dispatch guarantees more-specific types always win.
