# mapreducedim!   Existing GPUArrays.jl Architecture

## What's Already There

`mapreducedim!` is unique among the four PRs in this project: GPUArrays.jl already has a **complete orchestration layer** in [`src/host/mapreduce.jl`](https://github.com/JuliaGPU/GPUArrays.jl/blob/main/src/host/mapreduce.jl). The missing piece is not the plumbing   it is the kernel itself.

---

## The Complete Orchestration Layer (Already in Production)

### `neutral_element(op, T)`

Returns the identity element for standard reduction operators over type `T`.

```julia
neutral_element(::typeof(+), T) = zero(T)
neutral_element(::typeof(*), T) = one(T)
neutral_element(::typeof(max), T) = typemin(T)
neutral_element(::typeof(min), T) = typemax(T)
neutral_element(::typeof(|), ::Type{Bool}) = false
neutral_element(::typeof(&), ::Type{Bool}) = true
# ... and more
```

Handles integer, float, boolean types. Used to initialise the output array before reduction and to pad shared memory for out-of-bounds threads.

### `_mapreduce(f, op, A; dims, init)`

The internal dispatch function. Responsibilities:
1. Infer output element type from `f`, `op`, and `eltype(A)`
2. Compute output shape (reduced dimensions become size-1)
3. Allocate output array `R` with the correct shape
4. Look up `neutral_element(op, eltype(R))` if `init` not provided
5. Call `mapreducedim!(f, op, R, A; init)`

This function is complete and correct. It never needs to be modified.

### `Base.mapreduce` dispatches

All high-level reductions on `AnyGPUArray` are routed through `_mapreduce`:

```julia
Base.mapreduce(f, op, A::AnyGPUArray; dims=:, init=...) = _mapreduce(f, op, A; dims, init)
Base.sum(A::AnyGPUArray; dims=:, init=...) = ...        # → _mapreduce(identity, +, A)
Base.prod(A::AnyGPUArray; dims=:, init=...) = ...       # → _mapreduce(identity, *, A)
Base.maximum(A::AnyGPUArray; dims=:, init=...) = ...    # → _mapreduce(identity, max, A)
Base.minimum(A::AnyGPUArray; dims=:, init=...) = ...    # → _mapreduce(identity, min, A)
Base.any(A::AnyGPUArray; dims=:) = ...                  # → _mapreduce(identity, |, A)
Base.all(A::AnyGPUArray; dims=:) = ...                  # → _mapreduce(identity, &, A)
```

These dispatches are complete, correct, and already in production.

---

## The Stub (What PR #4 Replaces)

At the bottom of the orchestration chain sits one function:

```julia
# GPUArrays.jl/src/host/mapreduce.jl  (current state)
function GPUArrays.mapreducedim!(f, op, R::AnyGPUArray{T}, A::AnyGPUArray;
                                  init=neutral_element(op, T)) where {T}
    error("mapreducedim! not implemented for $(typeof(A))")
end
```

This is a deliberate placeholder. The comment above it in the original source reads:
> *"Implement this in your backend."*

All four vendor backends implement this function for their specific array types and never hit the stub. The stub only fires for: JLArray, future backends, user-defined GPU array types.

PR #4 replaces this stub with a KA tiled reduction kernel   making the placeholder real for all backends that lack a vendor implementation.

---

## Call Stack for `sum(A::JLArray)`

```
sum(A::JLArray)
  └─ Base.mapreduce(identity, +, A)
       └─ GPUArrays._mapreduce(identity, +, A; dims=:, init=0f0)
            allocates R = similar(A, (1,))
            └─ GPUArrays.mapreducedim!(identity, +, R, A; init=0f0)
                 ✗ BEFORE: error("not implemented")
                 ✓ AFTER:  KA tiled reduction kernel
```

The entire call stack above `mapreducedim!` is already correct. Only the bottom function needs to be filled in.
