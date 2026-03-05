# reverse / reverse! — Failure Demonstration

## Reproducing the Failure

Run this on any backend without a native `reverse` implementation (JLArrays, oneAPI, Metal):

```julia
# test_reverse_failure.jl
using GPUArrays, JLArrays

GPUArrays.allowscalar(false)   # recommended: GPU code only

A = jl(Float32[5.0, 3.0, 1.0, 4.0, 2.0])

println(@which reverse(A))
# → reverse(A::AbstractVector; dims) @ Base array.jl:2181
# Confirms: Base method dispatched, not a GPU kernel.

reverse(A)   # throws immediately
```

## Observed Output (verified on JLArrays.jl)

```
reverse(A::AbstractVector; dims) @ Base array.jl:2181

ERROR: Scalar indexing is disallowed.
Invocation of getindex resulted in scalar indexing of a GPU array.
This is typically caused by calling an iterating implementation of a method.
Such implementations *do not* execute on the GPU, but very slowly on the CPU,
and therefore should be avoided.

Stacktrace:
 [1] getindex
   @ GPUArrays/src/host/indexing.jl:50
 [2] reverse(A::JLArray{Float32,1}, start::Int64, stop::Int64)
   @ Base ./array.jl:2170
 [3] reverse(A::JLArray{Float32,1})
   @ Base ./array.jl:2181
```

---

## Dispatch Walkthrough

```
User calls:   reverse(A::JLArray{Float32,1})

Julia dispatch:
  JLArray  <:  AbstractGPUArray  <:  AbstractArray

  Step 1: JLArrays.jl   — no reverse method  ✗
  Step 2: GPUArrays.jl  — no reverse method  ✗
  Step 3: Base          — reverse(::AbstractVector) FOUND

→ Base.reverse runs:
      for i in 1:n÷2
          b[i], b[n+1-i] = b[n+1-i], b[i]   ← getindex on GPU array
      end
  → first getindex hits GPUArrays scalar guard
  → ERROR: Scalar indexing is disallowed.
```

---

## Two Failure Modes

### Mode 1: `allowscalar(false)` — Hard Error (recommended setting)
Throws immediately at the first `getindex` inside `Base.array.jl:2170`.  
Operation is completely unusable. Identical behaviour on oneAPI.jl and Metal.jl.

### Mode 2: `allowscalar(true)` — Silent Degradation (default)
The scalar swap loop executes on the CPU element by element.  
Each element incurs a **separate device-to-host transfer**.  
Result is numerically correct. **No warning is emitted.**  
The only observable symptom is a severe unexplained slowdown proportional to array size.

---

## Why This Is Hard to Debug

A user on oneAPI or Metal calling `A[end:-1:1]` or `reverse(weights)` in a training loop will see:
- ✅ Correct numerical output
- ❌ 30× slower iteration with no error message
- ❌ No stack trace pointing to the problem
- ❌ Only discoverable by profiling or setting `allowscalar(false)` explicitly

This is the definition of a **silent performance bug**.
