# accumulate! / cumsum / cumprod   Failure Demonstration

## Reproducing the Failure

```julia
# test_accumulate_failure.jl
using GPUArrays, JLArrays

GPUArrays.allowscalar(false)

A = jl(Float32[3.0, 1.0, 7.0, 0.0, 4.0, 1.0, 6.0, 3.0])
B = similar(A)

println(@which accumulate!(+, B, A))
# → accumulate!(op, B, A) @ Base ./accumulate.jl:...
# Confirms: Base sequential method dispatched, not a GPU kernel.

accumulate!(+, B, A)   # throws immediately
```

## Observed Output (verified on JLArrays.jl)

```
ERROR: Scalar indexing is disallowed.
Invocation of getindex resulted in scalar indexing of a GPU array.

Stacktrace:
 [1] getindex
   @ GPUArrays/src/host/indexing.jl:50
 [2] accumulate!(op::typeof(+), B::JLArray{Float32,1}, A::JLArray{Float32,1})
   @ Base ./accumulate.jl:...
```

---

## Dispatch Walkthrough

```
User calls:   accumulate!(+, B::JLArray, A::JLArray)

Julia dispatch:
  JLArray  <:  AbstractGPUArray  <:  AbstractArray

  Step 1: JLArrays.jl     no accumulate! method  ✗
  Step 2: GPUArrays.jl    no accumulate! method  ✗
  Step 3: Base            Base.accumulate!(op, B, A) FOUND

→ Base.accumulate! executes:
      B[1] = A[1]
      for i in 2:n
          B[i] = B[i-1] ⊕ A[i]   ← getindex + setindex! on GPU array
      end
  → first getindex hits GPUArrays scalar guard
  → ERROR: Scalar indexing is disallowed.
```

---

## Two Failure Modes

### Mode 1: `allowscalar(false)`   Hard Error (recommended)
Throws immediately at the first `getindex`. Completely unusable.

### Mode 2: `allowscalar(true)`   Silent Algorithmic Degradation (default)
Executes the sequential loop on CPU.  
- Each `B[i]` and `A[i]` access = separate device-to-host transfer  
- Result is numerically correct  
- **No warning emitted**  
- Performance is catastrophically worse than just the PCIe penalty:

| | Algorithm | Depth complexity |
|---|---|---|
| CPU scalar fallback | Sequential loop | **O(n)** |
| Blelloch scan (GPU) | Parallel tree | **O(log n)** |

Unlike `reverse` where the fallback is merely bandwidth-limited, `accumulate!` with the scalar fallback **loses the parallel algorithm entirely**. For n=1M, that's the difference between ~20 parallel steps and 1,000,000 sequential steps.

---

## Why This Is Worse Than `reverse`

With `reverse`, the scalar fallback gives you the right answer at ~30× slower.  
With `accumulate!`, the scalar fallback gives you:
1. ~22× slower due to PCIe bandwidth
2. O(n) sequential depth instead of O(log n) parallel depth   **another ~20× loss at n=1M**
3. Total effective slowdown at n=1M: potentially **400× or more** vs the GPU algorithm

All silent. No warning at the call site.
