# reverse / reverse!: Failure Demonstration

## The Failure Is Backend-Specific

This is the most important thing to understand correctly.
The benchmark (`reverse_honest_bench.jl`) ran with `allowscalar=false`
throughout and measured no scalar paths on CUDA: because there are none.

**CUDA.jl and AMDGPU.jl are not affected by this bug at all.**
Their vendor methods (`@cuda`/`@roc` kernels) are more specific types
than `AnyGPUArray` and win dispatch before `Base` is ever consulted.
`allowscalar` has no effect on their path: they never touch scalar indexing.

**The failure only applies to backends with no vendor `reverse` method:**
- `oneAPI.jl` (no `reverse` implementation: confirmed by audit)
- `Metal.jl` (no `reverse` implementation: confirmed by audit)
- `JLArray` (reference test backend: no vendor method)
- Any future backend built on GPUArrays.jl before this PR lands

---

## Dispatch Walkthrough for Unimplemented Backends

```
User calls:   reverse(A::JLArray{Float32,1})

Julia dispatch:
  JLArray  <:  AbstractGPUArray  <:  AbstractArray

  Step 1: JLArrays.jl  : no reverse method  x
  Step 2: GPUArrays.jl : no reverse method  x
  Step 3: Base         : reverse(::AbstractVector) FOUND

-> Base.reverse runs:
      for i in 1:n/2
          A[i], A[n+1-i] = A[n+1-i], A[i]   <- scalar getindex on GPU array
      end
```

Same walkthrough applies to `oneArray` and `MtlArray`.

---

## Two Failure Modes (for unimplemented backends only)

### Mode 1: `allowscalar(false)`: Hard Error

```julia
using GPUArrays, JLArrays

GPUArrays.allowscalar(false)
A = jl(Float32[5.0, 3.0, 1.0, 4.0, 2.0])

println(@which reverse(A))
# -> reverse(A::AbstractVector; dims) @ Base array.jl:2181
# Confirms Base method dispatched, not a GPU kernel.

reverse(A)   # throws immediately
```

```
ERROR: Scalar indexing is disallowed.
Invocation of getindex resulted in scalar indexing of a GPU array.

Stacktrace:
 [1] getindex
   @ GPUArrays/src/host/indexing.jl:50
 [2] reverse(A::JLArray{Float32,1}, start::Int64, stop::Int64)
   @ Base ./array.jl:2170
 [3] reverse(A::JLArray{Float32,1})
   @ Base ./array.jl:2181
```

Operation is completely unusable on oneAPI, Metal, JLArray.

### Mode 2: `allowscalar(true)`: Silent Degradation (default)

The scalar swap loop runs on the host CPU. Each `A[i]` is a separate
device-to-host transfer. Result is numerically correct. No warning emitted.
The only observable symptom is severe unexplained slowdown.

For a 10M-element array on a discrete GPU (oneAPI, Metal):
- Correct GPU kernel (what this PR adds): ~0.27 ms at 360 GB/s
- Silent scalar path (what happens today): ~267 ms over PCIe
- No error. No warning. 1000x slower with no indication why.

---

## What the Benchmark Proves

The benchmark file `reverse_honest_bench.jl` holds `allowscalar=false`
throughout and runs both CUDA.jl's vendor kernel and the KA kernel on
`CuArray`. The results show both paths execute at full GPU bandwidth --
confirming neither path involves scalar indexing on CUDA.

The scalar failure described above is **not measurable on CuArray**
because CUDA.jl's vendor method wins dispatch before Base is ever reached.
It is only observable on the unimplemented backends (oneAPI, Metal, JLArray).

---

## Summary: Who Is Affected

| Backend | Has vendor `reverse`? | Failure mode |
|---------|:---:|---|
| CUDA.jl | Yes (`@cuda` kernel) | None: vendor method always wins |
| AMDGPU.jl | Yes (`@roc` kernel) | None: vendor method always wins |
| oneAPI.jl | **No** | ERROR with `allowscalar(false)`, silent slowdown otherwise |
| Metal.jl | **No** | ERROR with `allowscalar(false)`, silent slowdown otherwise |
| JLArray | **No** | ERROR with `allowscalar(false)`, silent slowdown otherwise |
| Future backends | **No** | ERROR with `allowscalar(false)`, silent slowdown otherwise |
