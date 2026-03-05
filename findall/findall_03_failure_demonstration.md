# findall — Failure Demonstration

## Reproducing the Failure

```julia
# test_findall_failure.jl
using GPUArrays, JLArrays

GPUArrays.allowscalar(false)

bools = jl(Bool[true, false, false, true, true, false, true])

println(@which findall(bools))
# → findall(A::AbstractArray) @ Base array.jl:...
# Confirms: Base method dispatched, not a GPU kernel.

findall(bools)   # throws immediately
```

## Observed Output (verified on JLArrays.jl)

```
findall(A::AbstractArray) @ Base array.jl:...

ERROR: Scalar indexing is disallowed.
Invocation of getindex resulted in scalar indexing of a GPU array.

Stacktrace:
 [1] getindex
   @ GPUArrays/src/host/indexing.jl:50
 [2] findall(A::JLArray{Bool, 1})
   @ Base ./array.jl:...
```

---

## Dispatch Walkthrough

```
User calls:   findall(bools::JLArray{Bool,1})

Julia dispatch:
  JLArray  <:  AbstractGPUArray  <:  AbstractArray

  Step 1: JLArrays.jl   — no findall method  ✗
  Step 2: GPUArrays.jl  — no findall method  ✗
  Step 3: Base          — findall(::AbstractArray) FOUND

→ Base.findall executes:
      ys = eltype(keys(bools))[]        ← empty CPU Vector
      for i in eachindex(bools)
          bools[i] && push!(ys, i)      ← scalar getindex on GPU array
      end
  → first bools[i] hits GPUArrays scalar guard
  → ERROR: Scalar indexing is disallowed.
```

---

## Two Failure Modes

### Mode 1: `allowscalar(false)` — Hard Error
Throws at the first `getindex`. Operation completely unusable.  
Identical behaviour on oneAPI.jl and Metal.jl (neither has `findall`... wait, they do — see backend doc).  
Applies to JLArray and any future custom backend.

### Mode 2: `allowscalar(true)` — Silent Catastrophe (default)
The `push!` loop runs on CPU. Each `bools[i]` is a separate device-to-host scalar read.

For `n = 10^7` elements:
- **GPU two-pass algorithm:** ~18 ms (cumsum + scatter at 360 GB/s)
- **CPU scalar loop:** ~267,000 ms = ~4.5 minutes (10^7 individual PCIe reads)
- **No warning emitted. No error thrown. Numerically correct output.**

This is a silent performance bug that only shows up in profiling.

---

## The Logical Indexing Consequence

`findall` is not just called directly — it underlies `A[mask]`. Every vendor backend defines:

```julia
Base.to_index(::CuArray,  I::AbstractArray{Bool}) = findall(I)
Base.to_index(::ROCArray, I::AbstractArray{Bool}) = findall(I)
Base.to_index(::oneArray, I::AbstractArray{Bool}) = findall(I)
Base.to_index(::MtlArray, I::AbstractArray{Bool}) = findall(I)
```

Without `findall` at the `AnyGPUArray` level, for any unpatched backend:

```julia
A = jl(rand(Float32, 1000))
mask = jl(rand(Bool, 1000))
A[mask]    # ← BROKEN: falls to Base.to_index → Base.findall → scalar loop
```

This means the PR fixes not just `findall` but **all boolean/mask indexing** for future backends. It directly closes [GPUArrays.jl issue #178](https://github.com/JuliaGPU/GPUArrays.jl/issues/178) ("Logical indexing, filter, findall"), open since **2018**.
