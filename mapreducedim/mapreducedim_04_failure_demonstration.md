# mapreducedim! — Failure Demonstration

## The Unique Nature of This Failure

Unlike `reverse`, `findall`, and `accumulate!` — where the failure requires dispatch to fall all the way to `Base` — `mapreducedim!` fails **inside GPUArrays.jl itself**. The stub throws unconditionally, with no scalar indexing escape hatch, no silent degradation path, no workaround.

**Every call to `sum`, `prod`, `maximum`, `minimum`, `any`, or `all` on any non-vendor backend hard-errors. Always.**

---

## Reproducing the Failure

```julia
# test_mapreducedim_failure.jl
using GPUArrays, JLArrays

A = jl(Float32[1.0, 2.0, 3.0, 4.0, 5.0])

println("Attempting sum(A) on JLArray:")
sum(A)
```

Note: `allowscalar` setting is **irrelevant** — the error fires before any indexing occurs.

---

## Observed Output (verified on JLArrays.jl)

```
ERROR: mapreducedim! not implemented for JLArray{Float32, 1, ...}

Stacktrace:
 [1] error(s::String)
   @ Base ./error.jl:44
 [2] mapreducedim!(f::typeof(identity), op::typeof(+),
                   R::JLArray{Float32,0}, A::JLArray{Float32,1}; init::Float32)
   @ GPUArrays src/host/mapreduce.jl:...
 [3] _mapreduce(f::typeof(identity), op::typeof(+), A::JLArray{Float32,1};
                dims::Colon, init::Float32)
   @ GPUArrays src/host/mapreduce.jl:...
 [4] mapreduce(f::typeof(identity), op::typeof(+), A::JLArray{Float32,1})
   @ GPUArrays src/host/mapreduce.jl:...
 [5] sum(A::JLArray{Float32,1})
   @ Base ./reducedim.jl:...
```

---

## Dispatch Walkthrough

```
sum(A::JLArray{Float32,1})
  ↓
Base.mapreduce(identity, +, A)           ← GPUArrays override
  ↓
GPUArrays._mapreduce(identity, +, A)
  ↓  allocates R = jl(Float32[0f0])  (shape: 0-dim scalar)
  ↓  looks up neutral_element(+, Float32) = 0f0
  ↓
GPUArrays.mapreducedim!(identity, +, R, A; init=0f0)

  Julia dispatch:
    JLArray  <:  AbstractGPUArray  <:  AbstractArray
    Step 1: JLArrays.jl   — no mapreducedim! method  ✗
    Step 2: GPUArrays.jl  — AnyGPUArray stub FOUND   ← hits here

  → error("mapreducedim! not implemented")
```

The orchestration layer above (`_mapreduce`, `Base.sum`) is **completely correct**. The stub is the only broken piece.

---

## What This Breaks

Every line below hard-errors on JLArray (and any future backend):

```julia
A = jl(rand(Float32, 100, 100))

sum(A)                          # ERROR
prod(A)                         # ERROR
maximum(A)                      # ERROR
minimum(A)                      # ERROR
any(A .> 0.5f0)                 # ERROR
all(isfinite.(A))               # ERROR
sum(A; dims=1)                  # ERROR
maximum(A; dims=2)              # ERROR
sum(abs2, A)                    # ERROR
mapreduce(x->x^2, +, A)        # ERROR
norm(A)                         # ERROR (uses sum internally)
mean(A)                         # ERROR (uses sum internally)
```

For a new backend author implementing the GPUArrays.jl interface, the very first test they'd write — `@test sum(jl([1f0, 2f0, 3f0])) == 6f0` — throws immediately.

---

## Why No Silent Fallback Exists

`reverse` and `findall` silently fall to `Base`, which at least produces correct (if slow) output with `allowscalar(true)`. `mapreducedim!` has no such path: the stub throws before any data access. There is no workaround except `Array(A) |> sum` — an explicit host-side detour that the user must write manually.

This is the highest-priority fix in the project.
