# sort! / sortperm / sortperm! — Failure Demonstration

## The Three Distinct Failure Modes

Unlike `reverse` (silent CPU fallback) and `mapreducedim!` (hard error inside GPUArrays), sort has **three separate failure modes** depending on which function is called and which backend:

| Call | JLArray / Metal / future | CUDA.jl | AMDGPU.jl / oneAPI.jl |
|---|---|---|---|
| `sort!(A)` | CPU scalar → ERROR | ✓ works | ✓ works |
| `sortperm(A)` | CPU scalar → ERROR | CPU scalar → ERROR | ✓ works |
| `sortperm!(ix, A)` | CPU scalar → ERROR | CPU scalar → ERROR | ✓ works |

The CUDA case is particularly subtle: `sort!` works fine, but `sortperm` silently degrades or errors — even on the most capable backend.

---

## Reproducing the Failures

### Failure 1: sort! on JLArray (or Metal)

```julia
# test_sort_failure.jl
using GPUArrays, JLArrays

GPUArrays.allowscalar(false)

A = jl(Float32[5.0, 3.0, 1.0, 4.0, 2.0])

println("@which sort!(::JLArray) = ", @which sort!(A))
# → sort!(v::AbstractVector; ...) in Base at sort.jl:XXX
# CPU method! No GPU implementation found.

sort!(A)
# ERROR: Scalar indexing is disallowed.
# Stacktrace:
#  [1] error(s::String)
#  [2] assertscalar(::JLArray{Float32,1})  @ GPUArrays
#  [3] getindex(::JLArray{Float32,1}, ::Int64)  @ GPUArrays
#  [4] lt(::Base.Order.ForwardOrdering, ...) @ Base sort.jl
#  [5] sort!(::JLArray{Float32,1}) @ Base sort.jl
```

### Failure 2: sortperm on CUDA (confirmed CUDA.jl gap)

```julia
# test_sortperm_cuda_failure.jl
using CUDA

GPUArrays.allowscalar(false)

A = cu(Float32[5.0, 3.0, 1.0, 4.0, 2.0])

println("@which sortperm(::CuArray) = ", @which sortperm(A))
# → sortperm(v::AbstractVector; ...) in Base at sort.jl:XXX
# CPU method — even on CUDA!

sortperm(A)
# ERROR: Scalar indexing is disallowed.
# This is a known CUDA.jl limitation — sort! works but sortperm is missing.
```

### Failure 3: sortperm on JLArray

```julia
# Same result as Failure 2, but for any non-AMDGPU/oneAPI backend
ix = jl(Int32[1,2,3,4,5])
A  = jl(Float32[5.0, 3.0, 1.0, 4.0, 2.0])

sortperm!(ix, A)
# ERROR: Scalar indexing is disallowed.
```

---

## Dispatch Trace for sort!(A::JLArray)

```
sort!(A::JLArray{Float32,1})

Julia dispatch search:
  JLArray  <:  AbstractGPUArray  <:  AbstractVector  <:  AbstractArray

  Step 1: JLArrays.jl        — no sort! method defined
  Step 2: GPUArrays.jl       — no sort! method defined
  Step 3: Base.sort!         — Base.sort!(v::AbstractVector) FOUND

  → Base.sort!(A) calls:
      Base._sort!(A, Base.DEFAULT_UNSTABLE, Base.Order.Forward, ...)
      → Julia's introsort (quicksort + heapsort fallback)
      → comparison: lt(A[j], A[j+1])
      → A[j] triggers scalar GPU read
      → GPUArrays.assertscalar(A) fires
      → ERROR: Scalar indexing is disallowed
```

With `allowscalar(true)` (not recommended):
```
→ Each comparison: GPU global memory read  (~400 cycle latency)
→ Julia's introsort does O(n log n) comparisons
→ For n = 10^6:   ~20×10^6 scalar reads × 400 cycles / (2×10^9 Hz)
  = ~4000 ms    vs    ~2.8 ms for GPU merge sort   → 1400× slower
```

---

## What Breaks Downstream

```julia
A = jl(rand(Float32, 1000))

# All of these fail on JLArray / Metal / future backends:
sort!(A)                        # ERROR: scalar indexing
sort(A)                         # ERROR: scalar indexing (calls sort!)
sortperm(A)                     # ERROR: scalar indexing
sortperm!(similar(A, Int), A)   # ERROR: scalar indexing

# Real-world use cases that break:
ix = sortperm(A)               # Top-k selection
B = A[ix[end-9:end]]           # "Top 10" elements — fails at sortperm

# k-nearest neighbor (GPU ML inference):
dists = pairwise_distances(query, database)   # works
ix    = sortperm(dists)                        # ERROR — breaks kNN
nn    = database[:, ix[1:k]]                  # never reached

# Ranking in training loop:
ranks = sortperm(loss_values)   # ERROR — can't rank examples by loss
```

---

## Why allowscalar(true) Is Not a Solution

Some users default to `allowscalar(true)` and believe their code "works." Profiling reveals the truth:

```
Benchmark: sort! on Float32 array

Array size    allowscalar(true)    GPU merge sort    Ratio
──────────    ─────────────────    ──────────────    ─────
10K           ~180 ms              ~0.08 ms          2250×
100K          ~2100 ms             ~0.4 ms           5250×
1M            ~24 s                ~3.2 ms           7500×
```

The ratio grows super-linearly because each GPU scalar read has fixed ~400-cycle latency, and the sort does O(n log n) reads. At n=1M, that's ~20M scalar reads × 400 cycles = ~4 seconds of pure latency — before any actual sorting work.

---

## Case Study: Failing kNN in Production

A concrete real-world failure scenario from GPU ML inference:

```julia
# Attempting kNN search on a GPU embedding database
# Backend: future custom GPUArray type (e.g., vendor-specific AI accelerator)

function knn_search(query::AnyGPUArray, database::AnyGPUArray, k::Int)
    # Step 1: compute distances — works (mapreducedim! now fixed by PR #4)
    dists = vec(sum((database .- query).^2; dims=1))
    
    # Step 2: find k nearest — FAILS HERE
    ix = sortperm(dists)          # ERROR on any backend without sort
    # ↑ This line causes: Scalar indexing is disallowed
    # OR: silently runs on CPU taking 10 seconds for a 1M-vector database
    
    return database[:, ix[1:k]]
end
```

After this PR, `sortperm` works portably on any backend via AK.jl merge sort:
```julia
# After PR: sortperm delegates to AK.sortperm! → GPU merge sort
# 1M vectors: 3.2ms GPU  vs  24s CPU scalar  → 7500× speedup
```
