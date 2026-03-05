# sort! / sortperm / sortperm! — Algorithm Deep Dive

## What These Three Functions Do

| Function | Signature | Effect |
|---|---|---|
| `sort!(A)` | `sort!(A; lt, by, rev, order)` | Sort `A` **in-place**, return `A` |
| `sortperm(A)` | `sortperm(A; lt, by, rev, order)` | Return index array `ix` such that `A[ix]` is sorted; `A` unchanged |
| `sortperm!(ix, A)` | `sortperm!(ix, A; lt, by, rev, order)` | Fill pre-allocated `ix` with sort permutation; `A` unchanged |

`sortperm` is defined as a thin wrapper: allocate `ix = 1:n`, call `sortperm!(ix, A)`. So the real algorithmic work is in `sort!` and `sortperm!`.

---

## Why Sorting Is Hard on GPUs

Sequential sorting (insertion sort, quicksort on CPU) exploits cache locality and branch prediction. GPUs have neither in the same sense. The challenge is:

1. **SIMT execution** — all threads in a warp execute the same instruction. Divergent branches (e.g. `if A[i] < pivot`) reduce occupancy.
2. **No random write locality** — scattered writes to global memory thrash L2 cache.
3. **No recursion** (historically) — quicksort's recursive structure maps poorly to GPU. CUDA now supports *dynamic parallelism* (kernel launching kernels), which CUDA.jl exploits.
4. **Stability requirements** — `sortperm` must be stable (equal keys preserve original order) because users rely on it for reproducible rank ordering.

These constraints push GPU sort implementations toward **merge sort** (deterministic, stable, regular access patterns) or **radix sort** (no comparisons, integer keys only) rather than quicksort.

---

## Algorithm 1: CUDA.jl — GPU Quicksort with Dynamic Parallelism

Source: `CUDA.jl/src/sorting.jl` — 1062 lines, developed by @xaellison (Alex Ellison).

### High-Level Structure

```
quicksort!(vals, lo, hi)
  ├── If sublist fits in one block (L <= blockDim.x):
  │     bubble_sort(vals, lo, hi, stride=1)   ← terminal case, shmem
  ├── Else:
  │     pivot = bitonic_median(vals, lo, hi, stride)  ← pivot selection
  │     batch_partition(vals, pivot, lo, hi, parity)   ← parallel partition
  │     consolidate_batch_partition(...)                ← fix batch seams
  │     @cuda dynamic=true quicksort!(vals, lo, mid)   ← recurse left
  │     @cuda dynamic=true quicksort!(vals, mid, hi)   ← recurse right
```

### Phase 1: Pivot Selection via Bitonic Sort

Rather than picking a random or fixed pivot, CUDA.jl performs a **bitonic sort** on `blockDim.x` elements sampled with stride `(hi - lo) / blockDim.x` from the array. The median of these sorted samples is the pivot. This gives a statistically good pivot (close to the true median) even for adversarial inputs.

Bitonic sort on `M` elements in shared memory takes `O(M log² M)` steps — expensive, but `M` is fixed at 256, so it's a constant-time operation that runs entirely in shared memory (zero global memory traffic).

```
Bitonic sort network for M=8 (3 stages, each stage has log₂ stage passes):
Stage 1: compare distance 1
Stage 2: compare distance 2, then 1
Stage 3: compare distance 4, then 2, then 1
Result: median at position M/2
```

### Phase 2: Batch Partition

The array between `lo` and `hi` is divided into batches of `blockDim.x`. For each batch:
1. Each thread computes `flex_lt(pivot, vals[i], parity)` — whether its element should go right of pivot.
2. An in-shared-memory prefix sum (cumsum!) computes each element's destination.
3. Elements are scattered via shared memory, then written back to global memory.

`parity` alternates with recursion depth. When `parity=true`, elements equal to pivot count as "right". When `parity=false`, they count as "left". This ensures that arrays with many duplicates converge — equal elements get split between the two halves in alternating recursion levels, preventing the O(n²) worst case for all-equal arrays.

```
Example: pivot=5, parity=false
Input batch: [3, 7, 5, 2, 8, 5, 1, 6]
flex_lt(5, x, false):  [F, T, F, F, T, F, F, T]
                          ↓ cumsum
positions:              [1, 2, 3, 4, 2, 5, 6, 3]  (left/right destinations)
Output:                 [3, 5, 2, 5, 1, 7, 8, 6]  (left half | right half)
```

### Phase 3: Batch Consolidation

After batch-partitioning, each batch is internally partitioned but the seams between batches are not. `consolidate_batch_partition` uses one SM to walk through the batch boundaries, computing the true partition point using `find_partition` (binary search), then swapping elements across batch boundaries in-place.

### Phase 4: Dynamic Parallelism Recursion

CUDA.jl uses `@cuda dynamic=true` — the GPU kernel itself launches two child kernels for the left and right partitions. This avoids round-tripping to the CPU between recursion levels. The recursion depth is bounded by `log₂(n)`.

**Complexity:**
- Average: O(n log n), same as CPU quicksort
- Worst case: O(n²) theoretically, but the bitonic median pivot selection makes this astronomically unlikely
- Stack depth: O(log n) kernel launch levels

### Why sortperm! is Not in CUDA.jl

CUDA.jl's quicksort only sorts values. It does not produce a permutation index array. `sortperm` on CuArray falls to CPU (or errors with `allowscalar(false)`). This is a known limitation — confirmed by the audit `[CA-- -] sortperm!` in exhaustive_audit.txt. Users who need `sortperm` on CUDA must use workarounds.

---

## Algorithm 2: AcceleratedKernels.jl — GPU Merge Sort

Source: `AcceleratedKernels.jl/src/sort/` — 6 files:
- `merge_sort.jl` — core value sort
- `merge_sort_by_key.jl` — sort values, carry along a key array
- `merge_sortperm.jl` — sort an index array by values (sortperm)
- `sort.jl` — public API: `AK.sort!`, `AK.sortperm!`
- `utils.jl` — binary search, comparison helpers
- `cpu_sample_sort.jl` — CPU fallback for small arrays

### Why Merge Sort (Not Quicksort) for AK.jl

Merge sort has a fundamentally different structure that maps much better to GPU:

| Property | Quicksort | Merge Sort |
|---|---|---|
| Access pattern | Scattered (partition) | Sequential (merge) |
| Recursion | Data-dependent depth | Fixed log₂(n) passes |
| Stability | Unstable (typically) | Stable |
| Parallel efficiency | Hard to parallelize | Embarrassingly parallel by pass |
| sortperm support | Hard (scattered indices) | Easy (carry index array) |

### The Bottom-Up Iterative Merge Sort Algorithm

AK.jl uses a **bottom-up** merge sort — no recursion, purely iterative. This is the standard approach for GPU sort.

```
Pass 0: Sort blocks of size 1 → trivially sorted (nothing to do)
Pass 1: Merge pairs of size-1 blocks → sorted blocks of size 2
Pass 2: Merge pairs of size-2 blocks → sorted blocks of size 4
Pass 3: Merge pairs of size-4 blocks → sorted blocks of size 8
...
Pass k: Merge pairs of size-2^(k-1) blocks → sorted blocks of size 2^k
Stop when block size >= n
```

Total passes: `⌈log₂(n)⌉`. Each pass is a fully parallel GPU kernel — every pair of blocks is merged independently. No inter-block communication is needed within a pass.

### Concrete Example: n=8

```
Input:   [5, 3, 8, 1, 7, 2, 4, 6]

Pass 1 (block_size=1 → 2):
  Thread groups: [5,3] [8,1] [7,2] [4,6]
  Each TG merges 2 elements:
  Result: [3,5] [1,8] [2,7] [4,6]

Pass 2 (block_size=2 → 4):
  Thread groups: [3,5,1,8] [2,7,4,6]
  Each TG merges 4 elements:
  Result: [1,3,5,8] [2,4,6,7]

Pass 3 (block_size=4 → 8):
  Thread groups: [1,3,5,8,2,4,6,7]
  Final merge:
  Result: [1,2,3,4,5,6,7,8]  ✓

Total: 3 passes = ceil(log₂(8))
```

### The Merge Kernel

Each threadgroup is assigned one merge task (left half + right half → output). Within the merge, threads use **binary search** to find each element's destination:

```julia
# Each thread i handles output position out_start + i
# Find how many elements from left  are < output[i]: binary search in right half
# Find how many elements from right are < output[i]: binary search in left half
# These two counts sum to the output position → write element
```

This "merge path" strategy (from Green et al., 2012) gives every thread O(log n) work with no divergence and perfect load balance.

### sortperm! via merge_sortperm.jl

`sortperm!` is implemented by augmenting the merge: instead of sorting just values, the kernel co-sorts a companion index array initialized to `1:n`. Every comparison moves both `vals[i]` and `ix[i]` together. Since merge sort is stable, equal values preserve their original index ordering — the permutation is reproducible.

```julia
# Initialize
ix = 1:length(vals)

# Each merge step:
# When merging left[i] and right[j]:
#   if vals_left[i] <= vals_right[j]:  output vals_left[i], ix_left[i]
#   else:                               output vals_right[j], ix_right[j]
# Stability: <= (not <) ensures left element wins ties → preserves original order
```

---

## Algorithm Comparison: CUDA Quicksort vs AK Merge Sort

| Property | CUDA.jl Quicksort | AK.jl Merge Sort |
|---|---|---|
| **Passes over data** | O(log n) expected | Exactly ⌈log₂(n)⌉ |
| **Memory** | In-place (scratch only) | Requires output buffer (2× memory) |
| **Stability** | Unstable | Stable |
| **sortperm support** | ✗ Not implemented | ✓ Native |
| **Dynamic parallelism** | Required (CUDA only) | Not needed (portable) |
| **Performance (random)** | Slightly faster | Competitive |
| **Performance (sorted)** | Fast (good pivot) | Same as random |
| **Implementation size** | ~1062 lines | ~300 lines |
| **Portability** | CUDA-only | All backends via KA |

**For GPUArrays.jl**, the choice is clear: delegate to `AK.sort!` and `AK.sortperm!`, exactly as AMDGPU.jl and oneAPI.jl do. This gets stable, portable, feature-complete sort in 3 lines per function.

---

## Why Metal Has No Sort

Metal.jl has no sort implementation at all. The audit confirms `[CA-- -] sort!` (Metal absent). Metal compute shaders lack dynamic parallelism, making the CUDA approach impossible. AK.jl's merge sort works on Metal for other operations, but `sort!` / `sortperm!` delegates were never added to `Metal.jl`. This is the same gap GPUArrays.jl has.
