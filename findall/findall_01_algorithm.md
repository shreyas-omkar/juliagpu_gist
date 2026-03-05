# findall — Algorithm: Stream Compaction via Prefix Scan

## Why GPU findall Is Non-Trivial

On CPU, `findall` is a trivial sequential scan — walk the array, `push!` true indices into a growing list. On GPU, two fundamental problems block this:

1. **Unknown output size.** GPU memory must be allocated exactly before the scatter. A growable list does not exist on GPU.
2. **Race conditions.** If every thread tries to append to the output, two threads may write to the same position simultaneously.

The standard GPU solution is **stream compaction via prefix scan** — a two-step algorithm that solves both problems.

---

## Step 1: Prefix Sum (cumsum)

Compute the inclusive prefix sum of the boolean array:

```
indices = cumsum(reshape(bools, prod(size(bools))))
```

This gives two things for free:
- `indices[end]` = total number of `true` elements → **exact output size**, allocate now
- For each `true` at position `i`: `indices[i]` = its **unique write position** in the output → no two threads share a write slot

**Why write positions are unique:** the prefix sum is strictly monotone at true positions. If `bools[i] = true` then `indices[i] > indices[i-1]`, so no two true positions produce the same index.

---

## Step 2: Scatter Kernel

Launch one thread per input element. Each thread:

```
Thread i:
  if bools[i]:
      i′    = CartesianIndices(bools)[i]   ← recover ND index
      b     = indices[i]                    ← unique write slot
      ys[b] = i′                            ← scatter, no conflicts
  else:
      do nothing
```

Threads for `false` elements are idle. No synchronisation needed — scatter positions are provably unique.

---

## Full Pseudocode

```
Step 1:  indices = cumsum(reshape(bools, prod(size(bools))))
         n       = indices[end]                          ← exact output size
         ys      = Array{keytype(bools)}(undef, n)       ← allocate

Step 2:  parallel for i in 1:length(bools):
           if bools[i]:
               ys[indices[i]] = CartesianIndices(bools)[i]
```

---

## Worked Example: 8-element array

```
bools:    [ T  F  F  T  T  F  T  F ]
cumsum:   [ 1  1  1  2  3  3  4  4 ]
                                 ↑
                          indices[end] = 4  →  allocate ys of size 4

Scatter (threads for T positions only):
  i=1: bools[1]=T, indices[1]=1  →  ys[1] = CartesianIndex(1)
  i=4: bools[4]=T, indices[4]=2  →  ys[2] = CartesianIndex(4)
  i=5: bools[5]=T, indices[5]=3  →  ys[3] = CartesianIndex(5)
  i=7: bools[7]=T, indices[7]=4  →  ys[4] = CartesianIndex(7)

Output:  [ 1  4  5  7 ]  ← indices of all true elements
```

All four scatter writes happen in parallel, to unique positions. Zero conflicts.

---

## Dependency on PR #2 (accumulate! / cumsum)

Step 1 is exactly `cumsum(bools)`. Without a GPU `cumsum` at the `AnyGPUArray` level (PR #2), `findall` cannot be implemented portably. This is why PR ordering matters:

```
PR #1: reverse       (no dependencies)
PR #2: accumulate!   (no dependencies)
PR #3: findall       (depends on PR #2's cumsum)
PR #4: mapreducedim! (no direct dependency)
```
