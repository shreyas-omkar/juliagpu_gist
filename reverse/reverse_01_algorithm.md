# reverse / reverse!   Algorithm Deep Dive

## Core Idea: Index Reordering

Reversing an array is a pure index reordering problem   no arithmetic is performed on values, only on *where they are placed*.

For a 1D array of length `n`, element at position `i` maps to `n + 1 - i`.  
For N-dimensional arrays, the formula applies independently per dimension:

```
nd_out[d] = size(A,d) + 1 - nd[d]   if d in dims
nd_out[d] = nd[d]                    otherwise
```

---

## Out-of-Place: `reverse`

Each thread reads one source element and writes to one destination   **embarrassingly parallel**, zero data dependencies, no barriers needed.

```
Thread i:
  1. nd     = CartesianIndices(src)[i]       ← ND index from linear
  2. nd_out = apply mirror formula per dim
  3. dst[nd_out] = src[nd]
```

Launch: `ndrange = length(A)`, one thread per element.

---

## In-Place: `reverse!`

Cannot assign one thread per element naively   thread at position 1 swaps with `n`, thread at position `n` also swaps with 1, **undoing the first swap**.

**Fix:** Launch only ⌈n/2⌉ threads. Each thread owns one swap pair.  
Guard `lin_in < lin_out` skips the middle element of odd-length dims (its mirror is itself).

```
Thread i  (i ≤ ⌈n/2⌉):
  1. idx_in  = CartesianIndices(reduced_size)[i]
  2. idx_out = apply mirror formula
  3. if LinearIndex(idx_in) < LinearIndex(idx_out):
         swap(A[idx_in], A[idx_out])
     # else: middle element of odd-length dim   skip
```

`reduced_size` halves the last reversible dimension:
```julia
reduced_sz = ntuple(d -> d == half_dim ? cld(size(A,d), 2) : size(A,d), N)
```

---

## Worked Example: 3×3 Matrix, Reversing Along dim=1

```
Input:                        rev_dims = (true, false),  ref = (4, 4)
  [ 1  2  3 ]
  [ 4  5  6 ]
  [ 7  8  9 ]

Thread i=1  →  ND(1,1):  dim1: 4-1=3, dim2: keep 1  →  dst[(3,1)] = 1
Thread i=5  →  ND(2,2):  dim1: 4-2=2, dim2: keep 2  →  dst[(2,2)] = 5  (self-map, middle row)
Thread i=9  →  ND(3,3):  dim1: 4-3=1, dim2: keep 3  →  dst[(1,3)] = 9

Output:
  [ 7  8  9 ]
  [ 4  5  6 ]
  [ 1  2  3 ]
```

All 9 threads run simultaneously. No barriers, no shared memory, no inter-thread communication required.
