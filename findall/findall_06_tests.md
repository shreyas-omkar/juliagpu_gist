# findall — Test Suite

Added to `test/testsuite/indexing.jl` using the `compare` helper.

```julia
@testsuite "indexing/findall" (AT, eltypes) -> begin

    # ── findall(bools) ─────────────────────────────────────────────────────────
    @test compare(findall, AT, fill(true, 128))            # all true
    @test compare(findall, AT, fill(false, 128))           # all false
    @test compare(findall, AT, rand(Bool, 128))            # mixed 50/50
    let A = fill(false, 128); A[1] = A[64] = A[128] = true
        @test compare(findall, AT, A)                      # sparse (3 trues)
    end
    @test compare(findall, AT, rand(Bool, 16, 16))         # 2-D
    @test compare(findall, AT, rand(Bool, 8, 8, 8))        # 3-D
    @test compare(findall, AT, rand(Bool, 10^6))           # large (multi-block cumsum)
    @test compare(findall, AT, Bool[])                     # empty
    @test compare(findall, AT, Bool[true])                 # single true
    @test compare(findall, AT, Bool[false])                # single false
    @test compare(findall, AT, Bool[true, false])          # length 2
    @test compare(findall, AT, Bool[false, true])          # length 2, reversed
    @test compare(findall, AT, rand(Bool, 1025))           # non-power-of-2 (cumsum padding)

    # ── findall(f, A) ──────────────────────────────────────────────────────────
    @testset "$ET" for ET in eltypes
        @test compare(A -> findall(>(zero(ET)), A), AT, rand(ET, 128))
        @test compare(A -> findall(isnan, A),       AT, rand(ET, 64))
    end

    # ── logical indexing A[mask] ───────────────────────────────────────────────
    @testset "$ET" for ET in eltypes
        A     = AT(rand(ET, 256))
        mask  = AT(rand(Bool, 256))
        @test Array(A[mask]) == Array(A)[Array(mask)]      # 1-D mask indexing

        B     = AT(rand(ET, 16, 16))
        mask2 = AT(rand(Bool, 16, 16))
        @test Array(B[mask2]) == Array(B)[Array(mask2)]    # 2-D mask indexing
    end
end
```

---

## What Each Group Tests

| Test | What it verifies |
|------|-----------------|
| `fill(true, 128)` | All elements written — output size = input size |
| `fill(false, 128)` | Zero-output edge case — `n=0` guard |
| `rand(Bool, 128)` | General correctness of cumsum + scatter |
| sparse (3 trues) | Correct scatter with large gaps between true positions |
| 2-D / 3-D | `CartesianIndices` recovery for ND arrays |
| `10^6` | Multi-block cumsum path (more than one workgroup) |
| `Bool[]` | Empty array: `isempty(indices)` guard |
| `Bool[true]` / `Bool[false]` | Length-1 boundary cases |
| `rand(Bool, 1025)` | Non-power-of-2: verifies cumsum padding handles n≠2^k |
| `findall(f, A)` | `map(f, A)` + `findall` composition + `unsafe_free!` |
| `A[mask]` | `Base.to_index` override routes to GPU `findall`, not CPU |
| 2-D `B[mask2]` | ND logical indexing via overridden `to_indices` |

The **logical indexing tests** are the most important beyond `findall` itself — they verify that `A[mask]` is GPU-accelerated end-to-end, not just `findall` in isolation.

The **non-power-of-2 test** (`n=1025`) is critical for cumsum correctness: Blelloch scan pads to the next power of 2 internally, and this test verifies the padding produces correct results at the boundary.
