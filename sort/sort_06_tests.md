# sort! / sortperm / sortperm!   Test Suite

File: `test/testsuite/sort.jl` (new file, included from `test/testsuite.jl`)

```julia
@testsuite "sort" (AT, eltypes) -> begin
    @testset "$ET" for ET in eltypes

        # ── sort! basic correctness ────────────────────────────────────────────
        @test compare(sort!, AT, rand(ET, 128))
        @test compare(A -> sort!(copy(A); rev=true), AT, rand(ET, 128))

        # ── sort! edge cases ──────────────────────────────────────────────────
        @test compare(sort!, AT, ET[])                        # empty
        @test compare(sort!, AT, rand(ET, 1))                 # single element
        @test compare(sort!, AT, rand(ET, 2))                 # two elements
        @test compare(sort!, AT, rand(ET, 1025))              # non-power-of-2
        @test compare(sort!, AT, rand(ET, 10^6))              # large (many passes)
        @test compare(sort!, AT, sort(rand(ET, 128)))         # already sorted
        @test compare(sort!, AT, sort(rand(ET, 128); rev=true)) # reverse sorted

        # ── sort! with kwargs ─────────────────────────────────────────────────
        @test compare(A -> sort!(copy(A); by=abs),   AT, rand(ET, 64) .- ET(0.5))
        @test compare(A -> sort!(copy(A); rev=true), AT, rand(ET, 64))

        # ── sortperm ──────────────────────────────────────────────────────────
        @test compare(sortperm, AT, rand(ET, 128))
        @test compare(A -> sortperm(A; rev=true), AT, rand(ET, 128))

        # ── sortperm! ─────────────────────────────────────────────────────────
        @test compare((ix, A) -> sortperm!(ix, A),
                      AT, similar(AT(rand(ET, 128)), Int), rand(ET, 128))

        # ── Stability test: equal elements preserve original order ────────────
        # Build array with duplicates; sortperm must return increasing indices
        # for equal values (stable sort property)
        @testset "stability" begin
            A_cpu = ET[3, 1, 2, 1, 3, 2]
            A_gpu = AT(A_cpu)
            ix_cpu = sortperm(A_cpu)
            ix_gpu = Array(sortperm(A_gpu))
            @test ix_cpu == ix_gpu   # must match CPU stable sort exactly
        end

        # ── sortperm / sort! consistency ──────────────────────────────────────
        # A[sortperm(A)] must equal sort(A)
        @testset "sortperm consistency" begin
            A_gpu = AT(rand(ET, 256))
            ix    = sortperm(A_gpu)
            @test Array(A_gpu[ix]) ≈ Array(sort!(copy(A_gpu)))
        end
    end

    # ── Integer element types ─────────────────────────────────────────────────
    @testset "Int32" begin
        @test compare(sort!, AT, Int32[5, 3, 1, 4, 2])
        @test compare(sortperm, AT, Int32[5, 3, 1, 4, 2])
    end

    # ── Large array (exercises all log2(n) passes) ────────────────────────────
    @testset "large" begin
        n = 10^6
        @test compare(sort!, AT, rand(Float32, n))
        @test compare(sortperm, AT, rand(Float32, n))
    end
end
```

---

## What Each Test Verifies

| Test | What it verifies |
|---|---|
| `rand(ET, 128)` | Basic correctness on random data |
| `rev=true` | Reverse ordering kwarg passes through to AK |
| Empty array | No kernel launch on empty input |
| Single/two elements | Boundary conditions in merge pass logic |
| `n=1025` non-power-of-2 | Last merge block is smaller   padding logic |
| `n=10^6` large | All `ceil(log2(10^6))=20` passes execute correctly |
| Already sorted | Idempotency   sorting a sorted array gives same result |
| Reverse sorted | Worst-case input for some algorithms (not merge sort) |
| `by=abs` | Custom sort key passes through |
| `sortperm` | Index array is correct permutation |
| Stability | Equal elements preserve original index order |
| sortperm consistency | `A[sortperm(A)] == sort(A)`   fundamental invariant |
| Int32 | Non-Float element type works |

---

## Critical Test: Stability

Stability is the most important correctness property of `sortperm`. Unlike `sort!` (where equal elements are indistinguishable after sorting), `sortperm` is only correct if equal values produce indices in ascending order.

```julia
# Concrete stability check:
A = Float32[3.0, 1.0, 2.0, 1.0, 3.0, 2.0]
# Values and their original 1-based indices:
#  val:  3    1    2    1    3    2
#  idx:  1    2    3    4    5    6

# Stable sortperm result:
# sorted order: 1(idx=2), 1(idx=4), 2(idx=3), 2(idx=6), 3(idx=1), 3(idx=5)
# → ix = [2, 4, 3, 6, 1, 5]

# Unstable would give: [4, 2, ...] or [6, 3, ...]   wrong!
```

AK.jl's merge sort is guaranteed stable because it uses `<=` (not `<`) in the merge comparison, ensuring left-half elements win ties   which preserves original order across all passes.

---

## Critical Test: sortperm Consistency

The mathematical invariant that must hold for any correct `sortperm`:

```julia
@test A[sortperm(A)] == sort(A)
# Or with floating point tolerance:
@test A[sortperm(A)] ≈ sort(A)
```

This test catches bugs where the index array and value sorting get out of sync during the co-sort   a subtle bug that only appears when values and indices are merged separately.
