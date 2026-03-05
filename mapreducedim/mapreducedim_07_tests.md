# mapreducedim!   Test Suite

Extended from `test/testsuite/mapreduce.jl`.

```julia
@testsuite "mapreduce/mapreducedim" (AT, eltypes) -> begin
    @testset "$ET" for ET in eltypes

        # ── Whole-array reductions ─────────────────────────────────────────────
        @test compare(sum,     AT, rand(ET, 128))
        @test compare(prod,    AT, rand(ET, 16) .+ one(ET))      # avoid zero products
        @test compare(maximum, AT, rand(ET, 128))
        @test compare(minimum, AT, rand(ET, 128))
        @test compare(A -> any(>(zero(ET)), A), AT, rand(ET, 128))
        @test compare(A -> all(>(zero(ET)), A), AT, rand(ET, 128))

        # ── mapreduce with non-identity f (map fusion) ─────────────────────────
        @test compare(A -> sum(abs2, A),              AT, rand(ET, 128))
        @test compare(A -> mapreduce(x->x^2, +, A),  AT, rand(ET, 64))

        # ── Dimensional reductions (2-D) ───────────────────────────────────────
        @test compare(A -> sum(A; dims=1), AT, rand(ET, 16, 16))   # reduce rows
        @test compare(A -> sum(A; dims=2), AT, rand(ET, 16, 16))   # reduce cols
        @test compare(A -> maximum(A; dims=1), AT, rand(ET, 16, 16))
        @test compare(A -> minimum(A; dims=2), AT, rand(ET, 16, 16))

        # ── Dimensional reductions (3-D) ───────────────────────────────────────
        @test compare(A -> sum(A; dims=1),    AT, rand(ET, 8, 8, 8))
        @test compare(A -> sum(A; dims=3),    AT, rand(ET, 8, 8, 8))
        @test compare(A -> sum(A; dims=(1,3)),AT, rand(ET, 8, 8, 8))   # multi-dim

        # ── Edge cases ─────────────────────────────────────────────────────────
        @test compare(sum, AT, rand(ET, 1))                            # single element
        @test compare(A -> sum(A; dims=1), AT, rand(ET, 1, 16))        # size-1 dim (no-op)
        @test compare(sum, AT, rand(ET, 10^6))                         # large (multi-block)
        @test compare(sum, AT, rand(ET, 1025))                         # non-power-of-2
        @test compare(A -> mapreduce(identity, +, A; init=one(ET)),    # custom init
                      AT, rand(ET, 128))
    end

    # ── Boolean reductions ─────────────────────────────────────────────────────
    @testset "Bool" begin
        @test compare(any, AT, rand(Bool, 128))
        @test compare(all, AT, rand(Bool, 128))
        @test compare(any, AT, fill(false, 128))      # all-false: any → false
        @test compare(all, AT, fill(true,  128))      # all-true:  all → true
        @test compare(A -> any(A; dims=1), AT, rand(Bool, 16, 16))
    end
end
```

---

## What Each Group Tests

| Test | What it verifies |
|------|-----------------|
| `sum`, `prod`, etc. | Orchestration layer correctly routes all high-level reductions |
| `sum(abs2, A)` | Map fusion   `f` applied in-register, no temporary array |
| `dims=1`, `dims=2` | Dimensional reduction index arithmetic |
| `dims=(1,3)` | Multi-dimension reduction   multiple dims collapsed simultaneously |
| `rand(ET, 1)` | Single-element: trivial reduction, output = input |
| `dims=1` on `(1,16)` | Size-1 reduced dimension: output = input (neutral element + one value) |
| `rand(ET, 10^6)` | **Multi-block reduction**   exercises inter-block combine pass |
| `rand(ET, 1025)` | Non-power-of-2: threadgroup padding with neutral element |
| `init=one(ET)` | Custom init overrides `neutral_element`   verify pass-through |
| Boolean `any`/`all` | Neutral elements `false`/`true` initialised correctly |
| `fill(false, 128)` → `any` | All-false: verify `false` neutral element doesn't corrupt |
| `any(A; dims=1)` | Boolean dimensional reduction |

---

## Critical Test: Multi-Block Reduction (`n = 10^6`)

With group size 256, reducing `10^6` elements requires:
- `10^6 / 256 = 3906` threadgroups
- Each threadgroup produces one partial result
- A second pass combines 3906 partial results into the final scalar

This tests the multi-block path that is not exercised by any of the smaller test cases. Without it, a bug in the inter-block combine pass would go undetected for all "small enough" inputs.

---

## Critical Test: Non-Power-of-2 (`n = 1025`)

Group size 256. `1025 / 256 = 4.003...`   the last group only has 1 element, but must initialise `shared[2:256]` with the neutral element to avoid garbage in the tree reduction. This tests the padding logic in the kernel.
