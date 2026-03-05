# accumulate! / cumsum / cumprod — Test Suite

New file: `test/testsuite/accumulate.jl`

```julia
@testsuite "accumulate" (AT, eltypes) -> begin
    @testset "$ET" for ET in eltypes

        # ── accumulate! ────────────────────────────────────────────────
        @test compare((B,A) -> accumulate!(+, B, A),
                      AT, rand(ET, 128), rand(ET, 128))          # prefix sum
        @test compare((B,A) -> accumulate!(*, B, A),
                      AT, rand(ET, 128), rand(ET, 128))          # prefix product
        @test compare((B,A) -> accumulate!(+, B, A; init=one(ET)),
                      AT, rand(ET, 128), rand(ET, 128))          # with init
        @test compare((B,A) -> accumulate!(min, B, A),
                      AT, rand(ET, 128), rand(ET, 128))          # min scan
        @test compare((B,A) -> accumulate!(max, B, A),
                      AT, rand(ET, 64),  rand(ET, 64))           # max scan
        @test compare((B,A) -> accumulate!(+, B, A),
                      AT, rand(ET, 10^6), rand(ET, 10^6))        # large

        # ── cumsum ─────────────────────────────────────────────────────
        @test compare(cumsum, AT, rand(ET, 128))
        @test compare(cumsum, AT, rand(ET, 10^6))

        # ── cumprod ────────────────────────────────────────────────────
        @test compare(cumprod, AT, rand(ET, 128))

        # ── Edge cases ─────────────────────────────────────────────────
        @test compare(cumsum,  AT, rand(ET, 1))        # single element
        @test compare(cumprod, AT, rand(ET, 1))
        @test compare(cumsum,  AT, ET[])               # empty array
        @test compare(cumsum,  AT, rand(ET, 2))        # length 2
        @test compare(cumprod, AT, rand(ET, 2))
        @test compare(cumsum,  AT, rand(ET, 1024))     # exact power of 2
        @test compare(cumsum,  AT, rand(ET, 1025))     # non-power-of-2 ← critical

    end
end
```

## What Each Group Tests

| Test | What it verifies |
|------|-----------------|
| `accumulate!(+, ...)` | Core prefix sum dispatch and correctness |
| `accumulate!(*, ...)` | Non-additive operator works |
| `init=one(ET)` | Custom init value passed through to AK.jl |
| `min` / `max` scan | AK.jl handles non-arithmetic operators |
| `n=10^6` | Multi-block reduction (exercises full Blelloch tree) |
| `n=1` | Single-element edge: output = input, no reduction needed |
| `ET[]` | Empty array guard in AK.jl delegation path |
| `n=1024` | Exact power of 2 — Blelloch's natural input size |
| `n=1025` | **Non-power-of-2** — AK.jl must pad internally; verifies padding correctness |

The `n=1025` test is the most important: Blelloch assumes power-of-2 input. AK.jl handles this by padding to the next power of 2 internally. If padding is broken, this test catches it.
