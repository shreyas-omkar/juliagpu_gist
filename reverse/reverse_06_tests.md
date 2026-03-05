# reverse / reverse! — Test Suite

Added to `test/testsuite/base.jl` using the existing `compare` helper,  
which runs the operation on both CPU and GPU and asserts numerical equality.

```julia
@testsuite "base/reverse" (AT, eltypes) -> begin
    @testset "$ET" for ET in eltypes

        # ── Core correctness ───────────────────────────────────────────
        @test compare(reverse, AT, rand(ET, 128))                        # 1-D out-of-place
        @test compare(A -> reverse!(copy(A)), AT, rand(ET, 128))         # 1-D in-place
        @test compare(A -> reverse(A; dims=1), AT, rand(ET, 16, 16))     # 2-D dim=1
        @test compare(A -> reverse(A; dims=2), AT, rand(ET, 16, 16))     # 2-D dim=2
        @test compare(A -> reverse(A; dims=:), AT, rand(ET, 8, 8, 8))   # all dims (3-D)

        # ── Edge cases ─────────────────────────────────────────────────
        @test compare(reverse, AT, ET[])                                  # empty array
        @test compare(reverse, AT, rand(ET, 1))                           # single element
        @test compare(A -> reverse(A; dims=1), AT, rand(ET, 1, 16))      # single-element dim (no-op)
        @test compare(reverse, AT, rand(ET, 7))                           # odd length
        @test compare(A -> reverse!(copy(A)), AT, rand(ET, 7))            # odd length in-place
        #   ↑ middle element must be left untouched — tests the lin_in < lin_out guard

    end
end
```

## What Each Group Tests

| Test | What it verifies |
|------|-----------------|
| `rand(ET, 128)` | Basic 1D kernel correctness |
| `reverse!(copy(A))` | In-place swap logic, no double-swap |
| `dims=1`, `dims=2` | ND index arithmetic per dimension |
| `dims=:` on 3D | All-dimension reversal, ND kernel |
| `ET[]` | Empty array guard (`length(A) == 0`) |
| `rand(ET, 1)` | Single element — no swap needed |
| `rand(ET, 1, 16)` `dims=1` | Size-1 dimension is a no-op |
| `rand(ET, 7)` | Odd-length: middle element untouched |

The odd-length test is the most important for `reverse!` — it directly exercises the `lin_in < lin_out` guard that prevents the middle element from being swapped with itself.
