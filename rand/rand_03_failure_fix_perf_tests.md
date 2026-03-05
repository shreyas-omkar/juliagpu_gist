# rand / randn — Failure Demonstration, Fix, Performance, and Tests

## SECTION 3: Failure Demonstration

### The Failure is Different from Other PRs

For `reverse`, `sort!`, `mapreducedim!` — the failure is a scalar indexing error or a hard error. For rand/randn the situation is more nuanced:

**`rand!(A)` and `randn!(A)` already work on all backends** (the `G` flag in the audit). The gap is the **out-of-place allocation forms**.

### Failure Mode 1: JLArray (test backend / future backends)

```julia
using GPUArrays, JLArrays

# rand! works — GPUArrays kernel exists
A = jl(zeros(Float32, 1000))
rand!(A)          # ✓ works — uses GPUArrays.RNG Xorshift128+
randn!(A)         # ✓ works — uses GPUArrays.RNG Box–Muller

# Out-of-place rand FAILS
B = rand(JLArray{Float32}, 1000)
# ERROR: MethodError: no method matching rand(::Type{JLArray{Float32,1}}, ::Int64)
# Closest candidates: rand(::AbstractRNG, ...), rand(::Type{Float32}, ...)
# → Falls to Base.rand(Float32, 1000) → returns CPU Array, not JLArray!

# Silent wrong type — no error, wrong result:
C = rand(Float32, 1000)     # This is what actually runs — CPU Array!
typeof(C)                   # Array{Float32,1} — not on GPU at all
```

### Failure Mode 2: Wrong dispatch — silently returns CPU array

The most dangerous failure: when a user writes `A = rand(Float32, n)` intending GPU allocation but gets a CPU array, with no error message:

```julia
using GPUArrays, JLArrays

# User wants GPU random array:
function init_weights(n::Int)
    W = rand(Float32, n, n)   # ← INTENDED: GPU-allocated random weights
    return W
end

W = init_weights(1000)
typeof(W)  # Array{Float32,2} — CPU! No error.

# Later:
loss = sum(model(W))          # W is on CPU → silent performance disaster
                              # or type error when model expects JLArray
```

### Failure Mode 3: oneAPI local-function conflict

```julia
using oneAPI

# oneAPI.jl defines local rand, not Base.rand override
# So Base.rand dispatch misses it:
A = Base.rand(oneArray, Float32, 100)
# MethodError — the oneAPI.jl rand() is a module-local function, 
# not a Base method override.

# Only this works:
A = oneAPI.rand(Float32, 100)   # module-qualified call
# Generic user code `rand(Float32, 100)` on an Intel GPU falls through to CPU
```

### Dispatch Trace for `rand(JLArray{Float32}, 1000)`

```
rand(JLArray{Float32}, 1000)

Julia dispatch search:
  JLArrays.jl        → no rand method
  GPUArrays.jl       → no rand method  
  Base.rand           → Base.rand(::Type{Float32}, ::Int64) FOUND ← WRONG!
  
Result: CPU Float32 array, no warning, no error.
```

The type parameter is completely ignored — `JLArray{Float32}` as a type argument to `rand` doesn't trigger any GPU-specific dispatch.

---

## SECTION 4: The Fix

### Design Decisions

**1. What exactly is missing:** Only the out-of-place wrappers `Base.rand` and `Base.randn` at `AnyGPUArray` level. The kernel is already there.

**2. How to express "array type":** The dispatch needs to accept an array type as a value, not an instance. This is the pattern `rand(::Type{<:AnyGPUArray}, T, dims)`.

**3. Default element type:** `Float32` — matches all vendor defaults and ML conventions.

**4. File location:** Alongside existing rand! kernel in `GPUArrays.jl/src/host/random.jl`.

**5. No new dependency:** Unlike sort (AK.jl), this PR needs nothing — the kernel already exists.

### Complete Implementation

```julia
# FILE: GPUArrays.jl/src/host/random.jl  (ADDITIONS to existing file)

# Out-of-place rand: allocate GPU array, fill with U(0,1), return
function Base.rand(::Type{A}, T::Type, dims::Dims) where {A<:AnyGPUArray}
    arr = A{T}(undef, dims)
    return Random.rand!(default_rng(A), arr)
end

function Base.rand(::Type{A}, dims::Dims) where {A<:AnyGPUArray}
    return Base.rand(A, Float32, dims)    # default: Float32
end

# Dimension variadic forms
function Base.rand(::Type{A}, T::Type, dim1::Integer, dims::Integer...) where {A<:AnyGPUArray}
    return Base.rand(A, T, Dims((dim1, dims...)))
end

function Base.rand(::Type{A}, dim1::Integer, dims::Integer...) where {A<:AnyGPUArray}
    return Base.rand(A, Float32, Dims((dim1, dims...)))
end

# Out-of-place randn: allocate GPU array, fill with N(0,1), return
function Base.randn(::Type{A}, T::Type, dims::Dims) where {A<:AnyGPUArray}
    arr = A{T}(undef, dims)
    return Random.randn!(default_rng(A), arr)
end

function Base.randn(::Type{A}, dims::Dims) where {A<:AnyGPUArray}
    return Base.randn(A, Float32, dims)
end

function Base.randn(::Type{A}, T::Type, dim1::Integer, dims::Integer...) where {A<:AnyGPUArray}
    return Base.randn(A, T, Dims((dim1, dims...)))
end

function Base.randn(::Type{A}, dim1::Integer, dims::Integer...) where {A<:AnyGPUArray}
    return Base.randn(A, Float32, Dims((dim1, dims...)))
end
```

### Dispatch Table After Fix

```
BEFORE:
rand(JLArray{Float32}, 100)    → Base.rand(Float32, 100) ← CPU! Wrong type
randn(JLArray{Float32}, 100)   → Base.randn(Float32,100) ← CPU! Wrong type
rand(CuArray{Float32}, 100)    → ✓ (CUDA.jl has own method)
rand(ROCArray{Float32}, 100)   → ✓ (AMDGPU.jl has own method)
rand(oneArray{Float32}, 100)   → only via oneAPI.rand() — not Base.rand
rand(MtlArray{Float32}, 100)   → ✓ (Metal.jl has MPS method)

AFTER:
rand(JLArray{Float32}, 100)    → GPUArrays.rand → default_rng → Xorshift128+ ✓
randn(JLArray{Float32}, 100)   → GPUArrays.randn → default_rng → Box–Muller ✓
rand(CuArray{Float32}, 100)    → CUDA.jl (unchanged, more specific) ✓
rand(ROCArray{Float32}, 100)   → AMDGPU.jl (unchanged) ✓
rand(oneArray{Float32}, 100)   → GPUArrays fallback (now via Base.rand) ✓
rand(<future>{Float32}, 100)   → GPUArrays fallback ✓
```

---

## SECTION 5: Performance Analysis

### rand! Performance (Write-Only, Arithmetic-Bound for randn)

**rand!** — Xorshift128+ is extremely cheap (4 XOR ops + 1 add per sample). For Float32:
- Each thread produces one 32-bit sample per iteration
- Memory bound: pure write, 4 bytes per element
- At 360 GB/s write bandwidth: `n × 4 / 360e9` seconds
- n=10^7: 4×10^7 / 360e9 = **0.11 ms**

**randn!** — Box–Muller requires `log`, `sqrt`, `cos`, `sin` — GPU transcendentals run at ~4 ns each:
- 4 transcendentals per pair of outputs → 2 transcendentals per element
- At 4000 GFLOP/s transcendental throughput (RTX 3060): 2 × n / 4e12 seconds
- n=10^7: 2×10^7 / 4e12 = **0.005 ms** ← dominated by memory, not arithmetic!
- Memory bound: `n × 4 / 360e9 = 0.11 ms` — same as rand!

**CPU baseline for out-of-place rand(Float32, n):**
- CPU rand (xoshiro256++): ~2 ns/sample
- Plus memory allocation overhead
- n=10^7: ~20 ms

### Projected Performance (RTX 3060, Float32)

| n | CPU rand (ms) | GPU rand! (ms) | GPU randn! (ms) | Speedup (rand) |
|:---:|:---:|:---:|:---:|:---:|
| 10K | ~0.02 | ~0.01 | ~0.01 | ~2× |
| 100K | ~0.20 | ~0.01 | ~0.01 | ~20× |
| 1M | ~2.0 | ~0.03 | ~0.03 | ~67× |
| 10M | ~20 | ~0.11 | ~0.11 | ~182× |
| 100M | ~200 | ~1.1 | ~1.1 | ~182× |

The speedup plateaus at ~180× at large n because both CPU and GPU become memory bandwidth bound, with GPU having ~180× more bandwidth (360 vs 2 GB/s peak write).

**Important:** The "speedup" for the out-of-place fix is not GPU vs CPU — it's "correct GPU result" vs "wrong CPU result." The before state doesn't just run slowly — it returns the wrong type entirely.

### AMDGPU-Specific: rocRAND vs Box–Muller

rocRAND uses hardware-accelerated HRNG generators on AMD hardware. For `randn!`:
- rocRAND `generate_normal`: hardware CDF inversion, ~2× faster than Box–Muller
- Metal MPS: similarly hardware-accelerated on Apple Silicon
- GPUArrays Box–Muller: software only, but still GPU-fast

---

## SECTION 6: Tests

```julia
@testsuite "random" (AT, eltypes) -> begin

    # ── rand! in-place (already works — regression test) ──────────────────
    @testset "rand!" begin
        A = AT{Float32}(undef, 1024)
        rand!(A)
        @test all(0 .<= Array(A) .< 1)          # U(0,1) range
        @test length(unique(Array(A))) > 900     # not constant
    end

    # ── randn! in-place (already works — regression test) ─────────────────
    @testset "randn!" begin
        A = AT{Float32}(undef, 1024)
        randn!(A)
        v = Array(A)
        @test abs(mean(v)) < 0.1                 # mean ≈ 0
        @test abs(std(v) - 1.0) < 0.1           # std ≈ 1
    end

    # ── rand out-of-place (NEW — the fix) ─────────────────────────────────
    @testset "rand out-of-place" begin
        A = rand(AT{Float32}, 1024)
        @test A isa AT                           # correct type on GPU
        @test all(0 .<= Array(A) .< 1)
        
        # All dimension forms
        A2 = rand(AT{Float32}, 32, 32)
        @test size(A2) == (32, 32)
        @test A2 isa AT
        
        # Default Float32
        A3 = rand(AT, 100)
        @test eltype(A3) == Float32
    end

    # ── randn out-of-place (NEW — the fix) ────────────────────────────────
    @testset "randn out-of-place" begin
        A = randn(AT{Float32}, 1024)
        @test A isa AT                           # correct type on GPU
        v = Array(A)
        @test abs(mean(v)) < 0.15
        @test abs(std(v) - 1.0) < 0.15
        
        # 2D
        A2 = randn(AT{Float32}, 64, 64)
        @test size(A2) == (64, 64)
    end

    # ── Reproducibility: same seed → same values ──────────────────────────
    @testset "reproducibility" begin
        rng = GPUArrays.default_rng(AT)
        Random.seed!(rng, 42)
        A1 = rand(AT{Float32}, 256)
        Random.seed!(rng, 42)
        A2 = rand(AT{Float32}, 256)
        @test Array(A1) == Array(A2)             # deterministic
    end

    # ── Float16 and Float64 element types ─────────────────────────────────
    @testset "element types" begin
        @test rand(AT{Float16}, 100) isa AT{Float16}
        @test rand(AT{Float64}, 100) isa AT{Float64}
        @test randn(AT{Float64}, 100) isa AT{Float64}
    end

    # ── Empty array ────────────────────────────────────────────────────────
    @testset "empty" begin
        A = rand(AT{Float32}, 0)
        @test length(A) == 0
        @test A isa AT
    end

    # ── Statistical correctness: KS test for uniformity ───────────────────
    @testset "uniformity" begin
        n = 10_000
        A = Array(rand(AT{Float32}, n))
        # Kolmogorov-Smirnov: max deviation from uniform CDF
        sorted = sort(A)
        D = maximum(abs.((1:n)/n .- sorted))
        @test D < 0.02   # KS critical value at 99% for n=10000 is ~0.013
    end
end
```

### Key Test Rationale

| Test | What It Catches |
|---|---|
| `A isa AT` | The fundamental bug — without fix, returns CPU Array |
| `0 .<= A .< 1` | RNG output range correctness |
| Reproducibility | Seed/counter threading works correctly |
| Statistical mean/std | Box–Muller implementation correctness |
| KS uniformity test | Xorshift128+ not degenerate (e.g., no bias to 0.0) |
| Float16/Float64 | Type dispatch in fromint() conversion |
| Empty array | `isempty` guard in rand!/randn! kernels |
