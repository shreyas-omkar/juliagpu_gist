# `rand` / `randn` Failure Demonstration, Fix, Performance, and Tests

## SECTION 3: Failure Demonstration

### A Unique Failure Mode

For operations such as `reverse`, `sort!`, and `mapreducedim!`, the absence of a GPU fallback results in either a scalar indexing error or a hard `Not Implemented` error. For `rand` and `randn`, the failure mode is uniquely silent and dangerous:

**The in-place kernels `rand!(A)` and `randn!(A)` already function correctly on all backends** (confirmed by the `[CAOM G]` flag in the source audit). The architectural gap exists entirely in the **out-of-place allocation forms**.

### Failure Mode 1: JLArray (Test Backend / Future Backends)

~~~julia
using GPUArrays, JLArrays

# rand! works, the GPUArrays kernel exists and executes properly
A = jl(zeros(Float32, 1000))
rand!(A)          # ✓ Works: utilizes GPUArrays.RNG Xorshift128+
randn!(A)         # ✓ Works: utilizes GPUArrays.RNG Box-Muller

# Out-of-place rand FAILS
B = rand(JLArray{Float32}, 1000)
# ERROR: MethodError: no method matching rand(::Type{JLArray{Float32,1}}, ::Int64)
# Closest candidates: rand(::AbstractRNG, ...), rand(::Type{Float32}, ...)
# → Falls through to Base.rand(Float32, 1000) → returns a CPU Array!

# Silent wrong type — no error thrown, but incorrect memory residency:
C = rand(Float32, 1000)     # This executes Base.rand on the CPU
typeof(C)                   # Array{Float32,1} — not on the GPU at all
~~~

### Failure Mode 2: Silent Degradation to CPU

The most critical consequence of this gap occurs when user code intends to initialize a GPU array but silently receives a CPU array:

~~~julia
# Intended GPU random array initialization:
function init_weights(n::Int)
    W = rand(Float32, n, n)   # ← INTENDED: GPU-allocated random weights
    return W
end

W = init_weights(1000)
typeof(W)  # Array{Float32,2} CPU allocation. No error is thrown.

# Later in the computational pipeline:
loss = sum(model(W))          # W is on CPU → Silent performance disaster,
                              # or a Type Error when the model strictly expects a GPU array.
~~~

### Failure Mode 3: oneAPI Module-Local Conflict

~~~julia
using oneAPI

# oneAPI.jl defines a local rand function, not a Base.rand method override.
# Therefore, standard Base.rand dispatch misses it entirely:
A = Base.rand(oneArray, Float32, 100)
# MethodError: The oneAPI.jl rand() is a module-local function.

# Only a module-qualified call succeeds:
A = oneAPI.rand(Float32, 100)   
# Generic ecosystem code calling `rand(Float32, 100)` on an Intel GPU falls through to the CPU.
~~~

### Dispatch Trace for `rand(JLArray{Float32}, 1000)`

~~~text
rand(JLArray{Float32}, 1000)

Julia dispatch search:
  JLArrays.jl        → no rand method found
  GPUArrays.jl       → no rand method found
  Base.rand          → Base.rand(::Type{Float32}, ::Int64) FOUND ← INCORRECT TARGET
  
Result: CPU Float32 array, no warning, no error.
~~~

The type parameter is completely ignored. Passing `JLArray{Float32}` as a type argument to `rand` does not trigger any GPU-specific dispatch.

---

## SECTION 4: The Fix

### Design Decisions

**1. Identification of the Gap:** Only the out-of-place wrappers `Base.rand` and `Base.randn` at the `AnyGPUArray` level are missing. The underlying PRNG kernels already exist and are fully functional.

**2. Type Signature Expression:** The dispatch mechanism must accept an array type as a value, not an instance. This necessitates the implementation of the `rand(::Type{<:AnyGPUArray}, T, dims)` pattern.

**3. Default Element Type:** Following Julia `Base` and standard machine learning conventions, `Float32` is established as the default element type.

**4. Implementation Locus:** The wrappers are placed alongside the existing `rand!` kernel in `GPUArrays.jl/src/host/random.jl`.

**5. Dependency Profile:** Unlike the sorting and reduction PRs (which utilize `AcceleratedKernels.jl`), this implementation requires no new dependencies.

### Complete Implementation

**FILE**: `GPUArrays.jl/src/host/random.jl` (ADDITIONS to existing file)

~~~julia
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
~~~

### Dispatch Table After Fix

~~~text
BEFORE:
rand(JLArray{Float32}, 100)    → Base.rand(Float32, 100) ← CPU! Wrong type
randn(JLArray{Float32}, 100)   → Base.randn(Float32,100) ← CPU! Wrong type
rand(CuArray{Float32}, 100)    → ✓ (CUDA.jl has own method)
rand(ROCArray{Float32}, 100)   → ✓ (AMDGPU.jl has own method)
rand(oneArray{Float32}, 100)   → only via oneAPI.rand() — not Base.rand
rand(MtlArray{Float32}, 100)   → ✓ (Metal.jl has MPS method)

AFTER:
rand(JLArray{Float32}, 100)    → GPUArrays.rand → default_rng → Xorshift128+ ✓
randn(JLArray{Float32}, 100)   → GPUArrays.randn → default_rng → Box Muller ✓
rand(CuArray{Float32}, 100)    → CUDA.jl (unchanged, more specific) ✓
rand(ROCArray{Float32}, 100)   → AMDGPU.jl (unchanged) ✓
rand(oneArray{Float32}, 100)   → GPUArrays fallback (now via Base.rand) ✓
rand(<future>{Float32}, 100)   → GPUArrays fallback ✓
~~~

---

## SECTION 5: Performance Analysis

### Theoretical Hardware Bounds (Write-Only vs. Arithmetic-Bound)

**`rand!` (Uniform):** The `Xorshift128+` generator is computationally inexpensive (4 XOR operations + 1 addition per sample). 
* For `Float32`, each thread produces one 32-bit sample per iteration.
* The operation is strictly memory-bound (pure write, 4 bytes per element).
* At 360 GB/s theoretical write bandwidth, execution time is modeled as $n \times 4 / 360\times10^9$ seconds.

**`randn!` (Normal):** The Box-Muller transform requires transcendental functions (`log`, `sqrt`, `cos`, `sin`). 
* GPU transcendentals execute at approximately ~4 ns each.
* The algorithm uses pair-production (4 transcendentals per pair of outputs $\rightarrow$ 2 transcendentals per element).
* At 4000 GFLOP/s transcendental throughput (RTX 3060), computational time is modeled as $2 \times n / 4\times10^{12}$ seconds.
* For $n=10^7$, arithmetic time is $\approx \mathbf{0.005}$ **ms**.
* Memory time is $n \times 4 / 360\times10^9 = \mathbf{0.11}$ **ms**.
* **Conclusion:** Even with heavy transcendentals, `randn!` remains bottlenecked by memory bandwidth, not arithmetic throughput.

### Empirical Benchmarks

Because this PR rectifies an architectural dispatch gap, the silent CPU failure mode is benchmarked against the exposed `GPUArrays.RNG` (`Xorshift128+`) fallback and the native `CUDA.jl` (`Philox`) kernel.

| Size | CPU Silent Failure (ms) | Fallback `rand` (ms) | `CUDA` Philox (ms) | Ratio Fallback/CPU |
|:---:|:---:|:---:|:---:|:---:|
| 100\,K | [DATA] | [DATA] | [DATA] | [RATIO]$\times$ |
| 1\,M | [DATA] | [DATA] | [DATA] | [RATIO]$\times$ |
| 10\,M | [DATA] | [DATA] | [DATA] | [RATIO]$\times$ |
| 50\,M | [DATA] | [DATA] | [DATA] | [RATIO]$\times$ |
| 100\,M | [DATA] | [DATA] | [DATA] | [RATIO]$\times$ |

**Important Context:** The "speedup" demonstrated for this out-of-place fix is not purely "GPU vs CPU"; it represents "correct GPU execution" versus "incorrect CPU degradation." The prior state does not simply execute slowly it returns the wrong array type entirely. The fallback ensures correct return types and establishes immediate GPU residency for all subsequent operations.

---

## SECTION 6: Tests

~~~julia
@testsuite "random" (AT, eltypes) -> begin

    @testset "rand!" begin
        A = AT{Float32}(undef, 1024)
        rand!(A)
        @test all(0 .<= Array(A) .< 1)           # U(0,1) range
        @test length(unique(Array(A))) > 900     # Ensures variance
    end

    @testset "randn!" begin
        A = AT{Float32}(undef, 1024)
        randn!(A)
        v = Array(A)
        @test abs(mean(v)) < 0.1                 # mean ≈ 0
        @test abs(std(v) - 1.0) < 0.1            # std ≈ 1
    end

    @testset "rand out-of-place" begin
        A = rand(AT{Float32}, 1024)
        @test A isa AT                           # Correct type residency
        @test all(0 .<= Array(A) .< 1)
        
        # Variadic dimension forms
        A2 = rand(AT{Float32}, 32, 32)
        @test size(A2) == (32, 32)
        @test A2 isa AT
        
        # Default Float32 fallback
        A3 = rand(AT, 100)
        @test eltype(A3) == Float32
    end

    @testset "randn out-of-place" begin
        A = randn(AT{Float32}, 1024)
        @test A isa AT                           
        v = Array(A)
        @test abs(mean(v)) < 0.15
        @test abs(std(v) - 1.0) < 0.15
        
        # 2D initialization
        A2 = randn(AT{Float32}, 64, 64)
        @test size(A2) == (64, 64)
    end

    @testset "reproducibility" begin
        rng = GPUArrays.default_rng(AT)
        Random.seed!(rng, 42)
        A1 = rand(AT{Float32}, 256)
        Random.seed!(rng, 42)
        A2 = rand(AT{Float32}, 256)
        @test Array(A1) == Array(A2)             
    end

    @testset "element types" begin
        @test rand(AT{Float16}, 100) isa AT{Float16}
        @test rand(AT{Float64}, 100) isa AT{Float64}
        @test randn(AT{Float64}, 100) isa AT{Float64}
    end

    @testset "empty" begin
        A = rand(AT{Float32}, 0)
        @test length(A) == 0
        @test A isa AT
    end

    @testset "uniformity" begin
        n = 10_000
        A = Array(rand(AT{Float32}, n))
        # Kolmogorov-Smirnov: max deviation from uniform CDF
        sorted = sort(A)
        D = maximum(abs.((1:n)/n .- sorted))
        @test D < 0.02   # KS critical value at 99% for n=10000 is ~0.0163
    end
end
~~~

### Key Test Rationale

| Test | Verification Target |
|---|---|
| `A isa AT` | Verifies the fundamental bug is resolved; ensures a GPU Array is returned rather than a CPU Array. |
| `0 .<= A .< 1` | Confirms RNG output range correctness. |
| Reproducibility | Ensures deterministic seed/counter threading functions correctly across all parallel workers. |
| Statistical mean/std | Verifies mathematical correctness of the Box-Muller transform implementation. |
| KS uniformity test | Verifies `Xorshift128+` maintains distribution integrity (e.g., lacks bias toward 0.0). |
| Float16/Float64 | Confirms correct type dispatch during the integer-to-float conversion phase. |
| Empty array | Validates the `isempty` guard logic within the `rand!` and `randn!` kernels. |
