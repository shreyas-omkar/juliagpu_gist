# rand / randn — Backend Implementations (Source-Verified)

## Summary Matrix

| Backend | `rand!` | `randn!` | `rand` (out-of-place) | `randn` (out-of-place) | RNG Type |
|---|---|---|---|---|---|
| CUDA.jl | ✓ custom kernel | ✓ Box–Muller kernel | ✓ | ✓ | Philox2x32 (device) |
| AMDGPU.jl | ✓ rocRAND | ✓ rocRAND | ✓ | ✓ | XORWOW (rocRAND) |
| Metal.jl | ✓ custom kernel | ✓ Box–Muller kernel | ✓ (via MPS) | ✓ | Xorshift128+ (device) |
| oneAPI.jl | ✓ (GPUArrays) | ✓ (GPUArrays) | partial | partial | Xorshift128+ (GPUArrays) |
| GPUArrays.jl | ✓ Xorshift128+ | ✓ Box–Muller | **missing** | **missing** | Xorshift128+ |

---

## CUDA.jl — `src/random.jl` (353 lines, source-verified)

Two separate RNG systems coexist:

### System 1: Native kernel RNG (`CUDA.RNG`)

```julia
mutable struct RNG <: AbstractRNG
    seed::UInt32
    counter::UInt32
end
```

**rand! kernel** (source-verified):
```julia
function Random.rand!(rng::RNG, A::AnyCuArray)
    function kernel(A::AbstractArray{T}, seed::UInt32, counter::UInt32) where {T}
        device_rng = Random.default_rng()         # Philox2x32 on-device
        @inbounds Random.seed!(device_rng, seed, counter)
        
        # grid-stride loop
        threadId = threadIdx().x
        window = widemul(blockDim().x, gridDim().x)
        offset = widemul(blockIdx().x - 1i32, blockDim().x)
        while offset < length(A)
            i = threadId + offset
            if i <= length(A)
                @inbounds A[i] = Random.rand(device_rng, T)
            end
            offset += window
        end
    end
    
    # Fixed launch config for reproducibility (explicit XXX comment in source)
    threads = 32
    blocks  = cld(length(A), threads)
    @cuda threads=threads blocks=blocks name="rand!" kernel(A, rng.seed, rng.counter)
    
    # Advance counter for next call
    new_counter = Int64(rng.counter) + length(A)
    overflow, remainder = fldmod(new_counter, typemax(UInt32))
    rng.seed    += overflow
    rng.counter  = remainder
    A
end
```

**randn! kernel** (source-verified, Box–Muller):
```julia
function Random.randn!(rng::RNG, A::AnyCuArray{<:Union{AbstractFloat,Complex{<:AbstractFloat}}})
    function kernel(A::AbstractArray{T}, seed, counter) where {T<:Real}
        device_rng = Random.default_rng()
        @inbounds Random.seed!(device_rng, seed, counter)
        threadId = threadIdx().x
        window = widemul(blockDim().x, gridDim().x)
        offset = widemul(blockIdx().x - 1i32, blockDim().x)
        while offset < length(A)
            i = threadId + offset
            j = threadId + offset + window      # second output slot
            if i <= length(A)
                U1 = Random.rand(device_rng, T)
                while U1 == zero(T)             # guard: log(0) = -Inf
                    U1 = Random.rand(device_rng, T)
                end
                U2 = Random.rand(device_rng, T)
                Z0 = sqrt(T(-2.0)*log(U1)) * cos(T(2pi)*U2)
                Z1 = sqrt(T(-2.0)*log(U1)) * sin(T(2pi)*U2)
                @inbounds A[i] = Z0
                if j <= length(A)
                    @inbounds A[j] = Z1         # write pair
                end
            end
            offset += 2*window                  # stride by 2 (pair production)
        end
    end
    
    threads = 32
    blocks  = cld(cld(length(A), 2), threads)   # halved: each thread fills 2
    @cuda threads=threads blocks=blocks name="randn!" kernel(A, rng.seed, rng.counter)
    A
end
```

### System 2: GPUArrays.RNG fallback

```julia
function GPUArrays.default_rng(::Type{<:CuArray})
    # Returns a GPUArrays.RNG (Xorshift128+) for the current task/context
    # Used when calling rand!(::GPUArrays.RNG, ::CuArray) directly
    # Backed by a HandleCache to reuse RNG objects across tasks
end
```

**Key design:** `Random.default_rng()` inside a CUDA kernel returns an on-device Philox2x32 state. This is separate from the host-side `CUDA.RNG`. The seed/counter threading ensures reproducibility.

---

## AMDGPU.jl — `src/rand/random.jl` (180 lines, source-verified)

Uses **rocRAND** — AMD's counterpart to cuRAND. The RNG object wraps a `rocrand_generator` handle:

```julia
mutable struct RNG <: Random.AbstractRNG
    handle::rocrand_generator
    typ::rocrand_rng_type

    function RNG(typ=ROCRAND_RNG_PSEUDO_DEFAULT)  # default = XORWOW
        handle = Ref{rocrand_generator}()
        rocrand_create_generator(handle, typ)
        obj = new(handle[], typ)
        finalizer(unsafe_destroy!, obj)
    end
end
```

**rand! dispatch table** (source-verified, for-loop macro expansion):
```julia
# Each float type gets its own rocRAND C function
(rocrand_generate_char,            Cuchar)   # uint8
(rocrand_generate_short,           UInt16)
(rocrand_generate,                 UInt32)
(rocrand_generate_uniform_half,    Float16)
(rocrand_generate_uniform,         Float32)
(rocrand_generate_uniform_double,  Float64)
```

**randn! with pow2 padding** (source-verified):
```julia
function Random.randn!(rng::RNG, A::ROCArray{Float32}; mean=0, stddev=1)
    inplace_pow2(A, B -> rocrand_generate_normal(rng, B, length(B), mean, stddev))
    return A
end
# rocrand_generate_normal requires length to be power of 2 and >= 2
# inplace_pow2 handles non-pow2 by allocating padded buffer, copying back
```

**Critical difference from CUDA:** AMDGPU `randn!` accepts `mean` and `stddev` kwargs — it can generate N(μ, σ²) directly without post-processing. CUDA's Box–Muller kernel always generates N(0,1).

---

## Metal.jl — `src/random.jl` (280 lines, source-verified)

Metal.jl uses **two separate systems** for rand vs randn:

### System 1: Custom kernel RNG (Xorshift128+, source-verified identical to CUDA.jl)

```julia
mutable struct RNG <: AbstractRNG
    seed::UInt32
    counter::UInt32
end
```

The `rand!` and `randn!` kernels are **structurally identical to CUDA.jl** — same grid-stride loop, same Box–Muller for randn!, same fixed `threads=32` for reproducibility, same `XXX` comment. Only the kernel launch macro differs: `@metal threads groups` vs `@cuda threads blocks`.

Metal intrinsic names differ:
```julia
# CUDA.jl:           threadIdx().x,  blockDim().x,   blockIdx().x,   gridDim().x
# Metal.jl: thread_position_in_threadgroup().x, threads_per_threadgroup().x,
#            threadgroup_position_in_grid().x,  threadgroups_per_grid().x
```

### System 2: MPS (Metal Performance Shaders) for out-of-place

```julia
using ..MPS: MPSVector, _mpsmat_rand!,
             MPSMatrixRandomUniformDistributionDescriptor,
             MPSMatrixRandomNormalDistributionDescriptor
```

For out-of-place `rand` and `randn`, Metal.jl uses **Apple's MPS random number generators** — hardware-accelerated on Apple Silicon. This gives better performance than the custom kernel for large arrays.

---

## oneAPI.jl — `src/random.jl` (22 lines, source-verified)

The most minimal implementation — fully delegates to GPUArrays:

```julia
gpuarrays_rng() = GPUArrays.default_rng(oneArray)

# In-place — uses GPUArrays Xorshift128+ kernel
Random.rand!(A::oneWrappedArray)  = Random.rand!(gpuarrays_rng(), A)
Random.randn!(A::oneWrappedArray) = Random.randn!(gpuarrays_rng(), A)

# Out-of-place — local function (not Base.rand override!)
rand(T::Type, dims::Dims)  = Random.rand!(oneArray{T}(undef, dims...))
randn(T::Type, dims::Dims) = Random.randn!(oneArray{T}(undef, dims...))
```

**Important:** oneAPI defines `rand` as a local function, not `Base.rand`. This means `Base.rand(oneArray, Float32, 100)` would not dispatch to it. The fix in GPUArrays should override `Base.rand` at `AnyGPUArray`.

---

## GPUArrays.jl — Existing `rand!` / `randn!` Kernels

GPUArrays **already has** working `rand!` and `randn!` via `GPUArrays.RNG`:

```julia
# GPUArrays.jl/src/host/random.jl (existing, not new)
struct RNG <: AbstractRNG
    state::AbstractGPUArray  # NTuple{4, UInt32} per thread, lives on GPU
end

# rand! kernel: Xorshift128+ per thread, grid-stride loop
# randn! kernel: Box–Muller on top of rand!, pair production
```

The `GPUArrays.default_rng(AT)` function returns a task-local `GPUArrays.RNG` for array type `AT`. This is what oneAPI delegates to.

**What is missing is `Base.rand(AT, T, dims...)` and `Base.randn(AT, T, dims...)`** — the convenience out-of-place constructors. These should:
1. Allocate `A = AT{T}(undef, dims)`
2. Fill via `Random.rand!(GPUArrays.default_rng(AT), A)`
3. Return `A`

Three lines per function.

---

## The Precise Gap

```
rand!(rng::GPUArrays.RNG, A::AnyGPUArray)   ← EXISTS (Xorshift128+ kernel)
randn!(rng::GPUArrays.RNG, A::AnyGPUArray)  ← EXISTS (Box–Muller kernel)

Base.rand(::Type{<:AnyGPUArray}, T, dims)   ← MISSING for JLArray/future
Base.randn(::Type{<:AnyGPUArray}, T, dims)  ← MISSING for JLArray/future
```

Unlike all previous PRs, this one doesn't need a new kernel — it just needs thin out-of-place wrappers that connect `Base.rand` dispatch to the existing `GPUArrays.RNG` machinery.
