# rand / randn   Algorithm Deep Dive

## What These Functions Do

| Function | Signature | Output |
|---|---|---|
| `rand!(rng, A)` | Fills GPU array `A` in-place with U(0,1) samples | `A` |
| `randn!(rng, A)` | Fills GPU array `A` in-place with N(0,1) samples | `A` |
| `rand(T, dims...)` | Allocates new GPU array, fills U(0,1) | new array |
| `randn(T, dims...)` | Allocates new GPU array, fills N(0,1) | new array |

The default element type is `Float32` across all backends   matching GPU-native precision and ML conventions.

---

## The Uniqueness of rand/randn vs All Other Operations

Every other operation in this project (reverse, accumulate, findall, mapreducedim, sort) reads input → transforms → writes output. **rand and randn have no input array at all.** They are pure generation   each thread must independently produce a statistically independent random value from a shared seed.

This creates a fundamentally different problem: **how do you give n threads n different, statistically independent random numbers without communication?**

The answer is a **counter-based PRNG** where each thread's state is a deterministic function of (seed, thread_index, counter). No synchronization, no shared state, perfect parallelism.

---

## Algorithm 1: GPUArrays.jl Built-in RNG (Xorshift128+)

Source: `GPUArrays.jl/src/host/random.jl`   this is the only operation where **GPUArrays already has a working kernel** (`[CAOM G] rand!`). The `G` flag in the audit confirms this.

GPUArrays implements a **Xorshift128+** PRNG. This is the same algorithm used in JavaScript V8 and many browser implementations   extremely fast, good statistical quality for most purposes, 128-bit state.

### Xorshift128+ Algorithm

```
State: (s0::UInt64, s1::UInt64)    # 128 bits total

Step:
  t  = s0
  s0 = s1
  t  = t ⊕ (t << 23)   # xorshift
  t  = t ⊕ (t >> 17)   # xorshift  
  t  = t ⊕ s0           # mix with s0
  t  = t ⊕ (s0 >> 26)  # xorshift
  s1 = t
  return s0 + t          # + gives better statistical mixing than ⊕

Period: 2^128 - 1
Output: 64-bit integer, converted to float via IEEE bit manipulation
```

### Per-Thread State Initialization

```julia
# GPUArrays RNG: NTuple{4, UInt32} state per thread
# State array has one 4×UInt32 tuple per thread (max threads per block)

@kernel function rand_kernel!(state, A::AbstractArray{T}) where T
    i = @index(Global, Linear)
    
    # Each thread gets its own 128-bit state slot
    # Seeded from CPU: state[threadid] = hash(seed, threadid)
    rng_state = @inbounds state[mod1(i, length(state))]
    
    # Generate one sample per thread, grid-stride for large arrays
    offset = i
    while offset <= length(A)
        rng_state, val = xorshift128plus(rng_state)
        @inbounds A[offset] = fromint(T, val)
        offset += total_threads
    end
    
    @inbounds state[mod1(i, length(state))] = rng_state
end
```

The `state` array is stored in a `GPUArrays.RNG` object   `NTuple{4, UInt32}` per thread position, sized to `MAX_THREADS_PER_BLOCK` (typically 1024). Each task gets its own `RNG` via `GPUArrays.default_rng(ArrayType)`.

### Float Conversion

Generating a uniform float from an integer:
```julia
# Convert 64-bit random integer to Float32 in [0, 1)
# Method: set exponent bits to 127 (i.e. 1.0), randomize mantissa, subtract 1.0
# This gives perfect uniform coverage of all representable floats in [0,1)
function fromint(::Type{Float32}, x::UInt64)::Float32
    # Take top 23 bits for the mantissa
    mantissa = UInt32(x >> 41)
    # Set exponent to 127 (= 0x3F800000 = 1.0f)
    bits = mantissa | 0x3F800000
    return reinterpret(Float32, bits) - 1.0f0
end
```

---

## Algorithm 2: Box Muller Transform for randn!

The **Box Muller transform** converts two independent U(0,1) samples into two independent N(0,1) samples. It is exact (not approximate), numerically stable, and requires only `log`, `sqrt`, `cos`, `sin`   all of which are single GPU instructions.

### Mathematical Derivation

Given U1, U2 ~ U(0,1), independent:

```
Z0 = sqrt(-2 * ln(U1)) * cos(2π * U2)
Z1 = sqrt(-2 * ln(U1)) * sin(2π * U2)
```

Z0 and Z1 are **independent standard normal** samples.

**Why this works:** The probability transform theorem + the 2D Gaussian's rotational symmetry. The radius `r = sqrt(-2 ln(U1))` comes from the Rayleigh distribution (which is the radial component of a 2D Gaussian), and the angle `θ = 2π * U2` is uniform on the circle.

### Concrete Example

```
U1 = 0.3174    U2 = 0.7854
ln(0.3174) = -1.148
sqrt(-2 * -1.148) = sqrt(2.296) = 1.515

Z0 = 1.515 * cos(2π * 0.7854) = 1.515 * cos(4.934) = 1.515 * 0.2588 = 0.392
Z1 = 1.515 * sin(2π * 0.7854) = 1.515 * sin(4.934) = 1.515 * (-0.966) = -1.463

Both Z0, Z1 ~ N(0,1) ✓
```

### GPU-Specific Optimization: Pair Production

Because Box Muller produces **two** normals per call, the kernel can write to positions `i` and `j = i + stride` simultaneously, halving the number of RNG calls needed and maximizing arithmetic intensity:

```julia
# From CUDA.jl and Metal.jl randn! kernel (source-verified):
# grid-stride loop produces pairs
while offset < length(A)
    i = threadId + offset
    j = threadId + offset + window   # second output position
    
    U1 = rand(device_rng, T)
    while U1 == zero(T)              # guard: log(0) = -Inf
        U1 = rand(device_rng, T)
    end
    U2 = rand(device_rng, T)
    
    Z0 = sqrt(T(-2.0) * log(U1)) * cos(T(2π) * U2)
    Z1 = sqrt(T(-2.0) * log(U1)) * sin(T(2π) * U2)
    
    A[i] = Z0
    if j <= length(A)
        A[j] = Z1              # write second sample if in bounds
    end
    
    offset += 2 * window       # advance by 2× stride (pair consumption)
end
```

The `while U1 == zero(T)` guard prevents `log(0) = -Inf`. In practice this fires with probability 2^-24 for Float32   essentially never, but required for correctness.

### Complex randn!

For `Complex{T}` arrays (used in signal processing and quantum simulation):

```julia
# Complex Box Muller (from CUDA.jl and Metal.jl source):
Z0 = sqrt(-log(U1)) * cos(T(2π) * U2)   # note: no factor of 2 in sqrt
Z1 = sqrt(-log(U1)) * sin(T(2π) * U2)
A[i] = complex(Z0, Z1)
# Each complex element has real and imaginary parts both ~ N(0, 1/2)
# → |A[i]|² ~ Exponential(1)   correct for complex Gaussian
```

---

## Algorithm 3: Hardware RNG Libraries (CUDA/AMDGPU)

### CUDA: cuRAND + Philox2x32

CUDA.jl's primary RNG for rand! is **not** Xorshift128+. It uses cuRAND (NVIDIA's proprietary random number library) with the **Philox2x32** counter-based PRNG when called via `CUDA.CURAND.RNG`. However, the *native* CUDA.jl RNG (used when no explicit RNG is specified) uses a custom kernel with `Random.default_rng()` on-device   which maps to a Philox2x32 counter.

Philox2x32 is a **counter-based** PRNG from Random123:
```
State: (key, counter)    # both UInt32
Output: 2×UInt32 per step via Feistel network

Step (Philox2x32-10 = 10 rounds):
  for r in 1:10:
    hi, lo = mulhilo32(PHILOX_M2x32_0, counter[0])
    counter = (hi ^ key[0] ^ counter[1], lo)
    key = (key[0] + PHILOX_W32_0,)
  return counter
```

Key property: **skippable**   you can jump to the state at position N in O(1) time. This lets each thread start at `counter = thread_global_index` without any communication.

### AMDGPU: rocRAND

AMDGPU.jl uses **rocRAND**   AMD's counterpart to cuRAND, with identical high-level API:

```julia
# Source-verified from AMDGPU.jl/src/rand/random.jl
# Dispatch table for rand!:
rocrand_generate_uniform(rng, A, length(A))       # Float32
rocrand_generate_uniform_double(rng, A, length(A)) # Float64
rocrand_generate_uniform_half(rng, A, length(A))   # Float16

# For randn!:
rocrand_generate_normal(rng, A, length(A), mean, stddev)        # Float32
rocrand_generate_normal_double(rng, A, length(A), mean, stddev)  # Float64
rocrand_generate_normal_half(rng, A, length(A), mean, stddev)    # Float16
```

AMDGPU uses rocRAND's `ROCRAND_RNG_PSEUDO_DEFAULT` which defaults to **XORWOW** (a Marsaglia XOR-with-Weyl-sequence generator, period 2^192 - 2^32).

**Critical constraint:** rocRAND's normal distribution generator (`rocrand_generate_normal`) requires the output length to be a **power of 2** (≥ 2). AMDGPU.jl wraps this with `inplace_pow2()`:

```julia
function inplace_pow2(A, f)
    len = length(A)
    if len > 1 && ispow2(len)
        f(A)                              # fast path: in-place
    else
        padlen = max(2, nextpow(2, len))
        B = similar(A, padlen)            # allocate padded buffer
        f(B)                              # fill padded buffer
        copyto!(A, 1, B, 1, len)          # copy first len elements
        AMDGPU.unsafe_free!(B)            # free scratch
    end
    A
end
```

This means `randn!(A)` on an AMDGPU array of non-power-of-2 length silently allocates a larger temporary buffer. For n=10^6 (not a power of 2), padlen=2^20=1,048,576   only 4.8% overhead.

---

## What GPUArrays' `rand!` and `randn!` Already Do

The audit flag `[CAOM G] rand!` and `[CAOM G] randn!` means:
- `C` = CUDA has its own method (overrides GPUArrays for CuArray)
- `A` = AMDGPU has its own method (rocRAND)
- `O` = oneAPI uses GPUArrays' method (`gpuarrays_rng()`   confirmed by source)
- `M` = Metal has its own method (custom kernel, confirmed by source)
- `G` = GPUArrays provides the fallback kernel

The `G` flag means **GPUArrays already has working `rand!` and `randn!` kernels** using Xorshift128+. These serve JLArray (test backend), oneAPI, and any future backend.

**The gap is the out-of-place functions:** `[CA-M -] rand` and `[CA-M -] randn`   the `-` for oneAPI and the `-` for future backends means `rand(T, dims...)` (allocate + fill) is missing.

But looking at oneAPI's source: it **does** define `rand` and `randn` as local functions (not `Base.rand` overrides). For JLArray and future backends, there's no `rand(jl_arr_type, T, dims...)` dispatch at all.

---

## Why rand/randn are Architecturally Different from the Other PRs

All other operations transform existing data. rand/randn **create data from nothing**. The architectural difference:

1. **No input bandwidth**   no data to read from GPU memory, only writes
2. **Arithmetic-bound, not memory-bound**   each thread must compute `log`, `sqrt`, `cos`, `sin` (randn)   these dominate, not memory
3. **Task-local state**   the RNG state is per-task (not global), stored in `task_local_storage()`
4. **Reproducibility requirement**   same seed + same array size must always produce the same values, regardless of kernel launch configuration (noted with `XXX` comment in both CUDA.jl and Metal.jl sources)

This reproducibility concern   explicitly noted in source comments   drives the fixed `threads=32` choice in CUDA.jl and Metal.jl. Using a variable thread count would make results depend on GPU hardware capacity.
