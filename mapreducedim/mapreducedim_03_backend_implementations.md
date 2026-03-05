# mapreducedim! — Backend Implementations

All four vendor backends override `mapreducedim!` at their specific array type. The conceptual algorithm (tiled shared-memory tree reduction) is the same across all four. What differs is the vendor-specific intrinsic for intra-group communication and synchronisation.

---

## CUDA.jl — [`src/mapreduce.jl`](https://github.com/JuliaGPU/CUDA.jl/blob/master/src/mapreduce.jl)

- Dispatch: `mapreducedim!(f, op, R::AnyCuArray, A)`
- Launch: `@cuda` with manually computed thread/block counts
- Intra-warp reduction: **warp shuffle** (`shfl_down_sync`) for the last 32 threads, avoiding shared memory bank conflicts
- Shared memory: `CuStaticSharedArray(T, blocksize)`
- Sync: `sync_threads()`
- Two-kernel strategy: `reduce_kernel` (per-block) + `reduce_block` (final warp reduction)

The warp shuffle optimisation is NVIDIA-specific — it uses hardware-level communication within a 32-thread warp without shared memory, reducing the number of synchronisation barriers in the final stages.

---

## AMDGPU.jl — [`src/mapreduce.jl`](https://github.com/JuliaGPU/AMDGPU.jl/blob/master/src/mapreduce.jl)

- Dispatch: `mapreducedim!(f, op, R::AnyROCArray, A)`
- Launch: `@roc` with `groupsize` and `gridsize`
- Intra-wavefront reduction: AMD wavefront operations (`__builtin_amdgcn_ds_swizzle`) for intra-wave communication
- Shared memory: `ROCDeviceArray` allocated in group memory
- Sync: `barrier()`
- AMD GPUs have 64-thread wavefronts (vs NVIDIA's 32-thread warps) — the wavefront reduction handles 64 threads without shared memory

---

## oneAPI.jl — [`src/mapreduce.jl`](https://github.com/JuliaGPU/oneAPI.jl/blob/master/src/mapreduce.jl)

- Dispatch: `mapreducedim!(f, op, R::oneArray, A)`
- Launch: `@oneapi` with `items` and `groups`
- Intra-subgroup reduction: Intel subgroup operations (`sub_group_reduce`) where available
- Shared memory: `oneLocalArray(T, groupsize)`
- Sync: `barrier(oneAPI.CLK_LOCAL_MEM_FENCE)`
- Intel GPUs have variable subgroup sizes (8, 16, or 32 depending on hardware) — the implementation queries this at runtime

---

## Metal.jl — [`src/mapreduce.jl`](https://github.com/JuliaGPU/Metal.jl/blob/master/src/mapreduce.jl)

- Dispatch: `mapreducedim!(f, op, R::WrappedMtlArray, A)`
- Launch: `@metal` with `threads` and `groups`
- Shared memory: `MtlThreadGroupArray(T, maxthreads)`
- Sync: `threadgroup_barrier(MemoryFlagThreadGroup)`
- No subgroup/warp intrinsics available in Metal compute shaders — pure shared-memory tree reduction only
- Uses `threads_per_threadgroup()` and `thread_position_in_threadgroup()` for indexing

---

## KernelAbstractions.jl Equivalents (PR #4)

| Vendor concept | KA equivalent |
|---------------|--------------|
| `CuStaticSharedArray(T, n)` | `@localmem T (n,)` |
| `sync_threads()` (CUDA) | `@synchronize()` |
| `barrier()` (AMDGPU) | `@synchronize()` |
| `barrier(CLK_LOCAL_MEM_FENCE)` (oneAPI) | `@synchronize()` |
| `threadgroup_barrier(...)` (Metal) | `@synchronize()` |
| `threadIdx().x` / `workitemIdx().x` | `@index(Local, Linear)` |
| `blockIdx().x` / `workgroupIdx().x` | `@index(Group, Cartesian)` |
| `blockDim().x` / `workgroupDim().x` | `@groupsize()[1]` |

KernelAbstractions.jl compiles each of these to the correct vendor intrinsic at specialisation time. The source is written once; the compiler handles the rest.

The **warp shuffle / wavefront operations** used by CUDA.jl and AMDGPU.jl for the final reduction stage have no KA equivalent — these are vendor-specific hardware features. The KA fallback uses pure shared-memory tree reduction for all stages. This is slightly less optimal for those two backends (they have their own implementations anyway) but fully correct and efficient for all others.

---

## Summary

| Backend | Dispatch type | Launch | Shared mem | Sync | Final stage |
|---------|:---:|:---:|:---:|:---:|:---:|
| CUDA.jl | `AnyCuArray` | `@cuda` | `CuStaticSharedArray` | `sync_threads()` | warp shuffle |
| AMDGPU.jl | `AnyROCArray` | `@roc` | group memory | `barrier()` | wavefront ops |
| oneAPI.jl | `oneArray` | `@oneapi` | `oneLocalArray` | `barrier(LOCAL)` | subgroup reduce |
| Metal.jl | `WrappedMtlArray` | `@metal` | `MtlThreadGroupArray` | `threadgroup_barrier` | tree only |
| **GPUArrays PR #4** | **`AnyGPUArray`** | **KA `@kernel`** | **`@localmem`** | **`@synchronize`** | **tree only** |
