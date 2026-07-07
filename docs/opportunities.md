# Opportunities to close the performance gap

Functional extensions are out of scope here.

## High Priority

| # | Opportunity | Expected Impact | Notes |
|---|-------------|----------------|-------|
| O1 | **Fix `run_tiles` correctness** | +23-30% decode | 8 lines of code. The speedup was measured but output was wrong. Root cause: likely Mojo closure capture or atomic ordering in `fetch_add` work-stealing. Debug: single-thread for-loop test, then `parallelize` with 1 thread, then `AtomicFence()` after quantize, then static partitioning. |
| O2 | **SoA + FP32 weight layout** | +5-10% | Separate weight data (256B stride, cache-line aligned) from FP32 scales. Eliminates ~50% cache-line boundary crossings. Reduces L1 miss rate from 13% toward ~8%. Requires `gguf_loader.mojo` repack + GEMV kernel adjustment. |
| O3 | **`run_token` pattern** | +10-15% | 1 `parallelize` call per token (all 97 kernels in sequence) instead of 97 separate calls. Eliminates ~14% runtime overhead from mutex barriers. Requires spin barrier implementation. |
| O4 | **Persistent thread pool** | +5-10% | Pin threads to physical cores with `sched_setaffinity`. Eliminate 15K to 90 context switches. Requires FFI to POSIX. |

## Medium Priority

| # | Opportunity | Expected Impact | Notes |
|---|-------------|----------------|-------|
| O5 | **mmap + huge pages** | +2-5% | `mmap(MAP_HUGETLB)` for ~290MB weight scan. Skip `fread`+`memcpy`. `echo always > /sys/kernel/mm/transparent_hugepage/enabled` |
| O6 | **CPU governor = performance** | +5-15% | Single shell command: `cpupower frequency-set -g performance` |
| O7 | **Prefetch distance tuning** | +2-5% | Currently 8 blocks (272 bytes) ahead. Try 4-8 blocks with `PREFETCHT0` or `PREFETCHT1`. |
| O8 | **Tile-affinity work stealing** | +2-3% | Randomize tile access order to reduce L1 conflicts. Only if O1 is fixed. |

## Low Priority / Architecture

| # | Opportunity | Expected Impact | Notes |
|---|-------------|----------------|-------|
| O9 | **SIMD rmsnorm / silu** | <1% | Non-GEMV ops are only 11% of runtime. Not worth the complexity. |
| O10 | **iGPU TileTensor dispatch** | Potentially large | CPU pointer path (validated) to GPU TileTensor path (planned). `modular_ref/` has dispatch patterns. Requires DeviceContext infrastructure. |
| O11 | **Conv1d transpose** | Enables stdlib SIMD | Transpose `[C,K]` into `[K,C]` at load time. 1.7MB total across 18 layers. Unblocks `modular_ref` causal_conv1d_update_cpu. |
| O12 | **Native Mojo tokenizer** | removes pip dep | BPE uses Python's `regex` module (a pip dependency, not stdlib). Port to native Mojo to drop the Python interop overhead and the `regex` install requirement. |

## Dependencies

```
O1 (run_tiles fix) <- prerequisite for O8 (tile-affinity)
O2 (SoA layout) <- independent, orthogonal to O1
O3 (run_token) <- alternative to O1; pick one
O4 (thread affinity) <- independent
O5 (mmap) <- independent
O6 (governor) <- independent, instant
O11 (conv1d transpose) <- unblocks stdlib conv1d SIMD
```

## System-Level Tuning (instant, no code changes)

```bash
# CPU governor
cpupower frequency-set -g performance

# Transparent huge pages
echo always > /sys/kernel/mm/transparent_hugepage/enabled

# Thread affinity (if FFI available)
taskset 0x0F ./qwen35   # pins to physical cores 0-3
```

## Architecture Evolution

Longer-term directions beyond single-token decode optimization:

| Direction | What | Reference |
|-----------|------|-----------|
| Batched prefill | Process multiple prompt tokens in parallel | prefill is 5-10× slower than ik_llama |
| Continuous learning | Persist DeltaNet state between sessions | `research/continuous_offline_learning_init.md` |
| Radix tree KV cache | Content-addressable KV reuse + DeltaNet state snapshots | `research/continuous_offline_learning_with_radixtree.md` |
| Online LoRA | Gradient updates on LoRA params at runtime | `research/` folder |
| Larger models | Scale to 2B/9B/27B where i8×i8 wins | crossover at DIM >= 2048 |
