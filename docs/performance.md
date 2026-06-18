# Performance Engineering — QuickQwen

---

## Hardware Profile

| Property | Value |
|----------|-------|
| CPU | Intel i7-10610U (4 physical / 8 HT cores) |
| SIMD | AVX2 + FMA3 + F16C (**no AVX-512, no VNNI**) |
| Memory | DDR4-3200, ~42 GB/s theoretical |
| YMM registers | 16 (tight for 8-independent-chain kernels) |

---

## Performance Baseline (v13 vs ik_llama.cpp)

> **Note:** This section benchmarks against [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp),
> the optimized fork studied during development. The README's headline table uses **stock
> llama.cpp** instead — a different (slower) baseline.

| Metric | QuickQwen v13 | ik_llama.cpp | Ratio |
|--------|---------------|--------------|-------|
| Decode tok/s | 11.6–15.5 | 16.6–19.2 | 1.3–1.5× |
| Prefill tok/s | ~12 | ~18 | ~1.5× |
| IPC | 0.87 | 1.08 | 0.81× |
| L1-dcache miss | 12.9% | 8.0% | 1.61× |
| Context switches | 15,534 | 90 | 173× |
| Instructions | 43.8B | 39.9B | 1.10× |
| Binary size | 303 KB | ~60 MB | — |

Numbers vary by CPU governor (powersave ~898 MHz vs performance).

---

## Bottleneck Breakdown (v13 single token)

| Component | Share | Detail |
|-----------|-------|--------|
| GEMV kernels | ~49% | all matmul operations |
| Sequential ops | ~11% | norms, activations, residual |
| Runtime overhead | ~14% | 97× `parallelize` scheduling |
| `run_tiles` wrapper | ~5% | barrier management |
| Other | ~21% | tokenizer, init, pointer chasing |

Output projection (248320×1024) alone = 52.5% of FLOPs. Dominant bottleneck.

---

## GEMV Kernel: W8A8 SignedDot (v12+)

Per-block (32 int8 values), per-row instruction sequence:

```
1. Load 32 int8 weights from Q8_0 block
2. Load 32 int8 from quantized activation (x_q)
3. vpsignb  — sign manipulation for signed×signed
4. vpmaddubsw — int8×int8 → int16 multiply-add (16→16)
5. vpmaddwd   — int16×int16 → int32 multiply-add (8→8)
6. fp16_weight_scale × fp32_activation_scale → float32
7. Accumulate into per-row float32 sum
```

**Key insight**: 3 AVX2 intrinsics replace ~20 float32 instructions. +2.5× over f32×f32 GEMV.

Activation quantization (`quantize_f32_to_q8_0`): quantizes x once per GEMV call, reuses across all rows. 0.7% relative error vs f32.

### SIMD Intrinsics (AVX2)

| Mojo/LLVM Intrinsic | x86 Mnemonic | Operation |
|---------------------|-------------|-----------|
| `llvm.x86.avx2.psign.b` | `vpsignb` | Conditional negate by sign |
| `llvm.x86.avx2.pmadd.ub.sw` | `vpmaddubsw` | uint8×int8 → int16 madd |
| `llvm.x86.avx2.pmadd.wd` | `vpmaddwd` | int16×int16 → int32 madd |

Reference: `modular_ref/max/kernels/src/linalg/arch/cpu/vnni_intrinsics.mojo:185` — `_dot_i8_to_i32_8`.
**Beware**: this reference is uint8×int8, not int8×int8. Sign-extend both operands first.

---

## Weight Layout: Q8_0 Tiled (T=8)

```
Q8_0 block: [2B fp16 scale | 32B int8 values] = 34 bytes
Tile:       32 blocks × 8 rows interleaved = 8704 bytes (for DIM=1024)
```

8 consecutive rows' blocks interleaved → x-vector loaded once, reused across 8 rows.
Memory overhead: ~772 MB for repacked copies.

Load-time repacking in `gguf_loader.mojo` (+27 lines). +11% decode speedup.

---

## Quantization Crossover

Activation quantization to int8 helps when x doesn't fit in L1.

| Model | DIM | x size (f32) | Fits L1 (32KB)? | i8×i8 Verdict |
|-------|-----|-------------|-----------------|---------------|
| 0.8B | 1024 | 4 KB | Yes | **Slower** — overhead > bandwidth savings |
| ~2B | 2048 | 8 KB | Yes | Break-even |
| ~9B | 4096 | 16 KB | Borderline | Starts winning |
| 27B+ | 5120+ | 20+ KB | No | Clear win |

For 0.8B: f32×f32 GEMV is faster than i8×i8 despite lower arithmetic density.

---

## Fusion Patterns

| Pattern | What | Barriers Saved | Speedup |
|---------|------|---------------|---------|
| DN Mega-Projection | 3 `parallelize` → 1 (6144+16+16=6176 items) | 36/token | +42%* |
| Inter-layer chaining | `elem_add_rmsnorm()` fuses residual+norm | 49 points | included above |
| RMSNormGated | 4-pass → 2-pass per DN head | 18→8 per DN layer | included above |
| Streaming lm_head | Per-thread min-heap top-k, no 248K-logit buffer | ~1MB saved | +20% |
| Online softmax | FlashAttention-style single-pass, no 16KB score buffer | memory only | +14% |
| 16-row unrolling | Two consecutive T=8 tiles share x load | memory traffic 2× | +22% |

*Caveat: ±30% variance on short runs (4 decode tokens). Individual gains don't stack multiplicatively.

---

## Thread Parallelism

| Aspect | Detail |
|--------|--------|
| Mojo `parallelize` | Static chunk scheduling, mutex/condvar barriers |
| Per token | 97+ `parallelize` calls → 15,534 context switches (vs ik_llama 90) |
| `num_physical_cores()` | Pin to 4 physical cores, not 8 HT — avoids cache thrashing |
| Candidate: `run_tiles` | Atomic `fetch_add` work-stealing, +23-30% measured **but output incorrect** — correctness bug in closure capture / atomic ordering |
| Candidate: `run_token` | 1 `parallelize` per token, spin barriers — cleanest design |

---

## Profiling Methodology

```bash
# Hardware counters
perf stat -e cycles,instructions,cache-misses,context-switches ./quickqwen

# Hotspot analysis
perf record -g ./quickqwen && perf report

# Compare vs reference
perf stat -e cycles,instructions,cache-misses,context-switches \
    ik_llama -m model.gguf -c 4096 -p "Hello" -n 50
```

Key metrics to compare against ik_llama.cpp:
1. **IPC** (instructions per cycle) — codegen quality
2. **L1-dcache miss rate** — memory layout effectiveness
3. **Context switches** — threading overhead
4. **Total instructions** — kernel efficiency

---

## Memory Hierarchy Optimization

| Technique | Effect | Status |
|-----------|--------|--------|
| Software prefetch 8 blocks ahead | −5% when **removed** → keep it | applied |
| Row-interleaved T=8 repacking | x reuse across 8 rows | applied |
| Streaming lm_head top-k | no 1MB logits buffer | applied |
| SoA + FP32 weight layout | reduce L1 miss 13%→8%: separate weights (256B stride, cache-line aligned) + FP32 scales | not yet |
| mmap + huge pages | skip fread+memcpy for weight loading | not yet |

Q8_0 34-byte block causes ~50% cache-line boundary crossings. SoA layout eliminates this.

---

## Performance Anti-Patterns

| Anti-pattern | Why it fails |
|-------------|-------------|
| FMA in GEMV accumulation | 8 independent chains already hide mul+add latency — 0% gain |
| `exp_approx_f32` for silu | 1.4% error causes early EOS — don't approximate activations |
| x4 layout (sum4 aggregation) | 128 live YMM values overflows 16 registers → 18% slower |
| Pre-allocating quant buffers | Modern allocators serve small allocs in ~50-100ns from thread-local cache — 0% gain |
| Multi-row GEMV for DIM=1024 | x (4KB) already in L1 — no reuse benefit until DIM≥2048 |

---

## Mojo Performance Gotchas

| # | Issue | Impact |
|---|-------|--------|
| P1 | `parallelize` creates thread pool internally but uses mutex barriers | 173× more context switches than C |
| P2 | `clock_gettime` at 9.7% of samples (Mojo async runtime instrumentation) | ~10% overhead tax |
| P3 | No spin barrier primitive | Must use `parallelize` mutex or roll own atomics |
| P4 | Closure capture semantics in `parallelize` callbacks | Can silently break with atomic work-stealing |
| P5 | ` UnsafePointer` parameter alias | Mojo doesn't enforce — can alias read/write buffers silently |

---

## Reference Patterns in External Codebases

| What to learn | Where |
|---------------|-------|
| AVX2 int8 dot wrapper | `modular_ref/max/kernels/src/linalg/arch/cpu/vnni_intrinsics.mojo:185` |
| GGUF loading + quantized GEMV | `llamacpp_ref/ggml/src/ggml-cpu/` |
| Qwen3.5 DeltaNet forward | `llamacpp_ref/src/` — search for qwen35 or deltanet |
| Multi-row GEMV batching | `llamacpp_ref/ggml/src/ggml-cpu/ops.c` — `ggml_compute_forward_mul_mat` |
| Fused FFN (up+gate+silu) | `llamacpp_ref/ggml/src/ggml-cpu/` — search for `ggml_compute_forward_mul_mat_id` |
| Mojo stdlib RMSNorm/softmax/RoPE | `modular_ref/max/kernels/src/` |
| Mojo stdlib causal_conv1d | `modular_ref/max/kernels/src/` — `causal_conv1d_update_cpu` ~line 3102. Note: expects `[K,C]` layout, GGUF is `[C,K]` |
