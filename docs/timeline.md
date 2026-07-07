# Timeline — qwen35.mojo Development

Historical record. Dates are April–May 2026.

---

## Version Map

| Version | Architecture | Decode t/s | Status |
|---------|-------------|------------|--------|
| v2 | Monolithic, Q4_K/Q5_K/Q6_K, sigmoid scan | — | never coherent |
| v3 | Hybrid arch discovered, DN still elementwise | — | never coherent |
| v4 | DN state matrix + delta rule | — | never coherent |
| v5 | F32 GGUF, softmax replaces sigmoid scan | — | never coherent |
| v7 | **First coherent output** — 3 bug fixes (Sᵀ@q, Q/gate, RoPE dim) | 1.68 | ✅ correct |
| v8 | SIMD weight conversion, core pinning, accumulator unroll | 2.0–3.3 | ❌ **later found garbage** |
| v9 | int8×int8 merge + activation quantization | 4.45 | ❌ **structured but wrong** |
| v10 | `setup_model.py` transpiler, f32×f32 GEMV fallback | 6.98 | ✅ correct |
| v11 | v10 + repack + megafusion + streaming + online attn | 4.1–4.6 sustained | ✅ correct |
| v12 | W8A8 SignedDot kernel (psign→pmaddubs→pmaddwd) | 11.7 | ✅ correct |
| v13 | v12 + SIMD quantization | 11.6–15.5 | ✅ **latest working** |
| v13_threadpool | Atomic work-stealing `run_tiles` | ~19.7 | all numbers invalid |
| v13_exp_layout | x4 layout | — | correct but 18% slower, abandoned |
| v13_plus_phase2 | x4 phase 2 | — | ❌ garbage output, abandoned |

---

## Performance Trajectory (correct output only)

```
v7:   1.68 t/s   ← first coherent output
v8*:  ~3-4 t/s   ← SIMD + pinning + unroll (* later found broken)
v10:  6.98 t/s   ← transpiler rewrite + f32×f32 GEMV
v11:  4.1-4.6    ← merged optimizations (sustained; early 8.46 was 4 tokens only)
v12:  11.7 t/s   ← W8A8 SignedDot kernel (+2.5× over v11)
v13:  11.6-15.5  ← SIMD quant (+6%), FMA (0%), prefetch kept
ik:   16.6-19.2  ← ik_llama.cpp reference target
```

---

## Experiment Log

### Phase 1: Getting to Coherent Output (Apr 28 – May 1)

| Date | Event | Result | Later Corrected? |
|------|-------|--------|-------------------|
| Apr 28 | spec_v2 + 10 action specs written | Design only | spec_v2 assumed all layers same → WRONG |
| Apr 29 15:15 | First light: forward pass produces finite output | Synthetic input only | Bugs in every layer |
| Apr 29 17:00 | DN state discovered to be [128×128] matrix, not [128] vector | Fundamental rearchitecture | — |
| Apr 29 17:45 | DN research audit: 6 separate bugs in DN layer | Total rewrite needed | — |
| Apr 29 19:22 | Output garbled, 7 suspected causes identified | All non-stretch items implemented | Fix 6 (sigmoid→softmax) later confirmed; Fix 7 (outer product swap) found later |
| Apr 29–30 | 7 bugs fixed across 4 sessions | conv1d, sigmoid underflow, Q6_K, signed byte, ssm_a, softmax, outer product swap | All valid fixes |
| May 1 12:45 | v6 kernel replacements: rmsnorm F64, SIMD softmax, ComplexSIMD RoPE | 5 of 6 done, conv1d blocked | — |
| May 1 14:30 | Forward pass proven correct vs Python | Custom GGUF was broken, not code | — |
| May 1 16:00 | Q8_0 GGUF added, still gibberish | Thinking tokens missing | — |
| May 1 17:00 | trace_layer0.py found buggy | Echo chamber — validated wrong code against wrong reference | — |
| May 1 19:00 | ground_truth_trace.py via HF hooks created | DN L0-L2 now match HF | — |
| May 1 19:19 | **3 bugs fixed: Sᵀ@q, Q/gate interleaved, RoPE dim** | **COHERENT OUTPUT at 1.68 t/s** | — |

### Phase 2: Performance Optimization (May 1 – May 6)

| Date | Event | Result | Later Corrected? |
|------|-------|--------|-------------------|
| May 1 23:30 | Deep perf analysis vs ik_llama.cpp | 5.5× gap, root cause: per-element Q8_0 dequant | — |
| May 2 17:30 | v8: SIMD conversion + core pinning + 4-accumulator | 2.0–3.3 t/s | ❌ v8 output later found GARBAGE (SIMD sign-ext bug) |
| May 2 | 4 failed trials: pmaddubs, pmovsxbd, exp_approx, RoPE precompute | All failed for various reasons | exp_approx: 1.4% error → early EOS |
| May 3 12:15 | Instruction-level analysis v8 vs ik_llama | 7 instructions vs 4 per 32 elements; 2.4× bandwidth gap | — |
| May 3 13:00 | "Boulder roadmap" for v9 merge | Target: 9.27 t/s via int8×int8 + multi-row + repack | — |
| May 3 14:15 | v9 "big fucking merge" | 4.45 t/s | ❌ WRONG output; v8 ALSO found broken |
| May 3 19:00 | v10 transpiler rewrite | Compiles but output wrong (`<think` then stops) | — |
| May 3 22:45 | v10: swap i8×i8 → f32×f32 GEMV | **6.98 t/s, CORRECT** | v8 `comptime for` pattern was correct all along — misdiagnosed |
| May 4 01:00 | 4 parallel experiments: repack, i8i8, megafusion, streaming | +11%, −9%, +42%, +20% | megafusion +42% inflated by ±30% variance (4 tokens) |
| May 4 16:00 | v11 merged branch | 8.46 t/s (only 4 decode tokens) | ❌ Misleading: sustained is 4.1–4.6 t/s |
| May 4 19:45 | Buffer overflow in FA FFN fixed | `q_total` overflowed by 3072 floats → segfault at 120+ tokens | — |
| May 4 19:45 | v11 sub-experiments: fast exp, 16-row unroll, online attn, int32 accum | +0%, +22%, +14%, abandoned | — |
| May 5 10:30 | v13 x4 layout test | "garbage" | ❌ Actually CORRECT — was short prompt hitting EOS. But 18% slower. |
| May 5 10:30 | Pre-allocated quant buffers | −1% (noise) | — |
| May 6 01:25 | SIMD quantization, FMA, prefetch removal test | +6%, 0%, −5% when removed | — |
| May 6 12:30 | `run_tiles` atomic work-stealing | +23–30% decode | ❌ **GARBAGE OUTPUT** — all numbers invalid |
| May 6 18:45 | Profiling: L1 miss, context switches, IPC comparison | Root causes quantified | — |

### Key Misleading Results

| Claim | Reality | Impact |
|-------|---------|--------|
| v8 produces correct output at 3.3 t/s | v8 output is garbage (SIMD sign-ext bug) | All v8 numbers invalid |
| v9 produces structured output at 4.45 t/s | Structured but wrong (quantize/mask bug) | All v9 numbers invalid |
| v11 at 8.46 t/s | Only 4 decode tokens — sustained is 4.1–4.6 | Early v11 benchmarks misleading |
| v13_threadpool old at 19.7 t/s | Garbage output — work-stealing breaks correctness | All threadpool numbers invalid |
| v13 x4 layout "garbage" | Actually correct output — was short prompt hitting EOS | x4 still 18% slower, abandoned |
