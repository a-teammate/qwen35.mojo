# Correctness Playbook: qwen35.mojo

## Validation Methodology

| Method | What | When |
|--------|------|------|
| HF model hooks (`ground_truth_trace.py`) | Capture per-layer intermediates from HuggingFace model | Validate any kernel or layer (**primary oracle**) |
| Layer-by-layer element comparison | Compare norm2 + first-N element values at each step | Find first divergence point |
| llama.cpp / ik_llama.cpp cross-reference | Run same GGUF, compare first-token logits | Confirm GGUF validity |
| Single-token argmax | Compare Mojo argmax vs HF argmax for same input | Quick smoke test |

**Always compare element values, not just norm2.** Norm2 can match while signs are flipped.

## Architecture Pitfalls

### DeltaNet

| # | What | Correct | Mistake | Detection |
|---|------|---------|---------|-----------|
| D1 | State query | Column-wise Sᵀ@q: `out[j]=Σᵢ S[i·D+j]·q[i]` | Row-wise S@q: `out[i]=Σⱼ S[i·D+j]·q[j]` | ssm_norm output 7× too large |
| D2 | State shape | `[128×128]` matrix per group (18 layers × 16 groups = 288 matrices) | `[128]` vector per group (130× undersized) | outer product math impossible |
| D3 | State update | Delta rule: `S*=decay; r=v−Sᵀ@k; S+=β·outer(k,r)` | Additive: `S=β·S + α·k·v` | no self-correction, quality degrades |
| D4 | Position encoding | L2-norm: `q = l2_norm(q)/√128`, `k = l2_norm(k)` | Applying RoPE to DN q/k | wrong at pos≥0 |
| D5 | Conv1d activation | `silu(conv1d(qkv))`: activation **after** conv1d | `conv1d(qkv)` raw | DN layer divergence |
| D6 | ssm_a semantics | Stores `−exp(A_log)` (negative, e.g. −1.29). Use `decay = exp(ssm_a[g])` ∈ (0,1) | `exp(exp(ssm_a))` → values like 3e+36 | state explosion |
| D7 | ssm_norm convention | `ssm_norm.weight` has **NO +1** pre-added: apply as-is | Adding +1 (double correction) | subtle drift compounding |
| D8 | Conv1d layout | GGUF stores `[C,K]`, access `kernel[c·4+t]` | `kernel[t·C+c]` (transposed) | first DN layer wrong |

### Full Attention

| # | What | Correct | Mistake | Detection |
|---|------|---------|---------|-----------|
| F1 | Q/gate layout | Per-head interleaved: `[Q₀(256), g₀(256), …, Q₇(256), g₇(256)]` | Concatenated: `[Q_all(2048), gate_all(2048)]` | FA output 21× off reference |
| F2 | Attention mechanism | Standard softmax (3-pass numerically stable) | Sigmoid scan `sigmoid(q·k)` | incoherent text even with correct numerics elsewhere |
| F3 | RoPE dims | `rotary_dim=64` (partial_rotary_factor=0.25 × head_dim=256) | full `head_dim=256` | only visible at pos>0 |
| F4 | Per-head norms | Apply `q_norm`/`k_norm` RMSNorm per head after projection | skip per-head norms | subtle quality degradation |

### General

| # | What | Correct | Mistake | Detection |
|---|------|---------|---------|-----------|
| G1 | GGUF norm +1 | Most RMSNorm weights have +1 baked in. Use gguf value directly. | Adding +1 again (except ssm_norm, see D7) | numerical drift |
| G2 | Tied embeddings | `token_embd.weight` = output projection. Same tensor. | Separate output weight matrix | wrong logits |
| G3 | Signed byte | Q8_0 values are uint8 representing signed [−128,127]. Sign-extend: `(v ^ 0x80) − 0x80` | `Int8(UInt8)` → zero-extend | wrong negative weights |
| G4 | pmaddubs asymmetry | For signed×signed int8: sign-extend **both** operands before `pmaddubs` | Mask trick with wrong arg order (mask must be FIRST) | values off ~2× for negatives |
| G5 | Sampling format | Thinking model: expects `<think…` prefix, use `temp=0.7, top_k=20, top_p=0.8` | Missing top_p, wrong temp | generates `<think` then early EOS |
| G6 | Tokenizer input | v12+: stdin pipe (`echo "Hello" \| ./qwen35`). v10/v11: argv | v10/v11 `--prompt` flag splits into byte-like tokens | garbage tokens from wrong split |

## Mojo Implementation Pitfalls

| # | Issue | Detail |
|---|-------|--------|
| M1 | `Int8(UInt8)` zero-extends | Not sign-extend. Manual: `(v ^ 0x80) − 0x80` |
| M2 | Pointer vs SIMD types | `UnsafePointer[Int8, …]` for pointers, `SIMD[DType.int8, 32]` for SIMD. Never mix. |
| M3 | `bitcast` vs `.cast[]` | `bitcast` = bit reinterpret (same bitwidth). `.cast[]` = type conversion (with rounding). |
| M4 | Float16→Float32 | Use `bitcast[float16]()` then `.cast[float32]()`. The `vcvtph2ps` LLVM intrinsic was removed in LLVM 15+. |
| M5 | `pmovsxbd` intrinsic | Not found at link time in Mojo 0.26.2. Use XOR+SUB sign-extend instead. |
| M6 | No global variables | Work around with function params or config structs. |
| M7 | `deinterleave` comptime | `(nelts*2)//2 ≠ nelts` at comptime. Do not rely on this identity. |
| M8 | `parallelize` barriers | Uses mutex/condvar (not spin barriers). Each call adds scheduling overhead. |
| M9 | `__copyinit__` parameter | Must be named `copy` exactly. |

## Validation Pitfalls

| # | Issue | Detail |
|---|-------|--------|
| V1 | Hand-computed traces | Python traces (`trace_layer0.py` etc.) can contain the same bugs they validate. **Prefer HF model hooks** (`ground_truth_trace.py`). |
| V2 | Norm2 insufficiency | Norm2 can match while element signs are wrong. Always compare first-N element values. |
| V3 | Short-run variance | 4 to 10 decode tokens give +/-30% variance. Use 50+ decode tokens for reliable perf numbers. |
| V4 | SIMD quant rounding | Q8_0 SIMD quantization rounds differently than scalar. Both valid. Do not compare tokens across methods. |
| V5 | CPU governor | `powersave` ~898 MHz vs `performance` ~4 GHz. Always note governor in benchmarks. |
| V6 | First-token bias | Short prompts ("Hello") make the model hit EOS after 4 tokens. Use longer prompts for benchmarking. |
