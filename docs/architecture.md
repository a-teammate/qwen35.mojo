# Architecture Reference: qwen35.mojo

Qwen3.5-0.8B inference engine in Mojo.
Hybrid Gated DeltaNet (18 layers) + GQA Softmax Attention (6 layers).
Q8_0 GGUF, token-by-token decode.
`setup_model.py` Python transpiler generates per-shape-specialized GEMV kernels.

## Constants

| Constant | Value | Notes |
|----------|-------|-------|
| DIM | 1024 | Hidden dimension |
| N_LAYERS | 24 | |
| DN_LAYERS | 18 | layer%4≠3 |
| FA_LAYERS | 6 | layer%4==3 → layers 3,7,11,15,19,23 |
| VOCAB_SIZE | 248320 | |
| DN_GROUPS | 16 | DeltaNet heads/groups |
| DN_HEAD_DIM | 128 | |
| FA_N_Q_HEADS | 8 | |
| FA_N_KV_HEADS | 2 | GQA ratio 4:1 |
| FA_HEAD_DIM | 256 | |
| FA_ROTARY_DIM | 64 | partial_rotary_factor=0.25 × 256 |
| FFN_DIM | 3584 | SwiGLU up/gate |
| CONV1D_KERNEL | 4 | Causal conv1d in DeltaNet |
| KV_CACHE_CAP | 4096 | Sliding window |

## DeltaNet Forward (per token per layer)

| Step | Operation | Shape | Notes |
|------|-----------|-------|-------|
| 1 | h = rmsnorm(x, attn_norm) | [1024] | |
| 2 | qkv = attn_qkv @ h | [6144] | 16 groups × (q128 + k128 + v128) |
| 3 | qkv = silu(conv1d(qkv)) | [6144] | causal, kernel=4, per-layer state buffer |
| 4 | q, k, v = split(qkv) | [16,128] each | per group: qkv[g×384 : g×384+384] |
| 5 | q = l2_norm(q)/√128, k = l2_norm(k) | per group | **NOT RoPE** |
| 6 | decay = exp(ssm_a[g]) | scalar/group | ssm_a stores negative values (−exp(A_log)) |
| 7 | β = sigmoid(β_projected) | scalar/group | β from projection + sigmoid |
| 8 | S[g] *= decay[g] | [128×128] matrix | state decay |
| 9 | residual = v − Sᵀ@k | [128] | **column-wise**: out[j]=Σᵢ S[i×128+j]·k[i] |
| 10 | S[g] += β × outer(k, residual) | [128×128] | delta rule update |
| 11 | o[g] = Sᵀ@q | [128] | same column-wise pattern as step 9 |
| 12 | o = ssm_norm_weight × rmsnorm(o) | [128] | output gating (ssm_norm has NO +1) |
| 13 | z = attn_gate @ h | [2048] | per group: z[128] |
| 14 | o *= sigmoid(z[g]) | [128] | elementwise |
| 15 | out = attn_out @ concat(o) | [1024] | |
| 16 | x += out | | residual add |
| 17 | x += ffn(rmsnorm(x, post_attn_norm)) | | see FFN table |

## Full Attention Forward (per token per layer)

| Step | Operation | Shape | Notes |
|------|-----------|-------|-------|
| 1 | h = rmsnorm(x, attn_norm) | [1024] | |
| 2 | q_total = q_proj @ h | [4096] | 8×(256Q + 256gate) per-head interleaved |
| 3 | k = k_proj @ h | [512] | 2 KV heads × 256 |
| 4 | v = v_proj @ h | [512] | 2 KV heads × 256 |
| 5 | Q, gate = deinterleave(q_total) | [2048] each | block-level: extract alternating 256-dim heads |
| 6 | Q_h = rmsnorm(Q_h, q_norm), K_h = rmsnorm(K_h, k_norm) | per head | per-head RMSNorm |
| 7 | RoPE(Q, K) on first 64 of 256 dims | | scale=1/√256 |
| 8 | store K,V in cache at pos | | |
| 9 | scores = Q @ Kᵀ / √256 | [8, pos+1] | GQA: kv_h = q_h // 4 |
| 10 | attn = softmax(scores) | [8, pos+1] | 3-pass numerically stable |
| 11 | out = attn @ V → concat | [2048] | |
| 12 | out = out_proj @ out | [1024] | |
| 13 | out *= sigmoid(gate) | | output gating |
| 14 | x += out | | residual add |
| 15 | x += ffn(rmsnorm(x, post_attn_norm)) | | see FFN table |

## FFN (both layer types)

| Step | Operation | Shape |
|------|-----------|-------|
| 1 | up = ffn_up @ h | [3584] |
| 2 | gate_val = ffn_gate @ h | [3584] |
| 3 | ffn_out = up × silu(gate_val) | [3584] |
| 4 | ffn_out = ffn_down @ ffn_out | [1024] |

## Weight Tensor Contract (DeltaNet Layer)

| GGUF Name | Shape | Format | Convention |
|-----------|-------|--------|-----------|
| blk.{N}.attn_norm.weight | [1024] | F32 | +1 pre-added |
| blk.{N}.attn_qkv.weight | [6144, 1024] | Q8_0 | |
| blk.{N}.ssm_conv1d.weight | [6144, 4] | Q8_0 | layout **[C,K]** |
| blk.{N}.ssm_conv1d.bias | [6144] | Q8_0 | |
| blk.{N}.ssm_a | [16] | F32 | stores **−exp(A_log)** |
| blk.{N}.ssm_dt.bias | [16] | F32 | |
| blk.{N}.ssm_norm.weight | [128] | F32 | **NO +1** (raw) |
| blk.{N}.attn_gate.weight | [2048, 1024] | Q8_0 | |
| blk.{N}.attn_out.weight | [1024, 2048] | Q8_0 | |
| blk.{N}.post_attention_norm.weight | [1024] | F32 | +1 pre-added |
| blk.{N}.ffn_up.weight | [3584, 1024] | Q8_0 | |
| blk.{N}.ffn_gate.weight | [3584, 1024] | Q8_0 | |
| blk.{N}.ffn_down.weight | [1024, 3584] | Q8_0 | |

FA layers use the same FFN tensors, with separate Q/K/V projections + q_norm/k_norm + attn_output_gate.
Exact GGUF tensor names for FA differ from DN; see `setup_model.py` resolve_layer_weights.

## FLOP Budget (per token)

| Operation | Calls | FLOPs | Share |
|-----------|-------|-------|-------|
| Output proj 248K×1K | 1 | 509M | 52.5% |
| FFN up 3584×1K | 24 | 88M | 9.1% |
| FFN gate 3584×1K | 24 | 88M | 9.1% |
| FFN down 1K×3584 | 24 | 88M | 9.1% |
| DN QKV 6144×1K | 18 | 113M | 11.6% |
| DN out 2048×1K | 18 | 38M | 3.9% |
| FA Q+gate 4096×1K | 6 | 25M | 2.6% |
| FA K,V 512×1K | 12 | 13M | 1.3% |
| FA out 2048×1K | 6 | 13M | 1.3% |
| **Total GEMV** | **97+** | **~970M** | **100%** |

97+ `parallelize` calls per token. Each one is a mutex/condvar barrier.

## Memory

| Component | Size | Notes |
|-----------|------|-------|
| GGUF file | 776 MB | Q8_0 |
| Dequantized weights | ~2.4 GB | F32 in RAM |
| DeltaNet state | ~18 MB | 18 layers × 16 groups × 128² × 4B |
| KV cache | ~25 MB | cap=4096, FP32, sliding window |
| Conv buffers | ~1.7 MB | 18 layers × 6144 × 4B |
| Embedding | tied | token_embd = output projection |

## Special Conventions

- **Q/gate interleaved**: FA Q proj outputs `[Q₀(256), g₀(256), Q₁(256), g₁(256), …]`, **not** `[Q_all, gate_all]`
- **Tied embeddings**: `token_embd.weight` reused as output projection
- **Thinking model**: generates `<think…\n</think\n` prefix before response
- **Sampling**: temp=0.7, top_k=20, top_p=0.8
- **GGUF norm +1**: most RMSNorm weights have +1 pre-added; `ssm_norm` does **not**
- **DN uses NO RoPE**: L2-normalization on q and k
- **Conv1d layout**: GGUF stores [C,K], access `kernel[c*4+t]`

## Build & Run

```bash
python setup_model.py              # downloads model, generates code into build/, compiles ./qwen35
./qwen35 "prompt" [-n 128]
```

Mojo version: see `.mojo-version` (currently 0.26.2).

## Project File Map

| Path | Purpose |
|------|---------|
| `setup_model.py` | One-command setup: mojo check, model download, transpiler into `build/`, compile |
| `_components.mojo` | Shared: SIMD helpers, RMSNorm, softmax, sampling |
| `gguf_loader.mojo` | GGUF parser + Q8_0 tiled repacking |
| `tokenizer.mojo` | BPE tokenizer (Python `regex` interop) |
| `build/model_config.mojo` | Generated architecture constants + offset functions |
| `build/run_inference.mojo` | Generated inference engine (**do not hand-edit**) |

## External References

| Resource | What | Key Patterns |
|----------|------|-------------|
| [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) | ihor's llama.cpp fork | Qwen3.5 arch implementation, benchmark target. Use `-c 4096` to avoid 3GB alloc |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Stock C++ inference engine | GGUF loading, quantized GEMV, Qwen3.5 arch |
| HuggingFace `transformers` | Python reference model | Canonical forward pass for correctness validation |
