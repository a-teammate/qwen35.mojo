# quickqwen — a CPU inference engine for a *hybrid* transformer, built on a 15W laptop

A from-scratch inference engine for **Qwen3.5-0.8B**, written in [Mojo](https://docs.modular.com/mojo/).
No llama.cpp, no GGML — every matmul, attention kernel, and quantization routine is implemented directly.

Two things make this non-trivial:

1. **The model isn't a standard transformer.** Qwen3.5-0.8B is a **hybrid**: 18 layers of
   **Gated DeltaNet** (delta-rule linear attention with a recurrent 128×128 state matrix per
   group) interleaved with 6 layers of **GQA softmax attention**. Engines like llama.cpp and
   herbert-rs don't implement this architecture — it had to be reverse-engineered from the
   HuggingFace reference and the ik_llama.cpp source, one bug at a time.
2. **The target hardware has no fast paths.** This runs on an **Intel i7-10610U: 4 cores, 15 W,
   AVX2 + FMA3 — no AVX-512, no VNNI**. The interesting question is not "how fast can you go
   with the fancy instructions" but "how fast can you go when you *can't*." The answer, with a
   hand-written W8A8 kernel and a compile-time-specializing transpiler, is **~12.5 tok/s decode
   in a 303 KB binary**.

> This repo is a **learning artifact**, not a product. It was built to understand how a modern
> hybrid transformer actually runs and how far a from-scratch engine can be pushed on
> commodity silicon. The wrong turns are documented as carefully as the wins — see
> [The journal](#the-journal--wrong-turns).

---

## Demo

```
$ ./quickqwen "Write a haiku about autumn leaves." -n 48

</think>
Pale yellow fall leaves fall from the branches.

Prefill: 19 tokens in 1288 ms (14.75 t/s)
Decode:  12 tokens in 974 ms  (12.32 t/s)
```

It's a 0.8B model — fluent, not smart. The engine's job is to faithfully reproduce the
reference model's outputs, which it does (validated layer-by-layer against HuggingFace; see
[`docs/correctness.md`](docs/correctness.md)). Output quality is bounded by the model, not the
engine.

---

## Performance

All numbers on the dev machine: **Intel i7-10610U (4C/8T, 15 W, AVX2+FMA3, DDR4-3200)**,
Q8_0 weights, CPU governor = `performance`. The reference is **ik_llama.cpp** (a highly
optimized C++ engine by a specialist contributor).

| Phase | quickqwen | ik_llama.cpp | Ratio |
|-------|----------:|-------------:|------:|
| Prefill (~20 tokens) | ~15 t/s | ~75 t/s | ~0.20× |
| Decode (sustained) | **11.6–15.5 t/s** | 16.6–19.2 t/s | ~0.7× |
| Binary size | **303 KB** | ~60 MB | — |

Decode lands at roughly **0.7× of a hand-tuned C++ engine** — while being written from scratch
in a new language, on hardware that lacks the instructions (AVX-512/VNNI) those engines lean on.
Prefill is far behind because quickqwen does no batched GEMM yet (see
[Limitations](#limitations)).

### How decode got from 1.7 to 12.5 tok/s

| Version | Decode t/s | What changed | Correct? |
|---------|---------:|--------------|:--------:|
| v7 | 1.68 | First coherent output (3 bugs fixed: Sᵀ@q, Q/gate interleave, RoPE dim) | ✅ |
| v10 | 6.98 | `setup.py` transpiler → per-shape-specialized f32×f32 GEMV | ✅ |
| v11 | 4.1–4.6 | Merged: weight repack + fused FFN + streaming (early 8.46 was 4 tokens only) | ✅ |
| v12 | 11.7 | **W8A8 SignedDot kernel** (`vpsignb → vpmaddubsw → vpmaddwd`), +2.5× | ✅ |
| v13 | 11.6–15.5 | SIMD activation quantization (+6%), the **current version** | ✅ |

Full trajectory, including the versions that were silently broken:
[`docs/timeline.md`](docs/timeline.md).

---

## The kernel: W8A8 SignedDot

The decode bottleneck is memory-bandwidth-bound GEMV. The core inner loop, per block of 32
int8 weights × 32 int8 activations, is three AVX2 intrinsics that replace ~20 float32 ops:

```
vpsignb     # resolve signs for signed×signed
vpmaddubsw  # int8 × int8 → int16 madd (16 → 16)
vpmaddwd    # int16 × int16 → int32 madd (8 → 8)
            # then fp16 weight scale × fp32 activation scale → f32 accumulate
```

Activations are quantized to Q8_0 once per GEMV call and reused across all rows (0.7% relative
error vs f32). Weights are stored in a tiled Q8_0 layout (8 consecutive rows interleaved) so the
activation vector is loaded once and reused. Full kernel + layout notes:
[`docs/performance.md`](docs/performance.md).

### The transpiler

`setup.py` is a small Python program that reads the GGUF metadata and **emits
compile-time-specialized Mojo**: every GEMV is monomorphized to its exact `(rows, cols)` shape,
so tile sizes, unroll factors, and register usage are known to the compiler. This is what turned
a 4× decode win (v7→v10). It is currently specialized to the 0.8B shape (see Limitations).

---

## The architecture (why this isn't just "another llama.cpp port")

24 layers, split **18 / 6**:

- **DeltaNet layers** (`layer % 4 != 3`): causal conv1d → per-group q/k/v → **delta-rule state
  update** `S *= decay; r = v − Sᵀk; S += β·(k⊗r); out = Sᵀq`. State is a `[128×128]` matrix per
  group (16 groups × 18 layers = 288 matrices). No KV cache — the recurrent state *is* the memory.
- **Full-attention layers** (`layer % 4 == 3`): standard GQA softmax (8 Q / 2 KV heads, head dim
  256, partial RoPE on 64 dims, per-head RMSNorm), with a sliding KV cache.

Step-by-step forward passes for both layer types, with shapes and the exact gotchas, are in
[`docs/architecture.md`](docs/architecture.md). The catalogue of every bug that bit during
implementation — and the detection signature that found each one — is in
[`docs/correctness.md`](docs/correctness.md).

---

## The journal / wrong turns

Five published numbers in this project's history were **wrong**, and each was believed for a
while before being caught. They are kept on purpose — understanding *why* a fast-but-broken
version looks correct is the actual lesson of the project.

| Claim | Reality |
|-------|---------|
| v8: correct output at 3.3 t/s | Output was garbage — a SIMD sign-extension bug (`Int8(UInt8)` zero-extends, not sign-extends) |
| v9: structured output at 4.45 t/s | Structured but wrong — a quantize/mask bug. Looked coherent, wasn't. |
| v11: 8.46 t/s | Only 4 decode tokens — sustained was 4.1–4.6. Short runs have ±30% variance. |
| v13 threadpool: 19.7 t/s | Garbage output — atomic work-stealing broke correctness. All numbers invalid. |
| v13 x4 layout: "garbage" | Actually *correct* — a short prompt was hitting EOS. (Still 18% slower, so abandoned.) |

The recurring lesson: **a fluent-looking stream is not proof of correctness**, and **a fast run
on few tokens is not a measurement**. Both echo chambers (validating buggy code against a buggy
tracer) cost real time. See [`docs/timeline.md`](docs/timeline.md) and
[`docs/correctness.md`](docs/correctness.md).

---

## Run it

Requires [Mojo ≥ 0.26](https://docs.modular.com/mojo/manual/get-started/).

```bash
# 1. Obtain the model and place it as ./model.gguf
#    (Qwen3.5-0.8B, Q8_0 GGUF — e.g. via the HuggingFace Hub:
#     `pip install huggingface_hub && hf download ...`)
# 2. Build
mojo build run_inference-generated.mojo -o quickqwen -Xlinker -lm
# 3. Run
./quickqwen "Write a haiku about autumn leaves." -n 64
./quickqwen "your prompt" -n 128 --bench     # --bench: suppress streaming, just print stats
```

The engine loads `model.gguf` from the current directory. Input is a single positional prompt;
`-n` sets max decode tokens. Sampling is fixed to the Qwen3 thinking-model defaults
(`temp=0.7, top_k=20, top_p=0.8`).

To regenerate the specialized engine from GGUF metadata (after changing the model or the
transpiler):

```bash
python setup.py            # emits model_config_generated.mojo + run_inference-generated.mojo
```

---

## Limitations (honest)

- **One model.** Only Qwen3.5-0.8B (Q8_0). The transpiler is specialized to its exact shape;
  supporting other sizes means generalizing `setup.py` — intentionally out of scope here.
- **Prefill is slow** (~5–10× behind batched-GEMM engines). Decode is the optimized path;
  prefill still runs token-by-token.
- **Tokenizer via Python interop.** BPE uses Python's `regex` module, so the first run pays a
  Python startup cost. A native Mojo tokenizer is the obvious next step
  ([`docs/opportunities.md`](docs/opportunities.md), O12).
- **One CPU class tested.** Developed on a 15 W AVX2 laptop (no AVX-512/VNNI). Mojo's
  cross-backend story is the reason it was chosen, but only the x86-64 AVX2 path is exercised.
- **No batching, no server, no GPU.** Single-stream token-by-token decode only.

Unfinished directions (DeltaNet state persistence, radix-tree KV reuse, online LoRA) are written
up as research notes in [`docs/research/`](docs/research/) — aspirational, not implemented.

---

## Why Mojo?

Partly to learn it: Mojo pitches itself as a Pythonic path to systems performance (and a CUDA
alternative). This project was a stress test of that claim on a real workload. The language
delivered on per-shape specialization and ergonomic SIMD, and exposed real rough edges (mutex
barriers in `parallelize`, ~10% runtime instrumentation tax, silent pointer aliasing). Those are
logged in [`docs/performance.md`](docs/performance.md) under "Mojo Performance Gotchas."

---

## Repository layout

```
quickqwen/
├── run_inference-generated.mojo   # the engine: prefill/decode loop, all 24 layers (generated)
├── _components.mojo               # kernels: rmsnorm, gemv, W8A8 dot, rope, softmax, quantize
├── gguf_loader.mojo               # Q8_0 parser + tiled weight repack
├── tokenizer.mojo                 # BPE (Python regex interop)
├── model_config.mojo              # architecture constants
├── model_config_generated.mojo    # generated tensor offsets
├── setup.py                       # transpiler → per-shape-specialized Mojo
└── docs/
    ├── architecture.md            # hybrid DeltaNet + GQA forward passes, with shapes
    ├── performance.md             # W8A8 kernel, hardware profile, bottleneck breakdown
    ├── correctness.md             # every bug found + its detection signature
    ├── timeline.md                # version map + the wrong-turns ledger
    ├── opportunities.md           # ranked, unstarted optimization ideas
    └── research/                  # aspirational notes (continual learning, radix-tree KV)
```

---

## Related

- **[q2q](../q2q)** — the companion "understand before optimizing" half of this project: the
  Qwen3.5 execution path traced out of ik_llama.cpp, file by file, as readable annotated C++.
  Where quickqwen *builds* the engine, q2q *explains* the reference it was reverse-engineered
  from.
- [herbert-rs](https://github.com/xigh/herbert-rs) and Philippe Anel's
  ["CPU Inference" series](https://philippe-anel.fr/en/blog/) — the methodology that shaped how
  this project measures and reports itself.
- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) — the performance reference and the
  source of the Qwen3.5 architecture implementation that q2q studies.

## License

MIT — see [LICENSE](LICENSE). The engine code is original; the reference codebases studied
during development (llama.cpp, ik_llama.cpp, HuggingFace transformers, the Mojo stdlib) are
**not** included in this repository.
