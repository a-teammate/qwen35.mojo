# qwen35.mojo: running a hybrid transformer on a 15W laptop
> previous name 'quickqwen'

A from-scratch inference engine for **Qwen3.5-0.8B** that compiles down to ~300KB.
This model is not a pure transformer. It uses a hybrid architecture, which made it
a good test case.

The original motivation was to learn [Mojo](https://docs.modular.com/mojo/), a
new language pitched as a Pythonic path to systems performance and a CUDA
alternative that also targets CPU. I used it to build a SIMD-optimized inference
engine for my laptop.

Along the way I tried many optimizations: merged kernels, quantization at
different points in the architecture, layout experiments. The tiny model has
different cache locality than bigger ones. Compute is not the bottleneck;
shuffling memory around is. Many experiments did not make the cut.

## Target Constraints

1. **The model isn't a standard transformer.** Qwen3.5-0.8B is a **hybrid**: 18 layers of
   **Gated DeltaNet** interleaved with 6 layers of **GQA softmax attention**. DeltaNet is
   delta-rule linear attention that keeps a recurrent 128x128 state matrix per group. Engines like
   llama.cpp and herbert-rs don't implement this architecture, so I reverse-engineered it from the
   HuggingFace reference and the ik_llama.cpp source, one bug at a time.
2. **The target hardware has no fast paths.** The engine runs on an **Intel i7-10610U: 4 cores,
   15 W, AVX2 + FMA3** (no AVX-512, no VNNI). The question is not how fast you can go with the
   fancy instructions, but how fast you can go without them. The answer is **~12.5 tok/s decode in
   a 303 KB binary**, thanks to a hand-written W8A8 kernel and a transpiler that specializes at
   compile time.

DISCLAIMER:
> This repo is a **learning artifact**, not a polished product or component. I built it to understand how a modern hybrid
> transformer runs, and how far a from-scratch engine can go on commodity silicon. I document the
> wrong turns as carefully as the wins (see [The journal: wrong turns](#the-journal-wrong-turns)).

## Demo

```
$ ./qwen35 "Write a haiku about autumn leaves." -n 48

</think>
Pale yellow fall leaves fall from the branches.

Prefill: 19 tokens in 1288 ms (14.75 t/s)
Decode:  12 tokens in 974 ms  (12.32 t/s)
```

It's a 0.8B model: fluent, not smart. The engine faithfully reproduces the reference model's
outputs, validated layer-by-layer against HuggingFace (see
[`docs/correctness.md`](docs/correctness.md)). The model, not the engine, sets the ceiling on
output quality.

## Performance: okayish

Despite the name "quick qwen" we are not there yet. The main feature is the nano binary size and
slim codebase. On my dev machine the engine runs a bit slower than llama.cpp.

All numbers on the dev machine: **Intel i7-10610U (4C/8T, 15 W, AVX2+FMA3, DDR4-3200)**,
Q8_0 weights, CPU governor = `performance`. The reference is **llama.cpp**, the standard C++ CPU inference engine. I benchmarked a stock build, version b1-2083217 (built from source 2026-06-18, cmake Release, GGML_CUDA=OFF).

| Phase | qwen35.mojo | llama.cpp (stock) | Ratio |
|-------|----------:|------------------:|------:|
| Prefill (~16 tokens) | ~16 t/s | ~63 t/s | ~0.25x |
| Prefill (~95 tokens) | ~15.5 t/s | ~106 t/s | ~0.15x |
| Decode (sustained) | **12.1-13.0 t/s** | 18.1-20.7 t/s | ~0.65x |
| Binary size | **303 KB** | ~60 MB | n/a |

Decode lands at roughly **0.65x of llama.cpp**, written from scratch in a new language, on
hardware that lacks the AVX-512 and VNNI instructions those engines lean on. Prefill trails far
behind because qwen35.mojo does no batched GEMM yet (see [Limitations](#limitations)).

> **Note on methodology.** Each cell is the mean of 3 runs. Short-prompt prefill in particular has
> high variance (+/-25% on llama.cpp). Raw per-run samples are in
> [`docs/benchmark_results.json`](docs/benchmark_results.json).

### Changelog: how decode got from 1.7 to 12.5 tok/s

| Version | Decode t/s | What changed |
|---------|---------:|--------------|
| v7 | 1.68 | First coherent output (3 bugs fixed: S^T@q, Q/gate interleave, RoPE dim) |
| v10 | 6.98 | `setup_model.py` transpiler generates per-shape-specialized f32xf32 GEMV |
| v11 | 4.1-4.6 | Merged: weight repack + fused FFN + streaming (early 8.46 was 4 tokens only) |
| v12 | 11.7 | **W8A8 SignedDot kernel** (`vpsignb`, `vpmaddubsw`, `vpmaddwd`), +2.5x |
| v13 | 11.6-15.5 | SIMD activation quantization (+6%), the **current version** |

Full trajectory, including the versions that were silently broken:
[`docs/timeline.md`](docs/timeline.md).

## The kernel: W8A8 SignedDot

The decode bottleneck is GEMV, bound by memory bandwidth. The inner loop is three AVX2 intrinsics that replace ~20 float32 ops, one block of 32 int8 weights x 32 int8 activations at a time:

```
vpsignb     # resolve signs for signedxsigned
vpmaddubsw  # int8 x int8 -> int16 madd (16 -> 16)
vpmaddwd    # int16 x int16 -> int32 madd (8 -> 8)
            # then fp16 weight scale x fp32 activation scale -> f32 accumulate
```

The engine quantizes activations to Q8_0 once per GEMV call and reuses them across all rows (0.7% relative error vs f32). Weights sit in a tiled Q8_0 layout (8 consecutive rows interleaved), so the engine loads the activation vector once and reuses it. Full kernel + layout notes:
[`docs/performance.md`](docs/performance.md).

### The transpiler

`setup_model.py` is a small Python program (stdlib only) that reads the GGUF metadata and
**emits Mojo specialized at compile time**. It monomorphizes every GEMV to its exact `(rows, cols)` shape, so the compiler knows the tile sizes, unroll factors, and register usage.
That specialization turned a 4x decode win (v7 to v10). It specializes only to the 0.8B shape today (see Limitations).

## The architecture (why this isn't just another llama.cpp port)

24 layers, split **18 / 6**:

- **DeltaNet layers** (`layer % 4 != 3`): causal conv1d, then per-group q/k/v, then **delta-rule state
  update** `S *= decay; r = v - S^T k; S += beta * (k x r); out = S^T q`. State is a `[128x128]` matrix per
  group (16 groups x 18 layers = 288 matrices). No KV cache; the recurrent state is the memory.
- **Full-attention layers** (`layer % 4 == 3`): standard GQA softmax (8 Q / 2 KV heads, head dim
  256, partial RoPE on 64 dims, per-head RMSNorm), with a sliding KV cache.

Step-by-step forward passes for both layer types, with shapes and the exact gotchas, are in
[`docs/architecture.md`](docs/architecture.md). Every bug that bit during implementation, with the
detection signature that found it, is catalogued in
[`docs/correctness.md`](docs/correctness.md).

## The journal: wrong turns

Five published numbers in this project were **wrong**, and each sat unchallenged for a while
before I caught it. I keep them on purpose: understanding why a fast-but-broken version looks
correct is the lesson of the project.

| Claim | Reality |
|-------|---------|
| v8: correct output at 3.3 t/s | Output was garbage. A SIMD sign-extension bug (`Int8(UInt8)` zero-extends, not sign-extends). |
| v9: structured output at 4.45 t/s | Structured but wrong: a quantize/mask bug. Looked coherent, but wasn't. |
| v11: 8.46 t/s | Only 4 decode tokens. Sustained was 4.1-4.6. Short runs have +/-30% variance. |
| v13 threadpool: 19.7 t/s | Garbage output. Atomic work-stealing broke correctness. All numbers invalid. |
| v13 x4 layout: "garbage" | Actually *correct*. A short prompt was hitting EOS. (Still 18% slower, so abandoned.) |

The recurring lesson: **a fluent-looking stream is not proof of correctness**, and **a fast run on
few tokens is not a measurement**. Both traps (validating buggy code against a buggy tracer) cost
real time. See [`docs/timeline.md`](docs/timeline.md) and
[`docs/correctness.md`](docs/correctness.md).

## Run it

```bash
python setup_model.py          # installs Mojo (if missing), downloads the model (~800 MB),
                               # generates specialized code into build/, compiles ./qwen35
./qwen35 "Write a haiku about autumn leaves." -n 64
./qwen35 "your prompt" -n 128 --bench     # --bench: suppress streaming, just print stats
```

If Mojo isn't in `PATH`, it installs it via [`uv`](https://docs.astral.sh/uv/).

The engine loads `model.gguf` from the current directory. Input is a single positional prompt;
`-n` sets max decode tokens. Sampling defaults to the Qwen3 thinking-model values
(`temp=0.7, top_k=20, top_p=0.8`).

To regenerate the specialized engine after changing the model or the transpiler:

```bash
rm -rf build/                  # clean slate
python setup_model.py          # regenerates build/ and rebuilds
```

## Limitations

- **One model.** Only Qwen3.5-0.8B (Q8_0). The transpiler specializes to its exact shape;
  supporting other sizes means generalizing `setup_model.py`. Out of scope for now.
- **Slow prefill.** The prefill path runs at ~0.15-0.25x of llama.cpp (about 4-7x behind) because
  the engine has no batched GEMM. Decode is the optimized path.
- **Tokenizer via Python interop.** BPE uses Python's `regex` module (a pip dependency, not
  stdlib). The import itself is fast (<0.1 s), but it adds a runtime dependency on CPython.
  A native Mojo tokenizer would remove this ([`docs/opportunities.md`](docs/opportunities.md), O12).
- **One CPU class tested.** I developed it on a 15 W AVX2 laptop (no AVX-512/VNNI). I chose Mojo
  for its cross-backend support, but only the x86-64 AVX2 path is exercised here.
- **No batching, no server, no GPU.** Single-stream token-by-token decode only.

Unfinished directions (DeltaNet state persistence, radix-tree KV reuse, online LoRA) live as
research notes in [`docs/research/`](docs/research/): aspirational, not implemented.

## Why Mojo?

Partly to learn it: Mojo pitches itself as a Pythonic path to systems performance (and a CUDA
alternative). This project stress-tested that claim on a real workload. The language delivered
per-shape specialization and ergonomic SIMD. It also exposed rough edges: mutex barriers in
`parallelize`, a ~10% runtime instrumentation tax, and silent pointer aliasing. The gotchas are
logged in [`docs/performance.md`](docs/performance.md) under "Mojo Performance Gotchas."

## Repository layout

```
qwen35.mojo/
├── setup_model.py                 # one-command setup: mojo check + model download + codegen + build
├── _components.mojo               # kernels: rmsnorm, gemv, W8A8 dot, rope, softmax, quantize
├── gguf_loader.mojo               # Q8_0 parser + tiled weight repack
├── tokenizer.mojo                 # BPE (Python regex interop)
├── .mojo-version                  # pinned Mojo compiler version
├── .gitignore
├── LICENSE
├── README.md
├── build/                         # generated by setup_model.py (gitignored)
│   ├── model_config.mojo          #   architecture constants (generated)
│   ├── run_inference.mojo         #   the engine: prefill/decode loop, all 24 layers (generated)
│   ├── _components.mojo           #   copied from root
│   ├── gguf_loader.mojo           #   copied from root
│   └── tokenizer.mojo             #   copied from root
└── docs/
    ├── architecture.md            # hybrid DeltaNet + GQA forward passes, with shapes
    ├── performance.md             # W8A8 kernel, hardware profile, bottleneck breakdown
    ├── correctness.md             # every bug found + its detection signature
    ├── timeline.md                # version map + the wrong-turns ledger
    ├── opportunities.md           # ranked, unstarted optimization ideas
    ├── benchmark_results.json     # raw per-run benchmark samples
    └── research/                  # aspirational notes (continual learning, radix-tree KV)
```

## License

MIT, see [LICENSE](LICENSE)
