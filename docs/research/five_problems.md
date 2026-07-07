============================================================
FIVE PROBLEMS WE ARE SOLVING — RESEARCH NOTES
============================================================

This document captures the conceptual architecture that emerges when
you control BOTH the agent ReAct loop AND the inference engine, and
use a radix tree as the central index for model state, adaptation,
and self-supervised learning.

Companion document: research/continuous_offline_learning_with_radixtree.md
(contains the per-strategy details, LoRA inventory, S0 tuning data,
phase plan, and all original references)

============================================================
CONCEPT INDEX
============================================================

  Concept                             Origin / Reference
  ─────────────────────────────────── ──────────────────────────────────
  RadixAttention / KV cache reuse     SGLang, Zheng et al. 2024
    https://www.lmsys.org/blog/2024-01-17-sglang/
  DeltaNet state as implicit ICL      Mahankali et al. ICML 2024
    https://arxiv.org/abs/2406.02847
  ICL = fast weight programming       Schlag et al. ICML 2021
    https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf
  S0 Tuning (initial state PEFT)      Young, Apr 2025
    https://arxiv.org/abs/2604.01168
  Online-LoRA                         Wei et al. WACV 2025
  QLoRA (quantized base, FP32 LoRA)   Dettmers et al. 2023
  LoRA merging (task arithmetic)      Ilharco et al. 2023
  TIES / DARE merging                 Yadav et al. 2024 / Yu et al. 2024
  Self-consistency decoding           Wang et al. 2023
  Surprise-driven replay              OpenReview 2025
    https://openreview.net/pdf?id=IgZWU75BLL
  SHINE hypernetwork                  Liu et al. Feb 2026
    https://arxiv.org/abs/2602.06358
  Pijul (commutative patch VCS)       https://pijul.org/manual/theory.html
  Content-addressable file hashing    (design principle)
  Agent ReAct loop control            (our design)
  Subspaces as radix tree regions     (our concept)
  Tree-derived loss topology          (our concept)
  File-agent artifact control         (our concept)
  Typed code-mode language            (our design principle)
  Vector embeddings vs radix tree     (comparison below)
  Haystack problem / capacity         (recurrent state limitation)
  Progressive localization            (make remote → local)
  Trajectory steering                 S0 mechanism, Young 2025
  Execution trace as radix prefix     (our concept, replaces "messages")
  Incremental understanding via ΔNet  (our concept, additive not reset)
  Two-phase agent loop                (our design: localize → execute)
  File → DeltaNet state = compiled
    understanding (not RAG)           (our concept)

============================================================
FIVE PROBLEMS
============================================================

1. DOMAIN KNOWLEDGE WITHOUT RE-READING
2. EVOLVING AGENTS THAT IMPROVE AT RECURRING TASKS
3. GREAT TOOLCALLING FROM ACCUMULATED EXPERIENCE
4. PERSISTENT MEMORY WITHOUT CONTEXT WINDOW COST
5. CROSS-PROJECT KNOWLEDGE TRANSFER

============================================================
WALKTHROUGH: HOW EACH CONCEPT BUILDS ON THE PREVIOUS
============================================================

------------------------------------------------------------
STEP 0: THE STARTING POSITION
------------------------------------------------------------

A Mojo inference engine (qwen35.mojo) runs Qwen3.5-0.8B, a hybrid
model with 18 GatedDeltaNet layers (linear attention, fixed-size
recurrent state S) and 6 sigmoid scan attention layers (standard
KV cache). Weights are frozen at build time. All memory is
ephemeral — KV cache discarded after conversation, DeltaNet state
reset between sessions. Every conversation starts from zero.

We also control the agent ReAct loop — the orchestration layer
that decides when to call tools, when to think, when to fetch
resources. This is unusual: most systems treat the LLM as a black
box and the agent loop as a separate wrapper. We control both.

This changes everything.

------------------------------------------------------------
STEP 1: ELIMINATE "MESSAGES" AS THE FUNDAMENTAL UNIT
------------------------------------------------------------

Traditional LLM systems think in messages: system, user, assistant,
tool. The radix tree (from SGLang) indexes by message-sequence
prefix. But when you control both the agent loop and the model,
"messages" are a rendering artifact. The user sees a chat log, but
the model's real experience is a continuous stream:

  system_prompt → query → tool_call → tool_result → reasoning
  → correction → retry → success → next_query → ...

The radix tree indexes EXECUTION TRACES, not message lists. Two
agent runs that took the same tool-call path share a prefix even
if the user's phrasing was different. The tree captures BEHAVIORAL
PATTERNS — which tools were called, what results came back, what
the model did with them.

This is the first conceptual shift: from "message replay" to
"execution trace indexing."

------------------------------------------------------------
STEP 2: THE RADIX TREE BECOMES THE AGENT'S VERSION CONTROL
------------------------------------------------------------

A radix tree is append-only (nodes added, never overwritten),
preserves all alternatives (all branches coexist), is content-
addressable (states keyed by file hashes), and uses LRU eviction
(principled forgetting). These are exactly the properties of a
good version control system — specifically, properties shared
with Pijul (commutative patches, append-only, CRDT, no lost
alternatives).

Unlike git, which destroys alternative realities through merges
and rebases, the radix tree PRESERVES every experiment, every
failed approach, every dead end. Git hides history. The radix
tree keeps it, prioritized by actual reuse.

The radix tree IS the agent's VCS. No external dependency needed.
It tracks file versions, execution traces, model states, LoRA
checkpoints, and S0 snapshots — all in one structure.

The key property: nothing is ever destroyed, only deprioritized
by LRU eviction. Failed experiments contain valuable signal
("what NOT to do"). The tree preserves them until they're
genuinely irrelevant.

------------------------------------------------------------
STEP 3: FILE CHANGES ADD KNOWLEDGE, THEY DON'T RESET IT
------------------------------------------------------------

When a file changes (new content hash), the naive approach is to
discard the old DeltaNet state and start from zero. The right
approach: the model reads only the DIFF between old and new
versions, and the DeltaNet state updates incrementally.

This works because DeltaNet's recurrence IS additive:
  S(t) = β·S(t-1) + α·k(t)⊗v(t)

Processing diff(v1→v2) tokens on top of S_v1 gives S_v2. The β
decay preserves old understanding (gradually). The α gate adds
new understanding. The state at v3 contains trace understanding
from v1, weighted by β² — not a fresh start.

The radix tree stores the full state at each version. Restoration
is a single lookup. But the LEARNING SIGNAL is the delta:
S_v2 - S_v1 = "what the model learned from that diff." These
deltas are the training data for S0/LoRA updates.

The version chain is also a natural curriculum: v1→v2 was a simple
refactoring, v2→v3 was a bug fix, v3→v4 added a feature. Each
step's difficulty is organic — from real work, not synthetic data.

This means "file → DeltaNet state" is NOT RAG. RAG retrieves text
chunks. This restores PRE-COMPILED UNDERSTANDING — the model's
processing of a file is "compiled" into a compact state that can
be restored instantly. RAG says "here are some fragments that
seem relevant, re-read them." This says "here is my complete
understanding of this exact file, ready to use."

But compiled understanding does NOT replace session history. Five
reasons:

1. PRECISION OF RECALL: DeltaNet state gives "the gist" — topic,
   style, direction. It cannot retrieve "what filename was on line
   473." Session history gives verbatim recall. The state is a
   lossy compression of everything processed.

2. FULL ATTENTION NEEDS ACTUAL TOKENS: The 6 sigmoid scan layers
   can only attend to tokens physically in the KV cache. The radix
   tree caches KV from matching prefixes, but new context that
   diverges from all cached prefixes still needs explicit tokens.

3. THE HAYSTACK PROBLEM: DeltaNet state has fixed capacity (144 KB
   for 0.8B). As information accumulates, the β decay gate
   overwrites old knowledge. The model "sort of remembers" general
   patterns but cannot surface specific details. Information is
   present but irretrievable — and eventually overwritten entirely.

4. AUDITABILITY: Session history is human-readable. DeltaNet state
   is an opaque matrix. You cannot inspect what the model "knows"
   or verify correctness.

5. MODEL-SPECIFIC FRAGILITY: Session history is model-agnostic
   text. Change quantization, swap a LoRA checkpoint, adjust S0
   scaling — the state becomes invalid. History text survives.

The right framing: compiled understanding SUPPLEMENTS history, it
doesn't replace it. History gives precise recall and auditability.
Radix tree state gives accumulated patterns and speed. They handle
different things.

------------------------------------------------------------
STEP 4: SUBSPACES REPLACE SESSIONS
------------------------------------------------------------

Traditional systems have "sessions" — discrete conversations with
clear boundaries. In our design, the agent loop runs continuously.
There are no session resets. Instead, there are SUBSPACES —
regions of the radix tree where execution traces share similar
tool-call patterns, file contexts, and domain properties.

A subspace is NOT a separate conversation. It's a region of the
tree. "Code debugging for this project" is a subspace. "Search-
and-summarize tasks" is a subspace. "Working with library X" is
a subspace. These overlap — a task can be in multiple subspaces
simultaneously.

Each subspace accumulates its own:
  - LoRA adapter (learned patterns for that domain)
  - S0 initial state (trajectory steering for that domain)
  - DeltaNet state snapshots (understanding of files in that domain)
  - Execution statistics (fork rates, correction rates, hit rates)

Moving between subspaces is traversing the radix tree — finding
the closest matching prefix and restoring its state. No explicit
"context switch" needed. The tree IS the context switch.

------------------------------------------------------------
STEP 5: QLORA MERGING AT CONVERGENCE POINTS
------------------------------------------------------------

When two branches in the radix tree converge (different execution
paths, same successful outcome), their LoRA adapters can be merged.

This is well-studied: task arithmetic (Ilharco et al. 2023), TIES
merging (Yadav et al. 2024), DARE (Yu et al. 2024). LoRA adapters
are low-rank matrices — merging is just matrix arithmetic. Cheap.

What's novel here: the radix tree tells you WHICH adapters to merge
and HOW to weight them. Branches that converge to verified-successful
outcomes, weighted by hit rate (how often they're reused) and
confidence (low fork rate = high certainty). The merge weights come
from tree topology — no manual tuning.

After merging, the combined adapter is stored at the convergence
point. The two separate branch adapters are pruned. Each branch's
learning strengthens the merged result. This is continuous model
improvement through execution trace convergence.

QLoRA works on all layers, including DeltaNet. Base weights stay
quantized (Q4_K, Q5_K). LoRA adapters are FP32/BF16. Gradients
flow through dequantized weights but only update LoRA. The same
for every layer type — attn_qkv, ssm_alpha, ssm_beta, full
attention projections, FFN. The "FP32 for gradients" requirement
is about LoRA A,B matrices, not about which base weight they adapt.

For ssm_alpha [1024,16] and ssm_beta [1024,16]: LoRA at rank 8
adds ~8K params per matrix (half the original). Not very "low rank."
S0 tuning may be more parameter-efficient here.

------------------------------------------------------------
STEP 6: SELF-SUPERVISED LOSS FROM TREE TOPOLOGY
------------------------------------------------------------

The radix tree is not just a cache. Its structural features encode
learning signals without external labels:

  Fork points           → model uncertain, needs learning here
  Branch convergence    → consensus, high-confidence positive example
  Cache hit frequency   → utility reward signal
  LRU eviction events   → negative signal (nobody needed this)
  User corrections      → direct supervised (wrong→right) pairs
  Fork collapse         → learning succeeded
  New forks at novel
    prefixes             → new territory, model lacks knowledge

As the model learns, the tree reshapes. Forks collapse into single
confident paths. New forks appear at novel prefixes. The tree and
model co-evolve. The topology IS the curriculum.

These signals feed S0 tuning, LoRA updates, and guide which
DeltaNet state snapshots to retain vs. evict.

------------------------------------------------------------
STEP 7: LOCAL AND REMOTE TOOL SURFACES
------------------------------------------------------------

The agent interacts with tools. These split into two surfaces with
very different properties:

LOCAL TOOLS (fs, bash, build, test):
  - Deterministic: same input + same filesystem state → same output
  - Verifiable: exit codes, test results, compilation = ground truth
  - Fast: no network latency
  - Cacheable: (tool_call, fs_state_hash) → cached result

Every local tool interaction produces:
  1. A cacheable result (deterministic → memoizable)
  2. A verified training signal (exit code IS the label)
  3. A state checkpoint opportunity (snapshot after processing result)

Tool call memoization: same bash command, same filesystem hash →
skip execution entirely. Not just model inference caching — caching
the ACTUAL TOOL EXECUTION. The radix tree stores
hash(fs_state) + "make test" → exit_code=0 + captured understanding.

Free training pairs from execution traces: model suggests fix_1 →
fails. Model suggests fix_2 → passes. The radix tree captures the
entire trace. That's a verified (wrong, right) pair. No human
labeling. The compiler/test suite IS the labeler.

REMOTE TOOLS (web, fetch, search, download):
  - Non-deterministic: same URL → different content tomorrow
  - Unverifiable: no ground truth for web content
  - Slow: network latency
  - Cacheable with TTL and content-hash verification

THE "MAKE REMOTE LOCAL" PRINCIPLE:
Before execution, resolve all remote dependencies — download,
cache, index. Transform remote surface into local surface. This
is like compilation: transform high-level intent ("work with
library X") into concrete local state ("library X v2.3 downloaded,
docs cached, types resolved").

Once local:
  - Pre-resolved dependency trees become radix tree prefixes
  - Content-addressable cache: hash(doc_content) → understanding
  - Versioned knowledge: different library versions = different
    branches, each with its own cached understanding
  - Cross-project transfer: same library in different projects
    shares understanding via content hash

THE TWO-PHASE AGENT LOOP:
  Phase 1: LOCALIZE (remote → local, resolve, cache)
  Phase 2: EXECUTE (local-only, deterministic, memoizable,
            every outcome = training signal)

Phase 1 shrinks over time (more cache hits). Phase 2 gets faster
(more state restoration, more tool memoization).

------------------------------------------------------------
STEP 8: THE FILE-AGENT AND ARTIFACT CONTROL
------------------------------------------------------------

Instead of bash pipes and 100 small tools, the agent operates
through FILE-AGENTS — agents that control artifacts:

  - A file-agent for a folder (reads, writes, watches changes)
  - A file-agent for a browser (navigates, extracts, fills forms)
  - A file-agent for a shell (executes, captures output)
  - A file-agent for a radix tree of codebase forks (manages
    parallel experiments, merges successful ones)

When you want to make an edit, you don't call a bash tool. You
ask the file-agent for that artifact to do it. The file-agent:
  - Has full understanding of the artifact (DeltaNet state cached
    in radix tree keyed by content hash)
  - Can verify the edit (run tests, check compilation, etc.)
  - Can roll back if the edit fails (the radix tree preserves the
    previous state — no destructive overwrites)
  - Accumulates domain expertise (LoRA/S0 per subspace)

The file-agent is NOT a tool call. It's a persistent entity with
its own state, understanding, and adaptation history. The radix
tree indexes its accumulated experience with the artifact.

This connects to the "typed code-mode language" design: instead
of bash strings, tool calls are typed function calls. Every call
is cacheable by (function_id, args_hash, fs_state_hash). Same
function, same args, same environment → cached result, no re-
execution. The language provides the cache key structure.

------------------------------------------------------------
STEP 9: COMPARISON WITH VECTOR EMBEDDINGS
------------------------------------------------------------

Vector embeddings (ChromaDB, FAISS) and the radix tree solve
different problems:

  Property                Embeddings            Radix tree + ΔNet
  ────────────────────── ──────────────────── ────────────────────
  What's stored           Semantic fingerprint  Model processing
                          (vector)              state (understanding)
  What's matched          Approximate           Exact prefix or
                          similarity            content-hash match
  Granularity             Fixed at chunk        Natural (token
                          time                  sequence boundaries)
  Structure               Flat points in        Directed graph
                          vector space          (causal traces)
  Causality               None — points         Full — every edge
                          are independent       is a causal step
  Composability           No — embeddings       Yes — prefix
                          don't compose         sharing = understanding
                                                sharing
  Retrieval               "Find things LIKE     "Restore understanding
                          this"                 OF this exact thing"
  Versioning              No                    Yes — different
                                                hashes = different
                                                branches

Embeddings measure similarity. The radix tree captures CAUSALITY.
An embedding says "documents A and B are semantically close."
The radix tree says "processing A then calling tool X then
getting result Y led to outcome Z."

They should be COMBINED, not one replacing the other:
  - Embeddings: cold-start retrieval when no exact prefix match
    exists. "I've never seen this exact file, but I've processed
    similar ones — restore the closest understanding as a warm
    start, then update incrementally."
  - Radix tree: exact state restoration when a prefix or content-
    hash match exists. Zero approximation, zero re-processing.
  - Together: embeddings search CANDIDATES, radix tree provides
    EXACT matches. Embeddings handle the unknown; the radix tree
    handles the known.

------------------------------------------------------------
STEP 10: THE UNIFIED PICTURE — FIVE PROBLEMS, ONE ARCHITECTURE
------------------------------------------------------------

PROBLEM 1: DOMAIN KNOWLEDGE WITHOUT RE-READING
  Content-addressable state snapshots keyed by file hash.
  File → DeltaNet state → "compiled understanding." Open a
  project → restore understanding instantly. File changes →
  incremental diff processing, not full re-read. Different
  versions → different radix tree branches, each with cached
  understanding. Embeddings handle cold starts.

PROBLEM 2: EVOLVING AGENTS THAT IMPROVE AT RECURRING TASKS
  The radix tree captures execution traces. Successful patterns
  get reinforced via S0/LoRA updates. Failed patterns generate
  correction signals (from verified tool outcomes). The agent is
  measurably better at task X after doing task X N times. QLoRA
  merging at convergence points combines learning from different
  paths. Subspaces accumulate domain-specific adaptation.

PROBLEM 3: GREAT TOOLCALLING FROM ACCUMULATED EXPERIENCE
  Every tool call trace (context → call → result → model response)
  is in the radix tree. These ARE the training data for tool use.
  The model learns "in this context, with this filesystem state,
  calling tool X with args Y produces good outcomes." Local tools
  provide verified (exit code) training pairs for free. Remote
  tools, once localized, become deterministic and cacheable.
  File-agents accumulate expertise per artifact.

PROBLEM 4: PERSISTENT MEMORY WITHOUT CONTEXT WINDOW COST
  No more stuffing history into the prompt. Understanding persists
  as DeltaNet state + KV cache snapshots in the radix tree. The
  prompt can be tiny because the model already knows the context.
  Session history is still needed for: precision of recall (exact
  facts), full attention tokens, verbatim retrieval, auditability,
  and model-agnostic persistence. But the radix tree handles
  accumulated patterns, learned preferences, and speed.

PROBLEM 5: CROSS-PROJECT KNOWLEDGE TRANSFER
  Content-addressable cache: same library in different projects
  shares understanding via content hash. Learning from debugging
  library X in project A improves the model for library X in
  project B. The radix tree naturally enables this — no explicit
  transfer mechanism needed. QLoRA merging combines adapters from
  different projects that converged on the same knowledge.

============================================================
ONE ARCHITECTURE, NOT FIVE SEPARATE SYSTEMS
============================================================

  Radix tree (CPU-side index, mmap'd state storage)
    ├── Execution trace prefixes (not messages)
    ├── File content hashes → DeltaNet state snapshots
    ├── File content hashes → KV cache (6 full attn layers)
    ├── Per-subspace S0 initial states (48 KB each)
    ├── Per-subspace LoRA adapters (merged at convergence)
    ├── Tool call memoization cache (local, deterministic)
    ├── Remote → local compiled knowledge (content-addressable)
    ├── Experience replay data (compressed by prefix sharing)
    ├── Loss signal topology (fork, convergence, hit, eviction)
    └── Embedding index (cold-start retrieval over tree nodes)

  Agent ReAct loop (controls both orchestration and model)
    ├── Phase 1: LOCALIZE (resolve remote → local, cache)
    ├── Phase 2: EXECUTE (local-only, memoizable, verifiable)
    ├── File-agents (persistent, per-artifact, radix-tree-backed)
    ├── Typed code-mode language (cacheable function calls)
    └── No session boundaries — subspaces are tree regions

  Model adaptation surfaces (orthogonal, complementary)
    ├── S0 tuning → initial state (trajectory steering)
    ├── LoRA α/β → learning/forgetting rates (meta-learning)
    ├── LoRA QKV → state content (what enters recurrence)
    ├── LoRA FFN → channel mixing (standard transformer)
    └── All updates guided by tree-derived loss signals

============================================================
NEW REFERENCES (beyond V2 document)
============================================================

Pijul (commutative patch VCS, CRDT, append-only):
  https://pijul.org/manual/theory.html
  Relevant: patch theory informs radix tree design. Append-only,
  commutative, no lost alternatives. The radix tree already has
  these properties without Pijul as a dependency.

Jujutsu (jj) (Git-compatible VCS, better alternative handling):
  https://github.com/jj-vcs/jj
  Relevant: shows that preserving alternatives is achievable in
  practical VCS design. Git-compatible backend.

QLoRA:
  Dettmers et al., 2023
  Relevant: quantized base weights + FP32 LoRA. Works on ALL
  layer types including DeltaNet.

LoRA Task Arithmetic:
  Ilharco et al., 2023
  Relevant: merging LoRA adapters. W_merged = W_base + λ·ΔW.
  Our addition: merge weights from radix tree topology.

TIES Merging:
  Yadav et al., 2024
  Relevant: resolves sign conflicts in LoRA merging.

DARE:
  Yu et al., 2024
  Relevant: random drop + rescale for LoRA merging.

Self-Consistency:
  Wang et al., 2023
  Relevant: sample multiple completions, consensus as signal.
  Radix tree fork nodes are exactly these multi-sample points.

============================================================
END OF RESEARCH NOTES
============================================================
