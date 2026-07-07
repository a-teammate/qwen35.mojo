============================================================
CONTINUAL OFFLINE LEARNING — RESEARCH NOTES V2
============================================================

PROBLEM
============================================================

A Mojo-based inference engine like qwen35.mojo runs a static GGUF model.
Weights are fixed at build time. All "memory" is ephemeral:
  - KV cache: discarded after each conversation (6 full attn layers)
  - DeltaNet state: reset between sessions (18 DeltaNet layers)

Every conversation starts from zero. New knowledge requires external
retrain/fine-tune + new GGUF. Can the model evolve during inference,
accumulating knowledge from interactions without external retraining?

This would eliminate context compaction (sliding window eviction,
summarization) — the model would internalize what it learns.

Architecture context (from spec_v3.md):
  - Qwen3.5-0.8B: hybrid DeltaNet (18 layers) + sigmoid scan attention (6 layers)
  - DeltaNet: fixed-size recurrent state S, no KV cache
  - Full attention: FP32 KV cache, sliding window eviction
  - All execution compile-time specialized, no runtime config

============================================================
UNIFYING INFRASTRUCTURE: THE RADIX TREE
============================================================

A radix tree (space-efficient prefix trie) becomes the central data
structure connecting all learning strategies. Originates from SGLang's
RadixAttention (Zheng et al., 2024) for KV cache reuse, but its
applications here go far beyond caching.

Reference:
  SGLang: Efficiently Programming Large Language Models
  Zheng et al., Jan 2024
  https://www.lmsys.org/blog/2024-01-17-sglang/
  https://arxiv.org/abs/2312.07104
  Core idea: radix tree maps token-sequence prefixes → KV cache tensors.
  LRU eviction. Automatic prefix matching. Replaces ad-hoc cache
  management with a single data structure.

Radix tree stores at each node:
  - Token sequence (edge label) + node metadata
  - KV cache tensors (full attention layers, 6 of 24)
  - DeltaNet state snapshot (18 layers, ~144 KB per snapshot)
  - Optional: LoRA adapter weights (Strategy 1), experience data (replay)
  - Access statistics (hit count, last access time, fork count)

All operations are CPU-side (tree structure) with GPU-side tensor
storage managed by LRU eviction. Maintenance overhead is negligible.

------------------------------------------------------------
RADIX TREE APPLICATIONS
------------------------------------------------------------

1. AUTOMATIC KV CACHE REUSE (SGLang's original insight)
   Prefix matching across conversations. Shared system prompts, common
   question templates, repeated tool-call patterns — computed once,
   reused automatically. Up to 5x throughput improvement demonstrated
   by SGLang on similar workloads.

2. DELTANET STATE SNAPSHOTS AT PREFIX BOUNDARIES
   Map token-sequence prefixes → DeltaNet state snapshots. When a new
   conversation shares a prefix with a prior one, restore the DeltaNet
   state from that snapshot. The model "remembers" what it learned
   during prior context processing.
   This extends Strategy 3 from "one global persistent state" to "a
   tree of context-dependent states" — different topics produce
   different (and useful) state configurations, all indexed by prefix.

3. SMARTER EVICTION THAN SLIDING WINDOW
   Current spec uses sliding window eviction (spec_v3 #7). Radix tree
   LRU eviction is strictly better: evicts by reuse value, not age.
   System prompts and common templates stay cached forever. One-off
   queries get evicted first. Direct upgrade, no quality loss.

4. SPECULATIVE DECODING FROM CACHED BRANCHES
   During decode, if current generation matches a cached tree path,
   speculatively emit cached continuation tokens and verify in a
   single batch forward pass. If the model agrees (likely for
   deterministic/routine outputs), skip dozens of decode steps.
   Frees compute for learning operations (LoRA updates, extra
   self-consistency samples).

5. COMPRESSED EXPERIENCE REPLAY BUFFER
   100 conversations sharing a system prompt store that prefix once.
   Tree structure IS the compression. Leaves hold experience data
   (corrections, outcomes). Replay becomes nearly free because KV
   cache for shared prefixes is already in the tree.
   Directly addresses catastrophic forgetting in Strategy 1 without
   a separate replay buffer implementation.

6. LORA ADAPTER ROUTING BY PREFIX
   As the model accumulates LoRA updates from different domains
   (Strategy 1), the radix tree becomes a router: "prompts matching
   prefix X use LoRA_X." Tree naturally clusters conversations by
   topic. Domain-adaptive LoRA selection without a separate classifier.

7. MEMORY-MAPPED PERSISTENT STORE
   Radix tree as index into an mmap'd file of state data (KV cache +
   DeltaNet snapshots + LoRA weights). On startup, load only the tree
   (small, CPU-side). State data demand-paged from disk. LRU eviction
   manages GPU memory budget. Persistent memory without loading
   everything into GPU at once.

------------------------------------------------------------
SELF-SUPERVISED LOSS FROM TREE TOPOLOGY
------------------------------------------------------------

The radix tree is not just a cache — it is a self-supervised loss
signal generator. Its structural features directly encode learning
signals without external labels. The topology IS the curriculum.

  Tree feature                    Loss signal           Meaning
  ────────────────────────────── ───────────────────── ──────────────
  Fork points (same prefix →     Disagreement → high   Model uncertain
    multiple completions)         loss at that prefix   here, needs learning
  Branches converging to same    Consensus → low loss,  Multiple paths agree
    endpoint                      positive example      = high confidence
  Cache hit frequency            Hit rate = reward     Useful outputs get
    (how often a branch is        signal                revisited
    revisited)
  LRU eviction events            Eviction = negative   Nobody needed this
                                  signal               output, wasted compute
  User corrections at leaves     Direct supervised     Explicit (input,target)
                                  loss (wrong → right)  pairs, indexed by prefix
  Deep unbranched paths          Overconfidence        May be memorized, not
                                  signal               understood
  Fork collapse over time        Learning succeeded    Fork → single confident
                                                        path = uncertainty resolved
  New forks at novel prefixes    New territory         Model encountering
                                                        unfamiliar domain

This is recursive: as the model learns, the tree reshapes. Forks
collapse into confident single paths. New forks appear at novel
prefixes. The tree and model co-evolve. No external labeling needed.

This loss signal feeds into:
  - Strategy 1: which prefixes to prioritize for LoRA gradient updates
  - Strategy 2: which contexts the hypernetwork should adapt for
  - Strategy 3: implicit — tree topology guides which DeltaNet state
    snapshots to retain vs. evict

============================================================
STRATEGY 1: ONLINE LORA (EXPLICIT GRADIENT UPDATES)
============================================================

Add low-rank adapter matrices beside each frozen weight. Update ONLY
LoRA parameters during inference via backpropagation.

Mechanism:
  1. Forward pass produces output, compute loss against correction
  2. Backprop through LoRA params only (thousands, not billions)
  3. SGD/Adam step on LoRA matrices
  4. Base GGUF weights unchanged

Learning signal:
  - Explicit user corrections ("no, the answer is X")
  - Tool-call ground truth (search API returns fact)
  - Any supervised (input, target) pair
  - Tree-derived signals: fork frequency, hit rate, eviction events

Radix tree integration:
  - Compressed replay buffer: shared prefixes stored once, replay
    forward passes are nearly free (KV cache reuse for shared prefixes)
  - LoRA routing: tree clusters conversations by prefix → domain-specific
    LoRA selection without a separate classifier
  - Budget allocation: hot prefixes (high hit rate) get more update
    steps; cold prefixes get less. Tree statistics reveal where to spend
    limited gradient computation budget
  - Forward pass acceleration: gradient computation requires repeated
    forward passes over similar prefixes. RadixAttention makes these
    nearly free by reusing cached KV tensors for the 6 full attn layers

Memory architecture:
  - GGUF weights: LONG-TERM memory (frozen)
  - LoRA matrices: MEDIUM-TERM memory (slowly updated)
  - Radix tree (KV + DeltaNet state): MEDIUM-TERM (persisted, LRU-evicted)
  - KV cache / DeltaNet state: SHORT-TERM memory (per-conversation)

Forgetting mitigation:
  Online-LoRA regularizes LoRA weights to stay close to previous
  values. Per-parameter importance estimates adapt in real-time.
  Radix tree replay buffer provides additional replay of rare but
  important experiences (anti-forgetting without separate buffer).

Mojo implementation:
  - Backward pass (autograd) through forward computation, LoRA params only
  - DeltaNet state and KV cache are activations — no gradient needed
  - LoRA matrices stored in FP32 (gradient updates need precision)
  - LoRA injection into quantized GEMV: dequant W, add LoRAΔW, proceed

Reference:
  Online-LoRA: Task-Free Online Continual Learning via Low Rank Adaptation
  Wei et al., WACV 2025
  https://openaccess.thecvf.com/content/WACV2025/papers/Wei_Online-LoRA_Task-Free_Online_Continual_Learning_via_Low_Rank_Adaptation_WACV_2025_paper.pdf
  Summary: Online weight regularization for ViT models that adapts
  parameter importance estimates in real-time. No replay buffer.
  Demonstrated on streaming visual tasks. Extends to any architecture
  with LoRA injection points.

============================================================
STRATEGY 2: HYPERNETWORK CONTEXT → WEIGHTS (NO BACKPROP)
============================================================

A second small network reads context and directly PRODUCES LoRA weight
deltas in a single forward pass. No gradient computation at runtime.

Mechanism:
  1. Feed teaching context into the hypernetwork
  2. Hypernetwork outputs LoRA adapter (small A, B matrices)
  3. Apply LoRA to frozen model
  4. Generate with adapted model

Hypernetwork is trained OFFLINE to learn "text → useful weight changes."
At inference, pure forward pass.

Radix tree integration:
  - Cache hypernetwork outputs: same context prefix → same LoRA adapter.
    Store in tree to avoid re-running hypernetwork for repeated contexts
  - Routing: tree topology reveals which contexts need adaptation
    (high fork count = uncertain = good candidate for hypernetwork LoRA)
  - Training signal: tree statistics (fork/consensus patterns) identify
    which contexts the hypernetwork should be trained to handle better

Memory architecture:
  - GGUF weights: LONG-TERM memory (frozen)
  - Hypernetwork output (LoRA): injected per-context, cached in radix tree
  - Radix tree (KV + DeltaNet state): MEDIUM-TERM (persisted, LRU-evicted)
  - KV cache / DeltaNet state: SHORT-TERM memory

Advantages:
  - Zero gradient computation at runtime
  - One forward pass = instant adaptation
  - Different LoRA for different contexts/domains (tree-routed)

Limitations:
  - Hypernetwork must be trained for specific base model
  - Quality limited by hypernetwork capacity
  - Separate training pipeline required

Mojo implementation:
  - Load hypernetwork weights alongside GGUF
  - Run hypernetwork forward pass when context changes (cache miss in tree)
  - Inject resulting LoRA into main model's GEMV calls
  - Separate inference pass before generation

Reference:
  SHINE: Scalable Hyper In-Context Network
  Liu et al., arXiv 2602.06358, Feb 2026
  https://arxiv.org/abs/2602.06358
  Summary: Trains a hypernetwork that maps arbitrary text to LoRA
  parameters in a single forward pass. No fine-tuning of the target
  model required. Two-pass system: memory extraction then LoRA
  generation. Demonstrated on diverse NLP tasks.

============================================================
STRATEGY 3: DELTANET STATE AS IMPLICIT LEARNING (ZERO ARCHITECTURE CHANGE)
============================================================

For linear attention transformers, in-context learning IS mathematically
equivalent to gradient descent on weights. No new components needed.

Key result: linear attention performing ICL is literally running implicit
gradient descent. The attention mechanism IS the learning algorithm.
The key-value cache IS the training data. The attention weights ARE
the gradient.

For DeltaNet specifically, the recurrence:
  S(t) = β·S(t-1) + α·k(t) ⊗ v(t)
is a fast-weight update. State S is a learned representation modified
by every token. When the model processes a correction in context, the
DeltaNet state update IS the learning.

Mechanism:
  1. Do NOT reset DeltaNet state between conversations
  2. Feed corrections and feedback into the model as normal tokens
  3. DeltaNet layers update their state automatically
  4. Subsequent generation is influenced by accumulated state
  5. Optionally: periodically distill accumulated state into permanent
     weight changes (LoRA or full fine-tune)

No backpropagation needed for the online part. The DeltaNet forward
pass IS the learning. Qwen3.5's architecture already has this —
75% of layers are literally a learning algorithm.

Radix tree integration:
  - State snapshots indexed by prefix: instead of one global persistent
    state, the tree stores DeltaNet state snapshots at key prefix
    boundaries. New conversation matching a prior prefix restores the
    appropriate state — context-dependent memory, not global memory
  - Tree topology guides which snapshots to retain (hot prefixes) vs.
    evict (cold/unvisited). LRU eviction IS forgetting — automatic,
    principled, no manual tuning
  - Conv1d circular buffer (kernel=4, per DeltaNet layer) also
    persisted alongside state snapshots in the tree
  - This gives the 6 full attention layers their own persistent memory
    via KV cache reuse, and the 18 DeltaNet layers theirs via state
    snapshots — unified by one data structure

Memory architecture:
  - GGUF weights: LONG-TERM memory (frozen)
  - Radix tree (KV cache + DeltaNet state): CONTINUOUS memory
    (persisted across sessions, LRU-evicted)
  - Active DeltaNet state / KV cache: EPISODIC (per-conversation)

Limitations:
  - DeltaNet state has finite capacity (144 KB per snapshot for 0.8B)
  - Old knowledge overwritten by β decay gate
  - No explicit loss signal — learning is unsupervised (mitigated by
    tree topology signals)
  - Quality of "learned" knowledge unverified

Mojo implementation:
  - Simplest start: don't call state.reset() between sessions
  - Persist deltanet_state to disk between runs
  - May need to increase state capacity for longer retention
  - Conv1d circular buffer also needs persistence
  - Radix tree: CPU-side index, mmap'd state file on disk

Reference:
  Exact Conversion of In-Context Learning to Model Weight Updates
  in Linear Transformers
  Mahankali et al., ICML 2024
  https://arxiv.org/abs/2406.02847
  Summary: Proves mathematical equivalence between in-context
  learning in linear transformers and gradient descent on model
  weights. ICL tokens can be exactly converted to weight updates.

Reference:
  Linear Transformers Are Secretly Fast Weight Programmers
  Schlag et al., ICML 2021
  https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf
  Summary: Linear attention with outer-product updates is equivalent
  to a fast weight programmer. The attention layer reprograms its
  own weights based on context. Foundation for understanding DeltaNet
  as a learning system.

============================================================
FULL MEMORY ARCHITECTURE (ALL STRATEGIES COMBINED)
============================================================

  Tier       Storage                     Persistence      Update cost
  ───────── ─────────────────────────── ──────────────── ──────────────
  LONG       GGUF weights               Frozen           Retraining
  MEDIUM     Radix tree: KV cache       Disk-persisted   Zero (reuse)
             Radix tree: DeltaNet       LRU-evicted      Zero (restore)
               state snapshots
             Radix tree: LoRA weights   Per-domain       Backward pass
             Radix tree: experience     Compressed       User interaction
               replay data
  SHORT      Active KV cache            Per-conversation  Standard
             Active DeltaNet state      Per-conversation  Standard
  EPHEMERAL  Activation scratch         Per-token         Standard

The radix tree is the single index managing all medium-term memory.
One data structure, one eviction policy, one persistence mechanism.
LRU eviction manages GPU memory budget across all stored tensor types.

============================================================
SELF-SUPERVISED LOSS SIGNALS (ALTERNATIVES TO RL)
============================================================

RL (PPO, GRPO, DPO) is one way to get a learning signal, but overkill
for occasional finetuning. Alternatives:

1. TREE-DERIVED SIGNALS (novel, this work)
   Tree topology provides loss without external labels. Fork frequency
   = uncertainty. Convergence = consensus. Hit rate = utility.
   Eviction = waste. Co-evolves with model. See "Self-Supervised Loss
   from Tree Topology" section above.

2. SUPERVISED LEARNING ON CORRECTIONS
   User says "no, the answer is X" → (prompt, X) is a training
   example. Cross-entropy loss. Simple, effective. Stored in radix
   tree as experience data at the conversation's leaf node.

3. SURPRISE-DRIVEN REPLAY
   Track which tokens the model predicted poorly (high loss).
   Replay those examples with higher weight during update steps.
   Radix tree compression makes replay efficient — shared prefixes
   computed once. Focuses limited update budget on what the model
   doesn't know.

   Reference:
   Surprise-Driven Prioritised Replay for Continual LLM Learning
   OpenReview, 2025
   https://openreview.net/pdf?id=IgZWU75BLL

4. SELF-SUPERVISED CONSISTENCY
   Generate multiple completions from same prefix, check which are
   self-consistent, reinforce those. RadixAttention makes this cheap:
   prefix KV cache shared, only decode portion repeated per sample.
   Fork nodes in radix tree are exactly these multi-sample points.
   No external signal needed.

5. IMPLICIT LEARNING (Strategy 3)
   No explicit learning signal. The forward pass itself performs
   gradient-like updates. Closest to "the model evolves itself."
   Tree-derived topology signals provide optional supervision.

============================================================
RECOMMENDED PATH FOR QWEN35.MOJO
============================================================

Phase 0: Radix Tree Infrastructure
  - Implement radix tree (CPU-side, prefix trie with LRU eviction)
  - Replace sliding window KV cache eviction with radix tree LRU
  - Automatic KV cache reuse across conversations (full attn layers)
  - Disk persistence of tree index + mmap'd tensor storage
  - Observe cache hit rates and throughput improvement
  - Risk: implementation complexity, memory management

Phase 1: Strategy 3 + Radix Tree State Snapshots
  - Persist DeltaNet state between sessions (don't reset)
  - Store DeltaNet state snapshots in radix tree at prefix boundaries
  - Restore matching snapshots on prefix hit
  - Observe whether accumulated/restore state improves generation
  - Both layer types now have persistent memory via one data structure
  - Risk: state overflow, quality degradation, snapshot coherence

Phase 2: Tree-Derived Loss Signals
  - Instrument radix tree to track fork counts, hit rates, eviction events
  - Log tree topology statistics alongside generation quality metrics
  - Correlate tree features with generation quality (manual evaluation)
  - This is research: validate that tree topology actually predicts
    generation quality before building learning on top of it
  - Risk: correlation may be weak, signal may be noisy

Phase 3: Strategy 1 (Online LoRA) + Tree-Guided Updates
  - Add LoRA injection points to each weight matrix
  - Implement backward pass (autograd) for LoRA params only
  - Use tree-derived signals to prioritize which prefixes get updates
  - Use radix tree as compressed replay buffer for forgetting mitigation
  - Risk: autograd complexity, forgetting, gradient quality in Q4/Q5 regime

Phase 4: Self-Consistency + Speculative Decoding
  - Sample multiple completions at fork points (cheap via KV reuse)
  - Consensus as self-supervised training signal
  - Speculative decoding from cached tree branches
  - Freed compute cycles fund LoRA update steps
  - Risk: sample quality, verification cost

Phase 5: Strategy 2 (Hypernetwork) — if needed
  - Train or obtain SHINE-style hypernetwork for Qwen3.5
  - Cache hypernetwork outputs in radix tree
  - Route contexts to cached adapters via prefix matching
  - Risk: separate training pipeline, hypernetwork quality

============================================================
REFERENCES
============================================================

SGLang / RadixAttention:
  Zheng et al., Jan 2024
  https://www.lmsys.org/blog/2024-01-17-sglang/
  https://arxiv.org/abs/2312.07104

Online-LoRA:
  Wei et al., WACV 2025
  https://openaccess.thecvf.com/content/WACV2025/papers/Wei_Online-LoRA_Task-Free_Online_Continual_Learning_via_Low_Rank_Adaptation_WACV_2025_paper.pdf

SHINE (Hypernetwork → LoRA):
  Liu et al., arXiv 2602.06358, Feb 2026
  https://arxiv.org/abs/2602.06358

ICL = Gradient Descent (linear transformers):
  Mahankali et al., ICML 2024
  https://arxiv.org/abs/2406.02847

Linear Transformers = Fast Weight Programmers:
  Schlag et al., ICML 2021
  https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf

Surprise-Driven Replay:
  OpenReview, 2025
  https://openreview.net/pdf?id=IgZWU75BLL

============================================================
END OF RESEARCH NOTES V2
============================================================
