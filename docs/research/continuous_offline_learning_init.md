============================================================
CONTINUAL OFFLINE LEARNING — RESEARCH NOTES
============================================================

PROBLEM
============================================================

A Mojo-based inference engine like QuickQwen runs a static GGUF model.
The weights are fixed at build time. All "memory" is ephemeral:
  - KV cache: discarded after each conversation
  - DeltaNet state: reset between sessions

This means the model cannot learn from its interactions. Every
conversation starts from zero. To get new knowledge into the model,
you must retrain/fine-tune externally and produce a new GGUF.

The question: can the model "evolve" during inference, accumulating
knowledge from interactions without external retraining? This would
eliminate the need for context compaction (sliding window eviction,
summarization) — the model would internalize what it learns.

============================================================
THREE STRATEGIES
============================================================

------------------------------------------------------------
STRATEGY 1: ONLINE LORA (EXPLICIT GRADIENT UPDATES)
------------------------------------------------------------

Add low-rank adapter matrices (LoRA) beside each frozen weight.
Update ONLY the LoRA parameters during inference via backpropagation.

Mechanism:
  1. Forward pass produces output, compute loss against correction
  2. Backprop through LoRA params only (thousands, not billions)
  3. SGD/Adam step on LoRA matrices
  4. Base GGUF weights unchanged

Learning signal:
  - Explicit user corrections ("no, the answer is X")
  - Tool-call ground truth (search API returns fact)
  - Any supervised (input, target) pair

Memory architecture:
  - GGUF weights: LONG-TERM memory (frozen)
  - LoRA matrices: MEDIUM-TERM memory (slowly updated)
  - KV cache / DeltaNet state: SHORT-TERM memory (per-conversation)

Forgetting mitigation:
  Online-LoRA regularizes LoRA weights to stay close to previous
  values. No replay buffer needed. The regularization adapts
  per-parameter importance estimates.

Mojo implementation implications:
  - Need a backward pass (autograd) through the forward computation
  - Only track gradients for LoRA matrices, not full weights
  - DeltaNet state and KV cache are just activations — no gradient needed
  - LoRA matrices must be stored in FP32 (gradient updates need precision)

Reference:
  Online-LoRA: Task-Free Online Continual Learning via Low Rank Adaptation
  Wei et al., WACV 2025
  https://openaccess.thecvf.com/content/WACV2025/papers/Wei_Online-LoRA_Task-Free_Online_Continual_Learning_via_Low_Rank_Adaptation_WACV_2025_paper.pdf
  Summary: Online weight regularization for ViT models that adapts
  parameter importance estimates in real-time. No replay buffer.
  Demonstrated on streaming visual tasks. Extends to any architecture
  with LoRA injection points.

------------------------------------------------------------
STRATEGY 2: HYPERNETWORK CONTEXT → WEIGHTS (NO BACKPROP)
------------------------------------------------------------

A second small network (hypernetwork) reads context and directly
PRODUCES LoRA weight deltas in a single forward pass. No gradient
computation at runtime.

Mechanism:
  1. Feed teaching context into the hypernetwork
  2. Hypernetwork outputs LoRA adapter (small A, B matrices)
  3. Apply LoRA to frozen model
  4. Generate with adapted model

The hypernetwork is trained OFFLINE to learn the mapping
"text → useful weight changes." At inference, it's a pure forward pass.

Memory architecture:
  - GGUF weights: LONG-TERM memory (frozen)
  - Hypernetwork output (LoRA): injected per-context, ephemeral
  - KV cache / DeltaNet state: SHORT-TERM memory

Advantages:
  - Zero gradient computation at runtime
  - One forward pass of hypernetwork = instant adaptation
  - Can produce different LoRA for different contexts/domains

Limitations:
  - Hypernetwork must be trained for your specific base model
  - Quality of adaptation limited by hypernetwork capacity
  - Separate training pipeline required

Mojo implementation implications:
  - Load hypernetwork weights alongside GGUF
  - Run hypernetwork forward pass when context changes
  - Inject resulting LoRA into the main model's GEMV calls
  - Could be implemented as a separate inference pass before generation

Reference:
  SHINE: Scalable Hyper In-Context Network — Mapping Context to LoRA
  in a Single Pass
  Liu et al., arXiv 2602.06358, Feb 2026
  https://arxiv.org/abs/2602.06358
  Summary: Trains a hypernetwork that maps arbitrary text to LoRA
  parameters in a single forward pass. No fine-tuning of the target
  model required. Demonstrated on diverse NLP tasks. The hypernetwork
  uses a two-pass system: memory extraction then LoRA generation.

------------------------------------------------------------
STRATEGY 3: DELTANET STATE AS IMPLICIT LEARNING (ZERO ARCHITECTURE CHANGE)
------------------------------------------------------------

The deepest insight: for certain transformer architectures, in-context
learning IS mathematically equivalent to gradient descent on weights.
No new components needed.

The key result: linear attention transformers performing in-context
learning are literally running an implicit gradient descent algorithm.
The attention mechanism IS the learning algorithm. The key-value cache
IS the training data. The attention weights ARE the gradient.

For DeltaNet specifically, the recurrence:
  S(t) = β·S(t-1) + α·k(t) ⊗ v(t)
is a fast-weight update. The state S IS a learned representation
modified by every token. When the model processes a correction in
context, the DeltaNet state update IS the learning.

Mechanism:
  1. Do NOT reset DeltaNet state between conversations
  2. Feed corrections and feedback into the model as normal tokens
  3. DeltaNet layers update their state automatically
  4. Subsequent generation is influenced by accumulated state
  5. Optionally: periodically distill accumulated state into permanent
     weight changes (LoRA or full fine-tune) by running many examples

No backpropagation needed for the online part. The DeltaNet forward
pass IS the learning. Qwen3.5's architecture already has this —
75% of layers are literally a learning algorithm.

Memory architecture:
  - GGUF weights: LONG-TERM memory (frozen)
  - DeltaNet state: CONTINUOUS memory (persists across sessions)
  - KV cache: EPISODIC memory (per-conversation, for full attention layers)

Limitations:
  - DeltaNet state has finite capacity (144 KB for Qwen3.5-0.8B)
  - Old knowledge gets overwritten by β decay gate
  - No explicit loss signal — learning is unsupervised
  - Quality of "learned" knowledge unverified

Mojo implementation implications:
  - Simplest approach: just don't call state.reset() between sessions
  - Persist deltanet_state to disk between runs
  - May need to increase state capacity for longer retention
  - The conv1d circular buffer also needs persistence

Reference:
  Exact Conversion of In-Context Learning to Model Weight Updates
  in Linear Transformers
  Mahankali et al., ICML 2024
  https://arxiv.org/abs/2406.02847
  Summary: Proves a mathematical equivalence between in-context
  learning in linear transformers and gradient descent on model
  weights. Shows that ICL tokens can be exactly converted to weight
  updates. The attention mechanism implements an implicit learning
  algorithm.

Reference:
  Linear Transformers Are Secretly Fast Weight Programmers
  Schlag et al., ICML 2021
  https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf
  Summary: Shows that linear attention with outer-product updates
  is equivalent to a fast weight programmer. The attention layer
  reprograms its own weights based on context. Foundation for
  understanding DeltaNet as a learning system.

============================================================
ALTERNATIVES TO REINFORCEMENT LEARNING
============================================================

RL (PPO, GRPO, DPO) is one way to get a learning signal, but
overkill for occasional finetuning. Alternatives:

1. SUPERVISED LEARNING ON CORRECTIONS
   User says "no, the answer is X" → (prompt, X) is a training
   example. Cross-entropy loss. Simple, effective, most "learning
   from human feedback" systems actually do this.

2. SURPRISE-DRIVEN REPLAY
   Track which tokens the model predicted poorly (high loss).
   Replay those examples with higher weight during update steps.
   Combats forgetting without RL. Focuses limited update budget
   on what the model doesn't know.

   Reference:
   Surprise-Driven Prioritised Replay for Continual LLM Learning
   OpenReview, 2025
   https://openreview.net/pdf?id=IgZWU75BLL
   Summary: Replay buffer prioritized by prediction surprise.
   Addresses catastrophic forgetting in sequential LLM updates.

3. SELF-SUPERVISED CONSISTENCY
   Generate multiple completions, check which ones are self-consistent,
   reinforce those. No external signal needed.

4. IMPLICIT LEARNING (Strategy 3 above)
   No explicit learning signal. The forward pass itself performs
   gradient-like updates. Closest to "the model evolves itself."

============================================================
RECOMMENDED PATH FOR QUICKQWEN
============================================================

Phase 1: Strategy 3 (persist DeltaNet state)
  - Zero code changes to the architecture
  - Just persist state to disk between sessions
  - Observe whether accumulated state improves generation
  - Risk: state overflow, quality degradation

Phase 2: Strategy 1 (online LoRA)
  - Add LoRA injection points to each weight matrix
  - Implement backward pass (autograd) for LoRA params only
  - Update on explicit user corrections
  - Risk: complexity, forgetting

Phase 3: Strategy 2 (hypernetwork)
  - Train or obtain a SHINE-style hypernetwork for Qwen3.5
  - Integrate as a pre-generation step
  - Risk: separate training pipeline, hypernetwork quality

============================================================
END OF RESEARCH NOTES
============================================================
