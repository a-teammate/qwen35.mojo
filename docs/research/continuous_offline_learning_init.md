============================================================
CONTINUAL OFFLINE LEARNING: RESEARCH NOTES
============================================================

PROBLEM
============================================================

A Mojo inference engine like qwen35.mojo runs a static GGUF model.
The weights are fixed at build time. Every conversation starts cold.
All state is ephemeral:
  - KV cache: discarded after each conversation
  - DeltaNet state: reset between sessions

You cannot teach the model anything during inference. To add
knowledge, you must retrain or fine-tune externally and ship a new
GGUF.

The question: can the model learn from its interactions and
internalize knowledge without external retraining? If so, we can
drop context compaction (sliding-window eviction, summarization).
The model itself becomes the memory.

============================================================
THREE STRATEGIES
============================================================

STRATEGY 1: ONLINE LORA (EXPLICIT GRADIENT UPDATES)

Add low-rank adapter matrices (LoRA) beside each frozen weight.
Backprop updates only the LoRA parameters during inference.

Mechanism:
  1. Forward pass produces output; compute loss against a correction
  2. Backprop through LoRA params only (thousands, not billions)
  3. SGD or Adam step on the LoRA matrices
  4. Base GGUF weights unchanged

Learning signal:
  - Explicit user corrections ("no, the answer is X")
  - Tool-call ground truth (search API returns a fact)
  - Any supervised (input, target) pair

Memory architecture:
  - GGUF weights: LONG-TERM memory (frozen)
  - LoRA matrices: MEDIUM-TERM memory (slowly updated)
  - KV cache / DeltaNet state: SHORT-TERM memory (per-conversation)

Forgetting mitigation:
  Online-LoRA regularizes LoRA weights to stay near previous values.
  No replay buffer needed. The regularization adapts per-parameter
  importance estimates.

Mojo implementation implications:
  - Need a backward pass (autograd) over the forward computation
  - Track gradients only for LoRA matrices, not the full weights
  - DeltaNet state and KV cache are activations; no gradient needed
  - LoRA matrices must be FP32 (gradient updates need precision)

Reference:
  Online-LoRA: Task-Free Online Continual Learning via Low Rank Adaptation
  Wei et al., WACV 2025
  https://openaccess.thecvf.com/content/WACV2025/papers/Wei_Online-LoRA_Task-Free_Online_Continual_Learning_via_Low_Rank_Adaptation_WACV_2025_paper.pdf
  Summary: Online weight regularization for ViT models that adapts
  parameter importance in real time. No replay buffer. Shown on
  streaming visual tasks. Extends to any architecture with LoRA
  injection points.


STRATEGY 2: HYPERNETWORK MAPS CONTEXT TO WEIGHTS (NO BACKPROP)

A second small network (hypernetwork) reads context and produces
LoRA weight deltas in a single forward pass. No gradient work at
runtime.

Mechanism:
  1. Feed teaching context into the hypernetwork
  2. Hypernetwork outputs a LoRA adapter (small A, B matrices)
  3. Apply LoRA to the frozen model
  4. Generate with the adapted model

The hypernetwork is trained offline to learn the mapping from text
to useful weight changes. At inference it is a pure forward pass.

Memory architecture:
  - GGUF weights: LONG-TERM memory (frozen)
  - Hypernetwork output (LoRA): injected per-context, ephemeral
  - KV cache / DeltaNet state: SHORT-TERM memory

Advantages:
  - Zero gradient computation at runtime
  - One hypernetwork forward pass equals instant adaptation
  - Different LoRA per context or domain

Limitations:
  - Hypernetwork must be trained for your specific base model
  - Adaptation quality is bounded by hypernetwork capacity
  - Requires a separate training pipeline

Mojo implementation implications:
  - Load hypernetwork weights alongside GGUF
  - Run the hypernetwork forward pass when context changes
  - Inject the resulting LoRA into the main model's GEMV calls
  - Can run as a separate inference pass before generation

Reference:
  SHINE: Scalable Hyper In-Context Network: Mapping Context to LoRA
  in a Single Pass
  Liu et al., arXiv 2602.06358, Feb 2026
  https://arxiv.org/abs/2602.06358
  Summary: A hypernetwork maps arbitrary text to LoRA parameters in
  one forward pass. No fine-tuning of the target model needed.
  Shown on diverse NLP tasks. Uses a two-pass system: memory
  extraction, then LoRA generation.


STRATEGY 3: DELTANET STATE AS IMPLICIT LEARNING (NO ARCHITECTURE CHANGE)

For certain transformer architectures, in-context learning is
mathematically equivalent to gradient descent on weights. No new
components needed.

Linear attention transformers that do in-context learning are
running an implicit gradient descent algorithm. The attention
mechanism is the learning algorithm. The key-value cache is the
training data. The attention weights are the gradient.

For DeltaNet, the recurrence
  S(t) = β·S(t-1) + α·k(t) ⊗ v(t)
is a fast-weight update. State S is a learned representation that
every token modifies. When the model processes a correction in
context, the DeltaNet state update is the learning.

Mechanism:
  1. Do NOT reset DeltaNet state between conversations
  2. Feed corrections and feedback in as normal tokens
  3. DeltaNet layers update their state automatically
  4. Subsequent tokens benefit from the updated state
  5. Optionally distill accumulated state into permanent weight
     changes (LoRA or full fine-tune) by running many examples

No backprop needed for the online part. The DeltaNet forward pass
is the learning. Qwen3.5's architecture already has this: 75% of
its layers are a learning algorithm.

Memory architecture:
  - GGUF weights: LONG-TERM memory (frozen)
  - DeltaNet state: CONTINUOUS memory (persists across sessions)
  - KV cache: EPISODIC memory (per-conversation, for full-attention layers)

Limitations:
  - DeltaNet state has finite capacity (144 KB for Qwen3.5-0.8B)
  - Old knowledge is overwritten by the β decay gate
  - No explicit loss signal; learning is unsupervised
  - Quality of learned knowledge is unverified

Mojo implementation implications:
  - Simplest approach: do not call state.reset() between sessions
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
  weights. ICL tokens can be exactly converted to weight updates.
  The attention mechanism implements an implicit learning algorithm.

Reference:
  Linear Transformers Are Secretly Fast Weight Programmers
  Schlag et al., ICML 2021
  https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf
  Summary: Linear attention with outer-product updates is
  equivalent to a fast weight programmer. The attention layer
  reprograms its own weights from context. Foundation for treating
  DeltaNet as a learning system.

============================================================
ALTERNATIVES TO REINFORCEMENT LEARNING
============================================================

RL (PPO, GRPO, DPO) is one way to get a learning signal, but it is
overkill for occasional fine-tuning. Alternatives:

1. SUPERVISED LEARNING ON CORRECTIONS
   User says "no, the answer is X." Then (prompt, X) is a training
   example. Cross-entropy loss. Simple and effective. Most
   "learning from human feedback" systems do exactly this.

2. SURPRISE-DRIVEN REPLAY
   Track which tokens the model predicted poorly (high loss). Replay
   those examples with higher weight during update steps. Combats
   forgetting without RL. It spends the limited update budget on
   what the model does not know.

   Reference:
   Surprise-Driven Prioritised Replay for Continual LLM Learning
   OpenReview, 2025
   https://openreview.net/pdf?id=IgZWU75BLL
   Summary: Replay buffer prioritized by prediction surprise.
   Targets catastrophic forgetting in sequential LLM updates.

3. SELF-SUPERVISED CONSISTENCY
   Generate several completions, find the self-consistent ones, and
   reinforce them. No external signal needed.

4. IMPLICIT LEARNING (Strategy 3 above)
   No explicit learning signal. The forward pass itself performs
   gradient-like updates. Closest to "the model evolves itself."

============================================================
RECOMMENDED PATH FOR QWEN35.MOJO
============================================================

Phase 1: Strategy 3 (persist DeltaNet state)
  - No architecture changes
  - Persist state to disk between sessions
  - Watch whether accumulated state improves generation
  - Risk: state overflow, quality degradation

Phase 2: Strategy 1 (online LoRA)
  - Add LoRA injection points to each weight matrix
  - Implement a backward pass (autograd) for LoRA params only
  - Update on explicit user corrections
  - Risk: complexity, forgetting

Phase 3: Strategy 2 (hypernetwork)
  - Train or obtain a SHINE-style hypernetwork for Qwen3.5
  - Integrate as a pre-generation step
  - Risk: separate training pipeline, hypernetwork quality

============================================================
END OF RESEARCH NOTES
============================================================
