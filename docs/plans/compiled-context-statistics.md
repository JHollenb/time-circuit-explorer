# Test whether context admits a smaller streaming state

Status: **implemented and diagnosed in cycle6**, following the [complete lexical-prefix result](../raw-prompt-encoder-atoms.md) and Astra's cycle5 review. The [measured result](../compiled-context-statistics.md) preserves this original design: streaming/direct-kernel closure is about7e-17, but rank64 median whole-head error is183.3% versus97.8% uniform. Saved-only key diagnostics observe collapse; retrospective gauge conditioning helps but remains inaccurate. No image result or general small-state impossibility follows. The arbitrary-prompt zero-neural image goal remains active.

## The next unresolved mathematical assumption

The first encoder prefix now receives raw strings. Its token-local table is exact on the eight measured sequences, but a contextual attention read depends on the other tokens and their positions. Simply evaluating the full attention equation in NumPy would retain that neural computation. The useful next hypothesis is narrower and testable: **can a small reusable state preserve a contextual read and its effect on the next consumer?**

For one prespecified query head0 and its actual shared KV head0, put d=128 and use the completed raw-prompt operand store. Let x_i=q_i/d^(1/4) and y_j=k_j/d^(1/4), after the real position transform. Approximate

\[
\exp(x_i^\top y_j)\approx\phi(x_i)^\top\psi(y_j).
\]

With key-mask m_j and causality, maintain

\[
z_i=\sum_{j\le i}m_j\psi(y_j),\qquad
M_i=\sum_{j\le i}m_j\psi(y_j)v_j^\top.
\]

Then read

\[
\widehat h_i=\frac{\phi(x_i)^\top M_i}{\phi(x_i)^\top z_i}.
\]

This changes the executed representation from a complete pair matrix to a streaming state. It also exposes the approximation: a finite-rank kernel may lose important reads. Finite execution, current read accuracy and preservation after later consumers remain separate measurements.

## Freeze the candidate before held outputs

Start with ranks16/32/64 and one frozen Gaussian positive-feature construction. For fixed standard-normal rows omega_a independent of model weights,

\[
\phi_a(x)=r^{-1/2}\exp(\omega_a^\top x-\|x\|^2/2).
\]

The same construction on y yields an unbiased estimator of the unnormalized exponential dot kernel before ratio formation. The normalized ratio is not thereby unbiased. Freeze the feature-array bytes, ranks and arithmetic before inspecting held head outputs; do not select a favorable seed using the target.

Positivity preserves one useful native property: with a nonempty allowed prefix, the candidate read is a convex combination of the values. In real arithmetic each output coordinate stays inside the corresponding value range. This bounds its amplitude without establishing accuracy; finite-precision rescaling and final rounding require separate observations. It avoids discarding that bound merely to obtain a polynomial formula.

Represent feature logs and stream with per-feature maxima. When a new key changes a maximum, rescale that feature's previous z/M before adding it. During query reads, subtract one common maximum across feature contributions. Values and numerator may be signed; the denominator is positive when at least one allowed key exists. Declare an explicit output for an empty masked prefix and retain whether it occurs. Do not introduce epsilon clipping, fallback attention or target-dependent stabilizers invisibly.

Charge all feature formation. Online omega projections are additional contractions even though omega is not learned. Precompiling all token-position feature logs instead trades them for a large table: two FP32 feature tables for151936IDs ×512positions ×rank64 already occupy39,829,110,784bytes for one head. Neither option is free. This candidate remains model-derived approximate computation and is not the completed strict zero-neural goal.

## Smallest informative assay

- Consume fresh raw strings through the complete atom store; supply no prompt-level hidden states. Freeze varied order/repetition, unfamiliar combinations, Unicode and long contexts. Keep these separate from the eight development sequences.
- Preserve the actual causal key mask, grouped KV mapping and all512 query rows, including padding queries. The pipeline's later consumer receives512 conditioning rows.
- Run exact full softmax only in an offline reference process after candidate seals. A uniform allowed-key mean provides a no-routing control. Candidate serving must have no exact-attention or original-projection fallback.
- Record raw/BF16 head error, prompt-change error, signed cross terms, per-position behavior, denominator log range, feature concentration and state trajectories. Compare ranks as observations, not accuracy gates.
- Transport saved head errors through the original output-projection head slice offline. This is an ideal linear next-consumer diagnostic, not parity with the complete native fused projection, other heads, residual and later encoder. Keep quantization and consumer limitations explicit.
- Report state bytes, cached learned bytes, feature constants, all contractions/exponentials, first-use and warm latency, setup, RAM and offline reference cost. A smaller state is useful evidence even if this particular representation is slower or inaccurate; it does not justify concealing either result.

At rank64 the unscaled FP64 z/M state alone needs64 ×129 ×8=66,048bytes per head, plus512bytes of per-feature maxima and current operands. Compare that complete representation with the original prefix memory and actual work. Avoid a state-size claim that omits feature tables or formation.

Use one finite mrun lease near the retained compiled table, no automatic retry, source-bound inputs and complete prediction seals. Begin with a tiny mechanics organism, then the frozen full assay. No native image tail, semantic certification bundle or large benchmark is required to decide whether this contextual-state hypothesis is promising.

## Why the receiving computation matters

Astra reviewed the newest working [parallel-writer/next-consumer study](/Users/jakeholl/domains/obsidian/blog/2026-11-06-001355-the-next-consumer-changes-which-model-write-matters.md). Its MLP/attention ranking reverses at the next block, and clamped affine feature streams still exhibit finite-arithmetic interaction. The [time-circuit mathematics](/Users/jakeholl/domains/obsidian/blog/2026-11-05-153154-state-context-and-the-mathematics-of-time-circuits.md) asks for a future-sufficient state. The [bounded-normalization work](/Users/jakeholl/domains/obsidian/blog/2026-11-05-161109-the-one-assumption-that-would-make-model-computation-easier-to-understand.md) separates finite advancement from faithful advancement and favorable error cancellation. These are unpublished workspace links and scope-qualified evidence, not public release claims.

The next decision should turn on whether this representation retains the needed contextual dependence at affordable cost. A dense attention rewrite or a reduced learned model must not be renamed arbitrary-prompt zero-neural image generation. Later contextual layers, own-state diffusion and a general non-neural decoder remain required.
