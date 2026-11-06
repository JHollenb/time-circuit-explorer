# Does one shared finite-bank state preserve prompt changes?

Status: proposed after the [cycle7 source-frozen experiment](../source-conditioned-context.md) and Astra's result/newest-post review. Not staged or measured. Better broad reconstruction did not preserve several small contextual changes. The next discriminator tests finite-bank rank capacity rather than another Gaussian seed, mixture or centering adjustment.

## One common bank, not one fit per prompt

Use only the eight sealed cycle7 ports from `job-b372c2f27c66`. Deduplicate all query atoms, including padding queries, and active key atoms by `(token ID, absolute position)`. Repeated identities must have identical corresponding BF16 operands; verify key values too. Preserve each prompt's occurrence maps, multiplicities, masks and values. Do not merge positions or silently apply this token-local identity beyond the first layer.

Form one unmasked common-query × union-key kernel and normalize each common row globally:

\[
K_{uv}=\exp(q_u^\top k_v/\sqrt{128}),\qquad
P_{uv}=K_{uv}/\sum_w K_{uw}.
\]

Use stable FP64 arithmetic. Global row scaling cancels when restricting to a prompt's actual key occurrences and renormalizing. First verify that exact P under each causal/padding restriction reproduces the saved mathematical reference heads. Underflow or mismatched restriction is an instrument issue; diagnose it before interpreting rank errors.

Compute one actual thin SVD of P, fitting each unique bank row equally. Freeze ranks16/32/64/128 before execution:

\[
P_r=A_r B_r^\top.
\]

Retain the spectrum, numerical reconstruction and kernel Frobenius errors. Exact truncated SVD is optimal only for that matrix norm. A randomized approximation would need a different description. Do not fit rank64 separately to short prompts with fewer than64 active keys; that would produce a trivial result without a shared representation.

## Let the same statistic advance

For every rank, advance actual prompt occurrences through

\[
z_i=z_{i-1}+m_i B_r[k_i],\qquad
M_i=M_{i-1}+m_i B_r[k_i]v_i^\top,
\]
\[
\widehat h_i=\frac{A_r[q_i]M_i}{A_r[q_i]z_i}.
\]

The query coordinate is a fitted-bank lookup. This is not a feature rule for an unseen token/position or context. Compare streaming reads with direct restriction of the same rank-r matrix and retain finite-arithmetic differences. No oracle state refresh or exact-attention fallback enters this candidate statistic.

SVD factors generally define a **signed** kernel. Do not clip negative entries or renormalize a different nonnegative matrix silently: entrywise clipping can change rank and removes the declared shared bilinear law. Record negative mass, denominator sign and the continuous cancellation ratio `abs(sum(weights))/sum(abs(weights))`. Near cancellation is diagnostic, not an arbitrary accuracy gate. Zero/nonfinite optional reads remain labelled per row with their raw arrays and valid exact reference; retain other ranks/rows and continue safe state accumulation.

## Evaluate changes before endpoints

The primary outcomes are all four paired-change errors, signed cosines, absolute target/error RMS and ideal saved W_o-slice transport. Split active and padding rows. All four cycle7 pairs have equal active-token counts, allowing same-position comparisons without mask changes. Keep per-position counterexamples and rank trajectories; do not choose a rank from broad endpoint RMS and hide the paired results.

Then report whole-head raw/BF16 endpoint errors, kernel errors, singular spectrum, signed-mass diagnostics, state/factor bytes, offline kernel/SVD cost, online contractions and reference work. No new model/native inference or original checkpoint is necessary. Use a finite guarded CPU mrun lease near the saved arrays, bounded matrix memory, strict preflight, no duplicate accelerator work and no automatic retry. Tiny synthetic checks should verify duplicate identity, occurrence multiplicity, exact restriction, streaming factor closure and signed/zero-denominator handling before the real job.

## Claim boundary

All eight prompts contribute to the factorization. This is **target-assisted transductive finite-bank capacity**, not held-out prediction, a deployed model replacement or arbitrary-prompt serving. Signed rank capacity also differs from positive-feature rank capacity. Strong paired results would show a better shared representation exists on this finite bank than the tested Gaussian features. Poor results would not exclude every rank-r sufficient state: SVD optimizes a matrix norm, not normalized reads, output writes or prompt differences.

The [new image-continuation reference](/Users/jakeholl/domains/obsidian/blog/2026-11-06-014800-from-predicted-model-state-through-the-rest-of-the-image.md) now connects predicted states through remaining original equations to native VAE images. It strengthens a future reference consumer while retaining learned equations and supplied conditioning. It does not satisfy the zero-neural objective. Contextual encoding, general own-state denoising, a non-neural pixel decoder and independent prompt-faithfulness evidence remain required.
