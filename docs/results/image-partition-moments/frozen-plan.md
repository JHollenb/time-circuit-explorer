# Produce live image attention mass from full key moments

Status: **written and unexecuted**. The [partition-only feedback result](../partition-feedback.md) recovers two separate objects while retaining 32.93% first-velocity and 11.34% PNG error on one known scene. It shows a useful feedback interface but still forms every text-query×image-key score. The next candidate replaces that grid with source-defined current-key statistics; all learned projections, image attention, text recurrence, sampler and decoder remain paid.

## One current-state moment law

For each head, use its actual incoming image keys **after original head normalization and absolute RoPE**:

\[
\mu=\frac1{N_I}\sum_jk_j,\qquad
\Sigma=\frac1{N_I}\sum_j(k_j-\mu)(k_j-\mu)^T.
\]

Retain the full 128×128 covariance and the population denominator N_I. Compute in FP32 with autocast/TF32 disabled. For current text queries q, form

\[
\widehat{\log Z_I}=\log N_I+\frac{q^T\mu}{\sqrt d}+\frac{q^T\Sigma q}{2d},
\quad \hat\alpha=\sigma(\log Z_T-\widehat{\log Z_I}),
\quad \hat O_T=\hat\alpha O_{TT}.
\]

Text QK supplies one stable FP32 log-partition and text-only softmax/AV. Multiply each head’s FP32 O_TT by its own alpha **before** the BF16 text-output boundary. Keep image attention native with both complete value banks. Both streams consume their own current states; preserve joint sequential versus single parallel MLP ordering, reset C(E) per clock, and omit dead terminal text queries/MLP/output while retaining terminal K/V.

There are no supplied native alpha/log-partitions, fitted coefficients, seed/prompt/rank selection, covariance diagonalization, clipping or target-directed repair. This is a live coupled program, not an image-independent compiler. The image K bank is already formed for image attention; its learned production remains paid.

For scores s_j=qᵀk_j/sqrt(d), the exact finite-bank log-partition can be written

\[
\log Z_I=\log N_I+\kappa_1(s)+\tfrac12\kappa_2(s)+R(q).
\]

The approximation sets the residual R to zero. It is exact for an appropriate Gaussian score distribution in real arithmetic; current empirical keys need not satisfy that law. Higher cumulants and rare large scores remain unrestricted. Do not assume convergence of an infinite cumulant series at the demanded query or turn the approximation into an error bound. Mixture identities refer to common incoming Q/K/V, not the differences between diverged trajectories.

## Physical work and finite policy

Use explicit products for centered-key covariance, query×covariance, query×mean, TT scores and TT AV. The image-partition producer’s matrix account per live read is

\[
N_IHd^2+N_THd^2+N_THd=605{,}552{,}640,
\]

replacing N_TN_ID=1,610,612,736 score MACs. With 96 live reads, prospective matrix saving is **96,485,769,216**, yielding **21,120,701,956,096 first-image matrix/convolution MACs**, subject to actual instrumentation. Means, centering, elementwise quadratic row products/reductions, divisions, logs and sigmoid remain paid additional work. Do not double-count the q×covariance product or silently include unexecuted full image scores. Full FP32 mean/covariance storage is **1,585,152 bytes per block**, excluding keys, TT scores, outputs and diagnostics.

Splitting TT scores from the former full-key GEMM and separately multiplying alpha changes finite geometry and rounding. Declare scaling order, stable logsumexp/softmax, population moments, full covariance and BF16 output explicitly. This is neither native finite parity nor a proven Gaussian approximation. Record signed moment/partition diagnostics without quality gates.

## One bounded own-state image

Reuse authenticated E, noise/IDs/schedule and original native case A `job-394733a01129`. One strict-preflight CUDA lease, no automatic retries, new reference campaign, fit or sweep. Preserve the own four-clock trajectory and one actual VAE; seal and exit before existing native/cycle69 comparisons. Derive a fresh liveness/resource plan from the measured 8,442.8 MB RSS / 9,416 MB device history, rather than treating fewer nominal products as proof of lower latency or memory.

Tiny tests should cover the population denominator, full off-diagonal covariance, constant-key and translation laws, stable mixture formation, FP32 policy and a two-commit native-block reference for the **declared approximate** moment read. Agreement to this program is a mechanics result; its error against an exact finite-bank partition is an observation. Retain actual moments, alpha/text reads, both state chains, all physical counts, signed trajectory/pixel rows and full byte custody. A bad image remains evidence.

The question is whether current key moments can produce a useful dynamic gate without the full text→image score grid. An improvement would not establish zero-neural generation or general moment sufficiency; an adverse result would constrain this source-defined second-order law, with tails, higher moments and finite arithmetic as alternatives. Last full newest-post survey69 and actual storage maintenance69 are due again72 or earlier material evidence/pressure.
