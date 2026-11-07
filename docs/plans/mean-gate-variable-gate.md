# Separate common gate coefficients from changing-gate interactions

Status: **written and unexecuted**. No payload is staged or job submitted. The [own-state covariance diagnostic](../own-state-gated-covariance.md) finds that a paid best per-stream constant leaves 83.21% of pooled reconstructed discrepancy RMS, including 98.77% text and 80.10% image. The covariance mean already nearly reaches that constant-family optimum. The next question concerns row variation, not another anchor estimate.

## One fixed source and one new projection

Keep the same **clock3/single0** own-state receiver. Admit authenticated saved activation A, FP64 Up reference U*, its two stream means, original current support S, FP64 original Down map D, and residual gate g from cycle76. Original generation, captured current activation/gate/support, full Up reference and map promotion remain paid. Derive the two current activation means from A; do not supply them from a target or fit.

For row i in stream s, write

\[
\delta U_i=U_i^*-\bar U_s^*,\qquad
L_i=1_{\bar S}\odot\bar A_s\odot\delta U_i,\qquad
B_i=1_{\bar S}\odot(A_i-\bar A_s)\odot\delta U_i.
\]

In real arithmetic, the previous centered product R=L+B. Evaluate only

\[
Y_L=LD^T.
\]

This full-width FP64 Down contraction costs **43,486,543,872 prospective MACs**, 1,536×9,216×3,072. Reductions, elementwise products, transfers, fingerprints and I/O remain additional work. No new Up, attention, head, sampler, VAE, image, support change, fitting, sweep or additional receiver address.

The producer may read A, U*, stream means, S, D and g only. It must seal its own L, activation means, Y_L and necessary source/diagnostic fields, then exit **before** loading the old Y_R, Y_Q and Y_E comparisons. Keep those old output fields quarantined in the producer. Split source production from post-seal interpretation so a known error does not become a serving operand silently.

## Derive the comparison after release

Only after the producer exits, open the authenticated cycle76 fields and form

\[
Y_B^\Delta=Y_R-Y_L.
\]

This is a subtraction-defined changing-gate/reassociation term. Do not call it an independently evaluated BD^T; FP64 distributivity and reduction effects remain unseparated. Product-space L/B bookkeeping may be inspected after release without adding a Down contraction. Preserve the original Y_Q policy/reassociation boundary rather than reassigning its effects to changing gates.

Retain gated linear and derived changing-gate terms, their separate text/image means and centered row variation. Preserve signed energies, cosines, full Gram terms, reconstruction discrepancies and complete row outcomes. Centering makes mean/centered orthogonality a mathematical construction, not discovery of orthogonal model circuits.

Report both the full residual \(gY_E-gY_L\) and the residual centered within each stream. Do not silently subtract an oracle mean and describe that as a fully produced correction. An analysis with a supplied covariance mean must label that inherited paid operand explicitly. All rows and signed countertrends remain observations; no effect-size gate may stop evidence collection.

This tests whether current row variation has a useful interface through common receiver-conditioned coefficients. A strong common-gate component could motivate source factorization; a strong changing-gate interaction would motivate another conditional law. Neither outcome establishes a cheaper producer, a native finite concatenated block error, the cause of final image drift or arbitrary-prompt generalization.

## Compilation is not free

The real-arithmetic linear interface can be written

\[
Y_L=(Z-\bar Z_s)W_U^T
\operatorname{diag}(1_{\bar S}\bar A_s)D^T.
\]

The displayed equality additionally idealizes the previous finite FP64 Up and mean reductions; keep that finite distinction explicit. Physically selecting the 6,912 omitted channels, naively forming two 3,072×3,072 stream maps costs **130,459,631,616 MACs**. Applying those maps to 1,536 rows costs another **14,495,514,624**. These prospective costs are not executed here and are not a saving: coefficients and support vary with the receiver, with no demonstrated reuse or economical compilation.

Before staging, derive source/profile/custody joins, exact liveness/retention and resource admission from the implemented code and measured76 history. Use one finite strict-preflight CUDA mrun job, no retries, and a tiny independent scalar reference first. Keep full arrays near compute, compact scalar evidence locally, and run/verify the standard Saturn sweep. The strict arbitrary-prompt zero-neural goal remains active and unsolved.
