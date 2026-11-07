# Eliminate outputs with no consumer in the last FLUX single block

Status: **written and unexecuted**. Astra's [cycle63 source/result review](../results/source-reader-evidence/advisor.json) recommends moving from four demanded queries to one complete terminal image output. The [source-state reader](../source-state-reader.md) establishes a useful input boundary but still pays original full K and native source prefixes. This next cut uses native output liveness rather than another norm surrogate, fit or intermediate-write omission.

## Source-defined demand

Use the pinned FLUX.2 Klein 4B transformer source, SHA256 `0a3c56ed3ca28a416492b50953a433f20b8a89fd26c7b7a49b1b6b45d5236006`. In ordinary generation, without reference/cache modes, the final single block is single19. Immediately afterward the original transformer discards all text output rows and applies the final velocity head only to image rows.

Within that final block, current K/V from **every text and image row** can affect image attention. Image Q, image MLP and image output/residual writes remain live. Text Q, text MLP and text output/residual writes have no downstream consumer. No earlier block's text state is declared dead: it may affect later K/V. Reference/cache modes are outside this first specimen and must not silently inherit the elimination.

For incoming state X=[X_T;X_I] and actual clock context c, the demanded block output has the real-arithmetic form

\[
Y_I=X_I+g_c\odot W_o\left[
\operatorname{Attn}\big(Q_I,K_{T\cup I},V_{T\cup I}\big),
\operatorname{SwiGLU}\!\left(W_{\mathrm{GU}}N_c(X)_I\right)
\right].
\]

Here W_GU comprises the original fused gate/up projection rows; SwiGLU maps their 2M outputs to M coordinates. Then evaluate the original final normalization/modulation and velocity projection on actual own Y_I. Those learned operations are paid. The source proof removes dead row computations, not learned parameters globally or the live attention/value dependence on text.

## Authenticate before executing

Choose **one existing complete incoming single19 StateCut** with original modulation, full absolute RoPE, model/checkpoint identity, step, sequence geometry, native state and matched post-block/final-velocity target custody. First inventory metadata and byte availability. A matching saved cut is preferred. If the required cut is absent, explicitly account for one paid native prefix capture in a separate request; do not substitute an attention-input array for the declared residual/context interface or borrow a target state to fill a missing source slot.

Expected geometry is batch1, 512 text plus1,024 image rows, D=3072, 24 heads of128 and MLP width M=9216. Authenticate actual config, fused parameter layout, biases, normalization/gate conventions, selected weight row intervals and native finite profile before freezing science. Shape/model/source/custody mismatches are mechanics issues; unknown cut availability is an intake task, not evidence for or against output liveness.

## One physically reduced producer

1. Form the original full normalized input/context from the authenticated complete source state. All source formation is paid.
2. Physically slice the original fused projection's K/V weight rows and evaluate them for **all 1,536 rows**. Preserve native per-head Q/K RMS and absolute position factors. No stale, selected-only or reduced text K/V.
3. Physically slice its Q and MLP weight rows and evaluate them for the **1,024 image rows only**. Record actual input/weight/output shapes and affine MACs. Computing the full fused output and discarding columns does not establish the declared saving.
4. Use rectangular image-query attention against the complete current K/V. Evaluate the image SwiGLU, original output projection, gate and residual. No text output allocation/write or hidden full text Q/MLP computation.
5. Feed actual own image output through the original final velocity head. Capture/seal the emitted velocity and exit before the historical native target is numerically opened.

No VAE, new image campaign, fit, precision/rank/seed sweep or full denoiser run. No tolerance-based inheritance of historical images. If velocity differs, retain the discrepancy and stop this bounded measurement after reporting it; a later pixel consequence would require its own consumer transaction.

## Finite arithmetic is measured separately

The liveness statement is exact for the ordinary real-arithmetic graph. The executable producer changes fused GEMM width, row count and the SDPA query geometry. Those can change finite BF16/kernel behavior. Preserve exact byte comparisons where observed, plus whole velocity/row/channel signed errors and matched input/control fields. Never convert the algebraic dead-output proof into an unmeasured BF16 parity assertion.

Compare native versus candidate final-block image state and own final velocity only after sealing. No quality threshold stops valid exploratory evidence. Mechanics, instrument scope, numerical trend and any later terminal certification remain separate. A useful result may be exact elimination, small drift, large drift or an implementation/instrument limit with its proper label.

## Prospective work

For the declared geometry, excluding final velocity head, normalization, softmax, context/prefix formation and storage:

| Work | Native | Candidate |
|---|---:|---:|
| Fused input and output affine MACs | 188,441,690,112 | 135,291,469,824 |
| QK plus AV MACs | 14,495,514,624 | 9,663,676,416 |
| Total declared matrix MACs | 202,937,204,736 | 144,955,146,240 |

The candidate removes **53,150,220,288 affine + 4,831,838,208 attention MACs per call**, or 57,982,058,496 within this matrix account. Derivations are

\[
C_{aff,native}=ND(3D+2M)+N(D+M)D,
\]
\[
C_{aff,candidate}=ND(2D)+N_ID(D+2M)+N_I(D+M)D,
\]
\[
C_{att,native}=2N^2D,\qquad C_{att,candidate}=2N_IND.
\]

Final velocity-head work must be derived from authenticated actual maps and added to the executed ledger. All original learned maps and upstream states remain paid. These are nominal matrix counts, not measured wall-time savings or a whole-model percentage. Parameter sharing means no parameter is removed globally merely because text output work is dead here.

Use one guarded strict-preflight mrun CUDA-requested lease with finite timeout, retries disabled and actual selected-map/resource history. Resolve intake before choosing the smallest informative admission. Preserve complete source/target/own-state bytes near compute, publish compact lossless numerical results, then run the evidence plane and audit exact physical work. No source is staged or job submitted by this plan. Newest-post Astra review remains due65; actual storage cleanup/audit63, next66 or earlier material evidence/pressure. Arbitrary raw-prompt zero-neural generation remains open.
