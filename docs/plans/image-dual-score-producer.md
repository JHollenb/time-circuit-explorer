# Produce image logits through query-side dual coefficients

Status: **written, unexecuted**. This is a query-specific, baseline-assisted learned computation, not a zero-neural image generator or established speedup.

The [retained-key result](../stale-key-context-response.md) preserves a useful value-update direction but leaves 71.67% next-query response error. The next candidate updates the image score numerator while explicitly retaining the old scalar key-normalization denominator. Its purpose is to replace one actual image-key-producing operation rather than add another K/V factorial.

## Freeze the specimen and finite inputs

Use the same authenticated P/Q post-joint0 states, selected P query/residual, S=[4,11,12,13], joint1→joint2 consumer and 512×512 image geometry. Preserve exact native BF16 image ModLN formation; only then widen x to FP64. Use the original BF16 image-key matrix widened to FP64 during compilation, original learned key head weights gamma, and the transpose of the **actual stored** RoPE factors. Do not substitute ideal trigonometric values or assume the stored transform's transpose is its exact inverse.

Inventory image RoPE axis blocks and group membership from actual stored bytes before computation. The qualified geometry has four 32-channel axis blocks: T/L constant, H/W each 32 positions on the 32×32 image grid. Verify those counts and each row's group identity. Text has 512 distinct L positions and is excluded from the proposed transposed producer; do not broaden the candidate by an outcome-selected grouping.

## Move the smooth numerator onto the query side

Let q_ih be the already-positioned selected P query. For head h, axis a and stored position factor r, form

\[
u_{iha}(r)=W_{ha}^{T}[\gamma_{ha}\odot R_a(r)^Tq_{iha}].
\]

Gamma acts after applying the transpose of the stored positional transform. Sum the two constant-axis dual vectors. Retain one constant plus 32 H plus 32 W vectors. Then

\[
n_{ijh}=(x_j^Q)^T[u_{ih,0}+u_{ih,H}(h_j)+u_{ih,W}(w_j)]
\]

is an exact real-arithmetic factorization of the smooth image-key numerator. Evaluate the three dot contributions separately, using grouped original row addresses. Avoid materializing a query×head×image-row×input-width tensor. The identity does not preserve the original BF16 key projection, RMS and RoPE rounding boundaries; that finite-policy gap is part of the diagnosis.

## Pay for the retained denominator

The parent retained only normalized/rotated keys. Their pre-normalization magnitude cannot be recovered. A separate compiler must therefore compute one complete **FP64 P-image projection** from the actual P BF16 ModLN input widened to FP64 and retain

\[
s^P_{jh}=\sqrt{\operatorname{mean}[(W_hx^P_j)^2]+10^{-6}}.
\]

This is the explicitly declared smooth FP64 baseline denominator, not a claim to have recovered native BF16 raw-key RMS. The primary freezes it while using Q's numerator:

\[
\widehat\ell_{ijh}=n_{ijh}/(\sqrt{128}\,s^P_{jh}).
\]

The sole scientific approximation in that smooth law is retaining the P denominator. Native finite-policy discrepancies remain additional measured differences. There is no norm, gain, rank, coordinate or precision sweep and no fitted correction.

## Run one own-state candidate

The actor consumes the compiled dual table/P denominators, actual authenticated Q image state/context, paid text-only native Q keys, paid full Q values and the shared selected P operands. Restore selected text K/V to P exactly. Prepare separately authenticated text-key slices; the actor must not read the full Q image-key target or hydrate the image-key matrix. Compiler reads and historical source production remain charged.

Form stable joint softmax over all 1,536 text/image logits under one declared FP64 policy, contract the full supplied values and cast the head return at the declared boundary. Feed its own result into the unchanged original finite output map, sequential MLP and next-query consumer. Reuse the exact consumer law from cycle 56/57. Do not refresh a candidate field from an oracle or insert a Q-attention splice.

Seal the compiler before the fresh actor starts. Seal and exit the actor before opening image-key targets or computing Q-side reference denominators. Record actual typed input/output ABIs and compiled-data/source lineage, numerical read quarantine, physical projections/contractions and parameter/RNG closure.

## Diagnose the assumption after the candidate seals

One post-seal FP64 Q-image projection may compute the corresponding Q denominators and verify the transposed numerator against direct smooth projection. It is reference work and must not feed the primary. Separate retained-denominator discrepancy from the combined native finite-policy difference using the saved logits and native BF16 key operands. The latter includes projection/RMS/RoPE quantization and downstream attention reduction policy; do not label all of it a denominator error.

Retain absolute/signed per-row/head logit and partition effects, attention/write/MLP/T/next-query endpoints, actual context-response errors against the immutable PP/QQ references, zero/hold and the existing stale-key forecast. No extra consumer arm is needed. Unknown or heterogeneous effects remain observations rather than hard effect-size gates.

## Charge all work and storage

Expected nominal matrix work for the declared four-query geometry:

| Scope | MACs |
|---|---:|
| P-denominator compiler |9,663,676,416|
| Dual compiler |622,854,144|
| Actor grouped image dots |905,969,664|
| Actor text QK |6,291,456|
| Actor full AV |18,874,368|
| Actor original consumer affines |415,236,096|
| **Actor subtotal** |**1,346,371,584**|
| Post-seal Q projection reference |9,663,676,416|
| **Compiler + actor + reference** |**21,296,578,560**|

The dual table occupies **153,354,240 FP64 bytes**, larger than a full BF16 image-key field. P denominators, inputs, weights, references and captured arrays add storage. No memory saving, first-prompt saving or runtime speedup is established by these counts. Previous source states, selected query, native text keys/values, complete baselines and original matrices remain paid. Finalize physical geometry/admission from the smallest informative mechanics run; use one bounded mrun lease, actual backends and strict preflight, with no retries or full suffix/image.

This tests a concrete conditional source producer. Even if useful, it would leave arbitrary prompt conditioning, changing query families, complete own-state diffusion, learned consumer arithmetic and general pixels unresolved. The strict zero-neural objective remains unchanged.
