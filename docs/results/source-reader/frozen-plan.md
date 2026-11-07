# A four-query reader whose inputs are source states

Status: **written and unexecuted**. The [deferred value result](../deferred-value-projection.md) removes a supplied V bank, but its allocation still receives native queries/text keys, query-specific duals and retained P norms. The next primary removes those internal operand inputs, keeps deferred V, and pays the original full key path explicitly. This tests closure at a source-state interface, not another cheap norm approximation or strict zero-neural serving.

## The declared interface

Accept authenticated Q image/text/context and P text/context StateCuts, original parameter recipes and fixed support **[4,11,12,13]**. Derive selected residual R, modulation and rotary positions from those source pages. All selected Q/K/V/R/modulation/position arrays, native banks, prior normalized fields, query duals, retained norms, old probabilities/heads/states/queries and historical targets are denied as numerical inputs before sealing. Metadata and source-asset hashing remain paid and do not authorize numerical target use.

Preserve the existing contrast: selected P residual/query rows read Q's surrounding context with selected-P normalized text rows spliced into it. This is not a new claim about a fully changed arbitrary Q prompt. The source states remain native prefix outputs and include already-computed modulation and rotary factors.

## One executable primary

1. Form native BF16 ModLN for Q's full image/text fields and P's full text field. Form P's selected query ModLN separately on **[1,4,3072]**, matching the original `propagated_context.selected` geometry. Full P text ModLN before selecting K/V source rows and selected-row query ModLN are distinct finite operations.
2. Splice P's selected full-field normalized text inputs into Q's full normalized text field. Use the original full text/image K projections, native per-head RMSNorm and the original RoPE factors. No diagonal, random sketch, spatial interpolation, retained norm or new norm fit.
3. Project the P selected query using original `add_q_proj`, `norm_added_q` and its actual selected rotary factors. Compute full joint text/image QK and stable softmax from the new query and keys. No archived query, key, score or probability is read.
4. Reuse the frozen deferred-value kernel: each head forms separate text/image weighted normalized inputs, then applies its original V head slices. Preserve joint probability masses and FP64→FP32→BF16 H admission.
5. Feed own H and source-derived P residual/modulation through the unchanged receiving output projection/MLP into own T and next query. Derive next-query rotary/modulation operands from the same declared source pages, not archived selected arrays.

The actor loads only original maps needed for these operations: two K projections/two K norms, selected Q projection/norm, two V projections, and the five receiving/next-query tensors. Authenticate the original bias-free finite path and exact loaded names/payloads. Capture actual normalized inputs, spliced field, Q/K components, logits/probabilities, weighted value inputs/contributions, admitted H, own states/query, typed input/output fingerprints and explicit ABIs. Close parameter/input/RNG state and source-byte identities.

The identities about linear V remain real-arithmetic identities. Full hybrid K projection geometry can differ from separately projected-and-spliced native K, and selected Q's finite geometry must be respected. Retain all such discrepancies; do not assert native bit parity from algebra alone.

## Post-seal evaluation

Only after the one primary seals and exits, open already-sealed full-QQ/stale/P and cycle62 control operands and outputs. Compare source-derived selected rows/modulation/positions, Q and K fields, allocations and native consumer responses. Actual component/input mismatches are applicability observations with explicit locations and magnitudes, not erased failures. No newly formed full V bank, new dense reference campaign, additional MLP arm, suffix or image.

Keep primary endpoint and P→Q response errors separate across H/A/M/T/nextQ and all dependent rows/heads. Compare against the prior assisted cycle62 producer as well as zero/stale controls. Improvements and regressions remain visible independently of mechanics. This primary deliberately restores/pay full K to remove a demonstrated hard normalization dependency; its success condition for exploration is informative, valid execution, not a fidelity threshold.

## Prospective cost and limits

| Actor work | Nominal matrix MACs |
|---|---:|
| Full current image/text K formation | 14,495,514,624 |
| Selected P Q formation | 37,748,736 |
| Live joint QK | 18,874,368 |
| Deferred V inputs and head maps | 528,482,304 |
| Four receiving/next-query affine maps | 415,236,096 |
| **Total** | **15,495,856,128** |

Native state formation, RMS/ModLN/RoPE, softmax, original map storage and prefix generation remain paid in addition to these MACs. This actor is more expensive than the assisted cycle62 actor's 1.856 billion MACs. It exchanges caches and supplied internal operands for a reusable source-state computation; it is not a measured serving-speed improvement.

Use one guarded strict-preflight mrun CUDA-requested lease with CPU FP64 deferred V and native CUDA BF16 Q/K/consumer arithmetic, current exact selected-map resource basis, finite timeout and no retries. No parameter fitting, threshold search, precision sweep or extra candidate. Preserve complete evidence near compute and publish compact lossless results. Perform scheduled storage review in cycle63 or earlier pressure. Periodic newest-post Astra review is current through62, next65 or material evidence. Arbitrary raw-prompt zero-neural images remain open.
