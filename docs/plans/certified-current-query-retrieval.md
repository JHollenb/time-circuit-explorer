# Can we retrieve the same four keys with less score work?

Status: **implemented and measured in cycle16**, guarded job`job-d1b6e831e7fb`. The [result](../certified-current-query-retrieval.md) preserves ordered support/raw/BF16 output and avoids about10.6%of score products, but bound/sort overhead makes this implementation much slower. The original protocol below is retained. One pre-execution clarification uses the completed progressive dot itself as f, under the same error allowance, without an extra hidden dot. The [current-query result](../full-prefix-selector-objective.md) motivated this bounded cost assay; the next test returns to [tail mass/value](current-support-tail-factorial.md).

Freeze one protocol: the same saved operands,64source anchors/head, source gauge, routes, four-key budget, centroid read and causal masks. Use a per-KV-group coordinate order sorted by descending diagonal of the existing source key covariance; coordinate index breaks ties. Use eight blocks of16coordinates, with no prompt fitting or block-size sweep. Derive order only from the authenticated original source asset.

Advance an append-only causal index of actual unmasked key occurrences. Retain their centered FP64 keys and outward upper bounds for each block-tail norm. Keep actual values available for the unchanged reader. This is not a smaller-than-KV claim: count actual payload, index/norm/identity metadata and runtime source geometry. No oracle or target-output information may enter selection.

For each actual query, evaluate key scores progressively. For exact real arithmetic, a partial dot plus the Cauchy–Schwarz tail bound upper-bounds the complete score. The finite implementation must also bound differences between partial, candidate and old whole-prefix reductions; do not assume a one-row dot has the old GEMV bit pattern.

Let d=128, unit roundoff ε=2⁻⁵³, γ_n=nε/(1−nε), η=γ_(4d), and let A_j be an outward upper bound on ‖x‖‖u_j‖. With computed partial score p_j and an outward upper bound B_j on ‖x_tail‖‖u_j,tail‖, use

\[
U_j=\operatorname{nextUp}(p_j+B_j+2\eta A_j).
\]

For a separately evaluated full candidate dot f_j, the old full-dot score is enclosed by

\[
[\operatorname{nextDown}(f_j-2\eta A_j),
  \operatorname{nextUp}(f_j+2\eta A_j)].
\]

One error allowance covers the new reduction, one the old reduction. Build squared tail norms by positive reverse sums; do not subtract a nearly equal prefix from a total. Apply outward bounds through square root and multiplication. State the normal/finite FP64 assumptions and account for every operation in constructing bounds. Overflow, underflow/subnormal ambiguity, invalid intervals or unsupported arithmetic profiles trigger the old complete score call, not an optimistic prune.

Evaluate initial complete candidate dots deterministically in causal occurrence order until four incumbents exist. Thereafter prune only when U_j is strictly below the fourth-largest incumbent lower endpoint. Candidate membership and ordering must be unambiguous under the retained intervals. Resolve overlapping final membership/order intervals, including ties, by the original whole-prefix U@x call in the original coordinate/layout order; count its entire cost. Preserve the original earliest-occurrence tie law. The actual implementation must justify the finite enclosure before claiming certification; observed support equality alone is not a proof of a bound.

Seal every new support and raw/BF16 bank before loading old current-query forecasts or other references. Compare with the cycle15 complete supports, including their order. For exact support/order, run the same centroid read in the same order. Record word differences separately from floating closure; do not silently loosen equality. If the complete BF16 consumer input bank is identical, reuse its already measured original receiving-state result and authenticate that inheritance. No new original model/MLP execution is needed just to recompute an identical input.

Report evaluated score-coordinate products, partial and full candidate dot work, incumbent maintenance, norm construction, bound operations, fallback counts/work, sorting, retained bytes and wall/CPU time. A full dot must never hide inside a bound. Count cached norm construction once on key admission and query norms per query. Compare actual total scalar work and timing with the complete current-query scan under the same profile; reduced coordinate products alone do not establish a speedup. Report the unchanged centroid reader's full-prefix aggregation separately. Efficient selection alone would still not prove faster or smaller attention.

Tiny mechanics should include exhaustive comparison on random causal masks, adversarial near ties, negative scores, missing/short prefixes, tail-norm cancellation cases, fallback routing and grouped heads. Mechanical integrity may stop an invalid certificate or corrupted state. No accuracy or efficiency threshold stops evidence collection: if most keys survive or the bounds cost more, retain that negative result under the same frozen protocol.

Run one finite guarded CPU1 mrun lease near the retained arrays, strict preflight/default admission, no duplicate suite or automatic retry. Keep full arrays near compute and publish compact lossless observations. This is one bounded selection-cost assay. Its result must determine the next move: either use the measured index as a component or stop pursuing this optimization and return to the remaining tail/own-state/contextual-encoder gaps. Arbitrary-prompt zero-neural image generation remains the full open goal.
