# Supply the actual query, then separate selector objectives

Status: **proposed after cycle14 and Astra review; not staged or measured**. The [source-witness selector](../source-witness-support.md) slightly improves active state response but worsens padding and overall error, with lower correction-mass capture. No material implementation defect was found. Distinguish inadequate source-query coverage from a poor max-score objective before compiling another candidate.

Use the same retrospective saved pair,64anchors, routes, source mean c_g, four-key budget and centroid-tail law. For each allowed current causal key, let

\[
u_j=y_j-c_g,\qquad s_j=\ell^Tu_j,\qquad a_j=x^Tu_j.
\]

Compare two new diagnostic supports:

1. Current-query mass: top four a_j.
2. Actual-query envelope: top four max(s_j,a_j), zero offsets.

Use earliest occurrence for ties and no future or masked keys. Retain the existing correction-mass oracle's indices unchanged, alongside historical anchor/witness controls. Do not rerank that oracle or fit any selector from target head/consumer error.

These new selectors scan the complete current prefix. They are diagnostic selectors, **not compact serving algorithms**. The actual-query envelope supplies a missing direction to the failed max-score proposal while preserving its gauge/objective. Current-query versus envelope tests anchor protection. Both versus the old absolute-centered-tilt oracle test whether high raw scores miss important weight decreases or contextual corrections.

For every selected support, independently aggregate its complement under original anchor weights and use exactly the existing centroid read. Seal new supports and complete raw/BF16 head predictions before target access. No neural references need be rerun. Subsequent original output/RMS/MLP/state evaluation remains separately paid offline learned computation on all512rows. Retain parent exact/zero consistency checks and explicitly distinguish reused controls from new arithmetic.

Report endpoint and paired raw/rounded head errors, actual receiving-state errors, per-head countertrends and active/padding/at8/after8 strata. Measure overlap and absolute correction-mass capture. Also separate the positive and negative terms

\[
w_j=p_\ell(j)\operatorname{expm1}(t_j),\qquad
t_j=(x-\ell)^T(u_j-\mu_\ell).
\]

Report captured sums of max(w_j,0) and max(−w_j,0), their total masses, signed numerator/partition corrections and zero-mass counts. Keep undefined ratios explicit instead of dividing by a small arbitrary epsilon. High-current-score selection can miss keys whose influence should decrease. These diagnostic measures do not independently certify output fidelity.

Interpret narrowly. A better actual-query envelope supports source-geometry mismatch as a lead; it also removes three source witnesses, so this comparison alone does not isolate missing coverage from removing distractor directions. Better current-query ranking implicates the envelope's anchor protection. If both remain poor while correction-mass support does better, prioritize a correction-aware selector rather than more witnesses. None establishes a cheap selector, output-optimal support or universally sufficient four-key representation.

Tiny checks cover independent support ranking/ties/masks, fixed oracle indices, the unchanged centroid law, positive/negative/zero capture accounting and raw/BF16 record integrity. Run one finite guarded CPU mrun lease near retained data, strict preflight/default admission, no duplicate suite or automatic retry. Charge every full-prefix score scan, direct read, serialization and original consumer separately. Preserve all outcomes without an accuracy stop gate. Full arbitrary-prompt zero-neural images remain open.
