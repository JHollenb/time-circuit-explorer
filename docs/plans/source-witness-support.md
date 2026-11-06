# Source query witnesses for a fixed support budget

Status: **implemented and measured in cycle 14**. The [result](../source-witness-support.md) slightly improves active state error32.85%→31.29% but worsens overall95.96%→107.98%, with lower correction-mass capture. Original plan follows. The [matched tail-law result](../tail-centroid-tilt.md) improves eligible paired state error to 95.96%, but the frozen oracle reaches 43.67% under the same centroid reader. Test selection geometry while keeping that reader, four-key budget, 64 anchors per head and nearest-source routing fixed.

Reconstruct the parent's existing source query candidate pool from retained lexical atoms and candidate identities. Verify its original anchor identities/values. Assign candidates to the existing anchors using the existing key-covariance metric, with canonical identity/position ties. Current prompt queries and contextual targets must not enter compilation.

For each anchor ℓ, choose four frozen witnesses: q_ℓ0=ℓ, followed by three deterministic farthest-first candidates within its source cell, starting from the anchor. Use the same source metric and stable tie-breaking. Repeat the anchor for degenerate cells. Record identities, norms, cell occupancy and selection distances; no target-driven norm normalization, offset fitting or later repair.

Witnesses use the existing scaled coordinates x=q/d^.25 and y=k/d^.25. Use the existing immutable source key mean c_g as the declared gauge, with zero witness offsets. For an admitted key y,

\[
\rho_\ell(y)=\max_{b=0,\ldots,3}q_{\ell b}^T(y-c_g).
\]

Retain the four highest-priority occurrences, earlier clock winning ties. Store priority separately from the original anchor score

\[
s_\ell(y)=\ell^T(y-c_g).
\]

All positive tail Z/N/Y updates, evictions and centroid reads continue to use s_ℓ, never ρ_ℓ. Fixed witness priorities preserve a monotone top-four cutoff: rejected/evicted occurrences cannot re-enter, so the tail remains append-only. Repeated occurrences stay distinct and masks exclude keys. Bind checkpoint identity to witness bytes, policy, source gauge, budget and clock; preserve exact resumed execution.

This is a finite source-score envelope, not an attention probability. If x=Σ_b α_b q_ℓb with α_b≥0 and Σ_b α_b=1, then

\[
x^T(y-c_g)\le\rho_\ell(y).
\]

The four-witness convex hull has affine dimension at most three. Coverage of a general 128-dimensional source cell or an unseen query is unestablished. Even within the hull, keeping the largest envelope values is not a normalized-attention error guarantee. Ranking depends on the fixed c_g; source-gauge changes are a different algorithm. Report winning-witness frequencies and norms to expose dominance by one high-norm witness without retrospectively changing offsets.

Use the existing retrospective pair first. Compare anchor-score and witness-score priority under the identical centroid-tail read, advancing each candidate's own prefix state. Seal complete all32-head banks before target/reference/consumer access. Retain the previous frozen oracle-centroid result as a separately labeled reference; do not rerank or optimize it for this selector. No output-target fitting, covariance tensor, landmark increase or larger cohort is part of this cut.

Report paired rounded-head and actual first-block state responses on all512rows, including active/padding/at8/after8 splits and per-head countertrends. Retain support overlap/captured tilt diagnostics, canonical membership changes and zero/small target denominators. These are observations, not accuracy gates or independent causal membership tests. Original output/RMS/MLP equations remain separately paid offline consumer work.

Charge 65,536 extra FP64 priority bytes across32heads, 6,291,456 additional FP64 witness bytes if the original anchor is reused, and actual identity/diagnostic metadata. The new core is at least8,618,496bytes before further state/metadata; runtime geometry is at least9,445,376bytes before metadata. Four witnesses require approximately four times the admission score dot products. These costs offer no automatic efficiency advantage over the2,097,152-byte full BF16 GQA K/V payload. Separate source compilation, admission, readout, serialization and original-consumer timing.

Tiny mechanics checks should independently validate priority order/ties, original anchor-weight tail means after priority-based eviction, no future/masked-key access, degenerate witnesses, monotone cutoff, unchanged centroid law and identity-bound resume. Use one finite guarded CPU mrun lease near retained source data, strict preflight, default measured admission, no duplicate suite or automatic retry. Preserve raw rows and custody regardless of accuracy trends. A subsequent fresh raw-string own-state test is required before calling the selector reusable serving evidence.

This remains one reduced model-derived attention experiment. Later contextual operand formation, MLP replacement, own-state diffusion, general pixels and arbitrary-prompt zero-neural images remain open.
