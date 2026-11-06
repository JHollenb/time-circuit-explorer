# Separate tail mass from tail value under the useful support

Status: **proposed after cycle16/Astra review; not implemented, staged or measured**. [Certified retrieval](../certified-current-query-retrieval.md) preserves current-query top4 but is slower than a full scan. Stop that cost optimization and inspect the remaining approximation under fixed addresses. Use the same retrospective pair, source geometry, routes, current-query support/order and centroid law; do not increase support or add covariance state before seeing this result.

For selected support S and complement T, let Z_S,h_S be its exact mass/value mean under the current query. Let h_T⁰ be the anchor-weighted tail value mean, Z̃_T its centroid-predicted current mass, and h_T*,Z_T* the exact tilted tail mean/mass. Define

\[
w=\frac{\widetilde Z_T}{Z_S+\widetilde Z_T},\qquad
w^*=\frac{Z_T^*}{Z_S+Z_T^*}.
\]

Cross two positive mixture factors:

| Arm | Tail weight | Tail value |
|---|---|---|
|00existing centroid |w|h_T⁰|
|10mass repair |w*|h_T⁰|
|01value repair |w|h_T*|
|11exact |w*|h_T*|

Each h_ab=(1−w_a)h_S+w_a h_T^b. Keep all masses in one common unnormalized anchor/query gauge and aggregate in log space; do not mix normalized anchor and current masses. An empty complement makes all arms the selected read; retain that case and undefined tail ratios explicitly. Record logκ_T=logZ_T*−logZ̃_T, whose ideal nonnegativity follows from Jensen, while preserving finite negative deviations as observations.

The ordered smooth telescope is

\[
h_{00}-h_{11}=
\underbrace{(w-w^*)(h_T^0-h_S)}_{A\;\mathrm{mass}}
+\underbrace{w^*(h_T^0-h_T^*)}_{B\;\mathrm{value}}.
\]

Retain both complete vectors, their energies and signed2⟨A,B⟩cross-energy, including **paired vector differences across prompts**. The crossed interaction

\[
h_{11}-h_{10}-h_{01}+h_{00}
=(w-w^*)(h_T^0-h_T^*)
\]

shows dependence on intervention order; it is not unique causal credit. Separate each raw smooth result, BF16 rounding defect and downstream interaction. Do not infer consumer fidelity from one remainder norm, tail mass or logκ alone.

This is a full-prefix diagnostic repair using actual current keys/values. Freeze source/support identity, build both new hybrid banks and seal complete raw/BF16 arrays before target/control outputs or original consumer parameters are opened. Retain existing current/exact/zero banks and consumers unchanged; recompute/validate their bank identities under the same arithmetic before inheriting receiving results. No teacher refresh, target-output fit, new rawprompt or compact serving claim.

Execute only the two new hybrid banks through the original receiving output/residual/RMS/MLP/state equations on all512rows. Charge4new output/12MLP projections across2prompts, original parameter read, references and serialization separately; any additional control evaluation must be explicitly counted. Use the same paired-effect denominator and active/padding/before8/at8/after8 strata, perhead outcomes, endpoint countertrends and downstream factorial interaction. Original learned consumer evidence remains offline assistance.

Tiny checks should cover independent four-arm probability mixtures, exact telescope/crossed identity, empty tails, fixed support/order, extreme log masses and raw/BF16 records. One finite guarded CPU1 mrun lease near retained arrays, strict preflight/default admission, no automatic retry or duplicate suite. Preserve all effects without an accuracy stop gate. This diagnostic determines which tail information to try representing next; it does not justify another large covariance tensor, promise own-state closure or complete arbitrary-prompt zero-neural images.
