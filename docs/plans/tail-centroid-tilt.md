# Freeze support, then test the tail centroid

Status: **implemented and measured in cycle 13**. The [matched result](../tail-centroid-tilt.md) improves eligible paired state error 126.3%→96.0% at identical state/support; frozen oracle error slightly worsens 41.8%→43.7%. The original plan follows. The [sparse joint-tilt result](../sparse-joint-tilt.md) reduces core state 31.9 times and improves active responses, while eligible padding remains poor. Full-query-score oracle support improves the response substantially. That gap does not isolate support selection from the unit-tail approximation or membership changes.

Use only the existing retrospective pair, 64 source landmarks, nearest routes and m4 budget. Freeze both original eligible and original oracle supports at every query. Do not rerank the oracle for the new law. Compare unit versus centroid tail within each support class; no new selector geometry, coefficient fitting, dataset or covariance tensor.

Let Z_T,N_T,Y_T be the existing anchor-weighted tail state, h_T=N_T/Z_T, μ_T=Y_T/Z_T, and δ=x−ℓ. Replace the old unit tail factor with exp(δᵀ(μ_T−μ_full)). Retained entries keep exp(δᵀ(y_j−μ_full)). Their common centering factor cancels:

\[
\widehat h_{centroid}=
\frac{Z_Te^{\delta^T\mu_T}h_T+\sum_{j\in S}e^{s_j+\delta^Ty_j}v_j}
{Z_Te^{\delta^T\mu_T}+\sum_{j\in S}e^{s_j+\delta^Ty_j}}.
\]

Treat the tail as one weighted key/value centroid. This uses existing key sums, adds one scalar contraction at a query and no persistent core bytes. Combine positive log masses; preserve the empty-tail branch. No clipping, exact fallback or per-query target information is admitted in the eligible law. Old oracle support remains an offline diagnostic requiring full current-prefix key access.

The tail's exact mass is

\[
Z_Te^{\delta^T\mu_T}\kappa_T,\qquad
\kappa_T=E_T[e^{\delta^T(y-\mu_T)}]\ge1.
\]

The centroid law omits this partition curvature and the tail's tilted value response. At the anchor, exact minus centroid read derivative is

\[
D h_{exact}-D\widehat h_{centroid}
=p_T\operatorname{Cov}_T(v,y),
\]

where p_T is full anchor tail probability. Thus the approximation restores between-group mean transport while omitting within-tail interactions. Positivity and centering-gauge invariance are mechanics/properties, not accuracy gates. Empty support gives the anchor read, because its common centroid factor cancels; complete support gives exact attention.

Seal complete eligible candidate banks before exact references, oracle arms or original consumer reads. If replaying existing support, verify position-set equality against retained evidence; ignore heap-slot order and invalid sentinels. Full-query-score oracle indices must remain identical between unit and centroid arms. Retain own-state tail/retained updates, source/clock/budget identity and resume closure.

Measure paired rounded head-bank changes and original first-block state responses on all 512 rows, with active/padding/at8/after8 splits. Original learned output/RMS/MLP consumer remains separately paid offline work. Stratify paired head errors by unchanged versus changed support position sets across prompts. This stratification is descriptive and does not independently isolate a causal membership effect. Report raw RMS, cosine, signed errors and all countertrends; retain small/zero target denominators explicitly.

Tiny mechanics checks should cover a direct centroid-mixture reference, empty/full support and tail, gauge invariance under common key translation/centering, extreme log masses, unchanged state bytes and checkpoint/resume. Run one finite guarded CPU mrun lease near immutable retained data, strict preflight, no duplicate suite or automatic retry. Charge stream/read/serialization/oracle/consumer time separately. No new neural references or raw-string generalization claim is needed for this matched cut.

If eligible fidelity improves materially, tail treatment was material on this specimen. If a large eligible/oracle gap persists under the matched law, selection geometry becomes the next question. Neither outcome excludes other sparse laws or justifies a larger cohort by itself. Full contextual encoding, MLP replacement, own-state diffusion, general pixels and arbitrary-prompt zero-neural generation remain open.
