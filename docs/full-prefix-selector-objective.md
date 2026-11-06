# The current query selects a much better four-key read

Status: **measured retrospective component result; arbitrary-prompt zero-neural images remain open**. On the retained prompt pair, selecting the four largest current-query key scores reduces paired first-block state error from95.96% to41.76%, and active-row error from32.85% to8.33%. The source geometry, routes, four-key budget and centroid-tail equation are unchanged. This is a full-prefix diagnostic, not a compact serving algorithm or a new image generator.

The [frozen plan](plans/full-prefix-selector-objective.md) follows the [source-witness result](source-witness-support.md). That experiment improved active state response slightly but worsened padding and overall response. This contrast asks whether supplying the actual query helps, and whether protecting anchor-important keys helps it select a support.

## The retained computation

Use the first Qwen3 block bundled with FLUX.2 Klein4B:32query heads,8K/V groups, head dimension128,512output query rows. Both saved prompts have27active keys and485padding output rows. Their only changed token is at address8. These are supplied saved operands from an already studied pair, not fresh or prospectively withheld raw strings. Later query vectors and routes agree across the pair, so later predicted changes still respond to changed memory.

With scaled query x, source anchor ℓ and source-key mean c_g, write

\[
u_j=y_j-c_g,\qquad s_j=\ell^Tu_j,\qquad a_j=x^Tu_j.
\]

The two new supports are top4 a_j (**current**) and top4 max(s_j,a_j) (**envelope**). Scores use the fixed source gauge and zero offsets. Only current causal, unmasked keys are eligible; earlier occurrence breaks ties. The original source-anchor and four-source-witness results are reused, as are the previous correction-mass support indices. No oracle is reranked and no target-output error fits any selector.

For each chosen support S, independently aggregate the complement T under p_ℓ(j)=softmax(s)_j. Let α_T be its mass, μ_T its key mean and v̄_T its value mean. With δ=x−ℓ, every arm uses exactly the existing centroid law:

\[
\widehat h_S(x)=
\frac{\sum_{j\in S}p_\ell(j)e^{\delta^Tu_j}v_j+
\alpha_Te^{\delta^T\mu_T}\bar v_T}
{\sum_{j\in S}p_\ell(j)e^{\delta^Tu_j}+
\alpha_Te^{\delta^T\mu_T}}.
\]

The complete new raw/BF16 banks and supports seal before exact attention references, historical oracle arrays or original consumer parameters are read. Afterward, all512rows run through the original output projection, residual, RMS normalization, SwiGLU MLP and state addition. The original learned equations remain paid offline computation; zero framework imports do not remove them.

## Response through the actual receiving block

Errors compare the predicted prompt difference with the exact FP64/BF16 prompt difference, with the zero-attention arm subtracted where specified in the complete report. They are relative RMS ratios on dependent coordinates of one retrospective pair, not independent trial estimates.

| Selector | State change, all | Active | Padding | Rounded-head change, all |
|---|---:|---:|---:|---:|
| Current query, new |41.76%|8.33%|43.61%|37.13%|
| Anchor/current envelope, new |66.25%|20.52%|69.01%|55.76%|
| Source anchor, retained |95.96%|32.85%|99.87%|89.77%|
| Four source witnesses, retained |107.98%|31.29%|112.54%|105.06%|
| Correction-mass support, retained |43.67%|14.98%|45.45%|38.49%|

The shared all-row target state-change RMS is.02485758; active and padding values are.03174371 and.02441723. At the changed address8, current-query state error is4.19%, versus28.23% for source-anchor selection. Before8 the exact and predicted changes are zero, so relative errors are undefined, not epsilon-normalized.

Raw-head errors preserve the same ordering: current37.05%, envelope55.69%, anchor89.75%, witnesses105.05%, correction-mass support38.41%. The result is not explained by BF16 rounding alone.

Current improves rounded paired-head error on31/32heads overall,30/32active heads and27/32padding heads relative to the anchor. Padding heads0,14,20,23,29 worsen. The complete report keeps every head and region. Current also beats the correction-mass support on27/32heads overall, but that does not make either selector output-optimal.

Absolute endpoint comparisons give a useful countertrend. Current state error relative to the original attention effect is11.294% and11.377% across the two prompts; correction-mass support is slightly better at11.236% and11.149%. Relative to the entire endpoint state, current errors are12.618% and12.723%. A better paired change does not guarantee better endpoints, and these errors are still material.

## Signed corrections explain why capture is only a diagnostic

Let H=E_{p_ℓ}[v], μ_ℓ=E_{p_ℓ}[u] and t_j=δᵀ(u_j−μ_ℓ). Then

\[
w_j=p_\ell(j)\operatorname{expm1}(t_j),\quad
D=E_{p_\ell}[e^t]=1+\sum_jw_j,\quad
\beta=D^{-1},\quad c_j=w_j/D,
\]
\[
h(x)=\beta H+\sum_jc_jv_j,\qquad
\beta+\sum_jc_j=1.
\]

Positive w increases influence; negative w decreases it. The diagnostic retains each signed numerator vector ∑c_jv_j, partition correction ∑c_j, and their selected-support counterparts. It also retains log masses, so a tiny negative mass divided by a huge D does not silently become a defined zero-denominator ratio. Sign-capture fractions are computed in log space; truly zero total sign mass gives an undefined ratio with an explicit count.

| Selector | Positive capture, padding | Negative capture, padding | Absolute capture, padding |
|---|---:|---:|---:|
| Current |.76142|.15465|.69321|
| Envelope |.67044|.45301|.64823|
| Source anchor |.33186|.69273|.36785|
| Source witnesses |.33148|.46973|.34475|
| Correction-mass support |.76793|.27860|.72113|

These are equal-weight means over the30,892defined padding head-query rows across both prompts. Another148/31,040padding rows have zero correction mass; ratios are undefined. Active counts are1,664defined and64undefined of1,728rows. Every selector shares those denominators. The previous omitted-mass diagnostic encoded zero in the zero-total case; its historical numbers remain unchanged and should not be directly subtracted from these means.

The current selector captures much less negative correction mass than the anchor or envelope and nevertheless predicts the receiving response better. The correction-mass selector captures more absolute mass than current yet loses slightly on the paired state response. These are observations against treating capture as a scalar authority, not evidence that decreasing influences never matter.

The reason is visible algebraically. If the centroid approximation has numerator N̂ and denominator Ẑ, with omitted-tail errors r_N and r_Z, then

\[
h-\widehat h=\frac{r_N-\widehat h\,r_Z}{\widehat Z+r_Z}.
\]

The value direction and numerator/partition cancellation matter together. Capturing a large absolute change in weight does not order the error of this ratio or the nonlinear receiving MLP. This identity explains the possible failure of a capture heuristic; this run does not separately causally allocate those remainders.

## What changed in our interpretation

Actual-query envelope selection beats the old source witnesses. That supports source geometry as a lead, but the new arm both adds the actual query and removes three source directions; it does not isolate missing coverage from removing distractors. Current beats the actual-query envelope, suggesting that anchor protection can retain less useful keys under this read. The useful next direction is actual-query retrieval, rather than immediately growing the failed witness family.

This is still a bounded trend. Full-prefix access supplies information unavailable to the previous compact admission rule. Four-key sufficiency, a cheap retrieval method, a faithful own-state encoder and general pixels remain unestablished. The [next proposed assay](plans/certified-current-query-retrieval.md) tests whether the useful ranking can be recovered with less score work, with no approximation to its support membership and explicit fallback/storage costs.

## Compute, custody and instruments

Guarded CPU job`job-deaae907863b` succeeds on its first attempt in47.19scheduler seconds,45.287worker seconds, with1,911.1MB peak RSS and zero VRAM. NumPy FP64 arithmetic uses one configured math thread; mrun's default admission reserves4CPU slots/4GB. Beast is pinned because the operands and original checkpoint are host-local. Timeout600seconds, strict preflight, no automatic retry or duplicate suite.

| Scope | Measured |
|---|---:|
| New prediction child, both selectors/two prompts |3.3596s|
| Child after imports |3.2841s wall /2.7463s CPU|
| Child high-water RSS |210,704KiB|
| Source geometry arrays read |3,153,920bytes|
| Shared scanned key occurrences, per prompt |431,136|
| Shared anchor/current score MACs, per prompt |110,370,816|
| Original parameter verification/read |5.6240s|
| Post-seal exact/signed diagnostics, each prompt |8.0240s /9.8413s|
| Four original consumer arms, each prompt |4.8491s /4.8774s|

The MAC count includes only the shared anchor/current key-score pairs, not routing, sorting, direct tail aggregation, diagnostics, serialization or the consumer. There is no compact core-state claim, FLOP/energy measurement, repeated latency study or controlled native speed comparison. New consumer work contains8full output and24MLP projections; the three historical consumer arms are reused.

All routes agree. Exact reference BF16 words and exact/zero original-consumer words match their parents. Direct centroid closure is at most9.23×10⁻¹⁶ across new and historical arms; signed partition closure is1.89×10⁻¹⁵ and signed read versus independently evaluated exact raw attention closes within1.11×10⁻¹⁵. These are measured mechanics, not arbitrary fidelity gates. All74tiny mechanics tests pass.

An independent full-byte check covers34retained files totaling5,971,901,541bytes. This includes the4,967,215,360byte original checkpoint and overlapping parents;581,992,315verified bytes are in the new job root, including references/diagnostics/consumers, not all serving state. Complete arrays stay near compute. The [public bundle](results/selector-objective/provenance.json) is4,357,185bytes in11files, with byte-lossless gzip for the9.34MB diagnostic JSON and all38,400consumer observations. The [Saturn run/verify receipts](results/selector-objective-evidence/sweep.json) cover one observation stream with no fires, no terminal claim receipts and no physical address joins. They certify instrument execution, not image generation.

Request SHA256:`82820efbe67c20b77d6a10104917ea67f834e909393d6537f3ba40353dcc1539`. Prediction seal SHA256:`2f2516c7050acadcf30d8712e3639ed09b6cf7480972a126a25977dac5a5544a`.

Read the [complete report](results/selector-objective/report.json), [lossless signed diagnostics](results/selector-objective/signed-diagnostics.json.gz), [reference/parameter receipt](results/selector-objective/reference-receipt.json), [retained-byte manifest](results/selector-objective/retained-array-verification.json), [prediction code](../generator/selector_objective.py), [study](../generator/selector_objective_study.py), [publisher](../generator/publish_selector_objective.py) and [mechanics tests](../generator/test_selector_objective.py). The newest local [attention-kernel/MLP propagation post](/Users/jakeholl/domains/obsidian/blog/2026-11-06-054800-five-attention-values-become-a-different-hidden-state.md) strengthens the original-parameter finite reference. It does not eliminate learned computation or require this qualified FP64 diagnostic to reproduce every CUDA implementation bit-for-bit.
