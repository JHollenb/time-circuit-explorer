# What the first read still has to compute

The first-read crossover separates two jobs in prompt interpretation: deciding how strongly to read each address, and producing the value stored there. On this retained one-token edit, the allocation effect has a larger norm than the payload effect on 394 of 477 dependent padding-query rows; payload is larger on 34, and 49 have no change. The [compiled ordering program](source-ordering-program.md) supplies addresses within its finite domain. This experiment measures the weighted read those addresses must eventually support.

Arbitrary-prompt image generation with zero neural evaluations remains open. The local [reconstructed generator](../generator/README.md) accepts fresh prompts while evaluating the learned denoiser equations and using a native encoder and VAE. The hosted generator displays recorded runs. The separate [zero-network response chart](zero-network-image-response.md) serves four calibrated scene contexts and doses. None of those capabilities should be conflated.

## One authenticated read

We use the original first Qwen3 encoder reader, head 0 and GQA group 0, on its historical A/B pair. One token changes at position 8, ID 2115→1290. Both inputs have 35 active keys and 477 padding queries, with identical masks and positions. All 128 BF16 value coordinates at address 8 change; every other value word in this group remains equal. Each query has a captured ordered top-four support and four captured weights.

These are the approximate reader's weights, normalized within its selected four. They do not report full native attention mass. The 477 query positions share the same prompt and source history; they are dependent observations, not 477 semantic replications. This diagnostic supplies the weights and values from the retained source rather than predicting them from raw text.

Let \(p_X(j)\) be the captured weight at address\(j\), with zero outside the selected support, and let\(V_Y(j)\) be the value bank. The four reads are

\[
H_{XY}=\sum_j p_X(j)V_Y(j),\qquad X,Y\in\{A,B\}.
\]

The two diagonals are retained source endpoints. We compute just the cross reads\(H_{AB},H_{BA}\), seal them before opening the numerical endpoint members, and independently replay both diagonals in a separately charged account. The replay matches all 122,112 retained FP64 coordinates and all 122,112 admitted BF16 words exactly.

The source operation is a separate contiguous FP64[4] × fancy-gather FP64[4,128] contraction for each query. We preserve captured order and weights, then admit through FP32→BF16 round-to-nearest-even. There is no new QK evaluation, softmax, head projection, MLP, encoder suffix or image consumer.

## Factor the read change

An allocation-first decomposition and a payload-first decomposition give different intermediate attributions. We use their symmetric average:

\[
P=\tfrac12[(H_{AB}-H_{AA})+(H_{BB}-H_{BA})],
\]
\[
A=\tfrac12[(H_{BA}-H_{AA})+(H_{BB}-H_{AB})],
\]
\[
I=H_{BB}-H_{BA}-H_{AB}+H_{AA},\qquad\Delta H=H_{BB}-H_{AA}=P+A.
\]

Here\(P\) is the symmetric payload effect,\(A\) the symmetric allocation effect, and\(I\) their interaction. This chosen convention shares the interaction between the two main effects; it is not a unique semantic attribution.

Only one value address changes. Writing\(\Delta v=V_B(8)-V_A(8)\), we can factor

\[
P_{\mathrm{ideal}}=\frac{p_A(8)+p_B(8)}2\Delta v,
\qquad I_{\mathrm{ideal}}=[p_B(8)-p_A(8)]\Delta v.
\]

Both effects therefore lie along the same value-change direction in ideal arithmetic. That identity is established mathematics. The new evidence is its source-qualified effect account at this particular first read, including the small finite remainders and the allocation component that the identity does not supply.

| Raw dependent-row account | Measured value |
|---|---:|
| Edited address selected in both inputs |43|
| Selected only in B |385|
| Selected in neither |49|
| Ordered supports change |426|
| Nonzero read changes |428|
| Allocation norm larger / payload norm larger |394 /34|
| Negative payload–allocation inner product |145 rows|
| Pooled allocation norm / read-change norm |0.909508|
| Pooled payload norm / read-change norm |0.404401|
| Pooled interaction norm / read-change norm |0.671934|

These norm ratios describe vectors and do not divide semantic importance into percentages. They need not add to one. The signed account is

\[
\|\Delta H\|^2=\|P\|^2+\|A\|^2+2\langle P,A\rangle.
\]

Summing across rows gives 11.390648=1.862831+9.422403+0.105413, to the displayed precision. Although 145 rows have cancellation, the pooled cross term is positive. Keeping both facts avoids hiding the heterogeneous geometry behind one aggregate.

Queries 74 and 75 retain their exact ordered support but change the weights and both measured effects. At query 74, the edited address's weight changes 0.274604→0.510951; at 75 it changes 0.259813→0.493297. An address program alone cannot specify those weighted reads.

## The one-edit update has a smaller interface

The source's fixed padding query and token-local keys suggest another factorization. Let \(C\) be the common selected addresses excluding address 8. Their scores are unchanged by the token edit, so their relative weights should be unchanged in ideal arithmetic. A separate [public-scalar diagnostic](results/first-read-route-payload/common-weight-ratio.json) checks all 477 rows without another scientific lease: the largest conditional-weight difference is 1.6653×10⁻¹⁶. All 428 rows selecting address 8 in B have three common unedited addresses; the 49 neither-address rows have four and zero conditional difference.

For a common value mixture \(\mu_C\), let \(a=p_A(8)\) and \(b=p_B(8)\) when both supports contain the edited address. Then

\[
H_{AA}=(1-a)\mu_C+aV_A(8),\qquad H_{BB}=(1-b)\mu_C+bV_B(8),
\]
\[
\Delta H=(b-a)[V_A(8)-\mu_C]+b\Delta v.
\]

When address 8 enters B, let \(r\) be the dropped A address and \(a=p_A(r)\). The corresponding identity is

\[
\Delta H=b[V_B(8)-\mu_C]-a[V_A(r)-\mu_C].
\]

These are ideal conditional update laws. We have checked the scalar weight-ratio premise, not independently replayed these vector expressions. Using supplied \(b\) and old common conditional weights reconstructs the captured B weights within 2.7756×10⁻¹⁶; it does not produce either input. The diagnostic never replaces source weights or renormalizes a serving read. Its [helper](../generator/common_weight_ratio.py) and [additive receipt](results/first-read-route-payload/common-weight-publication.json) preserve that boundary.

For a standalone read, all four weights must come from the new input. A useful next [design question](plans/first-read-relative-mass-producer.md) is whether token identities and absolute positions can produce three relative score gaps or equivalent positive masses, without Q/K products, a per-history weight table or supplied old weights. No fit, table expansion or new native assay is queued.

## Preserve the rounding account

We keep inherited raw diagonals, replayed raw diagonals and admitted BF16 corners in three named panels. The first two agree exactly. Within each panel, normalize remainders by that panel's pooled read-change norm. The raw ideal payload identity's remainder is 8.53×10⁻¹⁷ and the interaction remainder 1.90×10⁻¹⁶; the admitted BF16 remainders are 0.002271 and 0.004591. This is rounding of a vector read, not a new semantic interaction.

The admitted A→B endpoints change 54,677 of 61,056 coordinates. Changing values while holding A's allocation changes 5,421 words; holding B's allocation changes 54,131. The weights determine where the value edit reaches the read. All 477 rows, including zeros and signed cross terms, remain in the [public row stream](results/first-read-route-payload/rows.jsonl.gz).

## Measured cost and remaining computation

The sole guarded CPU job, `job-21bbdd67b700`, takes 2.32 scheduler seconds with 73.4 MB peak sampled RSS and 0 VRAM. The worker records 0.316825 seconds after imports and before final report/provenance copies, including source authentication, diagnostics and array/row serialization. Its own high-water measure is 78,980 KiB; these are different instruments and scopes, neither an isolated steady-state memory estimate.

The two cross reads pay 488,448 scalar value products and 22.95 ms observed wall time, with 4.55 ms process CPU time. Independent endpoint replay pays 488,448 more products and 3.76 ms wall time. Those product counts exclude diagnostic norms/dots, factorial arithmetic, admission, member materialization, hashing and serialization. One run is not a latency distribution or matched speed comparison. Original lexical compilation, original QK/softmax and historical source acquisition remain paid.

Nine targeted tiny mechanics tests pass once in 0.025 s. The independent [publisher](../generator/publish_first_read_payload.py) verifies all 477 scalar rows, three aggregates, staged/collected identities, source and execution custody, and the complete compact collection. It freshly hashes 89,081,598 remote bytes: 73,440,516 source bytes plus 15,641,082 retained array bytes. The 13 original compact files total 252,819 B; full vectors remain near compute. The publisher does not independently rerun vector calculations, and its member allowlist is trusted code rather than an adversarial sandbox.

The remaining task is to produce prompt-dependent weights and values, consume the resulting read through the output projection and MLP, advance the actual contextual state, and eventually produce pixels. This experiment measures no downstream state or image consequence. The [original plan](plans/first-read-route-payload.md), [source-policy implementation note](plans/first-read-route-payload-implementation.md), [core](../generator/first_read_payload.py), [driver](../generator/first_read_payload_study.py), [report](results/first-read-route-payload/report.json), [scalar audit](results/first-read-route-payload/scalar-audit.json) and [review record](results/first-read-route-payload-evidence/README.md) preserve the bounded result.

Full research/storage review 96 remains current, next 99. This cycle adds a small near-compute array result and compact publication; no archive or deletion is needed. The goal stays active.
