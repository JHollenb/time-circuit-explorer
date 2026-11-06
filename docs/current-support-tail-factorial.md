# Tail mass and tail value both change the receiving state

Status: **measured retrospective diagnostic; arbitrary-prompt zero-neural images remain open**. With useful current-query top-four support fixed, repairing only omitted tail mass reduces paired first-block state error from 41.76% to 33.46%; repairing only tail value gives 33.77%. Value repair improves the paired rounded response of all 32 heads, including active and padded rows. Mass repair improves only 12 of 32 active-head responses. Their errors partly cancel, so a single pooled ranking would hide useful structure.

The [frozen protocol](plans/current-support-tail-factorial.md) follows [current-query selection](full-prefix-selector-objective.md) and its [retrieval-cost assay](certified-current-query-retrieval.md). Selection is useful but the frozen retrieval implementation is slower than a full scan. This experiment returns to the remaining approximation: what information does the unselected tail need to carry? It uses the same saved crane-before/crane-behind pair, all 32 first-layer query heads and all 512 receiving rows. It does not test fresh prompt generalization.

## Separate two factors without moving the addresses

For each current query, keep the ordered selected occurrences S, complement T, source anchor and route unchanged. The selected part has exact current unnormalized mass Z_S and mean h_S. The centroid approximation predicts tail mass Z̃_T and retains the anchor-weighted tail mean h_T⁰. Full-prefix mathematical attention supplies the exact current tail mass Z_T* and tilted mean h_T*. Define

\[
w=\frac{\widetilde Z_T}{Z_S+\widetilde Z_T},\qquad
w^*=\frac{Z_T^*}{Z_S+Z_T^*}.
\]

Four positive mixtures cross those factors:

| Arm | Tail mass | Tail value | Head read |
|---|---|---|---|
|00, existing centroid|Z̃_T|h_T⁰|(1−w)h_S+w h_T⁰|
|10, mass repair|Z_T*|h_T⁰|(1−w*)h_S+w* h_T⁰|
|01, value repair|Z̃_T|h_T*|(1−w)h_S+w h_T*|
|11, exact control|Z_T*|h_T*|(1−w*)h_S+w* h_T*|

All masses share the same unnormalized gauge and are aggregated in log space. Empty tails make all four arms identical; tail ratios remain undefined there. The first four causal query clocks produce 128 empty head/query tails per prompt. No epsilon turns an absent tail or zero reference change into a score.

The smooth head error factors exactly in real arithmetic:

\[
h_{00}-h_{11}=
\underbrace{(w-w^*)(h_T^0-h_S)}_{A\;\mathrm{mass}}
+\underbrace{w^*(h_T^0-h_T^*)}_{B\;\mathrm{value}}.
\]

The crossed interaction is

\[
I=h_{11}-h_{10}-h_{01}+h_{00}
=(w-w^*)(h_T^0-h_T^*).
\]

These are ordered vector identities, not independent scalar error budgets or unique causal credit. For paired prompt changes, subtract each complete A and B vector across prompts before measuring energy. The report retains their signed cross-energy, rather than summing magnitudes.

Canonical 00 uses the previous centroid arithmetic; canonical 11 uses the previous full mathematical attention arithmetic. Their tiny difference from the smooth mixtures is a separate control-arithmetic term C. BF16 boundaries add another vector R, giving rounded error A+B+C+R. Original control arithmetic is preserved so identity checks can support reuse of historical receiving results.

## The receiving computation distinguishes the repairs

Each prompt has 27 attended keys and 485 padded query rows. Only token 8 changes; later queries and source routes are identical. A causal key mask does not remove padded conditioning outputs: later FLUX computation consumes all 512 rows.

The primary ratio compares error in the paired state change with the exact paired **attention-caused** state change, using the same zero-attention consumer control for every arm:

\[
E=\frac{\operatorname{RMS}[(Y_b^m-Y_a^m)-(Y_b^{11}-Y_a^{11})]}
{\operatorname{RMS}[(Y_b^{11}-Y_b^0)-(Y_a^{11}-Y_a^0)]}.
\]

Here m labels the factorial arm and the superscript 0 denotes the zero-attention control, not arm 00. This denominator differs from the complete hidden-state change and is retained explicitly in the raw report.

| Arm | All 512 rows | 27 active rows | 485 padded rows | Changed row 8 |
|---|---:|---:|---:|---:|
|00, centroid|41.76%|8.33%|43.61%|4.19%|
|10, exact mass|33.46%|8.14%|34.91%|4.46%|
|01, exact value|33.77%|7.32%|35.25%|3.12%|
|11, exact control|0|0|0|0|

The exact-control zero is identity, not a general predictor achievement. Before token 8 both reference and prediction changes are zero, so relative error is undefined. The exact attention-effect RMS is .0248576 overall, .0317437 active and .0244172 padded. These are dependent rows from one retrospective pair, not independent trials.

At the rounded head-bank boundary, all-row paired error falls from 37.13% to 31.71% with mass repair and 29.54% with value repair. Active error falls from 16.52% to 15.69% and 11.68%, respectively. Value repair improves all 32 heads in all three all/active/padded strata. Mass repair improves 28 heads overall, 12 active and 27 padded. Thus a small pooled active improvement coexists with 20 worsening active heads. The report preserves every head and stratum.

Value repair also improves both absolute receiving endpoints more: endpoint attention-effect errors are 11.29%→9.38% and 11.38%→9.74%, versus 11.21% and 11.35% with mass repair. Whole-state endpoint denominators give different numbers and remain separately reported. Conversely, mass repair has the slightly better pooled paired state error. Neither metric alone selects a universal winner.

## Cancellation survives the original MLP consumer

For the complete paired rounded head error, normalize each component's mean-square energy by the total error energy:

| Component | Energy / total rounded error energy |
|---|---:|
|Mass A|.55014|
|Value B|.72580|
|Signed cross term 2⟨A,B⟩|−.28044|
|BF16 rounding R|.00417|

Control arithmetic and the remaining signed cross terms close the total; they are retained rather than rounded away in the [report](results/tail-factorial/report.json). The mass/value cross term is negative in active and padded strata too. Removing one approximation can remove cancellation as well as error.

Let F be the original output projection, residual, RMS normalization and SwiGLU consumer, including its declared BF16 boundaries. The measured factorial interaction is

\[
I_F=F(h_{11})-F(h_{10})-F(h_{01})+F(h_{00}).
\]

Its paired state RMS is .00492189, or 19.80% of the exact attention-caused state change; the corresponding active/padded ratios are 6.99%/20.60%. The ordered consumer telescope has normalized energies .56835 for mass-first and .64193 for value-after-mass, with signed cross-energy −.21027. These are actual hybrid consumers, not a linearized projection of head error. The interaction includes the crossed input construction and finite nonlinear consumption; it is not uniquely attributable to one MLP feature.

The tail weight itself shows a systematic missing mass: mean padded w is .17334/.17400, versus exact .28290/.28424. Mean padded logκ, where κ=Z_T*/Z̃_T, is .92069/.92240. Those averages are diagnostics, not sufficient statistics for the value or downstream state. Across all rows the minimum logκ is −1.33e−15, a retained finite-arithmetic deviation from ideal Jensen nonnegativity.

## Work, identity and evidence

The prediction child decodes only `supportPositions` and `nearestIndex` from the historical support files. It forms and seals all four complete raw/BF16 banks and signed diagnostic vectors before reading old control outputs or original consumer parameters. Whole-file hashing still reads bytes for custody. It computes full-prefix oracle repairs, including 64 mathematical exact-head reference evaluations; these are not a serving shortcut.

Both reconstructed 00/11 BF16 banks match the historical controls exactly, with zero support/order or route mismatches in any arm. That complete input identity allows inheritance of 00, 11 and zero consumers. Only 10 and 01 require new consumers: four full output and twelve MLP projections across the pair, each on all 512 rows. Original learned equations remain paid offline assistance. No native framework calls occur.

| Measurement | Observed |
|---|---:|
|Guarded CPU mrun job|`job-f1aeafd8d9a4`|
|Scheduler / worker wall time|31.88s / 29.534s|
|Peak sampled RSS / VRAM|1,945.4 MB / 0|
|Four-bank child, including startup|5.098s|
|Child after imports, wall / CPU|5.028s / 3.548s|
|Child high-water RSS|297,728 KiB|
|Original parameter read and verification|3.423s|
|Post-seal controls/consumer time, per prompt|7.207s / 6.147s|
|Runtime source geometry payload|3,153,920 bytes|
|New consumer dense contraction MACs|174,483,046,400|
|Exact-head reference QK/PV MACs|4,294,967,296|

The last two counts are derived from executed arm geometry and declared tensor shapes. They exclude softmax, normalization, SiLU, gating, routing, factorial arithmetic, hashing and serialization; they are not a complete FLOP, hardware-instruction or energy census. Parameter payload is 170,398,720 bytes; the verified original shard is 4,967,215,360 bytes. See the [accounting receipt](results/tail-factorial-evidence/accounting.json).

The job succeeds on its first attempt with CPU NumPy FP64, one math thread, default 4CPU/4GB admission, strict preflight, 600-second budget and no automatic retry. All 84 mechanics tests pass. Maximum smooth telescope/factorial closure is about 2.22e−16; canonical exact/smooth difference is at most 1.11e−15. These establish the instrument's arithmetic scope, not native CUDA parity.

Independent byte verification covers 31 files and 5,781,564,422 bytes, including the original checkpoint and overlapping inherited assets; this is not unique new serving storage. Complete arrays remain near compute. Public evidence is 10 files / 4,247,767 bytes, including all 38,400 losslessly compressed observations. The [Saturn run/verify sweep](results/tail-factorial-evidence/sweep.json) returns zero, with one stream and no watch fires. Its scope is `target-streams-only`, with no terminal image claims or physical-address joins.

Request: `272469f49d30d92dc0a8e7be40dba51af01445c83e8045fd2b080598c0ca89e8`. Prediction seal: `a483d9becd8abe14e80816b91398d5e894ddc3def936efc6635f241bf3ba5059`. Declared, sealed and executed guarded payload: `b936b0107785d9f43647d2642b63644a963feed25a50679b00b7624261e62b65`.

## The next mathematical cut

Astra's [review](results/tail-factorial-evidence/advisor.json) finds no material factorial implementation error and recommends one [joint positive-quadratic tail closure](plans/positive-tail-moment-closure.md). Both numerator and partition must come from the same positive finite measure. A covariance-only value correction can leave the value hull, while fitting another large state before testing its law would hide the present question. The proposed assay computes directional moments from retained arrays, without allocating a value–key–key tensor or claiming an efficient implementation already exists. A [follow-up](results/tail-factorial-evidence/advisor-followup.json) confirms the route after concurrent blog updates and clarifies the finite-mean cumulant diagnostic.

That [positive-quadratic follow-up is now measured](positive-tail-moment-closure.md), with paired state error 24.15% and retained head countertrends. A reusable streaming representation remains the next proposed cut.

The newest related [image-route intervention](/Users/jakeholl/domains/obsidian/blog/2026-11-06-064228-predicting-an-image-after-removing-one-hidden-state-route.md) predicts a genuine downstream image change using original parameters. It separates a direct H18 conditioning route from history still inherited by H27. That strengthens causal continuation evidence; it does not remove learned computation or make arbitrary-prompt zero-neural generation available.

Read the [provenance](results/tail-factorial/provenance.json), [retained-byte manifest](results/tail-factorial/retained-array-verification.json), [factorial code](../generator/tail_factorial.py), [study](../generator/tail_factorial_study.py), [publisher](../generator/publish_tail_factorial.py) and [tests](../generator/test_tail_factorial.py). Contextual encoding, complete own-state denoising, a general pixel decoder and independent prompt/image evidence remain required for the original goal.
