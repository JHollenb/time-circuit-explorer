# Making a state writer respond to context

Cycle 104 produces a nonzero first-state response to a prompt edit using its own raw input and non-neural serving. On the discovery cube left/right contrast, unchanged active-token H1 response error falls from 100% to 23.42%, with cosine 0.9723. That is a useful local change, with substantial limits: padding response error is 88.95%, the two vessel color contrasts still have zero padding response, and transfer endpoint accuracy worsens. No full encoder, denoiser or image is executed; arbitrary-prompt zero-neural image generation remains unsolved.

The [previous writer](contextual-writer-program.md) captured a common update but predicted zero change at every unchanged token. The new experiment changes the offline objective and coupled greedy fitting procedure while keeping its first writer, 155 features, depth-four/max-16-leaf grammar, minimum eight source rows and four discovery histories. It fits one candidate, with no parameter sweep, new teacher or feature expansion. The [frozen implementation contract](plans/contrast-writer-assay.md), [raw report](results/contrast-writer/report.json), [publication](results/contrast-writer/publication.json) and [source review](results/contrast-writer-evidence/advisor-source.json) define the actual scope.

## What changed mathematically

Let S be the actual rounded state after the frozen first writer and Y the native discovery H1. The second writer predicts S+AD before finite admission, where A selects a shared leaf constant from D. Define the retained FP64 residual R=Y−S. Let P take signed differences at unchanged learned-token positions in the two discovery prompt pairs. With separately normalized endpoint and response weights W,V, the relaxed objective is

\[
L(D)=\|AD-R\|_W^2+\|PAD-PR\|_V^2.
\]

Here q=PR is formed explicitly from the retained residual; direct loss evaluation uses that same target. Actual admitted-state responses are separately measured against native YB−YA. These floating evaluation orders are not silently interchanged.

For fixed routes the normal equations are

\[
M D=A^T\big(WR+P^TVPR\big),\qquad M=A^TWA+(PA)^TV(PA).
\]

Endpoint leaf mass is positive diagonal; the paired term is a graph Laplacian. There are at most sixteen leaves and 2,560 right-hand sides. The solve is offline learned constant fitting. Serving uses the unchanged branch/copy/add/FP32/BF16 program and never evaluates this matrix system or the original model's learned affine/gated maps.

A proposed split appends a right-subset correction t to the old parent leaf. With Q=W+PᵀVP, c=AᵀQt, d=tᵀQt, z=tᵀQR and the current fitted constants D, its relaxed improvement is

\[
s=d-c^TM^{-1}c,\qquad \Delta L=\frac{\|z-c^TD\|^2}{s}.
\]

Cholesky solves avoid an explicit inverse. A cancellation-dominated denominator triggers a direct partition solve rather than clamping. Each accepted split retains its direct loss before/after and the difference between its Schur score and realized relaxed decrease. No direct fallback is needed in this run; the largest recorded absolute gap is about 1.87×10⁻¹³. Actual finite rounding is evaluated separately.

The fitter streams one feature-prefix bank at a time and caches only eligible cut sums and local masks. It enumerates 2,901 feature cuts, retaining 2,497 distinct candidate partitions across the search; peak retained cut cache is 13,869,270 bytes. Eight splits produce nine second-slot atoms. The first slot still has fourteen atoms, giving 235,520 dictionary bytes versus the old endpoint program's 266,240. Fixed prefix, program and the 794.2 MB lexical/tokenizer assets remain additional. Equal maximum budget does not mean equal actual atom use.

## What now carries the edit

The old second tree used current-token coordinates and template/position features. The new tree selects three causal lexical-prefix means (coordinates 0, 2,080 and 960), two current lexical coordinates and three position splits. Its input-dependent routes can now differ at unchanged tokens.

| Unchanged-token response | Cube A/B | Vessel 00/10 | Vessel 01/11 |
|---|---:|---:|---:|
| Native H1 rows with nonzero response | 503 | 502 | 502 |
| Candidate H1 rows with nonzero response | 479 | 1 | 1 |
| Candidate pooled H1 response error | 86.92% | 99.31% | 99.30% |
| Old endpoint writer response error | 100% | 100% | 100% |
| Candidate unchanged-active H1 response error | 23.42% | 81.38% | 81.91% |
| Candidate unchanged-padding H1 response error | 88.95% | 100% | 100% |

Errors in this table divide by the corresponding native signed-response RMS. Unchanged-active excludes the two copied prefix rows; unchanged-token includes them. Some pre-edit native responses are zero, so pooled percentages are not individual-row error claims.

The same nonzero-row counts hold at Q/K/V. The cube active improvement is concentrated: only positions 9 and 34 respond, with one H1 row improving and one worsening; thirty other unchanged learned active rows keep the old zero prediction. For the cube unchanged-active panel, Q error is 51.37%, K 54.38% and V 20.70%, versus 100% for the old writer. Cube padding remains much less accurate. For vessel transfer 01/11, only unchanged position 9 responds; unchanged-active Q/K errors are 105.10%/104.62%, despite H1 improving to 81.91%; V is 89.70%. All vessel padding responses remain exactly zero. A nonzero state response is therefore insufficient evidence that the next consumer uses it correctly.

Four atoms serve discovery B alone, covering long position intervals. The result can include specimen-specific, factorized memorization; no fresh heldout semantic generalization is established. The vessel swaps preserve their token bag after both color IDs have occurred. The selected prefix-mean/current-token/position predicates then lose that assignment distinction, even though the full packet and available first/last occurrence features retain it. This diagnoses the selected routes, not an impossibility for all 155 features or the writer grammar.

The sole responding vessel row 9 also precedes the side edit. Its causal token prefix is identical in 00 versus 01 and in 10 versus 11, and its response scalars repeat exactly. Thus this response is a repeated causal context, not evidence of a new contextual transfer ability. Later padding would test a changed context but still predicts zero response.

## What the response objective traded away

Transfer 01/11 are previously public histories withheld from this fit and selection. The same observatory raw prompt executes without a native target. Endpoint errors below divide by each case's native H0→H1 increment RMS; the all512 panel includes two copied rows, while learned510 excludes them.

| Transfer H1 endpoint | Candidate 01 | Old endpoint 01 | Candidate 11 | Old endpoint 11 |
|---|---:|---:|---:|---:|
| All512 | 42.26% | 37.88% | 42.34% | 37.96% |
| Learned510 | 42.51% | 38.11% | 42.59% | 38.18% |
| Active | 68.29% | 50.45% | 68.37% | 50.36% |
| Padding | 36.98% | 35.69% | 37.09% | 35.81% |

Candidate H1 improves 73 rows of 01 but worsens 437, with two prefix ties; 11 has 70 improvements, 440 regressions and two ties. All pooled active/padding Q/K/V endpoint errors also worsen against the old endpoint program on both transfer histories. Candidate learned510 H1 error also exceeds the position control's 41.59%/41.64%; that control still has zero contextual response. The [raw report](results/contrast-writer/report.json) and [observations](results/contrast-writer/observations.jsonl.gz) preserve every case/port/stratum and row.

The continuation policy nevertheless selects candidate: its discovery score is 0.8861584684 versus 1.1338944839 for endpoint. This is the average across Q/K/V of separately normalized endpoint-increment and signed-response squared errors, balanced across learned active/padding and histories/pairs. It rewards contextual response enough to outweigh worse endpoints. It is neither an RMS percentage nor a terminal quality verdict. The old program and every rejected state remain evidence, and transfer is never used for selection.

The tradeoff has a simple interpretation. If e=AD−R is the relaxed state error, the paired term penalizes Pe: how the error changes between contexts. A common bias lies in the pair operator's nullspace. Therefore response can improve while endpoint bias increases. We need both a usable state and the right changes in it; optimizing their sum does not guarantee both improve.

## State response and consumer response are different contracts

For any actual consumer C, define endpoint errors eA=C(HhatA)−C(HA) and eB=C(HhatB)−C(HB). The exact response discrepancy is

\[
\big(C(\widehat H_B)-C(\widehat H_A)\big)-\big(C(H_B)-C(H_A)\big)=e_B-e_A.
\]

This identity remains true for the finite implementation. A large shared error may cancel between contexts; conversely a small state-space gain may rotate badly through the consumer. For intuition only, ignoring BF16 admission, input RMS normalization followed by a projection has derivative

\[
J_C(h)=W\,\mathrm{diag}(g)\left(\frac{I}{r}-\frac{hh^T}{d r^3}\right),\qquad r=\sqrt{\|h\|^2/d+\epsilon}.
\]

The candidate responds around its own biased base state. Vessel position 9 makes the distinction concrete: its H1 response error is only about 4.74% with cosine 0.9992, yet Q/K response errors are about 113.71%/111.32% on that same row. Both response direction and normalization/projection geometry can affect the result. This smooth derivative is not an exact Jacobian of quantized execution or a causal attribution of the observed vessel regression; map sensitivity, endpoint bias and finite arithmetic remain possible contributors. The actual paid Q/K/V measurements are the authority. Each port also has its own denominator, so their percentages cannot be read as attenuation of H1 error.

## Actual compute and evidence

One guarded Beast job, job-4a3243572a50, succeeds in 66.57 seconds, with sampled peaks of 2,481.8 MB RSS and 412.0 MB device VRAM. Torch peak allocated memory is 93,328,384 bytes, a different measure. Coupled tree search/constant fitting takes 18.47 seconds, excluding the worker's input-feature/target preparation; selected native-parameter authentication/load takes 2.95 seconds. The old lexical assets and exact old consumer outputs are reused; no full table is recompiled or copied into the development checkout. This is a different workload from cycle103, so its shorter time is not a matched speedup result.

Fresh initializer/discovery/final child wall times are 0.666/7.782/19.760 seconds. Final serving evaluates four methods over seven raw prompts. Outputs named `candidate` take 2.09–3.56 ms per prompt for feature formation and 0.597–0.863 seconds for the writer plus serialization/hashing, with 2,611,200 coordinate additions and 4,060–4,075 branch decisions. The separately timed `accepted` alias produces the same bytes but has different timings. Asset loading, tokenizer/H0 gathering and evidence costs are separate; child high-water RSS is not isolated steady serving memory.

New native work is six serial B1 inputnorm/QKV units: eighteen original linears, 48,318,382,080 nominal MACs and six normalizations. Twenty-four historical native/H0/endpoint/position consumer units are reused only after matching actual fresh input payloads, program and numerical profile. Their historical computation remains paid. No attention, MLP, full encoder, denoiser, VAE or image runs.

All eight supplied tiny mechanics checks pass once in 0.07885 seconds including imports. Declared/sealed/executed payload SHA agrees at a4d82bf59e9520f70c5f14a80155023a0da2f80e7de6322fdb6a3050fd83121f, 104,923 bytes. Strict preflight checks command/root/reservation; undeclared verifier/geometry hooks are skipped. Final own predictions seal before transfer numeric-field release. Canonical Saturn Frames/Acts bind actual H1 inputs, four separate native parameter tensors and actual Q/K/V payloads; immutable program restoration is byte-checked, not optimizer rollback.

The frozen publisher authenticates 27 compact members totaling 7,237,956 bytes, all 73,728 unique observation rows, 3,744 rebuilt pooled scalars and 190 fresh full-SHA remote files totaling 1,191,117,081 bytes. It checks native BF16 output headers/payload joins without local tensor hydration. Feedback error/reference scores are rebuilt; other feedback increment cosine/RMS fields are retained without an independent vector replay. [Mechanics](results/contrast-writer-evidence/mechanics.json), [preflight](results/contrast-writer-evidence/preflight.json), [publication](results/contrast-writer/publication.json) and the source-bound code preserve these scopes.

The next full research/storage review is cycle105, before another primary experiment. It should assess this tradeoff and order sensitivity against the broader evidence rather than automatically widening this tree. The [living log](../../obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md), [own-state MLP work](../../obsidian/blog/2026-11-06-111533-improving-an-mlp-on-its-own-predicted-states.md) and [ISA state-machine work](../../obsidian/blog/2026-11-06-004500-what-the-c-state-machine-learned-about-carry-and-context.md) connect the local result to the original goal: construct the distinctions from raw input, preserve them through actual consumers and continue to faithful pixels without neural serving evaluations.
