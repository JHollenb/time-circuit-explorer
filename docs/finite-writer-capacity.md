# The finite capacity of nine contextual writes

The nine existing write atoms leave most of the vessel prompt response unexplained even when a target chooses their routes. Removing the endpoint penalty barely changes that result. A better router can recover some error within this family, but cannot supply the missing H1 response on these specimens. This narrows the next representation question; arbitrary-prompt image generation with zero neural evaluations remains open.

The [contrast-aware writer](contrast-aware-contextual-writer.md) forms approximate first-layer contextual states from owned token history. Its second write uses nine constant vectors, selected by one shared decision tree. Both vessel contrasts previously produced zero response on all 475 unchanged padding rows. The [full research review](results/research-review-cycle105/README.md) therefore selected a [capacity diagnostic](plans/finite-writer-capacity.md): keep the available finite outputs and remove only the shared routing constraint.

## The finite output family

For each learned row, let S_i be the actual rounded first-write state and d_a one of the nine frozen FP32 second-write atoms. Its available outputs are

\[
C_i(a)=Q(S_i+d_a),\qquad a\in\{0,\ldots,8\},
\]

where Q means the executed binary64 addition followed by FP32 and BF16 rounding to nearest even. We reconstruct S and the existing output from the authenticated owned H0 and token packet. Both old route labels and the complete current H1 bytes match before enumeration.

The diagnostic may choose a different atom for every row using the native target Y. The first write, atoms, rounding, histories and two copied prefix rows stay fixed. Discovery A/B/00/10 and reused transfer 01/11 are scored separately. These six histories are already public; the new choices are target-assisted diagnostics.

For a paired unchanged-token row, the joint component is

\[
\ell_{ij}(a,b)=\alpha_i\|C_i(a)-Y_i\|^2+
\alpha_j\|C_j(b)-Y_j\|^2+
\beta_{ij}\|(C_j(b)-C_i(a))-(Y_j-Y_i)\|^2.
\]

Endpoint and response weights each balance their active and padding strata, with equal history or pair weight. Discovery keeps the previous fit's reference energies exactly; transfer uses its own analogous energies. A zero reference energy retains absolute squared error.

Each row appears in at most one pair, so the objective separates:

\[
L^*_{9}=\sum_{(i,j)}\min_{a,b}\ell_{ij}(a,b)
+\sum_{i\text{ unpaired}}\min_a\alpha_i\|C_i(a)-Y_i\|^2.
\]

All 81 pair choices and nine singleton choices are evaluated. Norms use explicit FP64 products and contiguous coordinate sums; component terms have a fixed multiplication/addition order, exact ties choose the smallest address, and outer scores use math.fsum. This is the minimum of those evaluated component scores. Historical global einsum reductions are retained separately, with their small reassociation differences.

## What target assistance recovers

| Population | Current endpoint term | Minimum joint assignment endpoint term | Current response term | Minimum joint assignment response term | Current joint score | Minimum joint score |
|---|---:|---:|---:|---:|---:|---:|
| Discovery, 2,040 learned rows | 0.366983 | 0.347324 | 0.607894 | 0.524144 | 0.974877 | 0.871468 |
| Reused transfer, 1,020 learned rows | 0.379211 | 0.355824 | 0.871424 | 0.807626 | 1.250636 | 1.163450 |

These are separately normalized squared-error terms, not percentages of rows or ordinary pooled RMS. The assignment changes 1,200 discovery labels and 840 transfer labels. None of the 1,535 components has an exact tie for its minimum.

The joint improvement is heterogeneous. Across all 3,060 learned endpoint rows, 1,714 improve, 326 worsen and 1,020 tie. Cube unchanged-learned response improves on 323 rows, worsens on three and ties on 183. The two vessel pairs improve on 142 and 145 unchanged-learned rows, with the other 366 and 363 tied. All row outcomes remain in the [observations](results/finite-writer-capacity/observations.jsonl.gz) and [derived scalar summary](results/finite-writer-capacity-evidence/public-scalar-summary.json).

Nonzero response counts alone would misdescribe the improvement. The cube's responding unchanged-learned rows fall from 479 to 306 while response error improves: removing a wrong response can help. Vessel padding gains responses on 141 and 144 of 475 rows, but most discrepancy remains.

## Remove the endpoint tradeoff

The saved norm tables also allow a separate response-only minimum:

\[
R^*_{9}=\sum_{(i,j)}\beta_{ij}\min_{a,b}
\|(C_j(b)-C_i(a))-(Y_j-Y_i)\|^2.
\]

This discards endpoint accuracy entirely. It asks whether any selection from these finite outputs can explain the measured response, including selections the joint objective would reject. It supplies no new serving program. The following padding RMS ratios reduce the published coordinate norms and native row-normalizer scalars within each pair.

| Pair and population | Current response error | Joint minimum assignment | Response-only minimum |
|---|---:|---:|---:|
| Cube A/B, 477 unchanged padding rows | 88.9469% | 79.2484% | 79.2477% |
| Vessel 00/10, 475 unchanged padding rows | 100% | 96.3972% | 96.3855% |
| Vessel 01/11, reused transfer, 475 rows | 100% | 96.5791% | 96.5761% |

The response-only minimum is almost the joint result. The large vessel discrepancy therefore cannot mainly be attributed to paying for endpoint accuracy in this frozen family. The available differences between finite outputs carry too little of this measured H1 response, even with target-assisted choices.

This is a bounded family result. Other dictionaries, additional state, another first write, a different symbolic program or a code designed for the future consumer remain open. The nine-output family does not establish a limit on model computation generally. Euclidean H1 response is also distinct from usefulness to the next normalization, Q/K/V maps, later conditioning and pixels.

## Computation and evidence

One guarded CPU mrun job, `job-459c49bb9118`, completes in 12.05 seconds on Beast. The worker timer after imports/request verification is 4.874 seconds; the two enumeration Acts take 0.787 and 0.809 seconds, including scalar emission and typed handling. Worker process CPU time is 2.894 seconds. Sampled job peak RSS is 1,309.1 MB; the process high-water report is separately 1,403,564 KiB. Numerical threads are one; the scheduler reserves four CPU threads and 4,096 MB through intent-first planning. GPU allocation is zero.

The assay evaluates 123,615 choices across 1,525 disjoint pairs and ten singleton rows. It forms 27,540 finite option states, paying 70,502,400 coordinate additions, 70,502,400 endpoint norm products and 316,224,000 response norm products. Reconstruction, reference-energy/ordinary-metric reductions, hashing and file work are additional. These operation counts are not FLOPs, learned MACs or a serving latency measurement. All historical neural preparation remains paid; no new learned operator or image runs.

Canonical Saturn Frames and Acts bind actual CPU tensor payloads. Torch CPU supplies views and custody; NumPy performs the diagnostic arithmetic. Eight tiny mechanics checks pass once. Strict preflight passes command/root/reservation checks; its undeclared verifier and geometry hooks are explicitly skipped. Declared, sealed and executed payloads match at SHA `a5ca38a33a152e3c63fdcb5d4dd0cc83286a0b0e4b351e62944d330c69c788cf`.

The [publisher](results/finite-writer-capacity/publication.json) checks every component scalar expression and canonical choice, all 9,216 observation rows and 468 pooled scalars, 30 tensor field ABIs, four label vectors and 18 byte-segment joins. It freshly hashes 52 remote files totaling 141,803,992 bytes. Coordinate norms remain producer measurements; scalar consistency and byte custody do not independently replay vectors. The standard [Saturn evidence sweep](results/finite-writer-capacity-evidence/saturn-instrument.json) reports an empty unsupported target for these gzip streams and gate-free typed records; it supplies no generic watch or claim coverage. The dedicated publisher checks remain separate. The 94,359,032 bytes of new full tensor payloads remain on Beast. Primary compact publication is 5,588,071 bytes, plus evidence and narrative files. No archive or deletion occurs; the full [storage review105](results/storage-audit-cycle105/README.md) remains current, with the next scheduled review108.

## The next representation question

Another routing fit with these nine atoms is a weak next step for the vessel padding response. Before changing the output family, the separate [consumer crossover proposal](plans/finite-consumer-crossover.md) asks why the earlier row9 H1 difference was nearly correct while its Q/K response was poor. It swaps the native and own response directions between their two baselines, retaining both finite admission remainders. This proposed local diagnostic is unimplemented and unrun; it cannot by itself repair padding or provide an image generator.

The larger condition remains the one in the [time-circuit account](../../obsidian/blog/2026-11-05-153154-state-context-and-the-mathematics-of-time-circuits.md): a useful code must advance from its own current state and preserve the distinctions its future consumes. The [MLP continuation work](../../cmachine/research/transition_laws/FINDINGS.md) and [state quotient counterexample](../../cmachine/research/state_quotient/FINDINGS.md) show why current-state fit and norm information alone are insufficient. This assay adds a complementary constraint: a router cannot recover a response absent from its finite output choices. A reusable state vocabulary and an economical resolver are both necessary; neither has yet closed raw prompt to pixels.

Sources: [complete report](results/finite-writer-capacity/report.json), [finite component scores](results/finite-writer-capacity/components.jsonl.gz), [source review](results/finite-writer-capacity-evidence/advisor-source.json), [actual result review](results/finite-writer-capacity-evidence/advisor-actual.json), [implementation contract](plans/finite-writer-capacity-assay.md), [living research log](../../obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md).
