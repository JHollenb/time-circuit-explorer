# A better raw state can still be a worse normalized state

The [joint writer assay](results/joint-writer/publication.json) constructs four new non-neural H1 programs and retains the original parent. Every candidate improves balanced raw-state endpoint loss, but worsens raw response loss and both normalized-state losses. The lowest candidate score is0.50964 versus the parent's0.48885, so the exact parent survives both slot decisions. This evaluates one approximate proposal-and-projection algorithm; it is not a capacity bound on the serving language. Arbitrary-prompt zero-neural images remain unsolved.

## What the construction changes

The [previous normalization audit](coupled-normalization.md) showed why a good raw change was insufficient: the response depends on the baseline state that normalization receives. The [execution contract](plans/joint-writer-assay.md) therefore balances finite H endpoints, paired H changes, finite Z endpoints and paired Z changes. Z is the original next-layer normalization/gamma operand, evaluated as paid offline feedback. The program serving H uses only its own raw-input lexical/history features, branches, constant-vector writes and finite additions.

Both slots may replace their routing and atoms, while the two-slot/depth4/maximum16-leaves-per-slot budget stays fixed. One sweep tries slot0 and then slot1, recomputing from whichever parent was actually selected. Each slot gets one new route with full and half proposal doses. All programs happen to retain14+9=23 occupied atoms, including the parent; no larger realized dictionary explains these results. The [earlier nine-atom bound](finite-writer-capacity.md) concerned the frozen second-write alphabet with the first write fixed. It does not bound these newly constructed vectors.

The four discovery histories contain2040 learned rows. History×active/padding endpoint strata have equal mass, as do pair×unchanged-active/padding response strata. Native energies use these same measures. With zero references handled as absolute squared losses, the score is

\[
L=\tfrac14\left(
\frac{E_H^{err}}{E_H^{native}}+
\frac{P_H^{err}}{P_H^{native}}+
\frac{E_Z^{err}}{E_Z^{native}}+
\frac{P_Z^{err}}{P_Z^{native}}\right).
\]

The [primitive energies and all four futures](results/joint-writer/futures.json) remain public. These are balanced squared-error terms, not percentages or the older increment-normalized endpoint scores.

| Program | H endpoint | H response | Z endpoint | Z response | Mean score |
|---|---:|---:|---:|---:|---:|
| Parent |0.363413|0.607894|0.401433|0.582676|0.488854|
| Slot0 full |0.348587|0.629228|1.221309|0.705557|0.726170|
| Slot0 half |0.335048|0.611041|0.554207|0.602181|0.525619|
| Slot1 full |0.354077|0.643148|0.839179|0.721933|0.639584|
| Slot1 half |0.345697|0.617583|0.475475|0.599807|0.509640|

Raw H endpoint improvement is real within this measure, but does not survive the complete objective. The best H endpoint candidate, slot0 half, lowers balanced native-relative RMS from60.28%to57.88%, while its Z endpoint RMS rises63.36%→74.45%. Every other candidate shows the same aggregate direction of tradeoff. This is not an experiment in which an arbitrary quality gate hid a winning candidate.

## A paired proposal still has to become an executable writer

The ideal normalization pullback is

\[
J(h)^Tu=\frac{\gamma\odot u}{s}-h\frac{\langle h,\gamma\odot u\rangle}{d s^3},\qquad
s^2=\|h\|^2/d+\epsilon.
\]

The proposal combines weighted H residuals with pulled-back Z residuals, retaining the pair errors in the same evaluated order as the objective. A positive two-row surrogate solves AΔ=−g. It couples paired rows rather than independently damping each row; an equal baseline update cancels in the H difference term. Its Z difference term still depends on the two rows' normalization scales. Both proposal records have solve-residual RMS1.156×10⁻¹⁸ and step RMS0.083619. Those small solve residuals verify the declared algebra, not a useful final program.

The next operation is approximate: fit a decision-tree route and write vectors to `oldSlot + doseΔ` in an H endpoint/pair metric. It does not minimize the complete paired H/Z surrogate. Changing routes also means that zero dose on the new route need not reproduce the parent. Half dose correctly refits `oldSlot + .5Δ`; it does not halve the complete dictionary.

The resulting program must rebuild H from its own H0 through both FP32/BF16 additions. Then the unchanged CUDA normalization/gamma consumer measures its actual Z. Positive-definite proposal algebra alone supplies no theorem that this route projection, finite admission and finite dose will lower L. The observed tradeoff places that mapping under scrutiny; it does not uniquely identify projection, curvature or finite effects as the cause.

## The retained state and transfer evidence

Both [selection records](results/joint-writer/selection.json) retain the exact parent Frame and copy its complete program bundle. Two fresh standard-library serving children reproduce the selected discovery H bytes before their Z fields are reused. The final child serves parent and accepted on all seven raw prompts, with NumPy/Torch blocked and no neural serving evaluations. Known transfer targets open only after its seal. Transfer is already historically observed, not newly held out; the seventh novel prompt produces an unscored H1 state.

Accepted therefore equals parent throughout. Cube unchanged-active H/Z response errors stay23.42%/23.39%, with padding88.95%/89.12%. Vessel discovery and transfer padding responses remain zero against nonzero native responses, giving100%relative response error. The full [report](results/joint-writer/report.json) and lossless observations retain endpoints, active/padding strata and transfer for parent and accepted. Rejected futures retain complete aggregate objectives and remote H/Z fields, without rejected-future per-row panels in that public stream. Aggregate regressions do not imply every row worsened. This assay adds no improved encoder or image.

## Compute and evidence scope

One guarded mrun job finishes in75.79seconds. Worker time after imports/request verification is71.83seconds and process CPU time49.93seconds. The two tree fits take21.49and18.40seconds; complete proposal Acts take23.12and19.15seconds. Fresh initial/final serving children take3.27and10.70seconds, including tokenization, asset reads, prediction and serialization. These scopes overlap the worker total and are not additive benchmark legs.

Actual work is30 original normalization/gamma units plus2 CPU proposal Acts, with0 original learned affine MACs and no QKV matrices, attention, full encoder, denoiser, VAE or image. Pointwise normalization, proposal/fitting arithmetic, serving and I/O remain paid. Gamma materialization is5120bytes; full authentication reads4.967GB of its shard in2.67seconds. Each program/case performs2,611,200 coordinate additions and at most4080 branch decisions, plus features and lexical reads.

The scheduler chooses4096MB RAM/4096MB VRAM/4 CPU slots; numerical threads are1. Sampled process peaks are2379.6MB RSS/296MB VRAM. The worker's own high-water RSS is2,264,456KiB and peak Torch CUDA allocation17,044,480bytes; sampling and allocation scopes differ. No matched runtime speedup is claimed.

Twelve supplied mechanics pass once in0.07613seconds including candidate imports. The [source review](results/joint-writer-evidence/advisor-source.json) precedes that execution; all43 reviewed pins are checked again before submission and publication. The [retained preflight](results/joint-writer-evidence/preflight.json) binds the actual command and payload, passing path/root/reservation checks; verifier-dry-run and geometry-predicate checks are explicitly skipped because none were configured. Actual geometry is instead bound by reviewed source, typed execution records and publication checks.

The publication verifies29 compact members/4.10MB,87 numerical fields/747.39MB retained on Beast,18,432 dependent raw rows,221 objective scalar/policy checks and936 rebuilt pooled metric fields. Fresh remote byte checks cover164files/6.613GB, including original assets, arrays, exact restores and serving outputs. Vector computation remains source-qualified producer measurement; scalar and byte checking is not an independent numerical replay. The standard [Saturn sweep](results/joint-writer-evidence/saturn-instrument.json) recognizes no compressed streams or gate-free Frame/Act claims and returns2 for both run/verify. That coverage gap remains distinct from the dedicated audit.

## The next question

After the scheduled full research/storage review111, Astra recommends a [saved-state directional audit](plans/writer-proposal-direction.md) before another fit. Follow signed local descent from the unconstrained Δ, through the actual FP32 slot-write change u, to the admitted state change v=Hcandidate−Hparent. The differences u−doseΔ and v−u separate program mapping from state admission under this declared decomposition. A negative unconstrained slope is not enough if the executable mapping loses it. A negative admitted local-model slope with worse actual loss still leaves curvature and finite/Jacobian mismatch unresolved.

This next audit is unimplemented and unrun. It requires no new fit, dose, original normalization or image. Full research/storage108 remains current, with111 next; large arrays stay near compute and no archive/delete occurs110. The complete raw-prompt conditioning program, autonomous denoising and general pixel decoder remain open.
