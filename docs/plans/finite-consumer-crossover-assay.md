# Finite consumer crossover implementation

Implementation contract for the separately reviewed [crossover proposal](finite-consumer-crossover.md). This source and contract are unexecuted at writing and freeze before the first mechanics execution. The arbitrary raw-prompt zero-neural image goal remains unsolved. The experiment uses supplied historical states and original learned projections for offline diagnosis; it is not serving.

## Specimen and computation

Use discovery vessel00→10, unchanged active row9 only. Native BF16 H1 endpoints a,b and accepted104 own endpoints p,q come from the source-qualified104 publication. Reuse exactly F(a),F(b),F(p),F(q), bound to full input payloads, original four parameter tensors, epsilon1e-6, Torch version, CUDA numerical profile and full[1,512,2560] geometry. The01→11 causal prefix repeats and is not an independent replication. No new prompt, teacher, fit, feature, atom or domain is added.

The CPU construction promotes complete BF16 states to FP64 before subtraction. It forms deltaN=b9−a9, deltaO=q9−p9, sumNO=a9+deltaO and sumON=p9+deltaN in that order, then admits each sum by FP32 round-to-nearest-even followed by BF16 round-to-nearest-even. Xno copies complete a and substitutes only row9; Xon copies complete p and substitutes only row9. Retain both unrounded sums, both admitted-minus-unrounded remainders, and A(a9+deltaN)−b9/A(p9+deltaO)−q9 as diagonal reconstruction diagnostics. Stored diagonal outputs remain authoritative.

One canonical CPU Frame/Act binds four actual full BF16 inputs, two full BF16 outputs and eight FP64 row outputs. The unchanged `contrast_writer_worker.Consumer` constructor/run path then performs exactly two qualified original CUDA RMS/QKV units. Its inherited method label “candidate” is retained as implementation metadata; each new consumer record additionally carries its explicit Xno/Xon branch and baseline/response semantics. The existing six-unit-only save method is not called. Parameter tensors are checked before and after; the selected shard is authenticated once on the worker before loading its four tensors, with a separate fresh publisher byte audit afterward.

The two new units pay six original learned linears and16,106,127,360 nominal matrix MACs. Four stored diagonal units are reused, not reevaluated. No attention, MLP, later encoder, denoiser, VAE or image runs. Historical native/own state preparation, lexical compilation, fitting and model costs remain paid. This does not claim zero neural evaluation: it explicitly evaluates six original learned maps for research.

## Signed finite analysis

Four canonical CPU analysis Acts cover H1,Q,K,V. Each binds the six actual full BF16 fields a,b,p,q,Xno,Xon at its port, promotes to FP64, then reads row9. Retain eleven FP64 vectors per port: N=Fb−Fa, O=Fq−Fp, Cno=FXno−Fa, Con=FXon−Fp, E=O−N, Uno=O−Cno,Vno=Cno−N,gno=E−(Uno+Vno),Uon=O−Con,Von=Con−N,gon=E−(Uon+Von). At H1 the consumer is identity and the same notation refers to actual state endpoints.

Norms and inner products multiply contiguous FP64 coordinates then use np.sum(dtype=float64). Keep all eleven squared norms; the N/O,N/Cno,N/Con and each U/V dot products; and each dot(U+V,g). Rebuild RMS, native-response-relative errors, response cosines, signed path cosines and five-term Gram sums from these scalars. Outer Gram sums use math.fsum. Retain norm(E)²−GramSum as an evaluated arithmetic diagnostic. Zero native-response energy yields undefined ratios and preserves absolute errors. No arbitrary scalar threshold determines scientific success.

The two paths include admission effects. They do not identify pure causal baseline/direction percentages or a consumer response to an unrounded state outside the BF16 ABI. Signed cross terms may cancel. Keep each port and both paths visible, including adverse or mixed results.

## Controls and evidence

For each new full consumer output, compare all512 rows to its authenticated stored baseline F(a) or F(p). There are3072 total rows: six intervention rows and3066 unmodified-row controls. Preserve all raw metrics plus18 pooled selections, and count any changed unmodified rows per branch/port. Unexpected collateral requires instrument/profile diagnosis before treating the result as a clean row9 intervention; it does not erase completed evidence or imply a scientific null.

Retain sixty new numerical fields: two crossed states, eight construction row fields, six original consumer outputs and44 signed analysis vectors. Full arrays stay near compute. Canonical Frame fingerprints, input/output ABIs and immutable before/after hashes bind actual payloads, not file descriptions alone. The post-analysis seal is not a prospective-prediction or target-quarantine claim.

The publisher authenticates submitted/collected/live guarded custody and request identity; parent104/source lineage; complete fixed branch/array/typed/scalar domains; actual original parameter and affine geometry; four analysis ports; all3072 raw comparison rows and108 pooled scalars; and188 signed summary scalars. It freshly hashes retained remote files and NPY payloads and checks the two complete crossed states outside row9 by four byte-segment comparisons. Norms, inner products and finite constructions remain source-qualified producer measurements rather than independent vector replay. No scientific tensors hydrate locally.

## Mechanics and resources

Complete source/math/resource/publisher review precedes eight tiny supplied mechanics checks: row-only construction, independent struct-based rounding/ties, FP64 promotion before subtraction, visible diagonal re-admission discrepancy, signed cancellation/Gram identity, zero response energy, scalar tampering and invalid geometry/nonfinite values. No scientific corpus or model runs locally. Executed source/test/plan bytes remain immutable; failures require distinct source versions.

Use one guarded host-local Beast CUDA lease because inputs and selected weights are retained there. Request CUDA, strict preflight, intent-first memory/reservations, one numerical CPU thread, no retries, no child leases and no automatic sweep or expansion. Seven logical Acts comprise one construction, two original consumers and four analyses. Expected runtime is below five minutes, with a900-second cap; that estimate is not a benchmark. Inspect live admission, logs and memory. A transient observation timeout is not permission to resubmit a live job.

The result concerns one causal row and immediate consumer. It cannot repair the fixed-nine padding capacity gap, establish a useful complete conditioner, prove future closure or generate arbitrary-prompt pixels. Full research/storage105 remain current; next scheduled full review108 or earlier material evidence/pressure.
