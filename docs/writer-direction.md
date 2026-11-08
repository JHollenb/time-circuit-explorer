# Local descent survives the writer, but the finite loss still rises

The [saved direction audit112](results/writer-direction/publication.json) resolves the next branch. All four completed110 futures retain negative aggregate local-model slopes through their actual FP32 writer and BF16 state updates. All four nevertheless increase the actual finite H/Z loss. Fitting removes most of the proposed descent magnitude, and some padding strata reverse sign, but an aggregate sign flip is not the explanation for these four losses. [Astra's actual-result review](results/writer-direction-evidence/advisor-actual.json) selects a construction that makes its finite step agree better with its residual-square model. Arbitrary-prompt images with zero neural serving evaluations remain unsolved.

## Follow the three actual representations

The [joint writer110](joint-state-writer.md) saved its FP64 gradient g and unconstrained step Δ, the actual program bundles and their complete H/Z states. The [review111](results/research-review-cycle111/README.md) supplied a counterexample showing that projection into a shared writer can reverse a descent direction. Cycle112 measures the actual directions rather than assuming that example explains our result.

For each recorded dose α, define

\[
d=\alpha\Delta,\quad
u=W_c(x)-W_p(x),\quad
v=H_c-H_p.
\]

W gathers the actual FP32 atoms under the actual trees on saved own lexical/history features. v subtracts the actual saved finite H states. Their differences isolate mapping m=u−d and additional state admission a=v−u. The account retains direct signed products and evaluated regrouping residuals; it does not infer a zero-dose program from two secants. Both original slot proposals actually share one parent, gradient, step and feature/measure payload, so common arrays load once while every consumed alias is authenticated.

The stored g omits a common objective factor. Reported values below are g·direction/2, the anchored ideal local-model slope of the four-term mean loss. They are not derivatives of the discrete BF16 objective or finite-dose forecasts.

| Future | Proposed slope | Actual writer slope | Admitted-state slope | Actual finite ΔL |
|---|---:|---:|---:|---:|
| Slot0 full |−1.346345|−0.189289|−0.189346|+0.237316|
| Slot0 half |−0.673173|−0.096872|−0.096916|+0.036765|
| Slot1 full |−1.346345|−0.095416|−0.095649|+0.150730|
| Slot1 half |−0.673173|−0.050776|−0.050903|+0.020786|

The admission contributions are−0.0000572,−0.0000445,−0.0002331 and−0.0001269 respectively. They make the aggregate local-model slope slightly more negative. This is not evidence that finite arithmetic is globally harmless: it describes this signed account on these four dependent futures. The finite objective still has authority over whether the actual program improves.

## Keep the countertrends in view

Vessel00 padding has positive admitted slope for every future, from+0.0001786 to+0.0070666. B padding also turns positive for both slot1 futures, +0.0032105/+0.0015076. The global negative products conceal these reversals. All8160 candidate-row observations and fifteen dependent groups per future remain public, using the original global weights/normalizers rather than refitting each stratum's scale. They are explanatory contributions, not independent specimens.

The four reconstructed gradient components agree closely with stored g: RMS discrepancy7.23×10⁻²¹ and maximum absolute3.47×10⁻¹⁸. Original ordered paired residuals are preserved. Small evaluated scalar regrouping residuals remain; vector closure maxima are1.11–2.22×10⁻¹⁶, rather than being replaced by symbolic zeros. The raw FP64 scalar rows pack back into the actual typed product-field payloads exactly. These checks strengthen the source-qualified account without independently replaying its vector dot products.

## A negative slope is not a finite loss guarantee

The [public-scalar supplement](results/writer-direction-evidence/public-scalar-finite-gap.json) subtracts each admitted local slope from the actual finite loss change. The resulting gaps are+0.426662,+0.133681,+0.246379 and+0.071689. These are inferred finite-minus-linear differences, not separately measured curvature terms or unique attributions to normalization, BF16 or Jacobian error.

A supplied one-dimensional example explains why. Let L(x)=(x−1)², start at0 and move by3. The local directional term is−6, but the actual loss change is+3; the quadratic remainder is+9. Even an exact linear consumer can overshoot because its squared loss has curvature. A direction can therefore be useful locally while its selected finite step is too large.

For the real explanatory geometry, let M_H and M_Z contain the frozen endpoint and pair measures, r_H/r_Z be parent residuals, v the admitted H change and ζ the actual finite Z change. In real arithmetic,

\[
\Delta L=\tfrac12\langle M_Hr_H,v\rangle
+\tfrac12\langle M_Zr_Z,\zeta\rangle
+\tfrac14\left(v^TM_Hv+\zeta^TM_Z\zeta\right).
\]

If J is the ideal normalization Jacobian used by g, subtracting the anchored local slope gives

\[
\Delta L-\tfrac12g^Tv
=\tfrac12\langle M_Zr_Z,\zeta-Jv\rangle
+\tfrac14\left(v^TM_Hv+\zeta^TM_Z\zeta\right).
\]

This identity separates a linearization mismatch from positive residual-square energy. The actual112 assay does not independently evaluate these two terms; FP64 regrouping also requires its own measured remainder in an implementation. The identity explains the alternatives that the aggregate gap leaves open. It does not turn an unexecuted decomposition into evidence.

## The old surrogate can violate the squared-loss floor

There is a useful source-qualified scalar inference. The110 proposal solves AΔ=−g+r, where A is its declared isotropic sensitivity surrogate. Its local mean-loss model at Δ is

\[
m(\Delta)=L_0+\tfrac12g^T\Delta+\tfrac14\Delta^TA\Delta
=L_0+\tfrac14g^T\Delta+\tfrac14\Delta^Tr.
\]

The retained products give approximately0.488854−0.673173=−0.184319. The recorded step/residual RMS values bound the last term near1.26×10⁻¹³ by Cauchy–Schwarz over5,222,400 coordinates. This is an inference from published scalar products and source-qualified solve diagnostics, not an independent A replay or exact-Hessian measurement.

A true residual-square Gauss–Newton model cannot become negative in exact arithmetic. Thus the old isotropic quadratic is not compatible with that nonnegative residual-square geometry. Its small solve residual certifies that its chosen system was solved accurately; it does not make that system a trustworthy finite-loss model.

## One concrete construction follows

[Astra's next decision](plans/parent-preserving-writer-step.md) retains both parent trees and their14+9 constants, forming one joint-atom Cauchy direction. Let B gather/add those leaf perturbations, D be a fixed positive leaf-mass diagonal, and

\[
r=-D^{-1}B^Tg,\quad q=Br,\quad
\beta=q^TM_Hq+(Jq)^TM_Z(Jq).
\]

Use the complete ideal Jacobian metric, so the one-dimensional residual-square model has α_model=−gᵀq/β. Cap that analytically chosen step by one frozen row-relative trust radius, then evaluate one actual finite candidate and retain or restore the parent. No large inverse, route refit, extra half-dose, feature expansion or diagnostic chain is selected. This is an unimplemented construction decision, not a new fit or result.

Frozen routes cannot separate histories with identical H0 and identical two-leaf assignments. The broad vessel-padding collision therefore remains. The purpose is finite agreement for one allowed state update before changing routes again. This connects [MLP own-state continuation](../../obsidian/blog/2026-11-06-111533-improving-an-mlp-on-its-own-predicted-states.md) to the [time-circuit architecture](../../obsidian/blog/2026-11-05-153154-state-context-and-the-mathematics-of-time-circuits.md): a proposed computation must become an actual state whose receiving computation improves. Neither local gradient fidelity nor a smaller dictionary establishes full contextual closure or pixels.

## Compute, verification and retained evidence

One model-free CPU mrun job completes in23.12seconds. Its canonical analysis Act takes8.574seconds; worker wall after imports/request verification is16.349seconds and process CPU11.172seconds. It pays647,577,600 direction/norm coordinate products plus5,222,400 gradient-discrepancy norm products, two ideal pullbacks/radius reconstructions,48,836 tree comparisons and byte/serialization work. Zero new original source-policy normalization calls does not mean zero normalization-related arithmetic. No tokenizer, lexical table, H0, checkpoint shard, transfer target, fit, dose, model or image runs.

Intent-first admission reserves4096MBRAM, zeroVRAM and four scheduler CPU slots; actual numerical threads are one. Scheduler peak RSS is1659.2MB; worker high-water RSS is1,679,492KiB. Historical native targets/states, original teacher work and the unchanged accepted parent remain paid.

Nine tiny mechanics pass once in0.07644seconds including candidate imports and test serialization, after exact Astra source review and root30-pin checks. The [publication](results/writer-direction/publication.json) checks25 compact members/5.906MB,70 fresh remote files/657.676MB,8160 raw rows,3120 scalar reaggregations and complete typed ancestry. Different scalar reduction orders remain diagnostics, with maximum absolute difference1.02×10⁻¹⁰. All five new numerical fields/3,427,840bytes stay on Beast. The source-reviewed stdlib scalar supplement runs once in0.00354seconds before serialization; neither timing is an image-generation speed.

Declared, sealed and executed payload d35f9bee match across166,818bytes. Strict preflight passes command/root/reservation checks; undeclared verifier/geometry hooks are explicitly skipped. The [standard Saturn run/verify](results/writer-direction-evidence/saturn-instrument.json) each return an unsupported empty target for compressed/gate-free records. That instrument limit stays separate from the dedicated audit; existing gzip and Frame/Act requests are upvoted once. Full research/storage111 remain current, last actual archive78, next periodic review114 or earlier material evidence/pressure. No archive/delete occurs here.
