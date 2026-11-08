# Full research review111: preserve the direction through the executable writer

The next useful question is whether our proposed update survives conversion into the program that actually writes the state. The [joint writer110](../../joint-state-writer.md) improves aggregate raw-state endpoint loss in all four futures while worsening raw-state response and both normalized-state terms. Its parent remains the accepted program. This review identifies an exact way that a sound surrogate solve can lose its descent direction, and a conditional construction that avoids that particular loss. It does not yet attribute the observed110 result to that mechanism. Arbitrary-prompt images with zero neural serving evaluations remain unsolved.

The [Astra survey](advisor-survey.json) repeats the twelve-repository inventory, including dirty and untracked work, filesystem modification times, HEAD/status and commits since108. All eleven external HEADs and status path/flag inventories match the previous survey. The prior127-path union has125 unchanged paths, two changed paths and none missing; the changes are the living log and Saturn requests. Time Circuit Explorer adds the108–110 commits. The35 content snapshots comprise20 complete and15 scoped reads. Hash-only comparisons remain distinct. Root independently [rechecks the149-path union](root-pin-verification.json):148 remain unchanged and the later111 living-log synthesis is the sole changed member. The survey's historical110 log pin remains intact. No newly changed external scientific implementation supplies a general contextual initializer or pixel program.

## An accurate solve is only the first representation of the update

Use the declared local surrogate

\[
q(z)=g^Tz+\tfrac12 z^TAz,\qquad A\succ0.
\]

Its unconstrained step is Δ=−A⁻¹g, so

\[
g^T\Delta=-\Delta^TA\Delta\le0.
\]

This sign concerns the chosen quadratic model. In110, A uses a scalar RMS-gamma/radius approximation to the normalization sensitivity, while g pulls finite residuals through an ideal Jacobian. Neither is the exact Hessian or gradient of the discrete finite objective. The stored g also omits a common objective factor: g·z/2 is the anchored ideal local-model slope of the four-term mean loss.

The executable writer has another restriction. Histories that share a leaf must share a write vector. On a fixed route let B map leaf coefficients to row writes; for full vectors, the corresponding map repeats that incidence structure over coordinates. Let M be the raw-H endpoint/pair metric used for the dictionary fit. In exact arithmetic, projection into a zero-containing linear perturbation space gives

\[
P_M\Delta=B(B^TMB)^{-1}B^TM\Delta.
\]

There is no general implication that gᵀP_MΔ is nonpositive when M differs from A. The [supplied exact witness](projection-geometry.json) makes this concrete:

\[
g=\begin{bmatrix}2\\-1\end{bmatrix},\quad
A=\begin{bmatrix}100&0\\0&1\end{bmatrix},\quad
M=I,\quad B=\begin{bmatrix}1\\1\end{bmatrix}.
\]

The two rows may write only one shared constant. Then

\[
\Delta=(-1/50,1),\quad P_M\Delta=(49/100,49/100),
\]

\[
g^T\Delta=-26/25,\qquad g^TP_M\Delta=+49/100.
\]

The proposal descends and its projected version ascends. Even A−M being positive semidefinite does not rescue that implication. This is an exact two-row counterexample supplied for mechanics; it is not a measured model direction or a proof that110 lost its sign here.

## What one compatible assumption would buy

Suppose the parent is preserved, the route is fixed, and allowed perturbations form a linear space containing zero. Fit in the same declared SPD metric A. The normal equation for z=Bδ is

\[
B^TAB\,\delta=-B^Tg.
\]

Multiplying by δᵀ gives

\[
g^TB\delta=-\delta^TB^TAB\delta\le0.
\]

For the witness, P_AΔ=(−1/101,−1/101) and the product is−1/101. A consistent pseudoinverse gives the same induced update when a redundant parameterization makes coefficients nonunique.

The useful assumption is compatibility between the proposal geometry and its allowed writes. It converts the writer fit into the restricted quadratic solve itself. It supplies a local-model sign guarantee before FP32 atom admission and BF16 state admission. It supplies no finite-loss guarantee, global capacity theorem or general prompt semantics. The parent-preserving requirement also has a cost: keeping the current route can provide it within the current slot budget; adding a separate parent-plus-correction program would have to respect that budget explicitly. A new route need not represent the parent.

## A changed route can move the state even at zero dose

The actual110 fit changes a route, then projects the entire old slot w plus αΔ. Before finite admissions, its change is

\[
u_\alpha=P_M(w+\alpha\Delta)-w
=\underbrace{P_Mw-w}_{u_0}+\alpha P_M\Delta.
\]

Thus decreasing α need not approach the original program. In the same supplied two-row space, take w=(0,1). Its route-change offset is u₀=(1/2,−1/2), with gᵀu₀=3/2. This exposes a second issue distinct from metric mismatch: the candidate family is affine around a different zero-dose state.

After finite state admission, even that affine identity need not survive. The [independent scalar witness](projection-geometry.json) uses two FP32→BF16 admissions, as the writer does. Start at256 with parent atom1 and other-slot atom0. The parent becomes256. Candidate atoms1,1.5,2 produce states256,258,258 at doses0,1/2,1. Consequently v₀=0, v_half=2 and v_full=2, while2v_half−v_full=2. Two admitted doses cannot recover the true zero-dose state. This fixture does not measure the magnitude of finite effects in110.

## The next empirical question stays small and constructive

Retain the [saved Δ→u→v audit](../../plans/writer-proposal-direction.md). For each existing future, compare the saved proposed αΔ, the actual FP32 dictionary/route write change u, and the actual admitted state change v. Preserve g·αΔ, g·u, g·v, the mapping term g·(u−αΔ), the admission term g·(v−u), evaluated regrouping residuals and the already measured finite loss change. Keep history and active/padding contributions visible.

If descent is lost in program mapping, the next construction should preserve the parent and fit allowed perturbations in the declared full surrogate. If the admitted local-model product stays negative while finite loss worsens, retain that countertrend and examine proposal/finite-objective agreement before changing feature or dictionary capacity. The matched-metric identity makes the mapping-failure branch actionable; it does not choose that branch before measurement. No additional fit, route, dose, original normalization, QKV, prompt or image runs in111. Exact source, field joins and tiny mechanics for the future CPU mrun assay remain unimplemented.

## The broader state-machine gap remains

The [MLP own-state work](../../../../obsidian/blog/2026-11-06-111533-improving-an-mlp-on-its-own-predicted-states.md) shows that a local gain can reverse through its finite receiver and later pixels. The [read-curvature work](../../../../obsidian/blog/2026-11-07-021005-the-read-bends-before-the-reader-runs.md) forecasts a bounded scalar with a native baseline and compiled original computations still paid. The [ISA executable symbols](../../../../isa/docs/executable-symbol-scan-lowering.md) expose reusable transitions on a finite organism, with its initializer assumed. The [state-quotient counterexample](../../../../cmachine/research/state_quotient/FINDINGS.md) shows how an off-space write can invalidate a proposed compact closure; it is not a general impossibility result for approximate or nonlinear states.

Together with the [time-circuit account](../../../../obsidian/blog/2026-11-05-153154-state-context-and-the-mathematics-of-time-circuits.md), these results sharpen the central requirement: produce a state from the actual input, advance that state through its own allowed writes, and preserve what its future consumes. We can loosen equality of internal coordinates when consumer equivalence is established. We should tighten agreement between a proposed update, the executable writer and the actual finite consumer. None of the reviewed pieces supplies the complete raw-prompt→contextual-state→autonomous-diffusion→general-pixel program.

The separate [local reconstructed generator](../../../generator/README.md) does accept new free-form prompts, while paying original learned denoiser equations and native encoder/VAE calls. The hosted site shows saved runs; the zero-neural chart varies calibrated scene contexts and doses. Those interfaces remain distinct from the active arbitrary-prompt zero-neural research goal.

## Work, timing and storage

The statically reviewed [stdlib helper](projection_geometry.py) executes once, passing three supplied exact/scalar cases in0.00023725seconds after imports and before serialization. That small interval includes source hashing, excludes process startup and is not an image-generation speed. Source17271b62 is frozen afterward. No scientific arrays, original learned operations, new model, fit, prompt or image run. Historical model preparation and the110 assay remain paid.

The [scheduled storage audit111](../storage-audit-cycle111/README.md) verifies93 pointers within their custody classes:84 older original-custody/type-size checks and nine fresh full-SHA snapshots totaling3,806,240,603bytes. It finds zero archive/log transports at or above100,000,000bytes in the two audited roots. No archive/delete occurs; last actual archive remains78. Shared free space is observed at26,004,680,704bytes, without exclusive cleanup attribution. Scientific arrays remain near compute. Next full research and storage review is114, or earlier route-changing evidence or storage pressure.
