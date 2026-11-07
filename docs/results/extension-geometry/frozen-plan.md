# Separate determined response from unconstrained extension

Status: **written and unexecuted**. Selected after the [minimum-norm result](../minimum-norm-source-response.md) and [Astra44 review](../results/minimum-response-evidence/advisor.json). One saved-array audit of the fixed map; no new fit, predictor selection, original model or image.

The all-mode map interpolates discovery exactly but produces large, poorly aligned held responses. This audit will determine which supplied-input geometry is fixed by source interpolation and where the source leaves extension freedom. It cannot causally identify missing receiver conditioning or certify a generator.

## The identifiability question

Use the retained X_source=UΣVᵀ, C=UᵀY_source and B_0=VΣ⁻¹C. Let P=VVᵀ. For any G of compatible shape, every exact source linear interpolant can be written

\[
B=B_0+(I-P)G.
\]

For one held normalized input matrix X_*, put

\[
Z=X_*V,\quad A=Z\Sigma^{-1},\quad
X_\parallel=ZV^\top,\quad r=X_*-X_\parallel.
\]

Its prediction under an arbitrary interpolant is

\[
X_*B=AC+rG.
\]

The current B_0 fixes G=0. Its amplification comes from the in-span A coordinates and their multiplication by C. Nonzero r describes an unconstrained extension of the source fit; it is not directly amplified by B_0. With rank1,888 in3,072features, the source leaves1,184 feature directions free in this linear family. Numerical rank and exact interpolation do not determine what a real model should do there.

## Exactly one audit over six archived held pairs

Freeze all six archived X_*, masks, Vᵀ, sigma, C and the cycle43 ridge filters. Use all modes. Compute Z,A,X_parallel,r and Yhat=AC once per pair. Do not construct T=A Uᵀ or load U: the identity X_*B_0=T Y_source explains the source-weighting interpretation without paying for another dense contraction.

Retain every query's input, projected and residual squared norms, signed finite cross term and energy-closure discrepancy, plus leverage ||A_row||². Source row leverage is1 in the exact full-row-rank algebra; label it as a reference identity rather than a new measured source assay. Bind selected-row order to the actual image-query index from the archived mask. Historical observation position fields are selected-row indices, not original image positions; preserve those old records unchanged.

For every mode retain ||A_column||², its existing continuous ridge filter, and diagonal output energy ||A_column||²||C_i||². One extra contraction D=AᵀYhat gives signed modal power

\[
s_i=C_i\cdot D_i,\qquad
\sum_i s_i=\|\widehat Y\|_F^2.
\]

Preserve negative powers and the aggregate cross power relative to the diagonal sum. Diagonal energies alone omit interference. Do not select directions, bins, thresholds, modes or a new predictor from these measurements.

Seal all six geometry/reconstructed fields and complete query/mode rows before opening the old minimum-norm predictions or error accounts. After sealing, verify reconstruction differences from the archived fixed program and join its already-measured errors/cosines/amplitudes. No new native response target is needed; scalar joins are retrospective diagnostics, not a new held test or consumer forecast.

Keep A, reconstructed predictions and signed-mode contractions near compute, with complete per-query/per-mode accounts and source-bound receipts. Transient projected/residual arrays may be summarized after their full finite arithmetic is checked; the immutable inputs/factors retain their reconstruction path. Public evidence stays compact. No model/table/full-array hydration locally.

## Execution and interpretation

Use one finite model-free CPU mrun analysis lease, default intent-first sizing, one configured arithmetic thread, strict preflight and no automatic retry. The four leading products cost, per pair,

\[
2(944\cdot3072\cdot1888)+2(944\cdot1888\cdot128),
\]

or **68,439,244,800 nominal MACs across six pairs**. Divisions, reductions, hashing, I/O and retention are additional. No new SVD, coefficient formation, original model/H0/VAE/image/noise computation or rank/gain/regularization sweep is allowed. Old factors, learned labels and native state production remain paid preparation.

Use a tiny independent example with known source row-space, nonzero orthogonal input and constructive alternative interpolants. Check the finite projection/cross/power accounts without an arbitrary scientific tolerance gate. Bind source identities, geometry, finite values, actor target/framework exclusion and full-byte custody. Keep all scientific magnitudes and heterogeneous rows.

Large in-span leverage can explain this map's extrapolation; nonzero r shows that exact source fitting leaves part of the linear extension uncalibrated. Neither proves that a receiver-conditioned feature is missing, that H0 is sufficient, or that another function cannot transfer. A new producer experiment should follow this distinction instead of another blind penalty or rank search. The arbitrary-prompt zero-neural goal remains open.
