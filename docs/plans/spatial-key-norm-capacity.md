# Fixed saved-only spatial reconstruction of the current key norm

Status: **written and unexecuted**. After the [fixed-sketch and paired-error results](../coherent-key-norm.md), test one different representation: preserve complete channel coherence at fixed physical image locations, then reconstruct only the scalar norm field. This is a retrospective capacity assay, not an implemented norm producer or new consumer run.

## Geometry must come from the actual native state

Authenticate the original Q StateCut and its top-level `img_ids` tensor, whose descriptor is retained in `sourceCuts.Q.cut.manifest.payload`. Existing extracted image/context/text pages do not by themselves provide that tensor. Resolve it through the actual native store, retaining full file/payload custody and matching the manifest descriptor. P and Q record the same native ID identity.

Verify the actual THWL axes and a one-to-one mapping of all 1,024 rows onto the 32×32 physical H/W grid. Preserve the original row order and full ID bytes. First-occurrence rotary-group labels are insufficient evidence of physical adjacency. If the actual ID tensor cannot be authenticated, report the geometry instrument as incomplete and resolve custody before interpolation; do not guess a row map.

Freeze knots at **{0,2,…,30}²**, independently for every head, with no data-dependent grid selection. At coordinate31 clamp to coordinate30. No extrapolation, periodic wrap, additional boundary knots, grid/density sweep or learned fit. This creates **256 sampled**, **705 unsampled-interior** and **63 outer-edge** cells per head: 6,144/16,920/1,512 of the complete 24,576 norm cells.

## Candidate access and the scalar law

A separate intake performs authenticated raw-byte extraction of exactly the 256 selected rows from the existing FP64 smooth Q norm field. It retains source file/header/offset/length and selected payload witnesses. The candidate receives only those knot norm words, physical IDs and the frozen rule. Full Q norms, P norms, raw keys, numerator, logits, probabilities, previous scores and consumer targets are denied as numerical operands until it seals all predictions and exits. Source byte hashing does not authorize numerical access to the unsampled field.

For each physical point, bilinear weights λ are nonnegative and sum to one. Interpolate the **squared smooth norm**, then take its root:

\[
\widehat s_{jh}=\sqrt{\sum_{k\in\text{four surrounding knots}}\lambda_{jk}s_{kh}^2}.
\]

The supplied smooth knot norm already includes epsilon. Equivalently interpolate s²−epsilon and add epsilon once; do not double-add it. The clamped last row/column follows the same fixed rule. At sample sites the formula closes to the supplied norm, subject to recorded FP64 square/root arithmetic. Preserve complete predicted norms, actual knot/row identities and all signed post-seal errors. No tolerance or fidelity gate controls evidence collection.

## Post-seal diagnostics

Only after sealing, open the complete authenticated Q/P norm fields and compare endpoint and Q−P response errors per cell, head and the three spatial strata. Zero response denominators remain explicitly undefined. P is an evaluation reference, not a candidate input. Record sampled equality as mechanics/capacity evidence rather than 256 independent predictions.

Then open the already-paid current query-side numerator and text logits. Form smooth reference and candidate image logits with identical text terms and full joint normalization. Define δℓ=ℓ_hat−ℓ. For each of the 96 dependent selected query-head addresses, preserve the full signed score error δℓ, reference probabilities p, unweighted errors, attention-weighted mean and centered variance, and exact smooth-distribution KL:

\[
\operatorname{Var}_p(\delta\ell)=\sum_jp_j(\delta\ell_j-\mathbb E_p\delta\ell)^2,
\qquad
KL(p\|\widehat p)=\log\mathbb E_p[e^{\delta\ell}]-\mathbb E_p[\delta\ell].
\]

Evaluate partition differences stably and retain finite arithmetic residuals. Text score errors are zero. These diagnostics test whether spatial errors lie where this reader is sensitive; they are **not measured AV, MLP, next-query or image fidelity**. Do not open value banks or run an attention-value contraction, learned map, extra consumer, full suffix or image.

## Costs and limits

The saved-only capacity run performs scalar interpolation and demand-weighted scoring from already-computed operands, with no new original-weight projection. Its inputs remain paid model outputs. It cannot establish cheap knot production or arbitrary-prompt coverage.

An eventual current-state producer would still pay **256×3072² =2,415,919,104 matrix MACs** for sampled original key projections, versus **9,663,676,416** for all 1,024 rows, plus original current ModLN/state formation, normalization, query-side compilation and receiving maps. The theoretical quarter-projection account is not measured work or a speedup. Keep those prospective costs separate from the saved-only assay.

Use one finite guarded strict-preflight mrun CPU lease, actual backend and no retries. Freeze candidate output before diagnostic access; keep complete arrays near compute and publish lossless observations. Exploration preserves heterogeneous results and negative trends. The next periodic Astra/newest-post review remains due62 or material evidence; storage review63 or earlier pressure. Arbitrary raw-prompt zero-neural images remain unresolved.
