# One fixed coherent output sketch for image-key normalization

Status: **written and unexecuted**. This is one source-only positive norm law, chosen after the [diagonal result](../diagonal-key-norm.md) and its saved-only moment diagnosis. It is not a rank, gain or random-seed search, and it cannot establish arbitrary-input or zero-neural fidelity.

## The computation

For each head h, choose one fixed signed sketch S_h of shape [16,128]. Compile from the original BF16 image-key weight widened to CPU FP64:

\[
B_h=S_hW_h,\qquad
\widehat s_{jh}=\sqrt{\frac{\|B_hx_j\|^2}{16\cdot128}+10^{-6}}.
\]

The actor's x is its actual original BF16 ModLN output widened to FP64. Keep the sealed cycle58 image-score numerator, native Q text keys/full values, selected P support [4,11,12,13], original head admission and own sequential MLP→own exit→own next query unchanged. The sole scientific change from the cycle58 retained-norm producer is the norm law. Cycle59's diagonal candidate is an additional saved comparison.

Set r=16 to target an eightfold reduction in matrix work versus a full 128-channel key projection, before observing any candidate result. Do not change r after looking at the result. Freeze seed **20261107** with NumPy PCG64. For head h, independently initialize PCG64 with SeedSequence([20261107,h]); draw random_raw(16*128) unsigned64 words, use the low bit to assign +1 for one and −1 for zero, then reshape [16,128] in C order. Retain every actual int8 sign matrix and its file/payload hashes. Widen signs to FP64 for compilation. No rerolls, gain calibration, selection across seeds, SVD/rank fit or target-derived basis.

The independent ideal Rademacher construction has

\[
\mathbb E_S\left[\frac{\|S_hz\|^2}{16\cdot128}\right]=\frac{\|z\|^2}{128}.
\]

Unlike the input-channel diagonal, each squared projection retains input-channel cross terms through S_hW_h. Unbiasedness is an expectation over hypothetical sketches, not an accuracy claim for this one fixed pseudorandom instance or for its spatial average. For ideal independent signs and general r,

\[
\operatorname{Var}_S\left[\frac{\|Sz\|^2}{128r}\right]
=\frac{2}{128^2r}\left(\|z\|^4-\sum_c z_c^4\right).
\]

Its relative energy standard deviation can approach √(2/16)≈35.36%. The square root is nonlinear; unbiased energy does not imply unbiased norms or consumer outputs. The fixed rank also creates possible blind directions. Retain every underestimated/overestimated/degenerate cell and adverse downstream response. Epsilon preserves a finite positive denominator; it does not certify useful attention.

## Isolation, continuation and observations

The compiler hydrates only original image-key W plus the pinned signs. It must not read source states, queries, prior duals, native norms/raw keys or targets. Seal B/signs before a fresh actor starts. The actor hydrates only the original five return/next-query weights already used in cycle59. Deny original image-key W/gamma and all saved P/Q norms/raw keys/old actor endpoints before actor seal and process exit.

Use typed Acts with explicit output ABIs for the compiler, original actual ModLN, sketch projection/energy/norm, same attention read, own MLP state and own next query. Preserve actual input→norm→head→state→query joins. Verify complete source/seal/custody/finite/shape contracts; no accuracy threshold stops evidence collection.

After sealing, compare only authenticated existing smooth-Q norms, logits/head references and original PP/QQ/stale/retained/diagonal consumer endpoints. No new full-key projection, full model, extra consumer arm, suffix or image. First verify five matched payload controls against cycle58 (actual input, numerator, text logits, text keys and values). If one differs, preserve all results but mark the norm-only attribution inapplicable.

Retain all 24,576 signed norm/energy cell errors, logit/log-partition errors and selected trajectory fields. For H/A/M/T/nextQ measure response and endpoint errors separately, plus all 96 dependent next-query heads and each whole selected row. Preserve direction, sign, scale, countertrends and row distributions. Do not treat 96 dependent heads as independent prompts. The squared norm is already exactly quadratic; this cut tests its representation, not a higher Taylor order around a supplied state.

## Prospective work ledger

| Work | Nominal matrix MACs |
|---|---:|
| Compile signed output coefficients |150,994,944|
| Actor norm projection |1,207,959,552|
| Existing numerator/read/original own-state consumer |1,346,371,584|
| Complete actor |2,554,331,136|
| Complete new compiler + actor |2,705,326,080|

FP64 B payload is **9,437,184 bytes**; actual int8 signs add **49,152 bytes**, before file headers/metadata. Norm work also pays 393,216 output squares, 368,640 reduction additions, 24,576 energy scalings, 24,576 epsilon additions and 24,576 roots, plus all existing ModLN/attention/finite-policy work. Charge original W reads/compiler hydration and all receiving parameters. Signed multiplication may be optimized into additions later; this plan claims only shape-based nominal matrix counts.

Use one finite guarded strict-preflight mrun lease, request actual CUDA for the original receiving maps, no retries, no duplicate run and no hand-estimated reservation unless the measured cycle58/59 backend exception applies and is explicitly documented. Inspect current fleet state before admission. Expected geometry remains one compiler and one four-row consumer actor, not full images. Keep large arrays near compute. Review storage at cycle60 before allowing more accumulated transports.

This law is **reduced learned arithmetic**. It still needs actual source-state production, a query-specific paid dual table, original text keys/full values and receiving maps. Changing queries may require recompilation. It does not claim a runtime speedup, first-prompt saving, new image, autonomous diffusion or arbitrary-prompt zero-neural generation. Whatever its quality, retain the result and then reconsider the representation against the full goal.
