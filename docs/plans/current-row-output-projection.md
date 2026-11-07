# Carry the complete output write through the same input-derived row coordinates

Status: **written and unexecuted**. No implementation, payload or job exists. The [current-input G/Up result](../current-state-gate-up.md) produces complete local operands with 18.00% actual state-increment error, while its original full output affine consumes 57.982 of 77.309 billion candidate MACs. Astra recommends one different cut: test whether the same source-derived row coordinates can carry the complete attention-plus-MLP write through that output map.

## One fixed law, with no output-derived basis

Use retained cycle82 coordinates C_s and own product, authenticated attentionRead, gate and incoming text/image residuals, and the original complete output weight. Preserve exact model/source/parameter/byte/custody bindings. Each stream retains rank 128, with 512 text or 1,024 image rows. Do not admit cycle82 updates/gated writes/exits, cycle73 targets, other bases, output-fit labels, means or channel supports before the new producer seals and exits.

Promote the retained FP64 coordinates directly and the actual BF16 read/product/weight values to FP64. Define

\[
C_s=U_sR_s,\qquad U_s=\operatorname{qr}_{\rm reduced}(C_s),
\]
\[
J_s=[\operatorname{attentionRead}_s,\operatorname{ownProduct}_{82,s}],\qquad
K_s=U_s^TJ_s,\qquad B_s=K_sW_{\rm out}^T,\qquad \widehat Y_s=U_sB_s.
\]

Use reduced FP64 Householder QR, canonical column signs from R's diagonal with zero→+1, all 128 columns retained, and a separately recorded implementation/profile. No new seed, rank selection, pivot-based pruning, output-derived basis, fit or power iteration. Rank/orthogonality diagnostics do not adapt the law. Source-defined row space is the subject of the test, not an established sufficient state.

In ideal arithmetic this is U_sU_sᵀJ_sWoutᵀ, an orthogonal row projection of the complete write without first evaluating that write. Nonlinear product formation or the attention read may escape the input-derived row space. This test measures their combined effect; it does not identify which component caused any error.

Freeze FP64 contractions, then admit the complete reconstructed update through FP64→FP32→BF16. Apply the existing separate BF16 gate multiplication and residual addition to the actual incoming fields. This replaces native output-affine arithmetic: grouping, precision and final admission are part of the contrast, not claimed bitwise reassociation equivalence. Derive exact complete-update ABI/layout and numerical/resource contracts from source before staging.

## Count compilation and diagnostics separately

| New output operation, both streams | Prospective MACs |
|---|---:|
| UᵀJ |2,415,919,104|
| Compressed rows through original full Wout |9,663,676,416|
| Complete update reconstruction |603,979,776|
| New output subtotal |12,683,575,296|

The matched cycle82 full output affine costs 57,982,058,496 MACs, so the prospective difference is 45,298,483,200. Two new reduced QR factorizations plus explicit Q formations add approximately 95.071 million algorithmic FLOPs, separately charged. Casts, concatenation, canonical signs, nonlinear/state arithmetic, parameter/data reads, memory, fingerprints and retention remain paid.

The prospective composed producer retains the entire cycle82 G/Up construction cost 19,327,352,832 and adds the new output path, for 32,010,928,128 MACs before QR/scalar/data work. The new bounded assay would execute the 12.684B output path using saved82 fields; 32.011B is a prospective composed-producer count, not a new measurement. Using those fields does not make their formation free, and neither this count nor a successful local result establishes first-prompt runtime savings. Common upstream source/QKV/attention, historical assay work and original learned maps remain paid. No cross-state reuse is assumed.

Any exact J reconstruction or UᵀU diagnostic must be declared and charged separately before freeze; do not silently append it to the serving subtotal. No diagnostic may increase rank, choose a seed or launch a correction candidate. Prefer the minimum diagnostics needed to interpret the declared row-space test.

## Release once, then compare both references

Retain complete U/R, projected inputs, compressed mapped rows, reconstructed FP64/BF16 update, gated write and actual exit/text/image outputs, with operation fingerprints. The producer must seal and exit before opening cycle82 update/gated-write/exit targets or cycle73 targets. No new G/Up projection, native reference, sampler continuation, VAE or image is needed.

Compare primarily with 82 to isolate this output cut and secondarily with 73 to retain total error. For each complete update and actual finite exit increment, keep full and within-stream-centered scores and every dependent row. Retain the signed identity

\[
e_{83,73}=e_{82,73}+\Delta_{83,82},\qquad
\|e_{83,73}\|^2=\|e_{82,73}\|^2+\|\Delta_{83,82}\|^2+2\langle e_{82,73},\Delta_{83,82}\rangle.
\]

Cast stored values before subtraction. Signed cancellation is not a positive causal share. Keep mechanics/instrument/trend/terminal status separate and preserve countertrends. No quality threshold stops collection or changes the program. Use one finite guarded CUDA lease, strict preflight, measured/buffer-derived admission, tiny independent references, no retries and the standard Saturn sweep on the actual raw compact result.

This is a proposed local learned-computation reduction from supplied current state. Economical arbitrary-prompt conditioning, evolving-state closure, broad prompt faithfulness and a general zero-neural pixel producer remain open.
