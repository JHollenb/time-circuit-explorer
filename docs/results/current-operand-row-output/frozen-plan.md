# Form row patterns after the current nonlinear product and read

Status: **written and unexecuted**. No implementation, payload or job exists. The [fixed-U diagnostic](../fixed-row-projection.md) finds that the recorded projection residual dominates the previous rewrite's discrepancy. The same input-derived coordinates should no longer be treated as sufficient for the complete write on this specimen. This one new cut changes the representation boundary rather than increasing rank or fitting output labels. There is no automatic continuation or image campaign.

For each text/image stream in original output-column order, define

\[
J_s=[\operatorname{attentionRead}_s,\operatorname{ownProduct}_{82,s}],\qquad
F_s=J_s\Omega_s,\quad U_sR_s=\operatorname{qr}(F_s),
\]
\[
K_s=U_s^TJ_s,\qquad B_s=K_sW_{\rm out}^{T},\qquad
\widehat Y_s=U_sB_s.
\]

Freeze rank 128 per stream. Generate one unscaled Rademacher sketch Ω_s with shape [12288,128] from NumPy PCG64(SeedSequence([20261107,85,s])), stream s=0 for text and 1 for image. Take random_raw words in C order, low bit 0→−1 and 1→+1, then convert signs to FP64. Use local generators; no global RNG advance, rank/seed search, centering, power iteration or output-derived basis. FP64 reduced Householder QR retains all 128 columns and canonicalizes signs from the diagonal of R, with zero assigned positive sign. It does not certify numerical rank.

Admit only cycle 82's own nonlinear product, authenticated paid attention read, common gate and incoming text/image, and the original complete output weight. Hash the actual BF16 J concatenation before FP64 promotion and preserve the original column order. Deny all old bases/coordinates, outputs/exits and comparison targets until the complete candidate seals and its producer exits. Form all eight candidate matrix products in FP64, then use cycle 83's one final FP64→FP32→BF16 admission, separate BF16 gate multiplication and residual addition. Derive exact layout/device/native-consumer contracts from the pinned source before staging.

Compare primarily with the retained cycle 82 complete output/exit, secondarily with cycles 83 and 73. Keep full and within-stream-centered updates, actual exit-minus-incoming increments, every row and signed error-difference/cross accounts. Preserve finite-policy changes; neither an accurate QR nor a small complete-state percentage certifies sufficient state. No new native reference, upstream computation, continuation, image or quality gate is required.

| Operation | MACs |
|---|---:|
| Current J sketch |2,415,919,104|
| UᵀJ |2,415,919,104|
| Compressed original output map |9,663,676,416|
| Complete reconstruction |603,979,776|
| Candidate subtotal |15,099,494,400|
| Post-seal UR reconstruction and UᵀU diagnostics |50,331,648|
| Complete assay |15,149,826,048|

Two QR factorizations and explicit Q cost approximately 95.071 million algorithmic FLOPs, separately from MACs. Prospective composition with the historical 19,327,352,832-MAC G/Up formation is 34,426,847,232 MACs, before QR/scalar/data and paid upstream work. This composition is not an executed full producer or measured runtime saving. Original output weights and all source/attention/nonlinear operand formation remain paid.

This is one bounded test of whether an economical current-operand basis carries missing row patterns before a learned receiving map. It is still learned computation and supplies no arbitrary-prompt zero-neural backend. Before freezing, derive source/ABI/resource contracts, run independent tiny references, obtain exact source/resource review and use one finite guarded mrun lease with strict preflight/no retries. Keep full arrays near compute. Full newest-post/storage reviews are current at cycle 84, next due 87 or earlier material evidence/pressure. Conditioning, autonomous evolving-state closure and a general non-neural pixel producer remain open.
