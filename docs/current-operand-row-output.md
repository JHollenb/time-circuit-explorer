# A row basis after the nonlinear product

Building the basis from the current attention read and MLP product improves the cheaper output approximation on the retained state. Actual state-increment error against the retained dense-73 path falls from **30.94% to 24.21%**, and 1,415 of 1,536 dependent rows improve. The more expensive cycle-82 output remains better at **18.00%**; 1,532 rows worsen against it. This is evidence that the representation boundary matters, with substantial remaining approximation error.

This is one supplied FLUX.2-klein-4B state at clock 0 / single block 0. It produces a complete local write and exit. There is no new prompt, native reference, continuation or image. Original learned maps, earlier state formation, attention and G/Up computation remain paid. Arbitrary-prompt image generation with zero neural evaluations remains unresolved.

![Measured full and centered state-increment errors](results/current-operand-row-output-figure/operand-row-output.png)

The [cycle-84 diagnostic](fixed-row-projection.md) found that projecting the complete write into the earlier input-coordinate row space caused most of the cycle-83 discrepancy. The input coordinates were useful for forming linear G/Up operands, but their span did not survive the subsequent nonlinear product and attention join. The [frozen plan](plans/post-nonlinearity-row-basis.md) therefore moved basis construction to the current output operand, keeping rank 128 per stream and one fixed sketch.

For a stream with rows indexed by tokens, write the current output operand as

\[
J=[R,P_{82}],\qquad Y=JW_o^\top.
\]

Here R is the paid attention read and P₈₂ is the exact retained, finite-admitted cycle-82 SiLU/product field. Its schematic real-arithmetic form is SiLU(G)⊙V, with G and V formed by the [cycle-82 producer](current-state-gate-up.md); the experiment consumes the actual BF16 field. Even if both linear operands are functions of coordinates C, their elementwise product need not remain in the column space of C. Reusing QR(C) after that product asks the earlier representation to preserve a computation that changes its row geometry.

The new source-only construction is

\[
F_s=J_s\Omega_s,\quad U_sR_s=\operatorname{QR}(F_s),\quad
K_s=U_s^\top J_s,\quad B_s=K_sW_o^\top,\quad \widehat Y_s=U_sB_s.
\]

Two independent local PCG64 generators use SeedSequence([20261107,85,s]). Each draws a 12,288×128 uint64 array in C order, maps its low bit to unscaled FP64 signs, and leaves global NumPy and Torch RNG state unchanged. Reduced FP64 QR retains all 128 columns, with deterministic diagonal-sign canonicalization. No seed, rank, centering, power or output-fitting search is performed. The producer authenticates the actual BF16 concatenation before promotion, physical sketch signs and every contraction handoff.

In exact arithmetic with orthonormal U, this computes P_U J W_oᵀ. The finite implementation additionally has FP64 reassociation, FP64→FP32→BF16 admission, a separate BF16 gate multiply, and a separate BF16 residual add. If A denotes the double admission and B denotes BF16 rounding, the observed increment is

\[
I(\widehat Y)=B\!\left(x+B\!\left(g\odot A(\widehat Y)\right)\right)-x.
\]

The report compares this actual exit-minus-common-incoming value. A large unchanged residual cannot hide update error in that denominator. Centering subtracts each stream's mean independently; it keeps changing row structure visible. QR reconstruction error is about 6.78×10⁻¹⁶ relative RMS and Gram max error is at most 1.78×10⁻¹⁵. These are finite numerical diagnostics, not a rank or output-sufficiency certificate.

| Actual increment vs retained dense-73 | All rows | Text | Image |
| --- | ---: | ---: | ---: |
| Cycle 82: complete original output | 18.003% | 18.065% | 17.966% |
| Cycle 83: basis from input coordinates | 30.936% | 31.606% | 30.532% |
| Cycle 85: basis from current read/product | 24.205% | 25.270% | 23.550% |
| Cycle 82, centered | 23.879% | 21.595% | 25.726% |
| Cycle 83, centered | 41.446% | 37.881% | 44.354% |
| Cycle 85, centered | 32.323% | 30.271% | 34.031% |

The primary question is how well the new output approximates cycle 82 using its exact same J. Increment discrepancy from 82 falls from 26.277% for cycle 83 to **17.827%** for cycle 85; the centered discrepancy is **24.097%**. On this primary comparison 1,409 rows improve and 127 worsen relative to cycle 83. Against dense 73, 1,415 improve and 121 worsen relative to 83, while only four improve and 1,532 worsen relative to 82. All 1,536 new rows beat no update against dense 73. These are dependent rows from one state, not independent prompt trials.

Text row 33, previously worse than no update, improves from 222.23% to 58.80% of its dense increment RMS. It remains worse than cycle 82. The four improvements against 82 are global rows 3, 1275, 1513 and 1524; the [row-localization receipt](results/current-operand-row-output-evidence/row-localization.json) preserves these exceptions and the largest regressions.

Signed vector accounting prevents us from assigning all improvement to a single scalar error. With I₈₅−I₇₃=(I₈₂−I₇₃)+(I₈₅−I₈₂), the measured full energy is

\[
11{,}502{,}037.557
=6{,}362{,}870.441+5{,}916{,}267.332-777{,}100.216.
\]

The last term is twice the inner product, not another positive loss. The image cross term is −805,136.414, while text has a small reinforcing +28,036.198; pooled cancellation is not uniform repair. Centered energy is 6,173,453.364+5,915,909.407−778,075.379=11,311,287.391. Relative to cycle 83, the new delta has energy 13,560,746.379 and cross term −20,847,400.080 against its old dense-73 error. Thus 18,788,691.258+13,560,746.379−20,847,400.080=11,502,037.557. Full and centered signed accounts are retained for all streams and every row; the vector subtraction closures are zero in the retained BF16-field accounting.

| Work actually executed | Matrix MACs |
| --- | ---: |
| JΩ sketches | 2,415,919,104 |
| UᵀJ | 2,415,919,104 |
| Compressed original output map | 9,663,676,416 |
| U reconstruction | 603,979,776 |
| Candidate subtotal | **15,099,494,400** |
| Post-exit UR and UᵀU diagnostics | 50,331,648 |
| Assay subtotal | **15,149,826,048** |

QR adds approximately 95.071M algorithmic FLOPs, separately from MACs. Scalar operations, signs, copies, casts, hashes and retention are additional. The matched cycle-82 full output contraction uses 57,982,058,496 MACs. Adding historical G/Up formation gives a **prospective 34,426,847,232-MAC composition**; that composition was not rerun. Reduced matrix work does not establish a matched runtime saving or remove the learned output map.

One guarded CUDA job, job-c042d4101661, succeeds in **34.97 scheduler seconds**, with sampled peaks **1,891.5MB RSS / 1,578MB device memory**. Producer wall time after imports is 9.485 seconds, including 6.349 seconds setup; scopes differ from the whole assay. All **357** tiny mechanics tests pass. Complete byte checks cover 55 files / 259,256,644 bytes and 28 NPY payloads, plus the original 75,497,472-byte output tensor, actual BF16 J and physical sketch signs. The publisher checks source/parent ancestry, typed fields, geometry, work, isolation, observed producer exit and all scalar aggregates. It does not independently replay real-array numerics.

The [evidence index](results/current-operand-row-output-evidence/README.md) links original results and checks. All 17 new arrays / 139,677,824 NPY bytes stay near compute; compact public metadata and lossless rows total 3.055MB. Saturn run and verify return 0 on the actual raw result: one stream, 1,536 records, zero fires, formal claims or qualified addresses, and one expected missing no-fire ledger. Full newest-post and storage review remain current at cycle 84, next due 87; no new archive or deletion occurs in 85.

The mathematical lesson is bounded: a representation chosen before a nonlinear operation can omit directions needed by its consumer; constructing it from the current operand improves the measured output approximation here. Cycle 85 does not independently separate projection residual from finite-policy, admission and commit effects. A compact input sketch is still not guaranteed to preserve the gated output metric or successor state. For the time-circuit goal we ultimately need a representation π satisfying both observation preservation O(x)=Ō(π(x)) and successor preservation π(F(x,c))=T(π(x),c) on states the replacement itself produces. This local approximation establishes neither closure nor arbitrary-prompt conditioning or pixels.

Astra recommends one [receiver-weighted sketch](plans/receiver-weighted-output-sketch.md) as the next representation test. It selects row directions after the original output map and current gate, without materializing the complete target write. That plan is written and unexecuted; its extra learned-map compilation remains paid.
