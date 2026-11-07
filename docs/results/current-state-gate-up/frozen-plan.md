# Produce complete gate and Up operands from a current-state basis

Status: **written and unexecuted**. No implementation, payload or job exists. The [support census](../rectifier-liveness.md) finds the combined rectifier coefficients dense on every row and tile. Stop that centered sparse-correction branch. Astra recommends one different question: can a basis formed only from current normalized input produce complete G/Up operands economically enough to preserve their native local consumer?

## Source boundary and isolated targets

Use the existing [cycle73 matched first-single capture](../matched-single-write.md), clock 0 / single block 0. Admit its complete normalized Z, paid attentionRead, gate and incoming text/image residual inputs, with exact source/custody/model/map identities. This is the cycle73 moment-attention path, not an independent full-native-model reference or a first-prompt source producer.

Before the candidate seals and its process exits, deny archived G/Up, activation, product, update and exits. Do not admit Abar, Ubar, L, channel supports or output-error labels at all. After exit, score against the already authenticated complete dense local targets; no new reference campaign. Revalidate all exact source contracts before implementation/staging rather than assuming the old metadata remains available.

## One fixed input-derived law

For text/image streams s=0/1, use all N_s=512/1024 rows of Z_s, d=3072, m=9216 and rank r=128. Promote the actual stored BF16 input and original gate/Up weights to FP64. Freeze Rademacher Ω_s of shape N_s×128 using NumPy `SeedSequence([20261107,82,s])`, PCG64 `random_raw` low bits in C order mapped to {−1,+1}. Retain the complete signs and generator identity. No centering, prefix landmarks, oversampling, power iteration or rank/seed sweep.

\[
Y_s=Z_s^T\Omega_s,\qquad Q_s=\operatorname{qr}_{\rm reduced}(Y_s),\qquad C_s=Z_sQ_s,
\]
\[
B_s=Q_s^T[W_G^T\;W_U^T],\qquad
[\widehat G_s\;\widehat U_s]=C_sB_s.
\]

Pin FP64 reduced Householder QR and its library/profile. Canonicalize each Q column's sign from the corresponding R diagonal, taking +1 for a zero diagonal; retain all 128 columns. Orthogonality, rank deficiency and input reconstruction error are diagnostics, not thresholds for changing the law. Invalid shapes, nonfinite arithmetic or a failed QR are mechanics/instrument findings. Do not fit or prune columns after seeing native outputs.

In ideal real arithmetic, the reconstructed operands apply the original maps to Z_s Q_s Q_sᵀ. A low-input-energy discarded direction can still be amplified by those maps or downstream nonlinearities; input reconstruction alone does not establish consumer sufficiency. Source BF16 rounding, new FP64 contractions, QR arithmetic and changed grouping remain explicit parts of the approximation.

## Preserve the receiving computation and finite geometry

The original fused `to_qkv_mlp_proj.weight` has gate rows 9216:18432 and Up rows 18432:27648. Compile those complete views into the basis. Never evaluate original full-width G/Up first and then compress their outputs.

Admit the reconstructed complete operands through **FP64→FP32→BF16**. Restore the archived GU view's shape, stride and storage offset before invoking the original `mlp_act_fn`. Charge any initialized backing/padding/copies. Then use the original full concatenated attention-plus-product output affine, gate multiplication and residual addition. The authenticated paid attention read stays fixed. Preserve exact native consumer source/profile/parameter identities; revalidate the archived geometry when implementing.

Seal complete G/Up, activation/product, update and exit outputs, plus bases, signs, projected maps, coordinates and actual operation receipts. The producer must exit before archived targets open. Retain full/centered operand/write/exit scores and every text/image row, including countertrends and numerical exceptions. No image or future sampler step is planned; a useful local result would not establish whole-trajectory closure.

## Charge current-state compilation before claiming economy

| Planned operation | Prospective MACs |
|---|---:|
| Two stream sketches |603,979,776|
| Project current rows into bases |603,979,776|
| Compile complete G/Up maps for both bases |14,495,514,624|
| Reconstruct complete G/Up operands |3,623,878,656|
| Original full concatenated output affine |57,982,058,496|
| Total before QR and scalar/data work |77,309,411,328|

The matched original G/Up-plus-output subtotal is 144,955,146,240 MACs: 86,973,087,744 for G/Up and 57,982,058,496 for the full output affine. Prospective reduction is 67,645,734,912 MACs, or 46.67% of this subtotal. This excludes common upstream QKV/attention/source preparation and is not a whole-block, trajectory or runtime saving.

Charge both Householder factorizations and explicit reduced-Q formation separately, approximately 397.1 million FLOPs at these shapes, with actual library/runtime recorded. Also charge signs, copies, casts, nonlinear elements, residual arithmetic, all original parameter reads, output retention and memory. Do not conflate FLOPs and MACs or treat input-dependent coefficient compilation as free offline work. The basis is recomputed for each current state; no cross-prompt or cross-clock reuse is assumed.

Derive exact ABIs, live-buffer memory, numerical contract and finite guarded mrun CUDA admission from the implementation and measured history. Use one lease, strict preflight, no retries and tiny independent factorization/cast/consumer-geometry references. Run the standard post-job Saturn sweep. Keep complete arrays near compute and compact evidence locally. No automatic rank increase, extra seed, tail correction, new reference or image campaign.

All learned source formation, original gate/Up/output maps and native consumer work remain paid. This is a proposed reduced learned computation, **not zero-neural generation**. Economical raw-prompt formation, evolving-state closure, prompt generalization and a general non-neural pixel producer remain open.
