# Separate the missing mean write from row variation on one own state

Status: **written and unexecuted**. No payload is staged or job submitted for this diagnostic. It follows the [mean-Up own trajectory](../mean-up-trajectory.md): first velocity and some red geometry improve, later velocity and PNG RMS mostly worsen, while PNG MAE slightly improves. Those mixed trends do not attribute image drift to covariance.

## One fixed receiver

Use **clock3/single0**, the first single MLP at the last sampler clock of cycle75. This architecture-defined address is fixed before examining per-block remainder magnitudes. It has 512 text and 1,024 image rows. Admit the saved actual normalized Z, SiLU activation A, current support S, selected Up, completed Up, product, means, anchors and residual gate from that one own-state transaction. All original generation/capture work remains paid. Bind the exact source checkpoint, input descriptors, candidate seal, completion Act and source U/D byte offsets.

No new image, prompt, own-state continuation, teacher refresh, hidden state replacement, selector adjustment, anchor fit or sweep. Other numerical receiver states and archived reference outputs remain quarantined. Source metadata and immutable custody may be inspected. This diagnostic cannot measure the final image's causal defect or a serving speedup.

## Three large products

Promote admitted Z and source U coefficients to FP64 and compute

\[
U^*=ZW_U^T,\qquad P^*=A\odot U^*,\qquad
E_P=P^*-P_{\mathrm{saved}}.
\]

A is the actual saved BF16 SiLU output, promoted without recomputing gates. U* is a new FP64 source-map reference, not the native BF16 full-Up contraction or a free operand. For each stream s, reduce the actual U* rows to \(\bar U_s^*\), and form

\[
R_P=1_{j\notin S}\,A\odot(U^*-\bar U_s^*),
\quad
Y_E=E_PD^T,\qquad Y_R=R_PD^T,\qquad Y_Q=Y_E-Y_R.
\]

D is the original MLP portion of the full concatenated output map. Evaluate Y_E and Y_R directly as two FP64 down contractions. Do not add a third Down or an optional two-row covariance projection. One full Up and two full Down products cost **130,459,631,616 prospective MACs**, three times 1,536×3,072×9,216. Elementwise products, reductions, copies, hashing and I/O remain additional work. There is no attention, original head, sampler or VAE call.

The product-space difference \(Q_P=E_P-R_P\) contains actual-anchor admission, selected-Up finite projection and product-multiplication rounding. Its elementwise bookkeeping can be expanded using the ideal completion C* (selected U*, omitted stream-mean U*):

\[
Q_P=A\odot(C^*-C_{\mathrm{saved}})
+(A\odot C_{\mathrm{saved}}-P_{\mathrm{saved}}).
\]

Keep selected and omitted admission observations separate in product space. Y_Q is a **combined policy/reassociation remainder**: its output subtraction also contains unseparated FP64 distributivity error. Do not call it an independently evaluated Q_P D^T. Record the product-space expansion discrepancy as an observation. None of these quantities is the actual native finite concatenated attention/MLP output error; current image attention is not supplied or recomputed.

## The representation question

Split Y_R within each stream into its mean \(\bar Y_{R,s}\) and centered rows \(Y_R-\bar Y_{R,s}\). In real arithmetic,

\[
\overline{R_P}_s
=1_{j\notin S}\operatorname{Cov}_{i\in s}(A_{ij},U^*_{ij}).
\]

Retain both sides and their measured FP64 reduction discrepancy. Do not add a tight arbitrary floating-point gate. Multiply the three output components by the saved residual gate: stream-constant mean, centered row variation and Y_Q. Preserve their complete three-term signed Gram matrix, energies, cosines, cross terms and per-row outcomes, separately for text, image and pooled rows. Negative or mixed cross terms are evidence.

Also compute the least-squares best supplied per-stream constant for the total gated missing write:

\[
B_s^*=\operatorname{mean}_{i\in s}(g\odot Y_{E,i}),\qquad
L_s^*=\sum_{i\in s}\|g\odot Y_{E,i}-B_s^*\|_2^2.
\]

This mean is optimal only within that declared constant-output family and unweighted squared-error geometry. The centered remainder gives that family's representational floor on this receiver. This is an oracle diagnostic computed from a paid full source reference, not a cheaply produced constant, a fitted candidate or an optimized image. Compare its floor with the gated mean-Y_R representation descriptively; neither has a quality gate or promotion authority.

## Execution and custody

Before staging, derive exact tensor liveness, retention and source-slice inventory from the implemented code. Use mrun intent/history planning, measured local diagnostic history and explicit CUDA admission; declare the actual FP64 backend, finite timeout and no retries. Run one small synthetic independent reference first, preserving all signed heterogeneous observations. No model arithmetic runs on the laptop or through SSH.

Keep full new arrays near compute, with compact scalar rows and receipts locally. Bind typed Acts and complete original U/D bytes, support, source inputs and emitted fields. Run and verify the standard Saturn sweep after the diagnostic. Mechanics, instrument status, exploratory trends and terminal claims remain separate. No effect-size threshold may stop evidence collection. The arbitrary-prompt zero-neural goal remains active and unachieved.
