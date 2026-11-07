# Produce image-key normalization from a source-defined positive quadratic

Status: **written, unexecuted**. This follows the [image-dual result](../image-dual-score-response.md). It tests one baseline-free normalization operand inside a still conditional learned computation.

The previous actor supplies a paid smooth P denominator. Its largest measured head-error term is retained-denominator plus contraction arithmetic. The next candidate replaces that supplied denominator with one positive law compiled from the original image-key weights alone. Keep the sealed cycle58 dual numerator, actual Q BF16 ModLN input formation, paid native Q text keys/full values, selected P restoration, original finite receiving MLP and own next-query consumer unchanged.

The exact smooth norm is

\[
s_h(x)^2=x^TG_hx+\epsilon,\qquad G_h=W_h^TW_h/128.
\]

Compile only its diagonal from original BF16 W widened to FP64:

\[
d_{ha}=\frac1{128}\sum_c W_{hca}^2,\qquad
\widehat s_{jh}=\sqrt{\sum_a d_{ha}(x^Q_{ja})^2+10^{-6}}.
\]

The approximation removes channel cross terms:

\[
s_h(x)^2-\widehat s_h(x)^2
=\frac1{128}\sum_{a\ne b}x_ax_b\sum_cW_{hca}W_{hcb}.
\]

It does not assume all columns are orthogonal, nor infer universal cross-term cancellation. A poor result would describe this one positive law on the tested state, not rule out an economical normalization producer. Preserve signed omitted energy, denominator/logit/partition effects, per-row/head countertrends and complete downstream response errors.

Compile one 24×3072 FP64 coefficient array, **589,824 payload bytes**. Count **9,437,184 weight squares**, their reductions and scaling separately from matrix MACs. At serving, actual BF16 own-state x widens to FP64, then requires **3,145,728 input squares**, **75,497,472 weighted-sum MACs**, 24,576 epsilon additions and 24,576 square roots. The actor's prospective matrix total becomes **1,421,869,056 MACs**. These are planned counts, not measurements or a runtime saving.

Explicitly materializing all dense G matrices would take **1,811,939,328 FP64 bytes**, **28,991,029,248 compilation MACs** and at least **231,928,233,984 dense evaluation MACs** for 1,024 rows before final row dots. Factoring G back through W recovers the original projection cost. Neither is a cheap exact substitute.

Seal the weight-only compiler before a fresh actor starts. Deny P/Q raw key projections, P/Q normalization targets, all archived candidate outputs and the original image-key matrix to the actor. The compiler must not read source states, target norms or an outcome-selected fit. Declare typed source/coefficient/input/output identities, numerical read closure and actual operation geometry. The actor must consume its own produced norm and advance its own emitted H→MLP→T→next query.

After actor seal and exit, reuse the already sealed cycle58 smooth Q norms and old PP/QQ/stale/cycle58 continuations for diagnosis. Execute no fresh full-key projection, numerical-policy sweep, fitted correction, extra consumer comparison arm, suffix or image. Use one smallest bounded mrun lease with measured admission; original sources, dual compilation, text keys/values and all receiving maps remain paid historical work. No claim of arbitrary query coverage, first-prompt savings, full own-state diffusion or zero-neural images follows from this component.
