# Test a reusable rank-one residual-score state

Status: **executed in cycle19**, job `job-49e50c9cb996`; [measured result](../rank-one-tail-state.md). The [full-direction positive quadratic](../positive-tail-moment-closure.md) reduces paired state error41.76%→24.15%, but reads the full tail to form directional moments. The source-defined streaming rank-one law closes against its direct calculation at every BF16 output with no fallbacks, while paired state error is39.02%. The following pre-execution construction remains the frozen protocol; no rank or polynomial-order sweep ran.

## Reduce only the residual score

Keep current-query exact ordered top4 selection, all source anchors/routes, the full-dimensional tail centroid and the positive quadratic law. For each GQA key group g, compile one pair of source-only vectors a_g,b_g. All four query heads in that group and all their anchors share them. With centered current keys u and current query displacement δ, use

\[
z_j=b_g^\top u_j,\qquad \beta=a_g^\top\delta,\qquad
\widetilde t_j=\beta(z_j-\bar z_T).
\]

Only t inside r₂(t)=1+t+t²/2 is reduced. Tail centroid mass still uses the full μ_T and δ; selected scores/values remain full-dimensional. Do not quietly project the centroid, rerank support or change the tail law.

Let C_y be the immutable source key covariance already used by the complete-bank geometry. Reuse the source-only query-candidate recipe and frozen routes; form the **uncentered** second moment D=E_source[δδᵀ], pooled equally across the group's four query heads, retaining the declared source lexical/padding mixture. Do not replace D by a centered query covariance or derive it from the scored prompt pair. The leading eigenvector v of C_y^(1/2) D C_y^(1/2) defines

\[
a_g=C_y^{1/2}v,\qquad b_g=C_y^{-1/2}v.
\]

Freeze eigensolver ordering/sign convention and source recipe before candidate serving. Document singular/roundoff-negative covariance eigenvalues, the represented support and pseudoinverse treatment; do not silently claim invertibility or consumer-optimality. The construction optimizes a source product-distribution score approximation, not downstream state error. Compilation accesses only source atoms, geometry and routes; its bytes, work and outputs must be sealed separately from prompt prediction.

## An actual prefix state

For every query-head anchor ℓ, advance the following state over admitted causal keys, in a stable common log gauge s_j=ℓᵀu_j:

\[
\sum_j e^{s_j}(1,z_j,z_j^2)\otimes(1,V_j),\qquad
\sum_j e^{s_j}u_j.
\]

All 64 anchors must be maintained as keys arrive, including anchors not selected by the current query. Masked keys do not enter; the query clock still advances over all 512 rows. Track state scale, admitted count, clock, source basis identity and retained selected-key occurrences explicitly. Do not rebuild prefix state from the whole tail for each query and call that streaming.

Subtract the selected four occurrences with their **original anchor scores and the same state gauge**. From the complement derive Z_T, the full μ_T, the scalar mean z̄_T and the three value/scalar moments. Contract them with r₂(β(z−z̄_T)); combine with unchanged exact selected mass/value in log space. Keep the measured finite centering defect and rounding separate.

At d=128, the declared moments use 3×129+128=515 FP64 scalars per anchor: **8,437,760 logical bytes** across 32×64 anchors. This excludes log scales, compensation, source bases, geometry, routes, outputs, work buffers and retrieval storage. A pair of 128-dimensional a/b vectors per eight groups adds 16,384 logical bytes. Allocated and actually used bytes must be measured. This footprint is not a saving relative to the active BF16 K/V payload.

Keep the full current K/V retrieval index and existing exact selector, explicitly charged. Supplying top4 positions does not eliminate selection work. No raw-KV saving or complete encoder closure is claimed.

## Controls and numerical complement behavior

Seal the complete streaming candidate raw/BF16 banks, source basis and state snapshots/traces before direct-tail diagnostics or original consumer parameters. After sealing, evaluate the direct rank-one positive law on the same ordered supports and operands; compare it with the streaming/downdated read. This separates source-rank approximation from state arithmetic and complement subtraction. Compare both with the existing full-direction quadratic and exact attention references; full-direction quadratic is not an accuracy bound.

Empty tails need an explicit selected-only path. Near-cancelling complement subtraction needs a declared numerical policy based on mechanics, not an arbitrary quality gate. Preserve signed moments, conditioning estimates, positivity/finite diagnostics and read discrepancies. Never clamp an invalid result into a legal distribution silently. Any direct-tail fallback must be explicit per row, counted with its key visits/products/time, and retained in the complete forecast provenance. Report pure-state rows separately so fallback-assisted output cannot establish standalone state closure.

Authenticate full-direction control banks/ports/parameter source before inheriting their receiving evidence. Evaluate the new streaming bank through the original full output/residual/RMS/SwiGLU/state consumer on all 512 rows, one new consumer per prompt unless a separately declared diagnostic arm is justified. Keep head/endpoint/paired-state errors, active/padded/before8/at8/after8 strata, signed errors/cancellation, exact-zero denominators and every per-head countertrend.

Tiny tests must independently check the source score objective, raw moment contraction against a direct positive law, causal append/rescale/masked clocks, exact selected-only tails, complement/cancellation behavior and actual state snapshot continuation. One finite guarded CPU mrun lease near immutable source atoms/ports, strict preflight, default admission, no automatic retry or duplicate suite. Charge source compilation, all-anchor updates, exact retrieval, moment reads, complement/fallback work, references and original consumers separately; run/verify the Saturn evidence sweep.

Pre-execution implementation amendment: keep absolute-moment envelopes alongside the core moments and cache each admitted key's original anchor scores, both explicitly charged. Use the documented FP64 engineering envelope in `rank_one_tail.py` to fall back to the same direct rank-one positive law when complement or polynomial normalization is unresolved. This assumes ordinary FP64 sum/product behavior and bounded exponential error; it is not a formal BLAS certificate. Retain denominator-aware full-centroid uncertainty and source eigengaps diagnostically. Core moment and envelope update/contraction counters are separate partial operation counts, not total FLOPs.

Pre-execution consumer amendment: evaluate one new streaming consumer per prompt. If the complete post-seal direct-rank-one BF16 bank is identical, inherit that receiving result; otherwise evaluate and count one additional direct-rank-one consumer per prompt. This diagnostic separates representation loss from downstream arithmetic/quantization effects. Existing full-quadratic, centroid, exact and zero receiving results are inherited only from the authenticated same-input parent; no fresh native-parity claim is implied.

The first assay remains the retained saved pair, for an explicit comparison to its full-direction reference. A miss limits this frozen rank-one representation, not all low-dimensional state methods. A useful result would establish one reusable component and motivate fresh raw-string replication. It would not establish arbitrary-prompt zero-neural images, contextual encoding through later layers, own-state diffusion or a general decoder.
