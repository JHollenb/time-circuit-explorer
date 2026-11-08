# Finite consumer crossover for the contextual state representation

Proposed after the [fixed-nine-atom capacity result](../finite-writer-capacity.md), with Astra as mathematical advisor. This plan is unimplemented and unrun. It selects one local representation diagnostic before another writer fit; it does not repair the vessel padding response or close the arbitrary-prompt zero-neural image goal.

## The bounded question

The [contrast-aware writer](../contrast-aware-contextual-writer.md) gives vessel row9 a nearly correct H1 difference, while its original next Q/K responses remain poor. Is the response defeated largely by its own baseline and normalization, or by the remaining response-direction error? Finite arithmetic requires two crossed evaluations to make that question concrete; neither crossed term alone identifies a pure causal share.

Use only the previously public discovery vessel pair00→10 and row9. The corresponding01→11 prefix repeats at this causal address, so it supplies no independent replication. Keep the accepted104 program, own states, native states, model/input identities and original consumer profile fixed. There is no new prompt, fit, teacher/domain expansion, atom, routing tree or serving program.

Let a,b be native row9 H1 endpoints and p,q the accepted104 own endpoints. Form the two directions in a frozen FP64 subtraction order:

\[
\delta_n=b-a,\qquad\delta_o=q-p.
\]

Let A apply the declared FP32 round-to-nearest-even and BF16 round-to-nearest-even admission to an FP64 state. Construct exactly two crossed states:

\[
x_{no}=A(a+\delta_o),\qquad x_{on}=A(p+\delta_n).
\]

Retain the unrounded FP64 sums and both state admission remainders

\[
r_{no}=x_{no}-(a+\delta_o),\qquad
r_{on}=x_{on}-(p+\delta_n).
\]

Also record any discrepancy in A(p+δ_o) versus stored q and A(a+δ_n) versus stored b. These are diagnostics, not replacements for the stored diagonal endpoints. The original responses below must use the authenticated historical outputs, not re-admitted diagonal states.

## The original receiving computation

F is the exact qualified layer1 input RMS normalization and Q/K/V profile used in104: FP32 RMS with epsilon1e-6, the retained BF16 admission/learned norm weight, and original BF16 Q[4096,2560],K[1024,2560],V[1024,2560] projections under the same backend and numerical policy. Verify actual profile and parameter/input/output payload identity before reusing F(a),F(b),F(p),F(q).

Preserve the full[1,512,2560] input geometry for each new evaluation. The native-base branch uses complete native00 H1 with only row9 replaced by x_no; the own-base branch uses complete own00 H1 with only row9 replaced by x_on. Keep the original layout, batch, dtype, device and kernel contract. The primary comparison reads row9 from the resulting Q/K/V; retain the unmodified-row comparisons as execution diagnostics. Do not silently substitute a row-sized GEMM with a different finite contract.

This is two new normalization/QKV units: six original learned linears and16,106,127,360 nominal matrix MACs. No attention, MLP, later encoder, denoiser, VAE or image is needed. Historical states, lexical/program construction, teacher traces and original consumer work remain paid. Actual resource geometry and backend placement must be frozen after source review; use one bounded mrun lease, no retries or automatic expansion. Original learned evaluations in this diagnostic are offline research costs, not zero-neural serving.

## Keep both finite decompositions

Define the stored diagonal responses and two crossed responses separately at each Q/K/V port:

\[
O=F(q)-F(p),\quad N=F(b)-F(a),\quad
C_{no}=F(x_{no})-F(a),\quad C_{on}=F(x_{on})-F(p).
\]

The total response error has two paths:

\[
O-N=(O-C_{no})+(C_{no}-N)
=(O-C_{on})+(C_{on}-N).
\]

These are algebraic decompositions of measured finite outputs. Admission effects are included in both crossed terms. An unrounded state lies outside this BF16 input ABI; the two new evaluations cannot separately identify a hypothetical admission-only consumer effect. Use descriptions such as “native baseline with own response after admission,” rather than assigning pure baseline or direction percentages.

At implementation, freeze subtraction, norm, dot-product and reduction order. For each path, retain both signed terms U,V and the actual floating regrouping remainder g=(O−N)−(U+V). Report their norms, cosines, signed inner product and cross terms. In exact arithmetic,

\[
\|O-N\|^2=\|U\|^2+\|V\|^2+2\langle U,V\rangle
+2\langle U+V,g\rangle+\|g\|^2.
\]

Evaluated finite closure remains a separately recorded measurement. Opposing terms may cancel; their magnitudes need not sum to the total error. Preserve each port's native-response denominator, absolute error when it is zero, and all countertrends. No arbitrary quality threshold selects or erases an observation.

## What this can change

If the native baseline with own response substantially recovers Q/K behavior while the own baseline with native response remains poor, that supports investigating baseline/normalization-sensitive state codes. The converse favors improving the represented response. Mixed ports, cancellation and finite remainders remain evidence rather than a binary verdict.

The result would concern one causal row and one immediate consumer. It would not demonstrate a useful full conditioning state, explain all475 padding rows, identify a general invariant, certify a larger writer family, prove future closure or generate pixels. Any implementation must retain canonical Saturn input/output custody, full source/profile identities, all reused versus new costs and a fresh actual-result review. No job is queued by this proposal.

Related: [capacity family contract](finite-writer-capacity.md), [actual104 consumer review](../results/contrast-writer-evidence/advisor-actual.json), [state quotient counterexample](../../../cmachine/research/state_quotient/FINDINGS.md), [MLP own-state work](../../../cmachine/research/transition_laws/FINDINGS.md), [living research log](../../../obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md).
