# Compile one MLP write into the state that receives it

Status: executed as guarded job `job-80041334757d`; see the [measured result and limitations](../compiled-mlp-write.md). Developed after Astra's review of both prompt-composition diagnostics. The original request and earlier failed harness run remain separate in the result bundle.

## Bounded question

Can we eliminate an original dense primitive from serving, while retaining its input-dependent product and the residual state update that consumes its output?

Use the existing32-row FLUX.2 Klein4B image-MLP corpus at pass2/joint1, source `job-f0cd6a040823`. Its [computation record](/Users/jakeholl/domains/saturn-pub/packages/mdb/docs/FLUX-MLP-STATE-LAW.md) binds two input endpoints, original parameter recipes and the recorded native CPU BF16 midpoint return. This is retrospective development evidence. It is not a new prompt, complete block or free-running denoising rollout.

## Compile products without retaining dense channels online

For the supplied line x(α)=x0+αΔx, form offline

\[
a(\alpha)=W_gx(\alpha)=a_0+\alpha a_1,\qquad
v(\alpha)=W_vx(\alpha)=v_0+\alpha v_1.
\]

With q=σ(a0) and d=αa1, use the baseline-local quadratic sigmoid

\[
\widehat\sigma(a_0+d)=q+q(1-q)d+
\tfrac12q(1-q)(1-2q)d^2.
\]

The retained SiLU product is a(α)σ̂(a(α))v(α), degree at most4. Project each coefficient once offline:

\[
\widehat m(\alpha)=W_{out}
\big[a(\alpha)\widehat\sigma(a(\alpha))\odot v(\alpha)\big]
=\sum_{j=0}^{4}c_j\alpha^j.
\]

Serve only the five projected coefficient arrays, selected parent residual U0, outer modulation gate gM and declared numeric metadata. Horner evaluation produces m̂; then the residual consumer computes

\[
\widehat I'=B\big(U_0+B(g_M\odot B(\widehat m))\big).
\]

The consumer receives the predicted write. Neither original matrix, expanded9216-channel branch, sigmoid/SiLU evaluator nor native module is available online. Compiling still uses original learned weights and nonlinear arithmetic; charge and disclose all of it as offline cost. A coefficient program derived from a model is not a solution to arbitrary prompts merely because its serving operations are small.

## Numeric and consumer scope

The polynomial describes smooth real arithmetic along the supplied line. The recorded native midpoint uses BF16-rounded input and staged BF16 projection/product/output operations. Their difference must remain in the reported error. Do not promise a polynomial exactly represents those discontinuous rounding boundaries. Keep unrounded write error and the final rounded write/residual error separate.

Use fixed U0 and gM for a common residual-consumer comparison: both predicted and recorded native MLP returns enter the same supplied update. This qualifies a bounded MLP-input intervention and its explicit consumer. It does not claim that an interpolated normalized input arose naturally from U0, or that the resulting exit is an observed full-block midpoint history.

## Minimal frozen assay

Bind the authenticated native collection, all input/parameter recipes and source identities before staging. Keep the midpoint native return and internal traces unavailable to compilation and serving. Only endpoint inputs, original parameter recipes, chosen common parent and modulation may supply the candidate. Native endpoint-linear and copy controls remain separate from the derived polynomial.

One finite guarded CPU mrun job compiles, starts a fresh coefficient-only serving child, seals complete output/write/consumer hashes for α=0,.5,1, then scores against the archived native calls. Do not run another neural evaluation just to reproduce existing truth. Additional doses without native truth may be illustrative outputs only.

Retain all32-row outcomes, absolute and relative RMS against native outputs and native changes, signed curvature residuals, rounding displacement, coefficient dtype/bytes, original logical parameter bytes, compilation wall/CPU/RSS, fresh serving arithmetic and end-to-end wall/RSS. Check the source/parameter identity, shapes, finite values, seal and custody; accuracy remains an observation rather than an abort gate. Run the standard evidence sweep and verify after collection.

This can establish a compact computation only on one supplied input line and context. General prompt operands, adaptive addressing, subsequent state transitions, composition across contexts and a non-neural image decoder remain open. If it works, the next extension must remove another supplied operand or compose another receiving operation rather than merely adding more independent calibration charts.

The seven-corner shape×color×relation assay remains a useful lower-priority interaction-transfer diagnostic. It does not remove an original computation from serving, so it is not the preferred next step for this goal.
