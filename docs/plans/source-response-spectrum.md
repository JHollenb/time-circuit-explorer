# Is the weak source fit attenuation or unresolved response structure?

Status: **executed and completed**, job job-7a556e85153a. The [result](../source-response-spectrum.md) preserves the [original frozen protocol](../results/source-spectrum/frozen-plan.md). Selected after the [response-only writer result](../source-native-response-writer.md) and [Astra42 review](../results/source-response-evidence/advisor.json). This is one saved-array diagnostic of the existing program, not a new predictor or image campaign.

Freeze the actual discovery X,Y, inherited scale and lambda=.01. There are1,888rows and3,072features. The source response error remains about82%, so interpreting the poor feedback solely as transfer is premature. A fixed ridge penalty can suppress directions that the matrix represents; numerical rank alone does not tell us which explanation matters.

## One spectrum and one existing prediction

Compute a thin FP64 SVD X=UΣVᵀ once. With tau=N lambda and C=UᵀY, the existing ridge prediction and source residual satisfy

\[
\widehat Y=U\operatorname{diag}
\left(\frac{\sigma_i^2}{\sigma_i^2+\tau}\right)C,
\]

\[
\|Y-\widehat Y\|_F^2=
\|Y-UU^\top Y\|_F^2+
\sum_i\left(\frac{\tau}{\sigma_i^2+\tau}\right)^2\|C_i\|^2.
\]

Reconstruct the already-sealed ridge prediction and retain its complete discrepancy from the original dual solve as a mechanics/numerical observation. A second solver can differ finitely; do not invent an ultra-tight cross-LAPACK equality gate. Bind input identity, dimensions, finite values and original output lineage exactly.

Report every singular value, coefficient filter, target energy ||C_i||², residual contribution and the pseudoinverse amplification diagnostic ||C_i||²/sigma_i². Exact zero denominators are undefined, not silently clipped. Retain full FP64 values and finite status rather than selecting modes. Include actual ridge coefficient amplification with sigma_i²/(sigma_i²+tau)² as a separate quantity.

Because thin U is square when N<d, its full orthogonal complement can be numerically empty even for rank-deficient X. A near-zero Y−UUᵀY is therefore not evidence of informative source span. Numerical-rank ambiguity must stay explicit. Retain the complete spectrum and cumulative target/residual energy against singular scale. If resolution-defined summaries are shown, label unresolved energy and its finite ridge-filter correction separately; do not turn one threshold into an irreducible scientific floor.

Nearly exact source interpolation can also be trivial at1,888<3,072. The diagnostic establishes neither useful unseen response information nor an economical decoder. It may reveal that the declared ridge suppresses source target directions, or that substantial response energy lies in numerically unresolved source directions. Either is a constraint on interpreting this fixed writer.

## Bound the work

Use one finite CPU mrun analysis lease with a model-free scheduling contract, no retries and no original parameters/frameworks in numerical execution. The original learned H0/label production and fitted response products remain preparation. No feedback/evaluation target, new predictor, gain/regularization/rank sweep, pseudoinverse serving, new H0/model/VAE/image or noise computation is allowed.

Use a tiny independent known-spectrum/ridge-residual oracle first. Measure SVD, projection, reconstruction and retained-array CPU/wall/RSS separately, including source/data I/O. Prefer intent-first admission and the measured compatible response job as resource history. Stop for integrity, finite, instrument or resource mechanics; retain all scientific magnitudes and countertrends without quality gates.

The result should determine whether it is informative to pay for a deeper receiver-conditioned producer. It must not select an alternative on historical held labels or present a better source fit as arbitrary-prompt zero-neural generation.
