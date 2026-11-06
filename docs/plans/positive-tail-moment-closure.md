# Couple tail mass and value through one positive directional measure

Status: **proposed after cycle17/Astra review; not implemented, staged or measured**. The [fixed-support factorial](../current-support-tail-factorial.md) finds both tail factors material. Value repair improves all 32 paired heads and both receiving endpoints, while mass repair has a slight pooled paired-state advantage. Preserve their cancellation and countertrends. Test one coupled approximation before expanding persistent state.

## One frozen candidate

Keep the same saved prompt pair, ordered current-query top4 support S, complement T, source anchors, routes and exact selected contribution. Under the anchor's normalized tail distribution p_T, define

\[
t_j=\delta^\top(u_j-\mu_T),\qquad
r_2(t)=1+t+\tfrac12t^2=\tfrac12[(t+1)^2+1]>0.
\]

Here μ_T=E_T[u]. The exact centered exponential factors are κ=E_T[eᵗ] and H*=E_T[V eᵗ]/κ. The sole new candidate is

\[
\kappa_2=E_T[r_2(t)],\qquad
H_2=\frac{E_T[V r_2(t)]}{\kappa_2},\qquad
Z_{T,2}=\widetilde Z_T\kappa_2.
\]

Combine Z_T,2 and H_2 with unchanged selected mass/value using positive log-mass aggregation. Direct positive weights are canonical arithmetic. Retain the actual centering defect E_T[t], rather than silently forcing it to zero in the denominator. Use stable scaling/log arithmetic and explicit finite/shape contracts. Empty tails remain exact selected reads and explicitly inapplicable to tail diagnostics.

The equivalent moment form is

\[
H_2=H+
\frac{\operatorname{Cov}_T(V,t)+\tfrac12\operatorname{Cov}_T(V,t^2)}
{\kappa_2},\qquad
\kappa_2=1+E_T[t]+\tfrac12E_T[t^2].
\]

This preserves the original value hull because numerator and partition use the same positive measure. Compute directional moments from saved arrays. Vt² generally requires a value–key–key statistic for arbitrary directions; this protocol does **not** allocate that tensor, fit a low-rank substitute or assume efficient serving follows from the identity. Full current-prefix access and moment/read work remain paid diagnostic costs.

## Diagnostic alternatives without new consumer arms

Record logκ−E_T[t]−Var_T(t)/2 and H*−H−Cov_T(V,t). Subtracting the measured mean separates finite centering defects from higher-cumulant error; retain the mean itself too. The zero-remainder pair is a joint-Gaussian cumulant closure, not a general second-order identity for finite attention tails. Keep signed vectors and errors without clipping a covariance prediction into the value hull. These two diagnostics do not receive a new learned consumer. This clarification comes from the [bounded advisory follow-up](../results/tail-factorial-evidence/advisor-followup.json); it does not change the candidate.

Record exact and quadratic tilted probability mass on t<−1, plus anchor mass there. In that region r₂ grows as t becomes more negative while eᵗ decreases. Preserve the resulting per-head/row countertrends and large-t effects. Quadratic weights also lack e^(a+b)=e^a e^b; positivity alone does not give the exponential's composition law. A poor result constrains this closure, not all moment representations.

## Seal, consume and account

One fresh guarded child forms and seals the sole new complete raw/BF16 bank before exact-tail diagnostic outputs, old control outputs or original consumer parameters are opened. Frozen support input decodes only positions/routes, with complete-byte custody separately retained. After the seal, compute exact/centroid diagnostics and authenticate complete control-bank identities, operand bindings, numerical profile and parameter source before inheriting their receiving evidence. Any control mismatch requires explicit diagnosis and separately counted computation; it must not be hidden by refreshing the candidate.

Evaluate only the new bank through the original output/residual/RMS/SwiGLU/state consumer, once per prompt: two full output and six MLP projections over 512 rows. Retain both endpoints, paired attention-effect denominator, all/active/padded/before8/at8/after8 strata and all 32 heads. Keep signed mass/value error vectors under the same ordered telescope, their paired differences/cross-energy, canonical arithmetic defects and BF16 rounding. Exact mass, value and whole-head comparisons remain diagnostics; no accuracy threshold may stop evidence collection or promote a universal claim.

Tiny mechanics checks should compare direct positive weighting with independent probability mixtures/moment identities, validate empty tails and extreme logs, preserve support/order, enforce candidate sealing and verify decoded paired consumer arithmetic. One finite CPU1 NumPy FP64 mrun lease near immutable operands, default admission, strict preflight, 600-second timeout, no automatic retries or duplicate full suite. Report child wall/CPU/RSS, parameter reads, actual new consumer count, contraction geometry, original reference work and retained/public bytes separately. Run and verify the Saturn observation sweep.

This is a retrospective test of a candidate tail law. It does not validate unseen prompts, compact serving, full contextual own-state closure or zero-neural images. A useful result would justify investigating a reusable representation of these directional moments; it would not establish that representation by itself.
