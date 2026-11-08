# Construct a baseline and response through one finite writer sweep

Execution contract for the [selected writer family](joint-baseline-response-writer.md), following the [normalization audit](../coupled-normalization.md). Before first import, the implementation, this contract and its twelve supplied mechanics require a source review by Astra. Executed source, test and contract bytes remain immutable. The question is whether changing both small write dictionaries and their routes can improve the complete finite state and its original normalized operand together. This is an exploratory local H1 construction, with no image or general prompt capability claim.

## Input and serving boundary

Use the authenticated cycle104 accepted program, its complete-vocabulary lexical bundle, original raw prompt strings, native H0/H1 descriptors and numerical profile. Authenticate the cycle109 publication and its request member before taking the gamma/shard descriptors from them. The family plan SHA256 is `abef41111855f8c620b5d3ab40ea7395ccbfc5d32e3e48d25a5922d04edb31b5`.

Discovery histories are A, B, 00 and 10, in that order. The two discovery edit pairs are A→B and 00→10. Known transfer is 01/11; those histories are already observed and are not newly held out. The seventh raw prompt is novel and unscored. Transfer numeric target fields stay quarantined until final fresh serving seals. No new prompt search, seed selection or image follows this assay.

Keep the current two-slot branch/copy/add/BF16 grammar: depth≤4, leaves≤16 per slot, minimum8 discovery rows per fitted leaf,16 sampled thresholds per feature and at most64 discovery literal features. The existing40 literals produce155 features: current position/mask, causal counts, literal history information and selected current/causal lexical coordinates. No case identity, future token, target H1/Z, native radius or whole-prompt output cache is a serving feature. The fixed first two prefix positions remain historically compiled constants. Each learned row is rebuilt from its own lexical H0 through **both** finite slot additions, even when only one slot changes.

For each addition, sum in FP64, admit to FP32 round-to-nearest-even, then admit to BF16 round-to-nearest-even. Never evaluate a slot0 change as `old H1 + change`: the intervening admission matters. A fresh standard-library child tokenizes raw text, gathers its own lexical rows and runs the resulting program. The trusted-source guard blocks NumPy/Torch imports and native/old prediction fields after tokenizer warmup; it is not an adversarial sandbox.

## Four finite objectives

Stack learned positions2…511 from the four histories, giving2040 rows. Endpoint weights give each history×learned-active/padding stratum mass1/8. Pair only unchanged-token learned rows, with each pair×active/padding stratum mass1/4. Edges are disjoint and oriented j−i. Preserve all endpoint and response measurements, rather than requiring each row to improve.

Let T and ZT be the native discovery H1 and its source-policy normalized operand; let H and Z be an actual candidate's outputs. Freeze native energies EH, PH, EZ and PZ using the same weights. If a native energy is zero, its denominator is1 and its term is an absolute squared loss. The continuation score is

\[
L=\tfrac14\left[
\frac{\sum_iw_i\|H_i-T_i\|^2}{E_H}+
\frac{\sum_{(i,j)}v_{ij}\|(H_j-H_i)-(T_j-T_i)\|^2}{P_H}+
\frac{\sum_iw_i\|Z_i-ZT_i\|^2}{E_Z}+
\frac{\sum_{(i,j)}v_{ij}\|(Z_j-Z_i)-(ZT_j-ZT_i)\|^2}{P_Z}
\right].
\]

The displayed divisions use the zero-reference convention above. Compute pair errors as written, rather than reassociating differences of endpoint errors. Score the complete finite state, not a Taylor prediction.

Original Z evaluation uses contiguous CUDA BF16 `[1,512,2560]`, FP32 square/mean, epsilon1e−6, rsqrt/rescale, BF16 admission and BF16 gamma multiplication, under the unchanged Torch2.13.0+cu130 deterministic profile. Hash the complete first4,967,215,360-byte parameter shard and header, then materialize only the2560-element/5120-byte gamma. No Q/K/V matrix is loaded. These original normalization/gamma evaluations are paid offline computation.

## Proposal algebra and paired solve

Use an ideal FP64 pullback to propose changes:

\[
J(h)^Tu=\frac{\gamma\odot u}{s}-h\frac{\sum_kh_k\gamma_ku_k}{d s^3},\qquad s^2=d^{-1}\sum_kh_k^2+\epsilon.
\]

Apply endpoint and pair graph weights to the actual H/Z errors, obtaining gH and gZ. Set g=gH+JᵀgZ. Omit the same common objective factor in g and A. This is neither a derivative of the finite admissions nor an exact Hessian or descent guarantee.

Set the heuristic isotropic scale ji=RMS(γ)/si and endpoint coefficient ei=wi/EH+wi ji²/EZ. For each edge, ph=v/PH and pz=v/PZ. Solve the two-row system

\[
A=\operatorname{diag}(e_i,e_j)+p_h(1,-1)^T(1,-1)
+p_z(j_i,-j_j)^T(j_i,-j_j),\qquad A\Delta=-g.
\]

Positive endpoint weights make this matrix positive definite. Scale coefficients before the solve and evaluate its determinant through the positive expansion

\[
e_ie_j+e_i(p_h+p_zj_j^2)+e_j(p_h+p_zj_i^2)+p_hp_z(j_i-j_j)^2.
\]

This preserves the common baseline mode that a diagonal preconditioner could suppress under a strong difference penalty. The H pair term cancels on equal updates; the Z term need not cancel when ji≠jj. Measure solve residuals on both paired and singleton rows as diagnostics, without a quality gate.

## One sequential sweep and exact restoration

Start with the exact104 parent. First construct slot0, then slot1, recomputing the second proposal from the actually selected first-slot parent. For each slot, fit one new depth-bounded route to `oldSlot + Δ` using the reviewed contrast tree fitter and its H endpoint/pair graph metric. This is an H-metric projection, not minimization of the full paired H/Z surrogate.

On that single proposed route, reuse one normal matrix/Cholesky and fit two right-hand sides: `oldSlot + Δ` and `oldSlot + .5Δ`. Admit each dictionary to FP32. Half dose does **not** halve the whole dictionary. Zero dose on a new route need not reconstruct the old program, so retain the exact parent as a separate candidate. Both full finite admissions and original Z are then evaluated on every discovery history. Select the minimum finite four-term score, with the parent first on ties. Copy the complete selected bundle and retain its exact Saturn program Frame. There is one sweep, two tree-fit attempts and four attempted futures, with no automatic extra dose, round, larger dictionary, QKV test or image.

Candidate-only numerical overflow is local to that future. A nonfinite dictionary admission records `numerically-inadmissible`, null score/hash and no fabricated program. If the frozen fitter raises its exact `Nonfinite FP32 atom admission`, both doses are `construction-incomplete`: retain gradient/step, do not invent a route. A state-addition overflow records the program, null score, completed history prefix and any already paid normalization calls. The other dose and exact parent remain eligible. Only a complete four-history finite score is selectable. Shared schema/custody/mutation errors, invalid instruments and unsafe device failures stop execution. No broad `ValueError` catch turns a shared failure into a losing candidate.

Finally, run a second fresh child serving parent+accepted on all seven raw prompts. Full-byte discovery H equality binds the actual serving implementation to the offline evaluated state before reusing its Z. Seal this child, then open known transfer targets and evaluate original Z for native, parent and accepted. Novel H1 remains an unscored output, not evidence of faithful image generation.

## Physical work and evidence

The fully valid path pays4 native discovery +4 parent discovery +16 candidate discovery +6 transfer normalization units=30. There are two CPU proposal Acts, so the valid path has32 registry executions. Partial futures reduce actual completed units; retain every paid partial call. There are zero affine MACs, QKV matrices, full encoder calls, denoisers, VAE calls or images in this assay. The two serving children execute inside one lease and make zero neural evaluations. Each program/case performs2,611,200 coordinate additions and at most4080 branch decisions, plus features, tokenization, lexical reads and serialization.

Retain all numerical proposal inputs, including H/Z/targets, old slot values, features, weights/edges, gamma and frozen energy vector, and gradient/step outputs. On the fully valid path this is87 fields:60 full H/Z BF16 fields, gamma and26 proposal fields, roughly0.75GB. All stay on Beast. Saturn input/output Frames, Act specifications, actual before/after input hashes, candidate SHA or explicit null, program ancestry, restoration bundles, fresh-child receipts and seals remain compact public evidence. This does not claim generic process isolation for the offline fitter's source-bound recipe closure.

Publish18,432 dependent scalar rows:12,288 endpoint rows and6144 pair rows;96 pooled endpoint and60 pair groups. Endpoints use actual native endpoints as their reference, not the older increment denominator. Independently rebuild every pooled scalar and discovery parent/accepted objective from raw scalar RMS metadata. Other candidate score arithmetic is rebuilt from primitive energies. Norms, gradients, tree fitting and vector computation remain source-qualified producer measurements; byte checking is not an independent numerical replay. Fresh full remote hashes cover arrays and NPY payloads, original shard, lexical/program assets, restored programs and fresh serving outputs. Do not hydrate arrays locally.

One intent-first, strict-preflight mrun CUDA lease on Beast uses numerical CPU threads1 and a900-second cap, with no retries. The source's seven logical operations group initialization, gamma loading, baseline scoring, two slot proposals, final serving and transfer scoring; physical call counts above remain authoritative. Historical104 fitting took18.47seconds for one tree; two fits plus guarded serving are expected below10minutes, without a runtime guarantee. Inspect the live handle/queue/logs and never duplicate a job merely because observation times out.

Run the standard Saturn evidence sweep and verification after collection; unsupported compressed/Frame–Act instruments remain instrument limitations, separately from the source/byte/scalar audit. Full research/storage108 remains current; next full review111 or earlier material evidence/storage pressure. Preserve rejected futures and historical artifacts. Arbitrary raw prompt to faithful image with zero neural serving evaluations remains the active goal.
