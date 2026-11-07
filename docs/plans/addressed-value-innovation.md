# Produce an attention innovation from a frozen value-address swap

Status: **executed**, job-b187c2bccbdd. The [result](../addressed-value-innovation.md) retains the complete own candidate, post-seal native reference, heterogeneous responses, actual work and preserved engineering failures. The [pre-execution plan](../results/value-innovation/frozen-plan.md) remains immutable. [Astra50](../results/normalization-innovation-evidence/advisor.json) selected this cut after the [shared normalization result](../shared-normalization-innovation.md). It removes a supplied B carrier and the full-grid innovation projection while retaining original A baselines and learned maps.

## Fix the event before observing its result

Use natural R0A, s60672.base.canonical, at clock 3 and first joint image block of FLUX.2 Klein 4B. This is the source-native donor A specimen, rather than the hybrid R1 receiver. Freeze image rows 264 and 792 as j/k before the new run. Record actual image IDs and their indices after the 512 text keys in the concatenated 1,536-key bank. Swap the complete V rows across all 24 heads; leave Q, K, text values, masks, rotary positions and clock unchanged.

This is a declared intervention on value/address association. It does not claim to be a natural prompt edit or a semantically identified object move.

The source-native archives retain H0 and output fields, not Q/K/V or attention probabilities. Acquire the missing operands freshly from authenticated conditioning, latent, IDs and clock inputs. Expose a source-qualified boundary before SDPA: the old source_native_reader.form calls SDPA unconditionally, so using it unchanged would incur an extra attention call. Freeze new acquisition code without modifying previously shipped sources. Charge the seven original affines, three head norms, modulation, rotary work and transfers. Bind model/source/profile identities and compare acquired input joins with their archived source.

## Factor the actual value event

For each head h, compute normalized probabilities over **all 1,536 keys** from the freshly acquired, actual rotated Q/K. Use declared CPU FP64 score/reduction arithmetic and the qualified 1/sqrt(128) scale, retaining the two required probability columns. A two-key softmax is a different operation.

\[
\gamma_{ih}=p^h_{ij}-p^h_{ik},\qquad d_h=v^h_k-v^h_j,
\qquad \Delta H_i^h=\gamma_{ih}d_h.
\]

The real-arithmetic innovation has rank at most 24 by construction. No SVD, fitted rank or measured target response chooses these directions. The native finite attention policy need not equal this FP64 law.

Push each 128-wide direction through its own physical head-column block of the original output matrix. Do not zero-pad each head to 3,072 and charge or execute a full matrix product.

\[
b_h=g_{\rm attn}\odot W_o^{(h)}d_h,\qquad
t_h=W_{\rm in}[d_{\rm MLP}\odot Cb_h],
\]
\[
\delta a_i=\sum_{h=1}^{24}\gamma_{ih}b_h,\qquad
J_i=\sum_{h=1}^{24}\gamma_{ih}t_h.
\]

C centers channels and dMLP is the retained actual BF16 one-plus-scale widened to FP64. Use FP64 for the factor projections and assemblies. The two small contractions assemble the complete candidate fields; they are not free because the factors have low rank.

With Q denoting the declared FP64→FP32→BF16 conversion, commit the candidate carrier and its own residual:

\[
\widehat a_B=Q(a_A+\delta a),\qquad
\widehat u_B=\operatorname{BF16}(E+\widehat a_B).
\]

From actual committed uA and candidate uB calculate FP64 standard deviations with epsilon 10⁻⁶. Reuse cycle 50's authenticated beta row and charge its historical compilation:

\[
\widehat F_B
=\frac{s_A}{\widehat s_B}(F_A-\beta)+\beta+\frac{J}{\widehat s_B}.
\]

Convert the fused prediction with Q, then run one native CUDA BF16 SwiGLU/down, clock gate and residual commit. The candidate sees no new native B carrier, fused operand, product, write or exit before the complete candidate is sealed and its process exits.

## Keep finite boundaries and baseline inheritance visible

The factor law is an exact real-arithmetic value-swap identity. It rearranges native attention, output projection, gate and conversion boundaries. Retain the carrier discrepancy, residual discrepancy and all downstream errors as observations. Do not require candidate B to match the new native B residual.

For the smooth normalized affine map the secant defect extends with rho=(candidate uB−uA)−delta a. That includes carrier conversion and residual commit defects; native attention/Wo mismatch is an additional upstream comparison. Do not label the complete error as rho-only or run extra attribution projections without a separate question.

After sealing, run an unchanged A attention sham through native SDPA, output projection, attention gate and residual addition. Compare its H0, carrier and residual payloads against archived A. Only exact complete consumed-input identity, bound to the same deterministic source-order local MLP implementation and profile, qualifies inherited A consumer outputs. This does not create the independently archived full-native-block-exit parity witness missing from cycle 48.

Then run one new native B local reference from the same acquired Q/K and swapped V: SDPA → output projection → gate/residual → native normalization/modulation → fused gate/up → SwiGLU/down → gate/residual. Stop at this local exit. If the A sham does not close, preserve its discrepancy and execute one additional post-seal A local MLP from its own residual for a usable A/B reference; do not rebuild or select the candidate. Charge that conditional extra work separately. A failed sham is an instrument/inheritance issue, not evidence that the value-event mechanism is absent.

Score carrier, residual, fused halves, product, down, gated write and exit. Keep absolute and native B−A response errors, signed response magnitudes/cosines, all query outcomes and zero-response controls. Weak denominators remain visible. Report the original fixed source-nonlocal mask for comparability, labelled as inherited, together with event-address/distance strata defined before scoring. There is one intervention on one specimen; queries are dependent observations.

## Charge formation, prediction and reference work separately

| Candidate operation | Nominal MACs |
|---|---:|
| Fresh seven-affine Q/K/V prefix | 51,136,954,368 |
| Full QK scores: 24 × 1,024 × 1,536 × 128 | 4,831,838,208 |
| 24 physical Wo head-block vectors: 24 × 128 × 3,072 | 9,437,184 |
| W_in on 24 directions | 1,358,954,496 |
| Assemble delta a: 1,024 × 24 × 3,072 | 75,497,472 |
| Assemble J: 1,024 × 24 × 18,432 | 452,984,832 |
| One native down | 28,991,029,248 |
| Candidate total | **86,856,695,808** |

The last five rows total **30,887,903,232 MACs**. Beta compilation is an additional historical 56,623,104 MACs. Baseline preparation is historical learned work too. Prefix matrices total 142,344,960 BF16 bytes; Wo adds 18,874,368 and W_in/down add 169,869,312, totaling 331,088,640 distinct BF16 parameter bytes for this selected path. FP64 widened copies, probabilities, buffers and validation are additional.

Two post-seal SDPA dispatches contain 19,327,352,832 nominal QK/AV MACs. The A sham Wo costs 9,663,676,416; the B Wo plus two MLP affines costs 96,636,764,160. That normal validation path totals **125,627,793,408 MACs**, and candidate plus validation totals **212,484,489,216**. If the sham requires an A MLP reference, add 86,973,087,744. Native modules may have further uncounted scalar operations, all of which remain paid. Record physical contractions/kernel dispatches, tensor shapes, norms, softmax scalars, transfers and loaded bytes separately from these analytical totals.

## Execute as one small, inspectable transaction

First run tiny independent factorization and conversion checks, including full-key normalization, sign/address handling and physical head-block geometry. Freeze the event, acquisition source and numerical profile before scientific execution. Use Saturn typed parent and candidate Acts with exact actual reads/writes, complete source identities and a seal-before-reference process boundary.

Use one guarded mrun CUDA lease, strict preflight, no automatic retry, and one resident selected parameter set where isolation permits. Request intent-first planning; if the full-alias estimate remains inapplicable, document selected-component geometry and current capacity rather than copying the last reservation blindly. Start with this one event and stop after the local reference and complete evidence capture. Expected work should remain a short assay; if a revised implementation exceeds fifteen minutes, obtain the required Luna critical review before launch.

Implementation clarification before staging: candidate and native reference use separate processes to make the target boundary explicit; each loads only its selected matrices. Candidate formation has 33 physical affine calls, one batched all-head QK contraction and two bank assemblies. Native validation reloads three matrices, so 331,088,640 distinct BF16 parameter bytes do not equal all process-level parameter reads. Restore archived native Q/K/V and Wo input layouts for the sham; keep layout differences visible. Event-near means Chebyshev image-grid distance at most two from either frozen address; all other queries are event-far. Record all four scopes, including the inherited source-nonlocal mask.

The measured cycle 50 process peaked at 2,719.9 MB RSS / 534 MB device usage. This new process adds native prefix parameters, a 75.5 MB FP64 Wo copy and up to about 604 MB of FP64 scores/probabilities, released before subsequent large output work. Use a documented selected-component exception of 6,144 MB RAM / 1,536 MB VRAM, four threads, 6 GB disk, 240 seconds expected and a 900-second hard cap, subject to current capacity preflight. It is not a latency or memory improvement claim.

Development record before the scientific candidate: the first partial stage failed on list-versus-wrapper host inventory before request sealing. The first submitted stage, job-0e973f3b508d, stopped at selected-parameter admission because the manually loaded Wo receipt omitted its header byte offsets. It formed no Q/K/V, candidate or native B reference. Preserve that stage, source and failure unchanged; the corrected primary/driver v2 supplies the same complete header receipt as the generic selected loader. The prediction law, event, precision, resource envelope and consumer remain unchanged. This is an engineering failure with no scientific outcome.

Keep arrays near compute and publish compact immutable receipts, lossless row summaries and one derived figure. Run and verify the Saturn evidence sweep. Do not use an effect-size threshold to suppress the result.

Success would establish a reusable fixed-Q/K value-event producer inside a conditional native system. Arbitrary prompt changes alter Q/K as well, and neither text-branch continuation, later own-state denoising nor general pixel generation is closed here. No zero-neural or end-to-end speedup claim follows from this proposed test.
