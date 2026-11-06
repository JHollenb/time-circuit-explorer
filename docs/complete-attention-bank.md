# A complete predicted attention bank exposes the missing exponential response

All 32 first-layer attention heads now run from raw text without reference-head replacement. Their complete candidate banks feed the original first-block output projection, residual, RMS, MLP and state update on all 512 rows. This removes the largest supplied dependency in the [previous one-head experiment](query-transport-context.md). It also exposes a stronger limitation: broad state reconstruction improves, but the tested prompt change remains inaccurate.

The [frozen protocol](plans/complete-attention-bank.md) tests one fresh pair: an orange paper crane stands **behind** versus **before** a silver key on a dark wooden table. Both strings have 27 active tokens and 485 padding positions. The complete forecasts seal before original parameter reads or exact contextual references. All compiled canonical operands match. The exact references use shared FP64/BF16 mathematical equations; native CUDA/framework parity is unmeasured.

This is a reduced attention representation plus an original learned consumer, not a reduced complete encoder or an arbitrary-prompt zero-neural image generator.

## Combine the errors at their actual consumer

Compilation preserves the existing source-only rule: 2,048 token IDs across 512 positions form eight KV-group covariances; 4,096 lexical query candidates plus all 512 padding positions select 64 landmarks separately for each query head. GQA shares K/V operands and source covariance work, not the distinct landmark moment banks. Nearest and transported readers advance their actual causal prefix state without clipping, target fitting or exact fallback.

The transported law remains

\[
\widehat h(x)=E_\ell[v]+\operatorname{Cov}_\ell(v,y)(x-\ell),
\qquad x=q/d^{1/4},\;y=k/d^{1/4}.
\]

Both complete candidate banks are BF16-rounded, concatenated and consumed through the full original W_o. Every arm recomputes residual, normalization and MLP from its own incoming bank. Exact attention and attention-deleted banks are separately paid diagnostic arms.

Primary consumer scores compare the paired change in F_pred−F_zero with the paired change in F_exact−F_zero. This keeps the attention contribution distinct from the embedding-only baseline. All rows, including padding, remain scored.

| Paired computation | Nearest error | Transport error |
|---|---:|---:|
| Complete rounded head bank | 99.24% | 105.06% |
| Output projection | 98.47% | 99.64% |
| Residual | 98.49% | 99.69% |
| Post-attention RMS | 95.36% | 89.19% |
| MLP write | 128.45% | 120.77% |
| Final first-block state | 125.11% | 116.03% |

The state improvement is real within this arithmetic and specimen, but the final paired error remains larger than the target change. Active-row final-state error improves 80.03%→72.10%; padding improves 128.54%→119.34%. The corresponding complete-head active error worsens 103.81%→135.31%. Consumers reorder errors; neither a head-only score nor a broad endpoint score can stand in for the demanded future response.

Broad final-state attention-effect errors improve from about 50% to 40% for both endpoints. Paired state cosine improves .435→.567. A target-assisted amplitude-only diagnosis would still leave 82.39% relative response error for transport, so one global gain cannot close the direction gap. That oracle scaling is not a serving coefficient.

Nineteen of 32 individual all-query projected head errors improve, while pooled head error worsens. Heads 19,28,31 carry 54.18% of transported projected paired error energy. Net cross-head cancellation explains little: sum of individual error energies divided by total projected error energy is .9993 for transport. Its signed padding cross-energy is negative but small. The [per-head record](results/complete-attention-bank/linear-head-effects.json) preserves signs, regions and every head, rather than declaring the whole mechanism present or absent from one score.

Signed extrapolation is stronger than in head0: value-hull excursions reach .99785, selected negative effective mass 1.70344 and selected displacement variance 21.9141. These are observations about this approximation, not proof of the source of its error. A second experiment measures that source directly.

## Factor the exact exponential remainder

For the same frozen anchor and actual prefix, define

\[
H=E_p[v],\quad \mu=E_p[y],\quad
t=(x-\ell)^T(y-\mu),\quad A=E_p[(v-H)t].
\]

Since E_p[t]=0, the omitted terms can be written

\[
R=E_p[e^t-1-t],\qquad
B=E_p[(v-H)(e^t-1-t)].
\]

The exact read and first-order error are

\[
h=H+\frac{A+B}{1+R},\qquad
\widehat h-h=
\underbrace{\frac{AR}{1+R}}_{P}
+\underbrace{-\frac{B}{1+R}}_{V}.
\]

P is the missing partition correction and V the missing value-weighted remainder. Both depend on the current query and actual memory. The [saved-only diagnostic](../generator/attention_remainder.py) evaluates them on every existing query/head row, without a new prediction or fit. It uses a log-sum-exp tilt and normalized remainder coefficients to avoid overflow, with expm1 near zero. Per-address coefficients remain retained near compute.

BF16 introduces a separate term Q:

\[
\operatorname{BF16}(\widehat h)-\operatorname{BF16}(h)
=P+V+
\underbrace{[\operatorname{BF16}(\widehat h)-\widehat h]
-[\operatorname{BF16}(h)-h]}_Q.
\]

The complete telescope closes to 5.33e−15 per endpoint, exact exponential reads to 8.89e−16, and all BF16 exact reference words agree. Numerical mechanics do not explain the large approximation error.

For the **paired** full-bank response, normalize energies by the exact rounded target-change energy:

| Term | Normalized energy |
|---|---:|
| Partition ΔP | 1.150080 |
| Value remainder ΔV | .780720 |
| Signed partition/value cross term | −.827484 |
| Rounding ΔQ | .000648 |
| Signed smooth/rounding cross term | −.000270 |
| Total error | 1.103694 |

The terms sum to the measured squared relative error, 105.06%². This is substantial **within-head remainder cancellation**, distinct from the small cross-head cancellation measured above. A variance or negative-mass correlation alone would miss it.

Removing P as an oracle vector diagnostic reduces all-query head error to 88.39%, active error to 95.96% and padding to 87.89%. Removing V alone instead gives 107.26% overall error. These corrections use actual exact exponential remainders and keep the original rounding defect fixed; they are not recomputed serving candidates or new consumer runs. Head19 is a countertrend: removing P worsens its error 110.02%→114.50%, while heads28 and31 improve. The partition matters, but treating it independently loses meaningful joint behavior.

## One changed address, unchanged later queries

The archived interfaces differ only at token position8, ID4815 (behind)→1573 (before). Q, K and V each change only at that row; masks agree. Every later query and source-nearest route is identical across the pair. Before position8, exact and predicted changes are exactly zero. Position8 is scored separately because its query changes as well.

For the 503 later rows, an exact one-address update separates key addressing and value payload. If p_a is baseline probability at address8 and r=exp(xᵀ(y_b−y_a)),

\[
\Delta h=
\frac{p_a(r-1)(v_a-h_A)+p_a r(v_b-v_a)}
{1+p_a(r-1)}.
\]

The implementation evaluates an equivalent stable form using the normalized unchanged remainder bank. All 32 heads close this identity on all 503 applicable rows to 7.59e−16. This is a baseline-assisted diagnostic, not a compact predictor or semantic decomposition. It demonstrates why later-row errors cannot be attributed to a between-prompt query-route jump. Their read must respond correctly to the changed K/V address.

After position8, transported paired head error is still 104.54%; deleting P diagnostically gives 88.43%. Both exponential terms must be understood as paired vectors at this changed memory interface.

## Compute, custody and the next constraint

| Scope | Measured cost |
|---|---:|
| Complete-bank lease | 74.93 s; 1,293.2 MB peak sampled process-tree RSS; 0 VRAM |
| Source compiler | 37.65 s, including 24.68 s gathering 35.67 MB of learned atoms |
| Eight covariance computations | 137,438,953,472 MACs |
| Fresh serving, two prompts | 9.208 s after imports; 304,596 KiB reported high-water RSS |
| Sum of per-head stream timings | 1.198 s and 1.147 s per prompt |
| Four original consumer arms | 4.839 s and 4.652 s per prompt |
| Saved-only remainder lease | 8.68 s; 689.3 MB peak sampled process-tree RSS; 0 VRAM |

Serving timings charge raw prefix work, both readers, final-state serialization and diagnostic traces; there are no repeated standalone latency or native speedup claims. Original reference work includes 64 exact heads, 8 full output projections and 24 MLP dense projections on all 512 rows per arm. Original parameters are read only after the complete forecast seal.

Logical core moment state is **272,663,040 bytes**, with sequential head execution lowering residency. Runtime source assets add 3,153,920 bytes; the external lexical table remains 2.645 GB. Two prompts retain 545,628,288 bytes of final-state NPZ files near compute, plus banks, ports and references.

For comparison, this model's complete BF16 K/V payload at 512 positions and eight groups is **2,097,152 bytes**. The moment core is about 130 times that payload. This is a storage comparison, not a measured total-runtime benchmark. The experiment has established a broader computational test, not an attention-efficiency advantage. Blindly adding higher tensor order would tighten the wrong constraint before the joint remainder is understood.

Independent byte checks cover 80 files/5,735,295,635 bytes for the complete bank, including the 4,967,215,360-byte original checkpoint shard, and 15 files/516,947,089 bytes for the saved-only diagnostic. Parent bytes overlap. Compact publications total 7,835,794 bytes. All 57 generator mechanics tests pass; both [Saturn run/verify sweeps](results/complete-attention-bank-evidence/sweep.json) return 0, with 15,360 and 32,768 observation records, no watch fires and `target-streams-only` scope. No terminal-claim receipts or physical address joins are harvested.

The [complete-bank provenance](results/complete-attention-bank/provenance.json) and [remainder provenance](results/attention-remainder/provenance.json) preserve guarded request/payload identities, compact raw reports and lossless observation streams. Full tensors stay near compute. Astra recommends the [sparse joint-tilt and append-only-tail test](plans/sparse-joint-tilt.md) next, removing the covariance tensor while preserving a coupled positive numerator/partition update. This is proposed, not measured. The active [research record](zero-neural-research.md) and living notebook keep later contextual layers, original MLP replacement, diffusion, general decoding and arbitrary-prompt image quality open.
