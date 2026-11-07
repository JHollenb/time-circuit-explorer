# Let the current gate select MLP computation

Status: **written and unexecuted**. No sources are staged and no job is submitted for this plan. The [image-key moment result](../image-partition-moments.md) preserves two objects with a small fidelity/work tradeoff, saving 0.455% of cycle69’s nominal total. Further partition refinement is not the next primary. MLP expansion/contraction accounts for 12.872 trillion nominal MACs, approximately 60.95% of that image path; the next experiment targets this operator class with a constructive current-input rule.

## One complete source, one fixed quarter

Use only clock3’s terminal-single19 incoming state from successful `job-34fbcd783e7c`, request `7f545c1670a42e228043d8046fd7c0271e1da9efae1137f880867f44926c1967`. Authenticate its guarded custody, full-byte receipt, candidate seal `71122d424dd57e0d3f306a8ff8c4ec7c9e816bf941358eb60a7c7949cb538457`, original parameters, native source and numerical profile. Inputs are `terminalRows3`, `singleMod3`, `temb3`, `cosine3` and `sine3`. Historical prefix production remains paid. The candidate cannot read `terminalImage3`, `velocity3`, other target outputs or supplied internal gate/up/attention activations before sealing and process exit.

For the actual normalized/modulated image rows X, write

\[
A=\operatorname{SiLU}(XG^T),\quad
Y=[A\odot XU^T]D^T.
\]

Pay every current gate and its original native SiLU. Rank channels using only admitted current gates, current input norms, and source-weight row/column norms:

\[
b_j=\|D_{:j}\|_2\|U_{j:}\|_2
\sqrt{\sum_i A_{ij}^2\|X_i\|_2^2}.
\]

Choose exactly **2,304 of 9,216 channels**, one common support across 1,024 image rows. Break score ties by original channel index, then restore ascending source-channel order for subsequent projections. No full-up calculation, native target, fitted coefficient, support search or budget sweep can influence this choice. Loading original up/down weights and preparing their norms remain paid.

Use the native BF16 admitted gate activation; widen it, normalized X and original norm operands to FP32 for score preparation with autocast/TF32 disabled. Declare actual reduction order, stable descending ranking and ascending source-index gather. Preserve the scores/support and exact physical parameter bytes. Do not use output error to modify support or normalize away the missing channels.

For fixed admitted A and real-arithmetic U/D,

\[
\|Y-Y_S\|_F
\leq\sum_{j\notin S}\|D_{:j}\|_2
\|A_{:j}\odot(XU_{j:}^T)\|_2
\leq\sum_{j\notin S}b_j.
\]

The second inequality applies rowwise Cauchy–Schwarz, then triangle inequality across omitted channels. Selecting the largest exact real b minimizes this particular omitted-sum bound. Operational FP32 scores approximate those values and can reorder near ties; their FP32 sum is a diagnostic estimate, not a certified numerical upper bound without outward error accounting. The real bound can be very loose because it ignores signed cancellation. It is not a quality gate, BF16 error certificate, final-head bound or whole-trajectory claim. Record the residual gate’s amplification separately if reporting an image-state bound.

## Physically omit the work

The native single block has two dense matrices: a fused input map and a fused attention/MLP output map. It does not have six independent MLP matrices. With D=3072 and M=9216, the fused input row intervals are Q `[0:3072]`, K `[3072:6144]`, V `[6144:9216]`, G `[9216:18432]`, U `[18432:27648]`.

Materialize full K/V for all 1,536 incoming rows; Q plus all gate rows for only the 1,024 image rows; selected U rows for those same image inputs; and output columns consisting of all 3,072 attention channels followed by selected MLP channels. Do not compute full U and discard it, or zero channels inside a full-width output GEMM. Keep complete current K/V, native Q/K RMS, absolute image-query RoPE positions, native image attention, residual gate, output stride and original final head. Attention and MLP remain parallel functions of the same incoming normalized state. Text Q/MLP/output stay dead at this last use.

| Computation | Nominal matrix MACs |
|---|---:|
| Full K/V | 28,991,029,248 |
| Image Q plus full G | 38,654,705,664 |
| Selected U | 7,247,757,312 |
| Attention plus selected-MLP output | 16,911,433,728 |
| Full image attention | 9,663,676,416 |
| Original final head | 421,527,552 |
| **Total** | **101,890,129,920** |

This omits **43,486,543,872 MACs** versus the matched 145,376,673,792 terminal/head computation. It is a prospective local account, not a whole-image saving or measured runtime improvement. Separately charge source-norm preparation (56,623,104 coefficient squares plus reductions/roots), all current SiLU, input norms, approximately 9.44 million weighted score accumulations, ranking, gathers, physical copies, transfers, hashing and I/O. Record cold preparation and any reusable norm constants separately; no compilation work disappears from the first-use account.

## Consumer and evidence

One source-bound terminal evaluation produces its own image state, and the original final norm/head consumes that actual state to produce its own velocity. Seal both and exit before opening cycle70’s existing terminal image/velocity outputs. Retain full signed errors, per-row distributions, support, gate/score summaries, actual operator shapes and all target-independent source identities. Changed GEMM widths and splitting G from U introduce finite differences; do not attribute the entire contrast solely to omitted channels.

Before real execution, run a tiny independent full-support reference and selected-channel reference, verify the source intervals and gathered down columns, tie handling and the real-arithmetic bound on supplied toy values. These are mechanics checks, not a demand for native finite parity. The real assay uses one fixed support budget and no new image, reference generation, prefix capture, encoder, decoder, fit or sweep.

Use one guarded strict-preflight mrun CUDA lease, finite timeout, no automatic retry, current admission and a fresh liveness/resource plan grounded in the earlier terminal-only measured history. Do not inherit full-image reservations blindly. Preserve partial progress if mechanics fail; all quality measurements remain observations. Source/custody/shape/finite-value integrity may stop execution; a loose bound or bad output may not erase valid evidence.

Astra reviewed this route after actual cycle70 evidence. The last full newest-post/git/mtime survey and actual artifact cleanup are cycle69, due again72 or earlier material evidence/pressure. This candidate still evaluates original learned maps and consumes a paid historical prefix; arbitrary-prompt zero-neural generation remains open.
