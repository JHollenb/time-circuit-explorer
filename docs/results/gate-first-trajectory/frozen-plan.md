# Let gate-first MLP selection change the whole trajectory

Status: **written and unexecuted**. No sources are staged and no job is submitted for this plan. [Cycle71](../gate-first-terminal-mlp.md) reduces one terminal/head computation by 29.91%, with 27.80% parent-write and 9.50% own-head velocity error. It supplies a constructive current-input selector, not evidence that errors stay small through many changed states. The next primary is one full composition test across all 20 single-block MLPs and four own sampler clocks.

## Freeze the same operator, widen its composition horizon

Keep cycle70’s original joint blocks, clock preparation/schedule, supplied original E/noise/IDs, current-key moment feedback, final head and VAE unchanged. Use the same source-defined gate-first score and fixed budget from71:

\[
A=\operatorname{SiLU}(XG^T),\qquad
b_j=\|D_{:j}\|_2\|U_{j:}\|_2
\sqrt{\sum_iA_{ij}^2\|X_i\|_2^2}.
\]

Admit native BF16 full gates; use FP32 source norms/current-input scores with autocast and TF32 disabled. Stable descending score order breaks ties by source index; restore ascending support before gathers. Always choose 2,304 of 9,216 channels, with no reweighting, renormalization, threshold, fit, budget sweep or dense-U fallback. The rounded score sum remains a diagnostic estimate rather than a certified numerical bound.

For single0–18, normalize/modulate the complete current concatenation of **512 text and 1,024 image rows**. Compute full current Q/K/V/G on all 1,536 rows. Select one common support from all those rows without balancing text and image contributions. Physically project only selected U rows and contract only selected MLP output columns, while retaining all attention output columns. Both emitted text and image states become the next block’s inputs.

The original fused input map’s Q/K/V/G rows form its first 18,432 rows, a six-D map; the selected U rows come from its remaining U bank. Use that physically reduced QKVG projection rather than full QKVGU followed by discarded U. Attention and MLP remain parallel functions of the same incoming normalized state. Q/K retain native head RMS and absolute RoPE. Image attention reads complete current text/image K/V; text attention retains cycle70’s own moment-produced alpha times conditional TT read before BF16 admission.

At terminal19, keep all 1,536 K/V rows, but Q/G, support formation, selected U and output computation use only the **1,024 live image rows**, as in71. Last-use text Q/MLP/output remain physically absent. The original final head consumes the actual emitted image state.

Recompute support at every block and clock: **80 selections**. Source U-row/down-column norms may be prepared once for each of the 20 blocks, with preparation and 1,474,560 bytes of FP32 norm vectors explicitly charged. Use ephemeral current-block physical maps. Retain scores, supports, actual map-byte digests, shapes and counts without holding 80 map sets or saving every full gate activation. Do not treat logical materialization bytes as simultaneous GPU residency.

## One actual own-state image

Authenticate cycle70’s original source E/noise/schedule and the existing case-A native comparator `job-394733a01129`; retain cycle70 `job-34fbcd783e7c` as a prior candidate comparator. Revalidate authoritative guarded custody, source/checkpoint/framework/numerical identity, existing complete-byte receipts and immutable source pins before staging. Historical native conditioning/noise and reference generation remain paid.

Start from the exact initial latent and supplied E. Every block consumes its own preceding text/image state; each original sampler commit consumes its own velocity and latent. No supplied hidden-state refresh, native shadow, target-conditioned support or reference regeneration. Decode the actual final own latent with one original VAE. Seal the entire candidate, actual PNG and all work records, then exit before a fresh comparison child opens existing native/cycle70 targets.

Retain per-clock velocity/latent/own-write discrepancies, complete signed rows and the actual PNG. Capture text/image trajectories and per-head alpha/moment diagnostics, plus source-index support and score distributions for all 80 selections. Support overlap/change across clocks and depth is descriptive; it is not a quality gate or proof of native sparse circuitry. An adverse image remains evidence about this fixed operator composition rather than a general exclusion of sparse MLP computation.

## Prospective work, with extra overhead visible

For N live MLP rows, the original G/U/down work is 3NDM. Full G plus quarter U/down costs 1.5NDM. Across \(76\times1536+4\times1024=120{,}832\) row-visits:

| Quantity | Prospective matrix MACs |
|---|---:|
| Full current G | 3,420,941,451,264 |
| Selected U | 855,235,362,816 |
| Selected down columns | 855,235,362,816 |
| Saving versus70 | **5,131,412,176,896** |
| Resulting whole path | **15,989,289,779,200** |

This is a prospective **24.2957% nominal whole-path reduction**, not a runtime prediction. With fused QKVG for live singles, expect 80 additional denoiser affine calls: 520 rather than 440. Native image attention and live text-moment product counts remain 100 and 480. Changed GEMM geometry remains combined with omitted channels.

Separately pay source-norm preparation: 1,132,462,080 coefficient squares and 368,640 roots if prepared once. Also pay 1,113,587,712 current gate activations, input norms, score squares/products/reductions, 80 sorts, gathers, physical copies, transfers, hashing and I/O. Cold first-use preparation and any reusable constants must be reported separately; a lower MAC count alone does not make the full instrument faster. Cycle71’s cold selection/preparation timings motivate measuring this overhead rather than assuming it disappears.

Run tiny complete/selected references for the all-row single-block path, independent two-stream state handoffs and terminal last-use path before the model run. No pretrained model is needed for these mechanics. Do not demand native finite parity from changed contractions or use an arbitrary image/activation threshold to stop discovery.

Use one guarded strict-preflight CUDA-requested mrun lease, finite timeout, no automatic retries or duplicate reference suite. Derive a fresh full-model liveness/resource plan from cycle70’s 8,308.8 MB RSS /9,476 MB device history plus norm/selection workspace and compact diagnostics; do not reuse the terminal-only reservation. Inspect live fleet state, queue note and actual progress. Preserve partial outputs if mechanics fail, and poll the same job handle until authoritative termination.

Astra recommends this route after reviewing actual71. Full newest-post/git/mtime synthesis and artifact maintenance are due by cycle72 or earlier material evidence/pressure; perform them before committing to the next large candidate. All original learned gate/key maps, joint MLPs, historical encoder-derived E, head and decoder remain paid. This tests reusable operator composition, while arbitrary-prompt zero-neural generation remains open.
