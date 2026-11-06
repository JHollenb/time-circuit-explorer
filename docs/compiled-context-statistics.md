# A small context state can be correct and still read the wrong token

The first raw-text streaming-context experiment is implemented, but its tested approximation is inaccurate. A 66,576-byte rank-64 state reproduces its own finite-feature attention kernel to about **7 × 10⁻¹⁷**. Its median whole-head relative RMS error against exact mathematical attention is **183.3%**, versus **97.8%** for a uniform read. This is evidence against the tested feature construction on this panel, not against compact contextual state generally. Arbitrary-prompt zero-neural image generation remains open.

The [complete lexical compiler](raw-prompt-encoder-atoms.md) supplies first-layer operands from new raw strings. This experiment asks whether context can then advance through reusable statistics rather than an explicit query/key-pair matrix. It follows the [frozen plan](plans/compiled-context-statistics.md), with a separate mechanics smoke followed by fourteen frozen strings. No accuracy threshold suppresses observations.

## The computation and its state

For positioned BF16 operands of Qwen3 query head0 and shared KV head0, let d=128, x=q/d^(1/4), y=k/d^(1/4). Exact causal attention is

\[
h_i=\frac{\sum_{j\le i}m_j e^{x_i^\top y_j}v_j}
{\sum_{j\le i}m_j e^{x_i^\top y_j}}.
\]

Fix nested ranks16/32/64 of one PCG64(seed6306) standard-normal array. The positive features are

\[
\phi_a(x)=r^{-1/2}e^{\omega_a^\top x-\|x\|^2/2},\qquad
\psi_a(y)=r^{-1/2}e^{\omega_a^\top y-\|y\|^2/2}.
\]

Their inner product estimates the unnormalized exponential kernel. The normalized attention ratio is not thereby unbiased. The serving state advances as

\[
z_i=z_{i-1}+m_i\psi(y_i),\quad
M_i=M_{i-1}+m_i\psi(y_i)v_i^\top,\quad
\widehat h_i=\frac{\phi(x_i)^\top M_i}{\phi(x_i)^\top z_i}.
\]

Implementation stores per-feature log maxima and rescales previous z/M when a new key increases a maximum. Reads subtract a common maximum across features. The candidate has no exact-attention or native fallback. An empty allowed prefix emits zero and records that condition. With a nonempty prefix, positivity makes the output a convex combination of allowed values; the measured range violations are only floating-point roundoff. This amplitude property does not establish reader fidelity.

Raw strings use the actual message template, tokenizer, truncation/right-padding to512, and the existing learned-output atom table. Selected Q/K operands apply the qualified eager BF16 half-RoPE arithmetic. All512 query rows, including padding queries, remain present. Canonical full-prefix assembly independently verifies every selected operand with zero word mismatches. Tiny synthetic checks also verify GQA mapping and exact continuation from the candidate's own saved state.

## Full observations

Relative RMS is RMS(candidate−reference)/RMS(reference). Each table cell is the median of per-prompt ratios over14 dependent sequences, or12 sequences for the padding region because two strings occupy all512 tokens. It is not a confidence interval or a semantic-image score.

| Method | All queries | Active queries | Padding queries | Ideal output-slice error, all queries |
|---|---:|---:|---:|---:|
| Rank16 | 180.5% | 130.8% | 188.5% | 164.0% |
| Rank32 | 180.5% | 132.7% | 188.5% | 164.5% |
| Rank64 | 183.3% | 134.3% | 189.9% | 174.3% |
| Uniform allowed-key mean | 97.8% | 60.0% | 101.6% | 89.2% |

The post-seal reference is FP64 causal softmax on canonical BF16 positioned operands, followed by a separately measured BF16 head rounding. It is not native SDPA/Flash parity. Output-slice errors multiply the head error by original W_o head0 columns, offline after forecasts seal. This is ideal linear error transport, not the complete fused native output projection, residual, later encoder or image.

Six prompt pairs expose information lost by a whole-head aggregate. Relative error here divides prediction-change error by the exact prompt-change RMS:

| Pair | Rank64 head-change error | Uniform head-change error |
|---|---:|---:|
| Bottle left/right of bowl | 100.0% | 87.2% |
| Fox scene, short/expanded | 100.0% | 97.2% |
| Reordered teapot/notebook/spoon | 502.0% | 89.8% |
| Repeated clock/station tokens | 99.8% | 118.6% |
| Unicode color reassignment | 100.0% | 94.4% |
| Long repeated order/reversal | 683.8% | 98.0% |

The first, second and fifth pairs produce essentially no candidate change at every tested rank. The repeat pair is a retained countertrend: the rank64 candidate beats uniform on this observable. The ideal output slice also changes some rankings; for the fox expansion, a near-zero predicted change has100% error versus109.5% for uniform. These exceptions do not repair the general reader, but belong in its evidence.

## Diagnose keys, not just features

The [first saved-only diagnostic](results/context-kernel-diagnostics/report.json) finds large operand norms and concentrated feature contributions. Median per-prompt allowed-pair ||x+y||² is about1,475. Reciprocal balancing alone lowers it to420; subtracting a key mean lowers it to61; subtracting query and key means lowers it to34. These centers use complete held operand arrays, including future/padding queries: they are retrospective conditioning probes, not deployable parameters.

For iid Gaussian features,

\[
\frac{\operatorname{Var}(\widehat K(x,y))}{K(x,y)^2}
=\frac{e^{\|x+y\|^2}-1}{r}.
\]

The report stores the logarithm of this quantity without overflowing. This identity concerns the unnormalized kernel estimator over random draws. Shared errors may cancel in the normalized ratio, so it cannot by itself diagnose head error.

Astra requested a direct key-probability check. The [second saved-only diagnostic](results/context-key-diagnostics/report.json) selects query clocks solely from masks: activeTokens//2, activeTokens−1 and511, deduplicated. This yields40 queries across14 prompts. It evaluates the same frozen feature kernel explicitly for diagnosis and recovers the sealed streaming heads within3.19 ×10⁻¹⁵. Rank64 places median largest key probability essentially at1, compared with0.2544 for exact attention. Median total-variation distance is0.9797; argmax keys agree at only2/40 selected queries. Several paired prompts collapse onto the same token/address.

This directly observes key collapse on selected queries. It is stronger than saying one feature dominates: feature coordinates and key addresses are different objects. The full key distributions are retained in [key-probabilities.npz](results/context-key-diagnostics/key-probabilities.npz).

## An exact symmetry offers a repair test, not a result to promote

For fixed centers a,b and positive scale c, transform

\[
x'=c(x-a),\qquad y'=(y-b)/c,
\qquad \log w_j=a^\top y_j.
\]

Then

\[
x'^\top y'+a^\top y
=x^\top y-x^\top b+a^\top b.
\]

The extra term is constant across keys for a given query, so exact normalized attention is unchanged. Add aᵀy to key-feature logs before state accumulation; the correction is independent of the query. With A=E||x−a||² and B=E||y−b||², c=(B/A)^(1/4) balances the two norm terms. A query-dependent center generally makes the key correction query-dependent and breaks this simple shared state.

The selected-query diagnostic checks this symmetry to8.33 ×10⁻¹⁶ in exact probabilities. Using centers and scale fitted retrospectively to each complete prompt improves rank64 median total variation from0.9797 to0.7557 and selected-head relative error from147.8% to109.4%. Large error remains. These are40 selected queries, not the whole-head medians above. They use lookahead and target operand geometry; they establish no frozen serving or generalization result. Increasing feature rank blindly or shipping prompt-fitted centers would not answer the missing question.

## Compute, storage and custody

| Measurement | Observed scope/value |
|---|---:|
| Core state, ranks16/32/64 | 16,656 /33,296 /66,576 bytes |
| Rank64 Gaussian constants | 65,536 bytes |
| Separate value-range diagnostics | 2,048 bytes |
| Retained learned atom table | 2,644,901,888 payload bytes |
| Unique lexical payload fetched in full serving child | 1,967,104 bytes,113 rows |
| Initialization after imports | .179s |
| First prefix assemblies in observed sequence | 1.08–917.22ms |
| Warm raw-text calls, rank16/32/64 | 9.31–16.24 /9.57–15.80 /9.88–17.97ms |
| Warm raw-text calls, uniform | 3.44–4.72ms |
| Serving child wall /CPU after imports | 8.344 /4.478s |
| Serving high-water RSS | 217,832KiB, about212.7MiB |
| Full worker /scheduler lease | 14.198 /17.09s |
| Scheduler sampled process-tree peak RSS /VRAM | 314.3MB /0 |
| Saved norm/gauge diagnostic lease | 2.27s,119.2MB sampled RSS |
| Selected-key diagnostic lease | 2.21s,100.9MB sampled RSS |

Core state includes z/M, per-feature maxima and two64-bit counters. Constants, current operands, lexical cache, outputs, traces and diagnostics are additional. Warm measurements include raw-text processing, selected RoPE, feature formation, streaming and diagnostics, with seven repeats per prompt/method. First-prefix timings are sequential first uses, not independent cold-cache trials. No controlled native speedup, FLOP total, energy or end-to-end image latency is claimed.

For rank64, feature projection costs4.37–8.39 million MACs per call, head-numerator reads4.19 million, partition reads32,768, and input norms68,224–131,072. Key-feature/rescale exponentials number2,688–65,536 plus32,768 query exponentials. State rescaling/value products/additions and diagnostic logarithms are separately counted per prompt in the seal. NumPy exponentials/projections remain computation. The online process imports no neural framework, evaluates no original dense projections, and consumes learned cached operands. It is not a representation-independent zero-neural image generator.

Reference work pays14 full mathematical attention evaluations,14 full lexical assemblies, six direct feature-kernel checks, and655,360BF16 payload bytes of the original output-projection slice. Authenticating its4.967GB source shard also reads that shard for hashing. The two diagnosis jobs use only saved arrays and fixed features; no original weights or new native inference.

Primary smoke: `job-f47a626860e2`; full: `job-cad6090b693f`; conditioning: `job-a9041e8934e1`; key diagnosis: `job-bfcdcc4108d6`. All succeeded with guarded declared/sealed/executed payload identity and authenticated collected inventories. An independent SSH byte check verifies88 retained full-assay files totaling60,159,514bytes without copying them locally. The [public primary evidence](results/compiled-context-statistics/provenance.json) includes every compact collected result; both JSONL streams are byte-lossless gzip with exact round-trip verification. They retain28,672 head-position rows and12,288 pair-position rows. Rows share prompts, parameters and trajectories; they are not40,960 independent semantic examples.

The first staged smoke proposal was never submitted: it was replaced before execution so references would consume canonical full-prefix operands rather than trusting candidate ports. It remains retained as an unexecuted proposal. Serving has import/checkpoint/network guards; the diagnosis scripts consume authenticated saved arrays. Zero-framework/weight counters describe source-audited trusted code, not hardware attestation or an adversarial sandbox. mrun's OS memory containment was active; its payload sandbox was off.

The Saturn sweep runs and verifies the primary's two streams with exit0 and `target-streams-only` scope. The key diagnostic's600-row stream produces42 entropy-change watch fires, then verifies its hash chain with exit0. Those automatic fires compare consecutive records across ranks, variants and prompts; they are not a calibrated temporal anomaly detector for this panel. They remain preserved rather than treated as additional scientific failures. The norm-only report has no recognized observation stream and is retained as report evidence, without claiming equivalent stream coverage. No terminal-claim receipts, native-image certification or physical-address joins are supplied.

The [raw fire ledger and separate interpretation](results/context-diagnostic-watch/interpretation.json) preserve those outcomes. The norm-only sweep returns exit2 with `evidence-incomplete-empty-target`, a reporting-instrument limitation rather than a scientific null.

## Continue from the actual gap

The experiment distinguishes an accurate implementation of a kernel from an accurate kernel for this model. The remaining question is what compact representation preserves *normalized contextual reads and their changes*, without exact-attention fallback, prompt-specific lookahead or relabeling learned arithmetic. Later contextual layers, own-state denoising, general non-neural pixel production and independent prompt-faithfulness evaluation all remain required.

Astra's follow-up recommends a [saved-only diagonal-geometry discriminator](plans/context-kernel-geometry.md). A fixed diagonal D and its reciprocal preserve the same exact gauge, while balancing coordinate moments rather than only total norms. This is proposed, not measured. The current prompt-fitted means/scalar result is not an optimized oracle bound. A common-bank low-rank capacity diagnostic remains a later option if geometry does not make this feature family useful.

Sources: [streaming state](../generator/context_statistics.py), [frozen assay](../generator/context_statistics_study.py), [norm diagnosis](../generator/context_kernel_diagnostics.py), [key diagnosis](../generator/context_key_diagnostics.py), [publication/retention verifier](../generator/publish_context_statistics.py), [active research record](zero-neural-research.md), and the local [living blog](/Users/jakeholl/domains/obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md). Astra's newest-post review connects this result to [complete own-state denoising](/Users/jakeholl/domains/obsidian/blog/2026-11-06-010513-predicting-a-complete-denoiser-pass-from-its-own-state.md), [bounded normalization and cancellation](/Users/jakeholl/domains/obsidian/blog/2026-11-05-161109-the-one-assumption-that-would-make-model-computation-easier-to-understand.md), [C-machine carry/context](/Users/jakeholl/domains/obsidian/blog/2026-11-06-004500-what-the-c-state-machine-learned-about-carry-and-context.md), and [reader families](/Users/jakeholl/domains/obsidian/blog/2026-11-06-003759-from-exact-readers-to-causal-computation-families.md). These notebook links include unpublished working files.
