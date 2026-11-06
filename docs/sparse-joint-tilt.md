# Sparse joint tilt: smaller state, an unresolved contextual response

Cycle 12 implements the [sparse-state plan](plans/sparse-joint-tilt.md) on the already-scored crane-behind/crane-before pair. The state with four keys per anchor uses **8,552,960 logical bytes across all 32 heads**, versus 272,663,040 for the preceding covariance state. It improves active-row paired state error from 72.1% to 40.8%, but padding error is 131.6% and overall error126.3%. A separately evaluated full-query-score oracle reaches 41.8% overall error with four keys. This is useful reduced-state evidence on one retrospective specimen, not a new image backend or fresh raw-prompt validation.

The [report](results/sparse-joint-tilt/report.json), [sealed primary predictions](results/sparse-joint-tilt/prediction-seal.json), [support diagnostics](results/sparse-joint-tilt/support-diagnostics.json) and [custody](results/sparse-joint-tilt/provenance.json) retain complete results. The implementation is [sparse_tilt.py](../generator/sparse_tilt.py); [sparse_tilt_study.py](../generator/sparse_tilt_study.py) stages, submits and collects the finite mrun experiment.

## Advance a compact, positive prefix state

Keep the parent's 64 source-frozen landmarks per query head and nearest-source route. Write x=q/d^.25, y=k/d^.25 and s_j=ℓᵀy_j for landmark ℓ. For every anchor, retain its m highest-scoring admitted occurrences, m=1,2,4. Scores are fixed after admission; earlier clock wins ties. Repeated occurrences remain separate and masked keys never enter. GQA shares K/V inputs, not different anchors' statistics.

The rest of the prefix is a tail T. Its log-max-rescaled state contains positive mass Z_T, value sum N_T and key sum Y_T. Retained entries carry score, occurrence position and BF16 K/V payload. An incoming occurrence either enters top-m and sends one eviction into the tail, or appends directly to the tail. The cutoff only rises, so rejected or evicted entries never need to re-enter. Tail updates use addition and running-maximum rescaling, with no subtraction of nearly equal totals.

Recover full anchor key mean μ from retained entries plus tail, and define

\[
\delta=x-\ell,\qquad t_j=\delta^T(y_j-\mu).
\]

Tilt retained keys exactly and leave the tail's factor at 1:

\[
\widehat h_S(x)=
\frac{N_T+\sum_{j\in S}e^{s_j+t_j}v_j}
{Z_T+\sum_{j\in S}e^{s_j+t_j}}.
\]

All terms share the anchor score scale; implementation combines positive log masses. Equivalently, with full anchor probabilities p_j and read H,

\[
\widehat h_S=
\frac{H+\sum_{S}p_j\operatorname{expm1}(t_j)v_j}
{1+\sum_{S}p_j\operatorname{expm1}(t_j)}.
\]

Numerator and partition change together. Empty support gives the anchor read and complete support gives exact attention. Partial support depends on the declared centering gauge: changing only retained log weights by a constant changes their relative mass against the unit tail. Neither positivity nor exact implementation of this law establishes fidelity to original attention.

## Charge the state and its assistance

For dimension d=128, rank r=64 and budget m, one head's core payload is

\[
r[16+2d\cdot8+m(16+4d)]+16\text{ bytes}.
\]

The two tail vectors are FP64; retained K/V use BF16 words. Independent per-budget banks are charged separately.

| Budget | Complete 32-head core bytes |
|---|---:|
| m1 | 5,308,928 |
| m2 | 6,390,272 |
| m4 | 8,552,960 |
| Previous covariance core | 272,663,040 |
| Complete 512-position BF16 K/V payload | 2,097,152 |

The m4 core is 31.9 times smaller than the covariance core, yet4.08 times the complete K/V payload. Runtime source assets add 3,153,920 bytes. Per-bank range/counter diagnostics add 65,536/512 bytes, and source-identity metadata adds 2,048 UTF-8 bytes across heads. Array/NPZ framing, workspaces, predictions and process residency are additional. This implementation duplicates anchor payload; no exact address-dedup saving is claimed.

Source geometry and saved lexical operands are inherited learned information. This experiment starts from the parent's saved ports, without retokenizing or qualifying new strings. Both prompts have 27 active and 485 padding rows; only token 8 differs, IDs 4815→1573. Later query rows and routes are identical, as independently qualified by the parent. All 512 query outputs still enter the downstream consumer.

## Compare the same prompt change through the actual consumer

Primary banks seal before diagnostics, oracle support selection or original consumer parameter reads. The separate oracle scans all allowed keys in the **current causal prefix**, ranking |p_j expm1(t_j)|. It uses no future keys or target output-error fit. It is not a compact serving rule, output-optimal subset search or bound on all sparse-support capacity.

Every new bank feeds original output projection, residual, RMS, gated SiLU MLP and residual state on all 512 rows. Eight arms per prompt include six sparse/oracle candidates plus exact/zero consistency controls:16 full output projections and48 MLP projections. Learned consumer equations are paid offline work despite zero framework calls. Parent nearest/transport consumer arrays are retained comparisons.

For F(h), the paired attention-deletion response is Δ[F(h)−F(0)]. Errors below are relative RMS against Δ[F(h_exact)−F(0)], not whole residual magnitudes. All/active/padding target RMS values are .0248576/.0317437/.0244172. Rows and heads are dependent observations from one pair.

| Method | State response, all | Active 27 | Padding 485 |
|---|---:|---:|---:|
| Parent nearest64 | 125.11% | 80.03% | 128.54% |
| Parent transport64 | 116.03% | 72.10% | 119.34% |
| Eligible m1 | 110.89% | 74.51% | 113.72% |
| Eligible m2 | 135.47% | 63.08% | 140.37% |
| Eligible m4 | 126.34% | 40.82% | 131.56% |
| Oracle m1 | 102.17% | 31.86% | 106.43% |
| Oracle m2 | 67.86% | 20.61% | 70.70% |
| Oracle m4 | 41.79% | 12.26% | 43.55% |

| Method | Rounded head-bank change, all | Active | Padding |
|---|---:|---:|---:|
| Eligible m1 | 96.26% | 103.71% | 95.77% |
| Eligible m2 | 119.06% | 99.92% | 120.17% |
| Eligible m4 | 115.62% | 76.27% | 117.68% |
| Oracle m1 | 102.29% | 52.28% | 104.67% |
| Oracle m2 | 64.34% | 34.06% | 65.80% |
| Oracle m4 | 37.63% | 18.79% | 38.52% |

Head-bank target RMS is .00435756. The m4/oracle4 state-response cosines are .5999/.9345 overall and.9237/.9925 on active rows. Broad endpoint state-effect errors are 47.66%/48.38% for eligible m4, versus 10.59%/10.43% for oracle4 across the two prompts. Those endpoints do not replace prompt-change measurement.

More support does not monotonically help the eligible rule: m2 is worse overall than m1, and m4 still worsens padding relative to transport. The oracle trend nevertheless shows that this particular sparse correction family can preserve substantially more of the demanded response on this specimen. It does not establish a reliable selector for unseen prompts.

## What the support diagnostics do and do not explain

Across padding rows, mean eligible/oracle support overlap rises from 16.2% at m1 to 38.7% at m4. Mean **fraction of total absolute tilt mass omitted** falls from .876/.620 to .629/.278 for eligible/oracle support. These are mass diagnostics, not probability-error bounds.

At m4, eligible support retains changed address 8 on14.47% of padding queries, versus 11.82% for the oracle. Simply retaining the changed token therefore does not explain oracle superiority. Mean captured normalized exponential-remainder fraction is .318 versus .721. That diagnostic records 0 when total computed remainder mass is exactly0; near-zero denominators can also be sensitive. Inspect retained normalizedRemainderMass alongside ratios.

Support selection is weak, but the comparison also changes effective tail composition and prompt-dependent support membership. Baseline errors pass through a nonlinear MLP. This experiment does not isolate those mechanisms. Astra recommends the [matched tail-centroid ablation](plans/tail-centroid-tilt.md) before changing selection geometry: freeze both existing m4 support sets, change only the tail law, and stratify rows by whether support membership changes across prompts. That ablation is proposed, not executed.

## Mechanics, time and custody

All actual reads are valid. Maximum measured convex-hull excursion is 1.73e−18; direct restricted-law closure is 1.33e−15; route mismatches from the parent are 0. Recomputed exact/zero consumers match the parent's BF16 words at every retained stage. All 61 generator mechanics tests pass, including direct-law equality, ties, repeated/masked occurrences, stable extremes, core-byte accounting, resume binding and separate oracle ranking. These checks protect the instrument; they do not certify accuracy.

Guarded CPU job job-f1e1fa692435 succeeds in53.35 scheduler seconds,51.757 worker seconds, with 1,472.2 MB sampled process-tree RSS and0VRAM. NumPy FP64 runs with one math thread. Primary child time is 8.848s, including8.778s after imports and4.895 CPU seconds; high-water RSS is 169,136 KiB. Per-prompt stream sums are .700–.864s across budgets, while m4 serialization-inclusive calls take2.486/2.676s. This is one saved-port process, not a repeated serving benchmark; extra serialization latency is measured without attributing a cause.

Original parameter read/hash takes 3.937s. Per-prompt direct/oracle diagnostics take4.894/6.563s; eight original-consumer arms take9.967/9.236s. No native timing comparison or speedup is established.

Request 9028caa1ece44b08b1566ff412694d07edffb2416508c304fbfdba2ab9710f35 binds the run. Prediction seal 83f3022f4b8b0838445a674cf5fb375aab00c3ebcb2eb3ddffcf3ba963039547 binds all primary banks and192 final per-head states. Independent complete-byte verification covers 222 files/5,764,072,602 bytes, including the4,967,215,360-byte original checkpoint and overlapping parent data. These are not all new forecast bytes.

Public evidence retains 11 files/2,751,634 bytes and all46,080 observations through lossless gzip. Full candidate/state/oracle-support/correction/consumer arrays remain near compute. The [Saturn sweep](results/sparse-joint-tilt-evidence/sweep.json) runs and verifies with 1 stream / 46,080 records, no watch fires, target-streams-only status, no terminal-claim receipts and no physical-address joins.

The newest workspace [raw-prompt-to-image reconstruction](/Users/jakeholl/domains/obsidian/blog/2026-11-06-042000-predicting-the-image-from-the-raw-prompt.md) joins Qwen,112 denoiser transitions and decoding under one forecast seal using original parameters and Torch primitives. It is a strong end-to-end reference on its tested specimen, with recovered packaging custody preserved. It does not eliminate learned computation. Arbitrary-prompt zero-neural images remain open.
