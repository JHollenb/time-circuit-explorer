# The tail mean matters, with support held fixed

Cycle 13 executes the [matched tail-law plan](plans/tail-centroid-tilt.md). At the same selected occurrences and prefix state, the centroid read lowers paired first-block state error **126.3%→96.0%**, active error **40.8%→32.9%** and padding error **131.6%→99.9%**. The frozen full-query-score oracle instead worsens **41.8%→43.7%** overall. Tail treatment matters on this specimen; a large support gap remains under the matched law.

This is one retrospective crane-behind/crane-before pair from the [sparse joint-tilt study](sparse-joint-tilt.md), supplied as saved first-layer operands. Both prompts have 27 active and 485 padding rows. All 32 heads are predicted; the original block consumer evaluates all 512 rows offline. There is no fresh raw-string validation, complete encoder or zero-neural image backend.

The [report](results/tail-centroid/report.json), [membership diagnostics](results/tail-centroid/membership-diagnostics.json), [prediction seal](results/tail-centroid/prediction-seal.json) and [custody](results/tail-centroid/provenance.json) preserve the results. Source: [centroid_tilt.py](../generator/centroid_tilt.py), [study](../generator/centroid_tilt_study.py), [tests](../generator/test_centroid_tilt.py).

## A different read from the same state

Each of 64 source landmarks per head retains four actual key occurrences by fixed anchor score. All others advance the same append-only positive tail. No selector, route, budget, covariance tensor or stored statistic changes. The two laws independently advance their own states and reproduce every parent state field. Frozen support position sets and routes have zero mismatches.

Let s_j=ℓᵀy_j, δ=x−ℓ, μ be the full anchor key mean, μ_T=Y_T/Z_T and h_T=N_T/Z_T. The old law gives the tail factor 1 and selected keys exp(δᵀ(y_j−μ)). The new tail factor is exp(δᵀ(μ_T−μ)). Their common centering factor cancels:

\[
\widehat h_C=
\frac{Z_Te^{\delta^T\mu_T}h_T+\sum_{j\in S}e^{s_j+\delta^Ty_j}v_j}
{Z_Te^{\delta^T\mu_T}+\sum_{j\in S}e^{s_j+\delta^Ty_j}}.
\]

Implementation combines centered positive log masses, with an explicit empty-tail branch. Empty support gives the anchor read; complete support gives exact attention. Logical core remains 8,552,960 bytes per complete law. A 3,153,920-byte source asset, diagnostics and workspaces add further memory. Executing and retaining both experimental arms costs additional memory/work.

The exact tail mass has an additional factor

\[
\kappa_T=E_T[e^{\delta^T(y-\mu_T)}]\ge1.
\]

The centroid approximation drops this partition curvature and keeps the tail value mean fixed. At the anchor,

\[
D h_{exact}-D\widehat h_C=p_T\operatorname{Cov}_T(v,y).
\]

It restores between-group mean transport while omitting within-tail value–key interactions. This derivative holds memory and support fixed while varying the query; the actual prompt pair changes memory too. It is not a complete attribution of that paired response. An independent finite-difference test verifies the derivative. Positivity and gauge invariance do not guarantee improved fidelity.

## Consume the predicted change

The oracle uses exactly the cycle 12 indices, with **no reranking**. It still requires the complete current causal key bank. It is neither compact serving nor an output-optimal support bound. Eligible banks seal before oracle, target or original parameter reads.

For consumer F(h), compare Δ[F(h)−F(0)] with its exact mathematical counterpart. State target RMS is .0248576 overall, .0317437 active and .0244172 padding. Head-bank target RMS is .00435756 overall. All observations come from one dependent pair.

| Support and tail | State error, all | Active 27 | Padding 485 |
|---|---:|---:|---:|
| Eligible unit | 126.34% | 40.82% | 131.56% |
| Eligible centroid | 95.96% | 32.85% | 99.87% |
| Frozen oracle unit | 41.79% | 12.26% | 43.55% |
| Frozen oracle centroid | 43.67% | 14.98% | 45.45% |

| Support and tail | Rounded head error, all | Active | Padding |
|---|---:|---:|---:|
| Eligible unit | 115.62% | 76.27% | 117.68% |
| Eligible centroid | 89.77% | 62.78% | 91.22% |
| Frozen oracle unit | 37.63% | 18.79% | 38.52% |
| Frozen oracle centroid | 38.49% | 18.48% | 39.42% |

Eligible head error improves for 26/32 heads overall, 23/32 on padding. Oracle improvement covers only 8/32 overall, but 23/32 active. Its active head error improves slightly while active state error worsens. Oracle overall state-response cosine rises .9345→.9369 despite its RMS error increase. No single proxy replaces the demanded consumer.

Eligible broad endpoint state-effect errors improve 47.66%/48.38%→36.17%/35.17%; oracle endpoints worsen 10.59%/10.43%→11.24%/11.15%. Baseline reconstruction and prompt-change fidelity remain distinct. For a smooth pre-quantization consumer,

\[
F_s(h+\Delta h)-F_s(h)=\int_0^1J_{F_s}(h+t\Delta h)\Delta h\,dt.
\]

Both baseline trajectory and changed head vector matter. Actual BF16 boundaries are discrete; this integral explains a possible dependence, not an exact identity for the finite program or a newly measured causal attribution.

## Membership strata are diagnostic

Canonical valid occurrence sets, rather than heap slots, are compared across prompts.

| Eligible stratum | Head-query rows | Unit error | Centroid error |
|---|---:|---:|---:|
| Unchanged, all | 12,248 | 89.58% | 88.07% |
| Changed, all | 4,136 | 128.19% | 90.73% |
| Unchanged, padding | 11,457 | 93.89% | 93.20% |
| Changed, padding | 4,063 | 127.98% | 90.21% |

The all-row target RMS differs between unchanged/changed strata, .0030433/.0069131. Different heads and target energies prevent interpreting the larger changed-set improvement as a causal membership effect. Oracle unchanged/changed counts are 13,093/3,291; errors are 35.09%→34.99% and 38.25%→39.33%. All regional/per-head records remain public.

Working inference: the unit tail was material for the eligible selector on this pair. Correcting its mean response helps without additional persistent state. The remaining matched-law gap motivates investigating support geometry; there is no universal preference for the centroid law. Astra recommends [four source-cell query witnesses per anchor](plans/source-witness-support.md), keeping this reader and budget fixed. Their frozen score envelope changes selection while original anchor scores continue to advance the tail. That next experiment is proposed, not executed.

## Computation and integrity

Guarded CPU job `job-f2b0ab14b656`, request `ad61466d465964461d25ae5dc10e8a81f10c5b4cd3ca47d0b5833c951894584d`, succeeds in 46.21 scheduler seconds and 42.920 worker seconds. Sampled process-tree peak is 1,537.1 MB RSS, 0 VRAM. NumPy FP64 uses one math thread; default admission reserves 4 GB/four CPU threads. Another small CPU audit was present, so this is not an isolated repeated benchmark.

Primary child time is 6.789 s, including 6.735 s after imports and 3.104 CPU seconds; high-water RSS is 178,652 KiB. Centroid stream sums are .712/.707 s per prompt; serialization-inclusive calls are 2.881/.748 s. The first unit stream includes a .738 s first-head cost. Order, first-use and serialization prevent a fair speed comparison between laws.

Parameter read/hash takes 5.429 s. Frozen diagnostics take 5.224/6.263 s per prompt; six consumer arms take 7.631/7.167 s. Original learned work totals 12 output and 36 MLP projections, all 512 rows. Zero framework calls do not eliminate those equations.

Unit banks and every exact/zero/unit consumer stage reproduce parent BF16 words. Every final state field, support set and route matches. Direct centroid closure is 7.22e−16, unit closure 1.33e−15; all actual reads are valid, with hull excursion at most 1.73e−18. All 66 mechanics tests pass.

Initial staging refused a historical parent because today's source had changed. Historical intake now verifies original staged bytes, request, custody and collected outputs without requiring later development files to match. Fresh publication retains the current-worker check by default. A regression proves altered staged bytes remain refused. That staging refusal launched no scientific lease; the experiment uses one guarded finite lease, strict preflight and no retry.

Seal `61e0f07c5b951ef897b89e6db4c99a3727f55464ebd263a7fb70a4ee8ae8f238` binds eligible banks and 128 final state files. Independent full-byte checks cover 221 files/5,822,071,055 bytes, including the original 4,967,215,360-byte checkpoint and overlapping parent data. Public evidence is 11 files/1,365,285 bytes with all 30,720 observations transported losslessly. Full arrays stay near compute. The [Saturn sweep](results/tail-centroid-evidence/sweep.json) verifies one observation stream with no fires, terminal receipts or physical-address joins.

The [storage review](results/storage-audit-cycle13/result.json) and [archive receipt](results/storage-cleanup-cycle13/result.json) preserve two older terminal-run transports totaling 1,628,748,559 bytes in checksum-verified cold storage. Extracted evidence stays local. Ten restoration pointers match receipts; remote type/size checks cover all ten, with fresh complete hashes for this cycle's two objects. Shared-volume free space is observed 9.524→11.118 GB during cleanup; concurrent/APFS effects prevent assigning that entire increase to this operation.

Full contextual encoding, MLP replacement, own-state diffusion and general pixels remain open. This experiment does not complete arbitrary-prompt zero-neural generation.
