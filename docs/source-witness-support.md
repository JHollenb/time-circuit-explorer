# Source witnesses change the keys, but worsen the main response

Cycle 14 implements the [source-witness plan](plans/source-witness-support.md) without changing the four-key budget, 64 anchors per head, source gauge, nearest routing or centroid read. Active-row state error improves **32.85%→31.29%**, but padding error worsens **99.87%→112.54%** and overall error **95.96%→107.98%**. The mechanically valid result is heterogeneous; it does not support this selector as an overall improvement on the saved pair.

The [report](results/source-witness/report.json), [source compiler](results/source-witness/compile-receipt.json), [prediction seal](results/source-witness/prediction-seal.json), [support diagnostics](results/source-witness/support-diagnostics.json) and [custody](results/source-witness/provenance.json) retain the evidence. Source: [source_witness.py](../generator/source_witness.py), [study](../generator/source_witness_study.py), [tests](../generator/test_source_witness.py).

## Separate retention priority from the measure being read

Reconstruct the existing 4,608 source query candidates from checksum-bound lexical atoms. Assign each to one of the original anchors using the original key-covariance metric. Each anchor gets itself and three deterministic farthest-first candidates from its source cell, with canonical identity/position ties and repeated-anchor fallback for degenerate cells. Reconstructed anchor values match exactly. No current-prompt queries, target heads or output-error fits enter this compilation.

With scaled coordinates x=q/d^.25, y=k/d^.25 and immutable source key mean c_g, retain the top four occurrences by

\[
\rho_\ell(y)=\max_{b=0,\ldots,3}q_{\ell b}^T(y-c_g),\qquad q_{\ell0}=\ell.
\]

There are no fitted offsets or norm normalization. Store ρ separately from the original anchor score s=ℓᵀ(y−c_g). Every eviction and positive tail Z/N/Y update continues to use s. Because priorities are fixed after admission, the cutoff only rises; the tail remains append-only. Repeated occurrences retain separate clocks and masked keys do not enter. Resume binds witness bytes/policy as well as source, budget and clock.

The centroid law from [cycle 13](tail-centroid-tilt.md) remains unchanged. For selected support S and remaining tail T, write u=y−c_g, δ=x−ℓ, μ_T=Y_T/Z_T and h_T=N_T/Z_T:

\[
\widehat h_S=
\frac{Z_Te^{\delta^T\mu_T}h_T+\sum_{j\in S}e^{s_j+\delta^Tu_j}v_j}
{Z_Te^{\delta^T\mu_T}+\sum_{j\in S}e^{s_j+\delta^Tu_j}}.
\]

Only selection changes. Independent direct reads close within 9.23e−16. The original anchor selector reproduces every parent state field, support set, route, rounded prediction and original consumer stage. All actual reads are valid, maximum raw hull excursion is 1.73e−18, and all 70 mechanics tests pass. These validate the computation, not its prediction quality.

## An upper score envelope need not preserve key ranking

If x is a convex combination of the witnesses, xᵀu≤ρ_ℓ(u). This is not a normalized-attention guarantee. Four witnesses have affine hull dimension at most three; coverage of a general128-dimensional cell is unestablished.

Even an included query can receive the wrong ranking. Take anchor/query x=(1,0), another witness (0,1), and keys u_A=(1,0), u_B=(0,100). Current-query scores are 1 and 0, while envelope scores are 1 and 100. The envelope prefers B. This is a mathematical ranking example, not an attention-output experiment. A valid bound can be much looser for some keys than others.

The remaining question therefore has two parts: whether source witnesses cover useful query directions, and whether selecting a maximum raw score is the right objective for preserving a prompt change. The [next proposed diagnostic](plans/full-prefix-selector-objective.md) supplies the actual query direction while keeping the read and budget fixed, then compares ranking objectives. It has not run yet.

## Measure the change through the actual first-block consumer

This is the same retrospective crane-behind/crane-before pair, supplied as saved operands: 27 active and 485 padding rows, all 32 heads. Complete anchor/witness banks seal before references and original parameters. The previous oracle-centroid bank and consumer are retained historical controls, without reranking or a new oracle consumer execution.

Paired state error compares Δ[F(h)−F(0)] against its exact mathematical counterpart. Target RMS is .0248576 overall, .0317437 active and .0244172 padding. These are dependent rows/heads from one pair, not independent prompt trials.

| Selector | State error, all | Active27 | Padding485 |
|---|---:|---:|---:|
| Anchor | 95.96% | 32.85% | 99.87% |
| Source witnesses | 107.98% | 31.29% | 112.54% |
| Retained correction-mass oracle | 43.67% | 14.98% | 45.45% |

| Selector | Rounded head error, all | Active | Padding |
|---|---:|---:|---:|
| Anchor | 89.77% | 62.78% | 91.22% |
| Source witnesses | 105.06% | 63.66% | 107.16% |
| Retained oracle | 38.49% | 18.48% | 39.42% |

Head target RMS is .00435756 overall. Witness head error improves on 12/32 heads overall and 9/32 active, despite the pooled active state improvement. Broad endpoint state-effect errors change only 36.17%/35.17%→36.35%/35.39%, while the paired response worsens. This again separates endpoint reconstruction from faithful contextual change.

Padding overlap with the old oracle drops .38694→.35717; omitted absolute tilt-mass fraction rises .62914→.65212; changed-address 8 retention falls .14468→.13731. These are post-seal diagnostics, not probability-error bounds. Absolute-mass ratios encode0when total mass is 0; that convention is preserved rather than retroactively rewritten.

Support membership changes across prompts on 4,136anchor head-query rows versus2,893witness rows, yet witness error is worse. Less churn is not sufficient for fidelity. Within the witness strata, unchanged/changed head error is 93.38%/110.99%, with different target RMS. The [membership record](results/source-witness/membership-diagnostics.json) is descriptive, not a causal membership intervention.

## Witness counts do not show simple global norm domination

Across 2,048 source cells, occupancy ranges 1–2,493; none is empty and 96 of the additional slots repeat their anchor. Mean witness norms are[5.7620, 5.6596, 5.6295, 5.6342]. Admission winner totals are[24,349; 30,174; 28,081; 27,988] across both prompts,32 heads,64anchors and 27admitted keys. The mean largest per-anchor winner share is .51385. These averages do not establish a single globally dominant high-norm witness; they also do not exclude local dominance. Full per-cell norms and counts remain available.

No witness norms, offsets or support rule were changed after seeing targets. This negative overall result remains evidence for this frozen selector/specimen, not an exclusion of source-only selection generally.

## Charge the additional computation

Witness priority adds 65,536 core bytes:8,552,960→8,618,496across32 heads. Additional witnesses add 6,291,456 runtime bytes, giving 9,445,376 bytes of geometry. Identity metadata arrays add 204,800 bytes offline. Winning-count diagnostics add 65,536 bytes, range/counters 66,048 bytes and identity strings4,096 UTF-8 bytes per witness bank. Array framing, workspaces and saved predictions are additional.

Admission score work rises 7,077,888→28,311,552 MACs per prompt. This excludes routing, state updates, readout and diagnostics; it is not a complete FLOP census. The full BF16 GQA K/V payload is 2,097,152 bytes, so no cache-efficiency advantage is established.

Guarded CPU job `job-614222d36ca1`, request `8d8e9c3e87dbdb997ea007e183dd2bc1f35f43bc93bc67db0f8ebc5767aa3c79`, succeeds in 62.53 scheduler seconds/60.707 worker seconds with 1,391.6 MB sampled process-tree RSS and 0VRAM. Actual NumPy FP64 math uses one thread; default scheduler admission reserves4 GB/four CPU threads.

Compiler time is 24.467s, including22.807s gathering1,747 unique learned atom rows/30,411,776 payload bytes. These are inherited learned outputs. Primary child time is 5.425s, including5.240s after imports and 3.070 CPU seconds; high-water RSS is 184,332 KiB. Witness stream sums are .714/.759s per prompt, serialization-inclusive.754/.801s. A 2.693s anchor serialization-inclusive call shows why single-process timings do not establish a law-speed gain.

Original parameter read/hash takes 6.938s. Post-seal support diagnostics take4.080/5.347s and four original-consumer arms take5.027/4.864s per prompt. Newly evaluated learned work is 8 full output and 24 MLP projections, all 512 rows. The oracle consumer is inherited. Zero framework calls do not eliminate those equations.

Seal `79bdbd89432b359961dbbe62fe2fa6f641eb600600408c31c73cff559519d8fa` binds the new banks and 128final states. Independent full-byte verification covers526 files/8,299,854,035 bytes, including touched source atom shards, original checkpoint and overlapping parents. These are not all new artifacts. Public evidence is 14 files/3,693,785 bytes with 23,040 observations transported losslessly. Full arrays remain near compute. The [Saturn sweep](results/source-witness-evidence/sweep.json) verifies one observation stream with no fires, terminal receipts or physical-address joins.

The newest [two-path image reconstruction](/Users/jakeholl/domains/obsidian/blog/2026-11-06-051300-predicting-two-attention-paths-and-their-image-futures.md) predicts original-weight padded/all-attended futures and their image difference. Its common-prefix finite drift reinforces numerical-profile qualification; it does not invalidate this fixed FP64 diagnostic or remove learned computation. Arbitrary-prompt zero-neural image generation remains open.
