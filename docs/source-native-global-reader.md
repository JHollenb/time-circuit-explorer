# One live global read does not yet supply a useful velocity update

The first joint block's image attention can be computed with seven original affines, followed by one small shared velocity writer. That program executes at 1.092% of a full final denoiser call's nominal affine MACs, with attention arithmetic counted separately. Its two images retain recognizable objects from their supplied prefixes, but have distorted surfaces and colors. Holding the previous velocity is substantially more accurate. On four historical distant-edit contrasts, the program improves the complete response slightly while missing the direction at unchanged queries.

This is a measured limit of this reader, writer and discovery corpus. It does not establish that a first attention read is insufficient for every better writer. It also remains learned computation: original projections, RMS weights, the fitted writer and VAE are paid. Arbitrary-prompt images with zero neural evaluations remain open.

The [frozen request and plan](results/source-native-reader/frozen-plan.md), [report](results/source-native-reader/report.json), [pre-run Astra review](results/source-native-reader-evidence/advisor-before-run.json), [implementation](../generator/source_native_study.py) and [mechanics audit](../generator/audit_source_native.py) preserve the executed scope.

## The reader and its receiver

At final clock3, let Z contain 1,024 packed image rows and E contain 512 contextual text rows. Original input projections, source layer normalization and compiled clock modulation form the first joint block's image/text receivers. Original image Q/K/V and text K/V maps, head RMS weights and absolute RoPE then give

\[
H_0(Z,E)=\operatorname{Attention}
\left(Q_i,[K_t;K_i],[V_t;V_i]\right).
\]

All 1,024 image queries read all 1,536 keys/values through 24 heads of width128. Text queries, attention output projections, MLPs and later blocks do not run. Image-query-only dispatch can differ finitely from full native query geometry, so source equations alone do not establish bit parity.

The candidate writer is rowwise and shared across positions:

\[
\phi_i=[Z_i,v_{\mathrm{prev},i},H_{0,i}],\qquad
\widehat v_{3,i}=Q_{\mathrm{BF16}}
\left(v_{\mathrm{prev},i}+\widetilde\phi_iB+b\right).
\]

There are3,328 input features and128 outputs. Discovery-only FP64 means and standard deviations define the standardized feature; an exactly zero deviation becomes one. The finite output policy is FP64→FP32→BF16 round-to-nearest-even. No prompt/seed identifier, target lookup, full-denoiser fallback or fitted gain enters serving.

## One inspectable proposal

Eighteen old even-seed discovery fields supply native velocity innovations Y=v_native−v_previous. Their own source-native H0 reads are recomputed and charged. The writer minimizes

\[
\frac1N\|XB+\mathbf1b^\top-Y\|_F^2
+\frac1M\|\Delta XB-\Delta Y\|_F^2
+.01\|B\|_F^2.
\]

The two paired discovery contrasts exclude the bottom-right8×8 edit and its four-neighbor halo, leaving944 queries per pair. Local Z/previous operands match exactly on that mask. There are18,432 ordinary rows and1,888 paired rows. A single FP64 centered normal-equation solve gives B and an unregularized intercept; there is no epoch loop, gain/ridge search or feedback selection.

Saturn Frames/Acts bind the discovery packet and coefficient write. The zero-writer baseline and fitted result are immutable branches; selecting the original Frame again verifies baseline identity, without claiming restoration of a mutated solver or model. Coefficients seal before development inputs become available. Fresh CPU fit/readout actors reject original model/framework/network access and undeclared target files, including three canary reads. The resident GPU producer still performs every declared live H0 read. Thirty typed output handoffs bind emitted velocities to their Z/previous/H0 and coefficient seal.

Eighteen odd-seed feedback fields and eight separately retained evaluation fields cannot affect fitting. The two main inputs use held v1, while canonical discovery generally uses v2. Their images therefore include history and scene transfer; failure does not isolate reader representation. Foreign comparisons use velocity only because their old CPU VAE normalization differs from the main GPU consumer. All candidates seal before scoring. Historical labels remain retrospective even when file access is delayed.

## What the original consumers show

| Case | Hold velocity RMS error | Candidate velocity RMS error | Raw-H0-zero velocity error | Hold PNG RMS error | Candidate PNG RMS error |
|---|---:|---:|---:|---:|---:|
|00|.41461|1.17838|2.78453|9.46357|31.37151|
|10|.43715|1.15378|2.78872|13.06402|31.79950|

Errors use the full same-state native final read HN as reference. Mean per-field velocity RMS is.58636 for the18 feedback candidates versus.40429 for hold; the eight evaluation fields give.57001 versus.40368. Every feedback/evaluation field loses to hold, so main held-v1 transfer cannot explain the entire gap. These are descriptive dependent-field averages, not independent task estimates.

![Original candidate and historical images](results/source-native-reader-figure/source-native-reader.png)

The candidate retains pitcher/can silhouettes, turns teal surfaces largely gray, and adds red/pink color and patterned surface artifacts. Those silhouettes already exist in the supplied latent after paid native prefixes. Their presence does not establish that the new reader learned to construct objects from noise.

Signed main color-response errors are74.30% in velocity,56.96% in final latent and54.40% in PNG. Hold gives59.59%,45.67% and25.59%. The same writer with raw H0 set to zero has worse endpoint errors but62.13% velocity-response error, better than the complete candidate. This control sets raw H0=0 before standardization; it is neither a zero standardized contribution nor a matched refit. Endpoint usefulness and response usefulness differ.

On all four distant-edit evaluation contrasts, complete velocity-response error is93.60–93.81%, slightly below hold's100%. At944 unchanged queries, error is147.11–159.38%, with response cosine from−.0105 to.0093. The apparent complete-response improvement does not establish a useful nonlocal response. Per-field and33,792 dependent row observations remain available without quality filtering.

## The mathematical gap this exposes

At an unchanged query, local Z/previous differences vanish, so before final quantization the current shared writer predicts

\[
\Delta\widehat v_i=\Delta H_{0,i}B_H.
\]

The current experiment evaluates exactly that response function, rather than inferring usefulness from H0 similarity. Its near-zero response alignment can reflect the chosen source representation, a restricted linear writer, discovery coverage, or transfer to the new edit location. Those causes are not separated by this run.

For an attention read A=Σ_j p_jv_j, its real-arithmetic differential is

\[
dA=\sum_jp_j\,dv_j+
\sum_jp_j(v_j-A)\,d\ell_j,
\qquad \ell_j=q^\top k_j/\sqrt d.
\]

That shows how global availability becomes a particular query-dependent response. It does not guarantee that later native MLPs/blocks express their velocity innovation as one fixed linear image of the first read. The next question is which part of the measured response can be expressed by the available features, before adding another learned layer or changing the producer.

## Compute and evidence

Job job-1892a00329e9 succeeds in65.57scheduler seconds with3,867.7MB sampled RSS and1,802MB VRAM. It performs46 source-native reads:18 discovery and28 subsequent own-input reads. Their322 original affines cost2,352,299,900,928 nominal MACs;46 attention calls cost444,529,115,136 QK/AV MACs. Summed instrumented reader time is2.0693seconds, including1.9103seconds for discovery and.01134seconds for the two main reads. First-use and instrumentation effects prevent a matched speedup claim.

Per served read, original affine cost is51,136,954,368MAC plus436,207,616 for the writer, totaling51,573,161,984 versus4,723,927,154,688 for the full final call. Attention adds9,663,676,416MAC separately. Ten selected original tensors occupy142,344,960bytes; fitted FP64 coefficients/statistics occupy3,462,144bytes. Clock constants, current state, workspaces, full asset files and VAE are additional costs.

The one fit takes6.9783seconds/11.1903CPU seconds. Its Gram contractions cost225,055,866,880 nominal MACs, RHS8,655,994,880, bias425,984; the dense-LU solve estimate is27.408billion FLOPs. Two post-fit training products add8,655,994,880MAC. These are shape-derived arithmetic accounts, not hardware counters. The30 serving writer products add13,086,228,480MAC and take.8496seconds/2.9200CPU seconds in aggregate including field retention.

Two native VAE decodes cost80conv/linear calls and2,481,224,417,280 nominal MACs, with2.8179/.0535seconds recorded. Four historical main-prefix full denoisers, original encoding and all old discovery-label production remain paid. There are no new full denoisers, encoders, reference images or noise draws.

All228 mechanics tests pass. Four new tests check the reduced normalization/RoPE/attention against an independent tiny NumPy calculation, paired ridge against augmented least squares, raw-head deletion/two-stage quantization, and fresh-process file quarantine. The publication audit checks46 typed native reads,30 writer handoffs, two exact final-consumer joins and33,792 lossless rows. Independent byte verification covers518files/8,688,976,825bytes. Large arrays remain near compute; the public bundle has29files/6,982,540bytes before separately derived evidence/figures.

The guarded executed payload is b56d4f7a3029259f0967975c411d1af535575ad07e5922595b64a508faed0685,3,198,278bytes. Full manifests, original PNGs, fits, costs and seals are under [results/source-native-reader](results/source-native-reader/report.json). The source uses the qualified packaged Saturn runtime under its mdb namespace. The initial local staging import stopped before payload creation because this shared environment does not expose saturn.job_support; no model ran under that uncorrected import.

The initial Saturn scan of the compressed public bundle reports evidence-incomplete-empty-target with exit2. Scanning the original uncompressed collected stream instead returns run/verify0over33,792rows, with no fires or formal claim receipts. This qualifies observation-stream mechanics, not semantic quality. The additive [instrument receipt](results/source-native-reader-evidence/saturn-instrument.json) preserves both outcomes.

The broader connection is the [MLP-to-reader composition](../../obsidian/blog/2026-11-06-180411-a-cheap-mlp-feeds-a-cheap-attention-program.md): actual operand handoff can work while response fidelity remains inadequate. The [longer-context result](../../obsidian/blog/2026-11-06-183300-longer-chat-context-still-needs-the-right-relationships.md) makes the analogous point for retained memory and relationships. The [joint incoming-state/write result](../../obsidian/blog/2026-11-06-182756-binding-needs-incoming-state-and-new-writing.md) asks which writer and carrier preserve a binding through its next real consumer. These results favor specifying producer, read, response and future together; they do not turn a reduced learned calculation into arbitrary-prompt zero-neural serving.

Astra selects a [response-only saved-array writer contrast](plans/source-native-response-writer.md) next. Keep the discovery scales/pairs/mask/ridge/gain and fit only the mapping from head changes to native velocity changes, without endpoint loss or intercept. Compare raw responses with zero and the old writer on same-location feedback and top-left evaluation. Seed and location remain confounded. This written, unexecuted test needs no new reader or image computation and cannot itself supply a full velocity.
