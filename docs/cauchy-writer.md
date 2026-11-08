# Let the receiving computation choose the scale of a state update

One parent-preserving constant update lowers the actual finite discovery loss from 0.488854 to 0.488057, achieving 87.25% of its predicted reduction. It uses the full geometry of the receiving normalization instead of the earlier isotropic approximation. Three objective terms improve; raw-state response worsens. The fixed routes still produce no vessel-padding response, so this is a small successful update of one contextual-state approximation, with arbitrary-prompt zero-neural images unresolved.

The [previous audit](writer-direction.md) found the disagreement this experiment addresses: all four actual writer updates retained a negative local-model slope, but all four made the finite objective worse. The new candidate's direction and curvature belong to the same residual-square model. [Results and byte custody](results/cauchy-writer/publication.json) retain the complete future and its limitations.

The object being changed is a small non-neural program that approximates the first contextual text state H1. Its two routing trees select 14 and 9 constant vectors. The routes stay fixed. Only their 58,880 FP32 coordinates change. This is one component on the path from raw text to model state; it is not a complete prompt-to-image generator.

## The program determines which directions are available

Let θ contain the23 existing constant vectors, and let B gather the selected vector from each slot and add them into the discovery rows. For an ideal perturbation r of those constants, the state direction is q=Br. Several constant updates can represent the same state direction: B can have a null space. There is no reason to invert a large singular parameter Hessian just to choose a useful direction.

The saved gradient convention omits a common factor: the local objective derivative in state direction q is gᵀq/2. Give each leaf a diagonal mass D equal to the sum of the original positive endpoint weights of its discovery rows. Then

\[
r=-D^{-1}B^Tg,\qquad q=Br,
\]

\[
g^Tq=-(B^Tg)^TD^{-1}(B^Tg)\leq0.
\]

This chooses descent inside the program's existing language. It cannot create a missing distinction: two histories with the same lexical row and the same two leaf assignments still receive identical outputs after any constant update.

## The next consumer supplies the geometry

The next original computation normalizes each row and applies a learned coordinate scale γ. For the ideal real-valued map, with s=√(mean(h²)+ε),

\[
J_hq=\gamma\odot\left(\frac q s-
\frac{h\langle h,q\rangle}{d\,s^3}\right).
\]

The derivative is a diagonal scaling plus a rank-one correction. It couples the coordinates through one row dot product. Its transpose is different when γ varies across coordinates. For a response between two histories, the change is J_jq_j−J_iq_i: each row has its own baseline and Jacobian.

Let P take the original oriented pair differences, and retain the original endpoint and pair weights W,V and four native-energy denominators E. The two state metrics are

\[
M_H=W/E_{HE}+P^TVP/E_{HR},\qquad
M_Z=W/E_{ZE}+P^TVP/E_{ZR}.
\]

A zero reference uses denominator 1, as in the original objective. The joint constant-space curvature would be

\[
C=B^TM_HB+B^TJ^TM_ZJB.
\]

We need only its energy along r, not the matrix itself:

\[
\beta=r^TCr=q^TM_Hq+(Jq)^TM_Z(Jq).
\]

Evaluate four nonnegative endpoint/pair energies. The one-dimensional Gauss–Newton model is

\[
m(\alpha)=L_0+\frac\alpha2g^Tq+\frac{\alpha^2}{4}\beta,
\qquad\alpha_{model}=-g^Tq/\beta.
\]

Unlike the earlier isotropic surrogate, a coherent residual-square model cannot predict negative squared loss in exact arithmetic. That is a consistency advantage, not a guarantee that finite execution improves.

## A bounded future, followed by actual feedback

The experiment caps the ideal row-relative step at η=1/16. This is a declared exploratory choice, not a calibrated safety theorem:

\[
\tau=\max_i\frac{\operatorname{RMS}(q_i)}{\sqrt{\operatorname{mean}(h_i^2)+\epsilon}},
\qquad\alpha=\min(\alpha_{model},\eta/\tau).
\]

One candidate is admitted to FP32 constants and runs both original BF16 state additions. The finite program includes rounding at the intermediate state, so it is not the ideal affine map B. Its objective is a discrete function; the ideal Jacobian is an anchored explanatory relaxation, not its literal derivative.

Saturn records the proposal and actual normalization as typed transactions. Discovery A/B/00/10 supplies feedback. The exact parent wins ties and is restored if the candidate raises the actual four-term loss. Parent, candidate and accepted programs then run in a fresh tokenizer/stdlib process on seven raw prompts. Only after selection and those outputs are sealed do the historically known 01/11 transfer targets become available. A rejected candidate still receives its transfer diagnostic. The seventh novel prompt has no quality target.

The [frozen assay](plans/cauchy-writer-assay.md) permits at most six new original normalization/gamma evaluations, zero original affine MACs, one fresh serving child and no image. Native targets, the saved gradient, the lexical table and original gamma remain paid research inputs. The serving program performs branches, lookups and finite additions; there is no neural model in that serving child.

## The actual future improves modestly, with visible regressions

The analytic ray minimum selects α=0.0440808543; the trust bound would allow 0.342665508. The model predicts loss 0.487940817. Actual two-add BF16 execution followed by the original normalization gives 0.488057208, a 0.1630% relative reduction from the parent. The candidate is retained under the declared discovery policy. This is one observed improvement, not convergence or a general descent theorem.

| Original globally normalized term | Parent | Candidate | Direction |
|---|---:|---:|---|
| H endpoint |0.363412505|0.363342044|improves|
| H response |0.607894450|0.608181050|worsens|
| Z endpoint |0.401433121|0.400725171|improves|
| Z response |0.582676027|0.579980566|improves|

These are the original four loss terms, averaged equally for selection. Stratum contributions retain their original global weights. They are not separately renormalized into new selection objectives.

The aggregate gain does not cover every row or response. Discovery endpoint errors improve on 1,307 of 2,040 H rows and 1,353 Z rows; 733 H rows and 687 Z rows worsen. Known-transfer endpoint errors improve on 634 of 1,020 H rows and 654 Z rows, with 386/366 regressions. Aggregate H and Z endpoint errors improve slightly in all six histories. These relative errors use the corresponding native endpoint as denominator, rather than the H1−H0 increment. Transfer is historically known and was released after selection; it is not newly heldout evidence.

Cube unchanged-active Z response error falls 23.3867%→22.2347%, while cube unchanged-padding Z response error rises 89.1174%→89.2737%. Vessel unchanged-active Z response improves 90.5551%→88.8918%; its known-transfer counterpart improves 90.6636%→89.0207%. H response worsens in every pair. Vessel padding keeps exactly zero predicted H and Z response on 475 discovery and 475 transfer rows, with 100% relative response error. Changing constants cannot break the existing address collision.

The ideal and actual changes also differ. The largest ideal row-relative step is 0.8040%, while the actual admitted state reaches 0.9077%. The ideal local slope is −0.001826418 and the admitted slope −0.001853803. Squared change energy rises 0.230205→0.672759 after both state admissions; this is a measured discrepancy, not a causal attribution of the finite loss gap. The polynomial versus direct linearized residual-square score differs by 5.55×10⁻¹⁷. All floating regrouping and scalar-reaggregation residuals remain visible.

## Compute and evidence

One guarded Beast job, `job-914bc8f1fc5b`, finishes in 24.82 seconds. It pays six original full512-row normalization/gamma evaluations, zero original affine MACs and no image. The proposal Act takes 0.418 seconds; the fresh seven-prompt, three-program serving child takes 14.20 seconds including its launcher. The child reports 54,835,200 coordinate additions, 13.24 CPU seconds, 51 unique lexical rows and zero neural evaluations. Those 21 dependent demonstrations are not a per-prompt latency benchmark. No full model or checkpoint shard is loaded; saved original gamma occupies 5,120 bytes.

The scheduler observes 1,928.4 MB peak RSS and 296 MB VRAM; the worker reports 17,044,480 bytes peak CUDA allocation. The worker's 22.68 seconds excludes initial imports/request verification, so it is a different timing scope. The four-CPU-slot reservation contains one numerical thread. Intent-first admission reserves 4,096 MB each of RAM and VRAM.

Eight supplied mechanics checks pass once, including a nonuniform-gamma adjoint identity, a separately constructed dense Jacobian witness and exact signed-zero preservation. The first stage stops before any sealed request or job because a required analysis-family label is missing. A separately [reviewed metadata adapter](results/cauchy-writer-evidence/advisor-stage-v2-source.json) adds that label; the mathematical and worker sources stay unchanged. There is one scientific job and no scientific retry.

The first publisher verifies 29 compact members totaling 4,619,726 bytes, 102 fresh remote files totaling 1,268,921,378 bytes, 18,432 endpoint/pair observations and 2,040 admission rows. Public admission scalars pack exactly to their typed FP64 output. All 144 report panels and the original global-weight losses are independently reaggregated from public scalar rows; small RMS-resquaring and reduction-order differences are diagnostics. Vector construction and actual normalization remain source-qualified producer measurements, not independent vector replay. Seventeen new numeric fields totaling 115,650,560 bytes remain on Beast.

The generic evidence sweep separately reports an unsupported empty target for this custom compressed/typed format. Its `evidence-incomplete` result describes instrument coverage. It does not erase the dedicated custody, typed transaction or scalar checks, and it does not certify a model capability.

Astra's [actual-result review](results/cauchy-writer-evidence/advisor-actual.json) retains all response countertrends and confirms that no new responding row appears. All 475 vessel-padding rows share the same current token and selected leaves in both the discovery and known-transfer pair. Existing first/last token registers distinguish the swap, but the selected addresses ignore it. An in-place split of slot1 leaf8 would add a predicate level beyond its current depth four; that proposed change must state its actual budget.

The next full research/storage review is 114. Before repeating constant updates, the useful constructive question is which input-owned route distinction can separate the colliding histories while preserving the exact accepted program as a null. That extension is not implemented or admitted by this experiment.

The broader question remains the one developed in the [state and time-circuit mathematics](../../obsidian/blog/2026-11-05-153154-state-context-and-the-mathematics-of-time-circuits.md) and the [MLP work on predicted states](../../obsidian/blog/2026-11-06-111533-improving-an-mlp-on-its-own-predicted-states.md): can a compact computation construct the state that the next computation actually needs, from its own preceding state? A better local update would move that question forward. It would leave contextual coverage, full denoising, general pixels and independent prompt faithfulness unresolved.
