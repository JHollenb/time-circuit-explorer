# Execute fewer padding-query trajectories, then reconstruct demanded conditioning

Status: cycle35 next cut, written and unexecuted. [Astra's review](../results/padding-mlp-tangent-evidence/advisor.json) selects this after the [source MLP tangent](../padding-mlp-tangent.md) exposes large nonlinear curvature. Do not tune that lexical anchor or increase Taylor order on the known pair. Preserve complete nonlinear computation on selected own-state trajectories and test the reconstruction at the actual image consumer.

Use the existing square 00/10 raw strings, seed 271828 noise and historical full-attention targets. Original native attention runs in block 0 and all 27 blocks. Historical full is the primary comparison; historical top4 is secondary context. This is a new native compact-query assay, not a continuation of the top4 first-reader approximation. No new reference image, noise draw, fit or policy sweep.

## Exact structural condition

Let A be all mask-active rows, P all sorted original padding positions and S=A∪Q for a retained subset Q⊂P. Require that every key read by a retained query is also in S. Padding keys must remain masked at every layer. Normalization and MLPs are rowwise; preserve parameters, original positions, mask, RoPE and execution mode. Under those conditions,

\[
\pi_S F_\ell(H)=F_\ell^S(\pi_S H)
\]

in real arithmetic. Active rows depend on active causal predecessors; retained padding queries depend on their own row state and the active bank. Omitted padding trajectories cannot alter retained trajectories. This is an architectural closure condition, not an approximation-quality assumption.

Authenticate the actual native mask and check its retained-query reads before execution. Slice both axes of the original mask. Gather the original position IDs and RoPE coordinates; never renumber selected padding rows, turn padding into keys or change causal eligibility. Use the compiled lexical H0 from the pinned source table and native contextual blocks. There is no original first-reader output substitution.

CUDA BF16 compact GEMM/SDPA can differ from the historical 512-row profile. Declare exact numerical flags and shapes before execution. Preserve every selected state through all layers and compare active/retained rows afterward. Active differences measure finite compact execution and its propagated effects, because omitted padding queries have no real-arithmetic path to active states. Do not call the compact result bit-exact without actual evidence.

## One fixed selection and expansion

If |P|≤16, retain all padding rows. Otherwise set

\[
Q=\{P_{\lfloor j(|P|-1)/15\rfloor}:j=0,\ldots,15\}.
\]

This includes both endpoints and uses no target values. Keep every active row. For each omitted padding position, copy the retained padding row at the nearest original index; ties go to the earlier index. Define that fixed expansion operator R once from mask/positions.

Execute all 27 original blocks on the retained rows' actual own states. At H9/H18/H27 expand each tapped field to all 512 original rows, preserving active rows at their own addresses, and pack the full conditioning E. Intermediate layers use compact own states, never expanded or reference states. Full original FLUX and VAE consume this complete E and the same inherited noise/schedule/compiled clocks. Only two new candidate images are generated. Seal both complete trajectories and images before diagnostic reference values become available.

Copying rows preserves row count and representative multiplicity. It does not preserve the original attention partition, mass or payload. Do not multiply an embedding by its multiplicity. Whether copied padding conditioning suffices is the consumer question.

## Diagnose numerical execution and reconstruction separately

Let P_sel denote selection on the tapped historical full conditioning, distinct from the padding index set above. For actual compact taps E_small and the full native target E,

\[
\widehat E-E
=R(E_{\rm small}-P_{\rm sel}E)+(RP_{\rm sel}-I)E.
\]

Only after sealing, retain both complete terms, energies, signed cross term, row distributions and finite closure discrepancy. The first includes compact numerical execution and all its propagated effects; the second is the fixed reconstruction limitation on historical native taps. Neither diagnostic term may enter candidate serving. Compare endpoints and signed color response through E, first velocity, final latent and PNG. Inspect original images for object/color/order and existing native defects. Approximation measurements remain observations, not exploration gates.

Keep full observed masks, original selected indices, every actual state handoff and denoiser E/noise/clock input join. The consumer must receive exactly the candidate E that was sealed. Reference states are paid historical diagnostic resources, not supplied candidate receivers.

## Compute, custody and stopping scope

For the current 37 active rows, 16 padding queries give 53 retained rows. Each native layer uses 100,925,440 nominal affine MACs per row. All 27 compact blocks cost 144,424,304,640 MACs per prompt, compared with 1,395,193,282,560 for512 rows:89.65% less encoder affine arithmetic, approximately 6.16% of the inherited encoder-plus-denoiser affine total. Count attention, normalization, copies, buffers, lexical storage, checks and original image work separately. Native parameter bytes remain loaded, full E/image buffers remain, and no speedup is assumed.

Two prompts pay 378 encoder linears,54 native attention calls,8 denoisers and 2 VAE decodes. This is learned computation on fewer trajectories; it does not meet the strict zero-neural objective. A successful result would justify further structural compilation; a poor result would expose how much demanded padding variation the consumer needs.

Use one finite CUDA mrun lease with measured serialB1 history, exact source and guarded payload custody, no retries and no duplicate GPU suite. Qualify selection/mask slicing, nearest expansion, source identity, actual own-state continuation and the diagnostic telescope with small mechanics checks. If expected duration exceeds15minutes, obtain the required Luna execution review. Keep full arrays near compute, independently verify bytes, preserve countertrends, update the living blog and retain the full original goal.
