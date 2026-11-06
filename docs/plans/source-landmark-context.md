# Can source-frozen attention readers preserve new prompt changes?

Status: **implemented and measured in cycle9**. The [source-landmark result](../source-landmark-context.md) retains this design and its interface-freeze implementation detail. The source-only rule runs on eight new strings, but its broad normalized-reader mixture loses changes. A separate retrospective bandwidth/capacity diagnostic finds both excessive smoothing and a remaining fixed-response-bank restriction. This is reduced model-derived attention, not faithful full conditioning or zero-neural images.

The smallest next experiment is64 fixed source-query landmarks with live prefix statistics. This evaluates reduced model-derived attention. It does not satisfy the original zero-neural image objective merely by replacing a dense operation with a smaller learned computation.

## Freeze a rule before new prompt outputs

Use the existing complete lexical atom store, source-qualified template/tokenizer,512-position policy, head0/KV0 and the cycle7 source ID list. All source operands are from the first token-local prefix; later contextual layers cannot use `(token ID, position)` as their full state identity.

Define scaled operands once:

\[
x=q/d^{1/4},\qquad y=k/d^{1/4},\qquad d=128.
\]

Then the original score is xᵀy. Compile the full source key covariance C_y over the declared uniform2,048-ID×512-position distribution using stable FP64 block moments. The source score-distance is

\[
\delta_a^2(x)=(x-\ell_a)^\top C_y(x-\ell_a)
=\operatorname{Var}_{y\sim\mathrm{source}}[(x-\ell_a)^\top y].
\]

This measures variation in the source score displacement. It is not a bound on normalized-read error for fresh prefixes. With unscaled q/k the equivalent expression divides by d, not √d. Preserve the scaling convention in the compiler, routing and receipts.

For a bounded candidate pool, sample4,096 distinct token/position pairs uniformly without replacement from that source grid and add every padding-token position. Freeze the PRNG seed and exact identities before test-port or head reads. Deduplicate identical identities, retain lexical/padding sampling weights, and record any overlapping selections. The candidate source-query mean uses a50/50 lexical/padding mixture. Choose the first landmark nearest that mean in the covariance metric, then choose63 more by deterministic farthest-first coverage in the same metric; break ties by token ID and position. No head target selects a landmark.

Set τ² to the median strictly positive nearest-other-landmark squared distance. Use the smooth all-landmark routing law

\[
w_a(x)=\frac{\exp[-\delta_a^2(x)/(2\tau^2)]}
{\sum_b\exp[-\delta_b^2(x)/(2\tau^2)]}.
\]

Use stable normalization and retain routing entropy/distances. Declare and record a finite degenerate-source policy if no positive distance exists; do not choose a new bandwidth from held errors. A covariance square root may accelerate distances; retain its numerical reconstruction and any roundoff-only PSD repair. Source artifacts, their construction and routing arithmetic are paid information and computation.

## Advance exact landmark readers on the actual prefix

For each allowed key/value occurrence, maintain

\[
Z_{a,i}=\sum_{j\le i}m_j e^{\ell_a^\top y_j},\qquad
N_{a,i}=\sum_{j\le i}m_j e^{\ell_a^\top y_j}v_j.
\]

Each N_a/Z_a is the exact mathematical attention read for one fixed landmark on the **current** allowed prefix. Preserve repeated occurrences and all512 query rows, including padding queries. No head output or exact current-query fallback enters the candidate.

Use per-landmark running maxima g_a, scaled partitions z_a and scaled numerators M_a. For an admitted score s_a, let g′_a=max(g_a,s_a), rescale old statistics by exp(g_a−g′_a), and add exp(s_a−g′_a) times the new value. The primary read is

\[
\widehat h_i(x_i)=\sum_a w_a(x_i)\frac{M_{a,i}}{z_{a,i}}.
\]

It is a convex combination of actual prefix values in exact arithmetic. That removes signed-denominator cancellation; it does not guarantee fidelity to the current query. Empty-prefix and nonfinite-input behavior belong to the mechanics contract, without an oracle substitution.

Two controls reuse precisely the same state. The nearest-landmark read separates quantization from smooth interpolation. The kernel-mixture control is

\[
\widehat h_{K,i}(x_i)=
\frac{\sum_a w_a(x_i)e^{g_{a,i}}M_{a,i}}
{\sum_a w_a(x_i)e^{g_{a,i}}z_{a,i}}.
\]

Evaluate its cross-landmark sums stably. Mixing M/z and mixing kernels are different laws: the latter changes route weights according to each landmark's true partition. The individual maxima must not be omitted merely because they cancel inside each normalized landmark read. Uniform allowed-key mean remains a separate control.

## Test fresh strings and small changes

Freeze a small new cohort of four paired raw-string changes, including Unicode and a long-order case, after source specification but before exact references. Freeze actual strings, tokenizer/mask results and all artifact/source hashes in the request. The cycle7 eight strings are development controls, not fresh evidence. The pairs should preserve equal active-token counts where practical; report mask differences explicitly rather than hiding changed support.

Fresh serving receives raw text, source artifacts and the complete lexical store, then forms its own operands and advances its own statistics. It receives no saved target Q/K/V packets or heads. Seal full candidate heads, route diagnostics and selected state checkpoints before source-qualified canonical-port and exact-attention scoring. Independent mathematical attention is an offline reference, not an online speed trick or a claim of native Flash parity.

Lead with each paired-response error/cosine, absolute exact/error RMS, active/padding splits and ideal saved W_o-slice transport. Retain whole-head errors, maximum value-range deviations, routing entropy, source-distance coverage and nearest/kernel-mixture controls. No aggregate accuracy gate removes rows or defines mechanism absence. A positive read can still erase or distort a small contextual change.

## Charge the whole computation

At64 landmarks and value width128, the core FP64 g/z/M plus clock/count state is66,576bytes. Charge landmarks, covariance/routing transforms, source artifacts,2.645GB lexical store, workspaces and outputs separately. Count activeKeys×64×128 landmark-score MACs, exponentials, state updates, query metric/routing work and value reads. Source compilation and warm/cold serving timing remain separate. This method may cost more than exact attention at a short prefix; no speedup is assumed.

Use tiny synthetic checks for stable/resumed updates, masks/multiplicity, convex weights and the difference between normalized-reader and kernel-mixture laws. Run the real source compilation/fresh assay through one guarded finite CPU mrun lease near the retained store, strict preflight and no automatic retry. Publish compact complete observations and verify retained-byte identities; do not hydrate the full store onto the development machine.

An informative result would supply a source-defined rule that consumes unseen raw strings and retains their changes. It would still cover one first-layer head. Later contextual conditioning, autonomous diffusion, general non-neural pixels and independent image/prompt evaluation remain required. First-layer routing/features depend on a finite token/position domain and might eventually be compiled; do not build another large table before this bounded reader is informative.

Related: [active research record](../zero-neural-research.md), [source-frozen geometry](../source-conditioned-context.md), [common-bank raw evidence](../results/common-bank-context/provenance.json), and the local [OVC15 positivity countertrend](/Users/jakeholl/domains/obsidian/blog/2026-11-05-161109-the-one-assumption-that-would-make-model-computation-easier-to-understand.md).
