# Compose mask-dependent K/V reduction through the complete image demand

Status: executed in cycle26 as `job-394733a01129`; [result and computation account](../complete-masked-encoder-image.md). The protocol below records the original frozen scope. Both candidates match all retained computed fields and original PNG bytes for this pair; most learned computation remains. Selected by Astra after the [same-noise image consequence](../conditioning-image-consequence.md). No further precision search or intermediate image-attribution study is included.

Before tokenization, freeze these two fresh strings:

1. **A:** `A red cube to the left of a blue sphere on a plain white tabletop. Two separate objects, studio photograph.`
2. **B:** `A red cube to the right of a blue sphere on a plain white tabletop. Two separate objects, studio photograph.`

Use the complete lexical atom store to begin each raw-prompt path. Preserve exact tokenizer/template, positions and 512-row mask identities. Reuse the existing qualified block0 lexical-prefix construction; disclose every learned block0 operation that remains. In every remaining demanded Qwen block **1…26**, form K/V only for mask-permitted key rows, using the already selected K/V-only accumulation policy. Query, residual, normalization, output and MLP computations retain all512 rows, including padding. Do not interpret masked keys as dead queries.

For A={j:m_j=1}, the reusable dependency rule is

$$
K_\ell=W^K_\ell N(H_\ell)_A,\qquad V_\ell=W^V_\ell N(H_\ell)_A,
\qquad \ell=1,\ldots,26.
$$

Each block consumes its own preceding H; pack its own H9/H18/H27 into E. No cached contextual native state, layer-specific rescue, reference replacement or added approximation is allowed. Omitted K/V rows remain unconsumed; changed GEMM geometry means this finite implementation remains approximate. Per prompt, nominal eliminated projection work is

$$
26(512-|A|)\,2560\,(2\cdot1024)\quad\text{MACs},
$$

with actual |A|, primitive geometry, gathers/scatters, logical bytes and runtime accounted separately. MACs are operation geometry, not measured energy or device FLOPs.

Render candidate A/B with original pinned native Klein4B/BF16/512px/four-step FLUX and VAE. Reuse the already sealed seed314159 noise from cycle25, cloned independently, and identical fresh schedulers. Seal both complete candidate outputs before executing corresponding full-formation references. Full references retain the same numerical profile and source prefix; disclose any independent native prefix check within their cost. All four branches own their latent trajectories. If memory requires separate encoder/image phases or repeated model loads, account for them; do not hide them in speed comparisons.

Use four images total: candidate/full for each prompt. Retain and independently verify complete conditioning, final latent, raw decoder, processed float RGB and native PNG. Bind actual H handoffs, E reads, initial noise, schedule and native consumer counts. Inspect endpoint differences, signed differences and the paired prompt response

$$
\Delta I=I_B-I_A,\qquad
\widehat{\Delta I}=\widehat I_B-\widehat I_A,\qquad
e_\Delta=\widehat{\Delta I}-\Delta I.
$$

Report absolute/relative RMS, MAE, reference amplitudes and spatial collateral with undefined zero denominators preserved. Inspect whether the requested left/right relation changes coherently in both reference and candidate. Treat these two dependent prompts as exploratory examples, not a powered semantic-quality test. Do not discard rows or stop collection for numerical accuracy.

Implement one finite guarded mrun CUDA lease with strict source preflight, no retries and an honest physical B1 geometry. Inspect intent-first sizing and compatible measured history before launch; the generic diffusion B64 preview is known to be inapplicable. Freeze implementation and resource details before staging. No duplicate full suite on the accelerator. Full arrays stay near compute; compact evidence and original images are public.

This is a bounded first-prompt computation reduction and a test of reusable composition. Remaining original learned encoder, FLUX and VAE work is paid. Arbitrary-prompt generation with zero neural evaluations remains the unchanged, unsolved objective.
