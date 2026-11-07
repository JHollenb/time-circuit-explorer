# Hold the first joint text write fixed across the image-carrier response

Status: **written and unexecuted**. Astra46 selects this single follow-up to the [carrier–receiver crossing](../carrier-receiver-crossing.md), before another fit, gain/rank scan or derivative sweep.

The first joint text attention reads the image keys/values. Identical initial text conditioning therefore does not hold its output write fixed across different image receivers. Cycle46 kept that path native. The next question is how much of its measured response difference changes when this one concrete co-carrier is fixed.

Reuse the two complete image payloads H_A/H_B, R_1 receiver Z, common inputs, native numerical profile and all169 transformer tensors from cycle46. Reuse the already archived R_0 textWrite tensor at the output of `transformer.transformer_blocks[0].attn.to_add_out`. Its source is independent of the A/B image clamp because this native text projection executes before the image projection boundary. Bind exact BF16 shape [1,512,3072], words, source hashes and lineage.

Run only two ordinary B1 native calls. Both retain the existing complete image-input clamp. Both also replace the first text-projection output with the R_0 textWrite payload using a forward output hook. Preserve native output shape/dtype/device/stride/storage offset in separate storage, replace the entire tensor rather than adding a delta, and verify one write at each of the two declared addresses per call. Native text attention/projection and image attention still compute before being discarded; all of that work is paid. Use Saturn Frames/typed Acts to bind the two writes and immutable input/model closure.

Define G(r,t,h) as the declared suffix with receiver r, supplied first text write t and supplied image carrier h. From cycle46,

\[
d_0=G(R_0,T_0,H_B)-G(R_0,T_0,H_A),
\quad d_1=G(R_1,T_1,H_B)-G(R_1,T_1,H_A).
\]

The two new velocities supply

\[
d_{10}=G(R_1,T_0,H_B)-G(R_1,T_0,H_A).
\]

Seal both new complete BF16 velocities before reading old velocity targets for scoring. Then retain FP64 d10 and the exact ordered decomposition

\[
I=d_1-d_0=(d_{10}-d_0)+(d_1-d_{10}).
\]

The first term is the remaining receiver-context response difference with the first text write held at T0. Later text–image feedback remains live, so it is not a pure image-path term. The second is the response change from restoring T1 in receiver R1. They are reference- and order-dependent contrasts, not universal causal shares. Retain their full signed fields, energies, inner product, cancellation, all1,024query observations and the predeclared944source-nonlocal subset. Do not turn the terms into positive percentages or select a model based on an arbitrary magnitude gate.

Primary inputs may contain R0's archived textWrite and H_A/H_B, but no old velocity targets. Keep old velocity references quarantined until the two new outcomes seal. No new donor production, text encoder, fit, noise, image scheduler update, VAE, image, reference campaign or branch search is needed. Historical donors/conditioning/velocities remain paid preparation.

Expected work is two original transformer calls,218affine calls,50attention dispatches,**9,448,195,620,864 affine MACs** and**724,775,731,200 QK/AV MACs**, with all original clock/image/text work included. Use one guarded CUDA mrun lease, strict preflight, no retries, unchanged scalar geometry and the measured cycle46 resource history rather than the inapplicable generic B64 alias estimate. Keep the initial instrument below15minutes; if that expectation changes, apply the workspace's longer-run review requirement.

First run a tiny independent two-boundary hook/stride/offset/alias/byte-consumption mechanics check. Only integrity, finite, source/profile, hook delivery and resource failures may stop execution. Preserve valid outcomes of either sign and size. Full scientific byte custody, original-source freeze, lossless per-query publication and post-job Saturn applicability checks follow execution.

This intervention remains on a supplied-carrier domain and can create unnatural hybrids. It does not show text alone is sufficient, identify the unique missing state, establish a cheap receiver-conditioned operator, prove natural H0 information insufficient, or supply arbitrary-prompt zero-neural images. Its purpose is to separate one existing co-carrier before proposing a more complicated producer.
