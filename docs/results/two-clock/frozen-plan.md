# Omit an earlier velocity read, then test downstream recovery

Status: **written and unexecuted**, selected by [Astra37](../results/last-velocity-evidence/advisor.json) after the [last-velocity result](../last-velocity-forecast.md). This is a known-own-prefix component experiment; arbitrary-prompt zero-neural generation remains open.

## Question and common parent

Can the final native reader recompute what an omitted earlier reader would have supplied? Reusing the final velocity approximately freezes an already formed clean-latent estimate. Test one earlier omission and its downstream recovery, rather than tune the failed secant or merely reproduce hold on another scene.

Use the authenticated cycle36 own Z2 immediately before clock2, v1, the same complete conditioning E, latent IDs, compiled clocks, numerical profile and four-step sigma schedule. Contexts00/10 and seed271828 are known development evidence. Let U_i be the qualified finite native scheduler commit at clock i.

## Fixed four-corner policy

| Clock2 / clock3 | Rule | Evidence |
| --- | --- | --- |
| NN | Native v2, then native v3 | Existing cycle36 |
| NH | Native v2, then hold v2 | Existing cycle37 |
| HH | Hold v1 at both clocks | New |
| HN | Hold v1, then native read of its produced state | New |

Both new branches share exactly

\[
Z_3^H=U_2(Z_2,v_1).
\]

Then

\[
Z_4^{HH}=U_3(Z_3^H,v_1),\qquad
v_3^H=F_\theta(Z_3^H,\sigma_3,E),\qquad
Z_4^{HN}=U_3(Z_3^H,v_3^H).
\]

HH holds v1 again and never receives v2. HN must consume the emitted Z3H, without a historical state refresh. Both use the same original finite scheduler, VAE preparation/decoding and pixel processing. Preserve BF16 scalar/product/add/cast boundaries; do not collapse two commits using real-arithmetic cancellation.

NN and NH may be inherited only with exact source/profile/clock/noise/full-interface identity. They are not recomputed merely to create new references.

## Release and physical work

Before candidate sealing, allow only authenticated Z2,v1,E, IDs, schedule, compiled-clock assets and the required original transformer/VAE parameters. Deny historical v2,Z3N,v3N,Z4 and all target images/arrays. Prediction receives its actual own continuing state. Freeze source and complete byte identities before execution.

Generate four new images: two contexts by HH/HN. Share the single Z3H commit per context, not duplicate neural prefixes. Run two new final denoiser evaluations and four native VAE decodes. The supplied two-call prefix per context remains charged: four historical denoisers across the pair, plus original encoding. There is no new reference campaign, noise draw, coefficient fit or policy sweep.

After all four images seal, perform four scheduler-only mechanics replays: reconstruct native Z3N from Z2/v2 and the inherited NH final latent from Z3N/v2 for each context. Require exact stored-word agreement for this declared finite contract. These are paid primitive checks, with no additional denoiser or reference decode.

Potential serving costs are two denoisers per HH image and three per HN/NH image. Original weights and VAE remain. Report physical calls, affine operations, complete weight/input bytes, timing, CPU and sampled RAM/VRAM separately; do not infer matched speedup from operation counts.

Use one guarded finite mrun CUDA lease, source/custody preflight, no retries and sequential B1 branches at the qualified shape. Reuse measured compatible image-consumer history when automatic generic full-model sizing is inapplicable; document any exception and verify physical guard fit. Approximation errors are observations, not stopping gates.

## Mathematical observations and limits

For each complete observable Y, retain

\[
J_Y=Y_{NN}-Y_{NH}-Y_{HN}+Y_{HH}.
\]

This is a policy interaction, not isolated denoiser curvature: the held velocity changes with history. Keep all endpoint and paired color-response results, full signed fields, finite commit remainders, pixel/latent interaction with inherited encoder error, and original images. Include the same-work HN versus NH comparison and both contexts independently.

Since each late-reader comparison shares its own incoming Z3, final-latent interaction factors as the final sigma gap times velocity interaction plus the signed interaction of the finite final-commit remainders. Retain all three vectors and their telescope. A nonzero policy interaction can occur even for an affine final reader; it does not by itself establish nonlinear curvature.

Retain candidate failures and every completed required branch; release saved target values only after all four complete candidate images seal. Authenticate actual conditioning, emitted Z3H, clock3 operand and both final commits. Perform and verify the Saturn observation sweep without treating it as semantic certification.

Success would establish bounded downstream recovery or useful two-call execution on a known pair. Failure would distinguish an important earlier update from the already useful final hold. Neither outcome supplies arbitrary-prompt interpretation, a complete weight-free velocity law or a non-neural pixel decoder.
