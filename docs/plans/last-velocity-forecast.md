# Forecast the fourth denoiser velocity from its own preceding trajectory

Status: **executed in cycle37**; [measured result](../last-velocity-forecast.md). Hold gives 17.66% signed PNG response error against the own parent, versus 54.52% secant. Preserve the original law and immutable payload snapshot. Selected after the cycle 36 retained-query result and [Astra review](../results/retained-queries-evidence/advisor.json), this remains a component replay with a paid three-pass prefix, not arbitrary-prompt zero-neural generation.

## Why this boundary

The [retained-query executor](../retained-padding-queries.md) reduces nominal encoder affine work by 89.65%, but its unchanged denoiser still accounts for about 99.24% of the remaining counted encoder-plus-denoiser affine work. Test one complete missing velocity evaluation before tuning another encoder reconstruction rule. FLUX position-dependent text rotations prevent an exact duplicate-token collapse from copied conditioning values alone.

The specimen is the authenticated cycle 36 own image trajectory for square 00/10, seed 271828, job job-a9e0a0f370fb, report SHA256 82c3e9e25df7c11a93ded490b3d34803237502f63169e1a06f95df493a8c17bc. Both contexts and their outcomes are known development evidence. No held-out or fresh-prompt claim follows.

## One forecast and one control

Index the four denoiser calls 0 through 3. Supply the actual own incoming state Z3 and preceding outputs v1,v2. Their recorded sigma sequence is

\[
(\sigma_0,\sigma_1,\sigma_2,\sigma_3,\sigma_4)
=(1,\ 0.9580853581428528,\ 0.8839818835258484,\ 0.7174965739250183,\ 0).
\]

Freeze the temporal secant

\[
a=\frac{\sigma_3-\sigma_2}{\sigma_2-\sigma_1}
=2.2466599638045444,\qquad
\widehat v_3=v_2+a(v_2-v_1).
\]

Use previous-velocity hold, \(\widehat v_3=v_2\), as the single control. No fitting, clipping, blend, coefficient search or policy selection follows scoring.

Compute the ratio in FP64 from the stored FP32 sigma values, round the coefficient to FP32 once, then perform subtraction, multiplication and addition as separate FP32 operations with an FP32 result at each boundary. Round the final forecast to BF16 once. Hold preserves v2's original BF16 words. Retain the raw FP32 forecast and emitted BF16 operand.

This is an extrapolation along the actual changing latent trajectory. In real arithmetic a smooth velocity along that path has derivative

\[
\frac{d v}{d\sigma}
=\partial_\sigma v+J_zv\,\frac{dz}{d\sigma}.
\]

The secant does not recover those two terms individually, and smoothness over this large interval is a hypothesis. The extrapolation coefficient exceeds 2; the final Euler gap is \(\sigma_4-\sigma_3=-0.7174965739250183\). Success is not presumed.

## Actual finite commit and image consumer

Fork both policies from the same immutable Z3. Apply the pinned native deterministic FlowMatch Euler scheduler step, then the unchanged native VAE preparation, decode and pixel postprocessing. The scheduler source SHA256 is 8b96f25a6170480fb02decb58904e026250d6d68be5de8a7b8785b4c721e9d80. Preserve native sigma/device/scalar behavior, the BF16 product boundary, FP32 sample addition and final BF16 cast; do not replace them with a single FP32 multiply-add. A small exact replay using the saved native v3 may validate the commit only after all forecasts and candidate images seal.

The real-arithmetic last-state defect would be

\[
\widehat Z_4-Z_4=(\sigma_4-\sigma_3)(\widehat v_3-v_3).
\]

Retain the actual finite discrepancy and its difference from that smooth expression. VAE image error remains a separately measured consumer response.

## Inputs, release and work

Before forecasting, authenticate source/request/payload lineage and complete bytes for each allowed prefix field. Prediction reads only own Z3,v1,v2, schedule and required decoding context. Deny native v3, Z4, target images, other target arrays and all transformer tensors. VAE parameters are allowed for the declared decoder only. No encoder or denoiser object is constructed. All four forecast/latent/image branches must seal before target values are released for scoring; a failed required branch cannot silently release them.

Produce four candidate images: two policies by two contexts. This replay performs zero new encoder or denoiser evaluations, four paid native VAE decodes and no new reference image, noise draw or teacher campaign. The original three denoiser calls per context remain explicitly charged inputs. Replaying their saved outputs is narrower than demonstrating a fresh three-pass serving implementation.

If incorporated into an otherwise identical generator, this rule would remove two of eight pair denoiser calls, 206 affine calls and 9,447,854,309,376 nominal affine MACs, or 25% of denoiser affine work. Those are prospective operation counts, not measured runtime savings. The encoder, three original denoisers per image and VAE remain; no whole-generator zero-neural or model-memory claim follows.

Use one finite guarded mrun CUDA lease for the four sequential decodes, strict source/custody preflight and no retries. Let planning/history select resources; declare the VAE/backend and actual geometry. Freeze new worker sources rather than editing cycle 36's executed payload. Stop only for evidence-integrity, invalid finite inputs or safe-execution failures; approximation errors remain observations.

## Measurements

Compare primarily with the same cycle 36 parent v3/Z4/PNG, so this cut measures the last-call replacement on its actual own prefix. Compare historical full images secondarily, retaining interaction with the existing encoder error:

\[
\widehat I-I_{\rm full}
=(\widehat I-I_{\rm parent})+(I_{\rm parent}-I_{\rm full}).
\]

Keep complete signed fields, endpoint and paired color-response errors, cross terms, row/scanline distributions and all four original candidate PNGs. Preserve undefined relative errors where the reference effect vanishes. Inspect geometry, object roles and material defects alongside colors and layout. Report arithmetic, scheduler, VAE, loading, end-to-end wall time, CPU time and sampled RAM/VRAM separately. Perform the Saturn observation sweep and verify it; its stream mechanics do not certify semantics.

The output is evidence about a fixed temporal law. Failure narrows this law and interval; success would justify a subsequent fresh own-prefix execution. Neither outcome resolves arbitrary-prompt image generation without neural evaluations.
