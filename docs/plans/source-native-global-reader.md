# One source-native global read and a velocity-innovation writer

Status: **written and unexecuted**, selected by [Astra40](../results/rotating-writes-evidence/advisor.json) after the [rotating-write result](../rotating-residual-writes.md). This tests a much smaller learned velocity producer. It does not implement zero-neural generation, arbitrary-prompt encoding or a non-neural pixel decoder.

## Preserve a specific response function

The neighboring [global-reader study](../../../obsidian/blog/2026-11-06-174218-a-small-global-reader-needs-the-right-response-basis.md) transmits remote changes but exposes poorly aligned response directions. Our coarse and rotating whole-stack approximations also fail. Stop geometry, phase and write-budget variants. Test one source-native read whose operands can be formed directly from the actual current input.

At final clock3, form only the first joint block's image attention read:

\[
H_0(Z,E)=\operatorname{Attention}
(Q_{\rm image},[K_{\rm text},K_{\rm image}],
[V_{\rm text},V_{\rm image}]).
\]

Use all 1,024 image queries, all 1,536 current keys/values, original absolute positions and original source-qualified BF16 normalization, modulation, Q/K/V and dispatch equations. Image/text input projections and first-joint pre-attention normalization/modulation are paid. Compiled clock operands supply the fixed time dependence. Flatten 24 heads to 3,072 coordinates per image row.

Omit text queries, both attention output projections, every MLP and every later block. Do not supply a captured native H0 at serving. A latent/conditioning/clock write invalidates the read; compute it from that branch's own actual Z/E each time.

Predict one complete final velocity:

\[
\widehat v_3=Q_{\rm BF16}
\left(v_{\rm prev}+[Z,v_{\rm prev},H_0]B+b\right).
\]

The choice of first joint read reflects cheap direct operand formation, not established sufficiency. No native hidden-state refresh, full denoiser fallback, prompt/seed identifier, target lookup or fitted gain is allowed.

## One discovery-only fit, then inspect its future

Reuse the authenticated old even-seed discovery states and native velocity labels from the neighboring campaign at artifacts/flux-global-reader-20261106, including its recorded own-state continuations and remote edits. Before staging, inventory exact records and bind model/checkpoint, conditioning, packed IDs, clock, state/velocity lineage and numerical contracts. Discovery, old odd-seed feedback and reused development evaluation records must be separately enumerated by immutable hashes. Incompatible or missing records are instrument limitations to resolve, not negative scientific evidence.

Only B, b and discovery normalization statistics are mutable. Freeze the original reader parameters, clock tables, source program, ridge .01 and unit gain. Standardize each of the 3,328 input features from discovery statistics only, with declared fixed numerical handling for zero variance. Fit one shared rowwise writer to native velocity innovations. Use the already-declared equal-average response loss:

\[
\mathcal L(B,b)=
\operatorname{mean}\|XB+b-Y\|^2+
\operatorname{mean}\|\Delta X B-\Delta Y\|^2+
.01\|B\|_F^2.
\]

Y is native velocity minus the corresponding previous velocity. Paired differences use discovery remote edits and share the same feature normalization; their intercept cancels. This is one bounded closed-form proposal, with no epoch, ridge, basis, gain or checkpoint sweep.

Use Saturn's typed state/Act and custody machinery to bind discovery operands, coefficient writes and final velocity writes. Capture the zero-writer/hold baseline and the fitted coefficient state as immutable, restorable proposals. There is no stochastic optimizer state or training cursor to conceal; record numerical solver, normalization, RNG use and fingerprints explicitly. Prove restoration of the baseline coefficient fingerprint when testing rollback. Evaluate both proposals through the real final scheduler and native VAE or an exactly qualified inherited hold result.

Old odd-seed consumer feedback is observational only. It cannot select coefficients, gains or checkpoints. Preserve endpoint, paired-response, continuation and visible-image countertrends even if the proposal is poor. Seal coefficients before a fresh optimizer-free serving child sees the development inputs; seal all candidate outputs before evaluation target access. No future target may enter fitting or selection.

## Small scientific panel

Serve the existing two Z3H/E/clock3 inputs with their exact previous velocities, then commit through the original finite scheduler and VAE. Existing same-state HN and hold HH are development references. Also retain separately frozen remote-edit velocity contrasts from the neighboring campaign; exclude those evaluation labels from fitting. These reused outcomes do not become fresh generalization evidence because access is delayed.

Retain H0, complete feature normalization and emitted velocity/latent/RGB fields, actual source operation counts, read invalidation bindings, candidate seals and native-consumer inputs. Measure hold, candidate endpoint error, signed color/remote responses, and the original images. A zero-read replay of the same sealed writer can diagnose its use of H0 without another learned reader call; do not refit a new control or select based on it.

Start with a tiny independent normalization/RoPE/attention/readout mechanics check. If a matched native component check is needed, limit it to one motivated first-joint attention read on a discovery input and charge it as preparation; do not run a new full-denoiser reference campaign. Changed query/GEMM/dispatch geometry does not imply native bit parity. Keep scientific errors as observations.

## Explicit cost

The existing full-call source census predicts seven original affines: 51,136,954,368 MACs per served call. The shared 3,328→128 writer costs another 436,207,616, totaling 51,573,161,984, or 1.092% of 4,723,927,154,688 full-call affine MACs. QK/AV costs 9,663,676,416 additional MACs. Ten original tensors occupy 142,344,960 bytes, including the three head-normalization weights, plus fitted coefficients, clock constants, current state, workspaces and VAE.

This remains eight learned maps, including the fitted writer. Offline source feature formation and fitting, original discovery labels, native prefixes, original encoding and VAE are all paid. Count them separately from serving; no runtime or speedup result exists yet.

Use one finite, strict-preflight mrun lease with actual CPU/CUDA declarations, one resident selective reader/VAE load and in-process logical branches. Use intent-first sizing and measured history/probes for a documented exception. No duplicate accelerator work or per-branch jobs. Stop only for source/release/custody/finite/resource mechanics. If expected work exceeds15minutes, obtain the mandated execution review before launch.

The result can test whether one cheaply executable source response function supplies useful velocity directions. Success would still leave arbitrary-prompt contextual operand formation, zero-neural velocity production, multi-clock own-state closure and general non-neural pixel generation unresolved.
