# Separate attention competition from image-value feedback

Status: **written and unexecuted**. The [image-independent text program](../prompt-program.md) restores red/blue object vocabulary but adds many extra objects. Its first velocity still has 64.22% error. Text-only attention removed both image-conditioned normalization and image-value payloads. The next candidate separates them on one existing prompt/seed, without a new reference, fit, sweep or target-directed repair.

For each incoming text query/head, define

\[
\alpha=Z_T/(Z_T+Z_I),\qquad
O_T^{native}=\alpha O_{TT}+(1-\alpha)O_{TI}.
\]

This identity holds for fixed common incoming Q, K and V in real arithmetic. It does not decompose the measured difference between the native and text-only trajectories, whose states subsequently diverge.

Use **O_T=alpha O_TT** for every live text write. Compute full current image keys from the candidate’s own incoming image state, and current text queries/keys/values from its own evolving text state. Preserve the original sequential joint attention→MLP and parallel single attention/MLP laws. Image attention keeps both complete current value banks. Reset text at C(E) independently for each clock. Omit the dead final text query, MLP and output, retaining final text K/V. Advance all four own sampler commits and actually decode the final latent once.

The omitted image-value term is exactly zero at the text write, but image keys still compete in its denominator. This deliberately gives up the previous image-independent compiler. Alpha is one scalar **per text query and head**, computed from the whole current image-key bank; its small output dimension does not make its formation free. No saved native text states, attention masses, hidden image states or reference refresh may enter the candidate.

## Declare the finite program and actual work

Use an explicit, source-frozen text attention law: form scaled full text-query/key scores in FP32 with TF32 disabled, compute a stable softmax over all text+image keys, multiply only its text columns by current text V in FP32, then cast the text attention output to native BF16 before the original output/write path. This implements the zero-image-value numerator with full key normalization in real arithmetic. Its explicit score/softmax/AV geometry and rounding differ from fused native SDPA; characterize that finite program in tiny fixtures and retain this limitation beside every causal inference. Image SDPA keeps its original primitive and its own incoming banks.

Instrument the actual score and value matrix products, not just calls to native SDPA. Passing zero image V to a full SDPA still executes the full AV product and cannot earn the omitted-product saving. Materializing dense FP32 text scores costs about 75.5 MB per layer at B1/512text/1024image/24heads, before softmax/workspace; keep only one layer’s scores live and derive a fresh liveness/resource plan before staging.

If image-value multiplication is physically absent, prospective nominal saving against the same original account is

\[
231{,}928{,}233{,}984_{\;terminal}
+96N_TN_ID=386{,}547{,}056{,}640,
\]

with prospective first-image account **21,217,187,725,312 MACs**, plus any implementation overhead or duplicate work actually executed. This restores 154,618,822,656 image-key score MACs relative to the text-only candidate. All original maps, live text updates, current image K/V, sampler and VAE remain paid. FP32 scores may be slower despite fewer nominal products. No matched speed, cheap feedback, amortization or zero-neural claim follows from this design.

## One guarded candidate and complete evidence

Use authenticated original E, noise/IDs, schedule and case-A comparator `job-394733a01129`. One strict-preflight CUDA lease, no retries or concurrent full suites. Do not reuse a stale reservation blindly: account for full model/VAE plus one explicit text-score/softmax workspace, retained state diagnostics and I/O. Inspect current fleet and exact source-bound history. Preserve partial progress on failure.

Before staging, test: text-only and image-key partitions share one stable denominator; zero image-value contribution does not erase image-key competition; joint and single updates consume the correct incoming/post-attention states; terminal text last-use omission; and two toy own-state commits. These are mechanics checks for the declared program, not a proxy quality gate.

Bind incoming state, actual alpha/text read, all output states, physical operator counts and own sampler/decoder inputs in typed records. Seal and exit before the existing native and prior-candidate comparisons. Retain actual PNG, signed row distributions, dose-free per-clock velocity/latent/write curves and complete byte custody. Report mechanics, instrument, trend and terminal status separately; a bad image remains evidence.

The question is whether dynamic attention competition alone improves this altered text law. An improvement would not prove that alpha alone controls count; a poor outcome would not universally exclude image-key feedback. Image-value payload, finite arithmetic and their interaction remain alternatives. The original arbitrary-prompt zero-neural goal remains open. Actual cleanup66 is due again69 or pressure; newest-post survey67 is due70 or material evidence.
