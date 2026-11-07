# Save the state before eliminating work

We now have complete, durable inputs and native outputs for the final FLUX.2 single block. This closes a missing experimental interface: the next producer can start from an actual incoming state, emit its own image rows and final velocity, then compare with a separately retained native target. The reduced producer has **not run yet**. Arbitrary prompt → image with zero neural evaluations remains open.

The successful capture takes **79.94 scheduler seconds**, with sampled peaks of **1,999.1 MB RSS** and **1,196 MB device memory**. It pays **4,870,914,441,216 nominal matrix MACs**, plus normalization, softmax, storage and other uncounted operations. This is reference preparation, not a generation saving.

## Demand determines what must be computed

The [source-state reader](source-state-reader.md) recovered its own Q/K and advanced a small set of queries. Its original states, learned maps and much of its surrounding computation still had to be supplied. A more useful next question is whether the native consumer itself makes some output computation unnecessary.

For the ordinary, nonreference FLUX.2 Klein graph, the final single block receives text and image rows, but the following velocity head consumes only image rows. Let

\[
X=[X_T;X_I],\qquad Z=N_c(X),\qquad N=N_T+N_I.
\]

Here c includes the actual modulation and absolute position factors. The demanded real-arithmetic output is

\[
Y_I=X_I+g_c\odot W_o
\left[
\operatorname{Attn}(Q_I,K_{T\cup I},V_{T\cup I}),
\operatorname{SwiGLU}(W_{GU}Z_I)
\right].
\]

All current text and image K/V remain necessary. Text Q, text MLP activations and text output/residual writes have no consumer after this final block. Earlier text outputs remain live because later blocks can read their K/V. This is an output-liveness argument on the source graph; it does not say text is irrelevant to the image.

At this specimen's N_T=512, N_I=1024, D=3072 and M=9216, physically omitting those dead rows would remove **57,982,058,496 nominal matrix MACs per final-block call**. The [frozen liveness plan](plans/terminal-output-liveness.md) derives the account. Changing fused projection widths, row counts and attention geometry can change BF16 arithmetic, so exact algebra does not establish executable parity.

## A source state must be complete

The [metadata intake](results/terminal-capture-intake/state-cut-census.json) authenticates 342 retained cuts in one qualified fresh-family corpus. They cover joint and project boundaries, with no single19 cut in that corpus. This is a bounded inventory, not a workspace-wide absence claim.

We therefore pay a separate native suffix from the original post-joint0 P state: joint1–4, single0–18, a complete incoming single19 capture, native single19, a complete post-block capture, then the original final normalization and velocity projection. One original block is resident at a time; no complete transformer is constructed.

Each new StateCut retains all thirteen slots: conditioning, context, cursor, image, image IDs, latent, noise prediction, phase, sigmas, step, text, timesteps and text IDs. Context contains the original modulation, time embedding and absolute RoPE. Both cuts roundtrip exactly through the actual deployed store. Their identities are:

| Boundary | StateCut ID |
|---|---|
| Before single19 | `6ad613490ff20c97e5ef42b6c946a349b73387b4b6899e30c2955329005c8955` |
| Before final velocity head | `d1a377245109d1016bb52fbcecc9e7c28e7c4eeccf1f79e0f4c1848ca6af2df0` |

The target's parent is the new source cut; the source's parent is the original P cut. Nine untouched slot descriptors remain exactly bound to P. The advancing image/text, phase and cursor are explicit. These are component-capture cuts with a declared execution manifest; they do not assert drop-in compatibility with the full native Session adapter or independent full-transformer parity.

## The failed attempt is part of the cost

The [first attempt](results/terminal-native-capture-failed-v1/README.md) fails after **69.80 seconds** at its first persistence call. Its wrapper expected an optional shared-store module that the deployed wheel does not contain. No arrays, cuts or compact result were persisted. The traceback position implies the preceding native prefix ran, but its counters were not durable; we do not invent an exact realized MAC count for that failure.

V2 preserves the original source and request. Separately named sources first run a tiny complete CPU StateCut through the exact worker package in a fresh child. The old-wheel local store path succeeds, the external-pool sentinel stays unopened, and denied/unlisted numerical inputs stay blocked. A second fresh child then performs the native capture. Its completed stages and cumulative work are fsynced after each stage. The repair qualifies persistence; it changes no learned equation.

## What the executed ledger establishes

| Item | Successful capture |
|---|---:|
| Native typed Acts / explicit output ABIs | 25 / 29 |
| Native affine / attention calls | 90 / 24 |
| Distinct original tensors | 146 |
| Affine MACs | 4,523,022,090,240 |
| QK plus AV MACs | 347,892,350,976 |
| Final-head affine MACs, included above | 421,527,552 |
| Worker wall time after imports | 52.531 s |
| Peak CUDA allocated / reserved | 686,592,000 / 924,844,032 bytes |

Per-stage wall times synchronize CUDA immediately before and after each operation, excluding the later fingerprint work. They are not isolated kernel latency measurements or a matched speed comparison. Scheduler time is the measured complete-job scope. The failed predecessor and original prefix/context production remain additional paid work.

The [complete byte verification](results/terminal-native-capture/retained-array-verification.json) checks 71 files totaling 7,819,725,566 bytes, including the inherited original checkpoint. Four NPY payload witnesses independently join the source combined residual to its text/image StateCut bytes, the target fields to their pages, and the target combined state and velocity to the typed outputs. The intermediate joint4→single0 concatenation is source-audited; its complete intermediate bytes were not retained for an independent replay.

There are no encoder, learned clock-network, scheduler, VAE, image, fit or reduced-producer calls here. The standard evidence sweep sees one 25-row stream and three applicable predicates, with no fires or formal semantic claims. Dedicated audits qualify the typed computation, byte custody and state joins. All **305 mechanics tests pass**.

## The next computation

The [execution intake addendum](plans/terminal-liveness-execution-intake.md) binds these actual cuts and original head maps to the unchanged liveness plan. Next, physically slice K/V rows for all tokens and Q/MLP rows for image tokens, compute rectangular attention, advance the producer's own image state and consume it with the original velocity head. Seal that result before numerically opening the historical target.

This is a path toward removing paid learned work with a source-defined reason. It is not a zero-neural image generator. The [evidence index](results/terminal-capture-evidence/README.md) preserves the source, failure, deployed-package mechanics, reviews and exact accounting.
