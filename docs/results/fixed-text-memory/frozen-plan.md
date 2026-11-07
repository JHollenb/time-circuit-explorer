# Ask whether image generation needs evolving text working state

Status: **written and unexecuted**. [Cycle66](../terminal-own-state-trajectory.md) removes dead terminal text output work while retaining every current K/V read, then advances four own sampler steps and actually decodes its final image. It removes about 1.074% of a matched nominal whole-trajectory matrix/convolution account. More precision repair or another similar picture would not resolve the dominant live dependency. Astra's next cut tests that dependency directly.

## One fixed, source-defined state intervention

For each denoiser clock form its actual original conditioning projection C(E). Hold that complete projected text state fixed at every joint and single block: all text residual writes are zero. Preserve all text rows, source order, absolute IDs and the original clock-dependent modulation. The candidate must derive C from the supplied E, not read any archived intermediate text state.

At every block compute full current text/image K/V, every image Q, image attention output, image MLP and image residual write. Use the actual own image state, then the actual own final head, native scheduler commit and next latent. Joint blocks retain their original ordered image attention and MLP substeps; single blocks retain the original parallel attention/MLP law. Only text working-state updates are omitted. No stale image K/V, reduced spatial grid, selected text support or oracle refresh.

Write the ordinary coupled block as

\[
(T_{j+1},I_{j+1})=B_j(T_j,I_j;E,t).
\]

The intervention executes

\[
\widehat T_j=C(E),\qquad
\widehat I_{j+1}=\Pi_I B_j\bigl(C(E),\widehat I_j;E,t\bigr).
\]

The image-row computation at the declared fixed text input must physically omit text Q, text attention output/projection and text MLP work. It must not compute those outputs and then clamp them. In real arithmetic this is exact demand elimination for the **intervened** fixed-text program, not equivalence to the ordinary native program before the terminal last-use boundary. Physical row/width slicing and rectangular SDPA also change finite execution, as cycle65 demonstrated. The changed future K/V source is the intended scientific intervention; the tested candidate includes those finite geometry changes too.

## A consequential deletion, with no fitted replacement

For N_T=512, N_I=1024, N=N_T+N_I, D=3072 and M=9216, removing text Q, output and gate/up/down work avoids

\[
\Delta C_{aff}=N_T(2D^2+3DM),\qquad
\Delta C_{att}=2N_TND.
\]

The joint block has separate attention-output and MLP-down maps; the single block has a concatenated output map. Both give the same nominal removed account at these widths. Each removed text path therefore saves 57,982,058,496 matrix MACs in the declared geometry. Across 25 blocks and four clocks, prospective deletion is **5,798,205,849,600 MACs**, including cycle66's already removed terminal work. Incremental deletion beyond cycle66 is **5,566,277,615,616 MACs**. Verify actual shapes and counts rather than promoting these prospective numbers into execution evidence.

Every image projection, live text/image K/V, normalization, attention, activation, clock, head and decoder still uses its original learned computation. Historical E/noise and original reference production remain paid. This is not zero-neural completion. Its value is testing whether one entire evolving state branch can be removed without replacing it with supplied hidden state, a fit or a neural student.

## One own trajectory and one existing comparator

Use the same authenticated native case A, job `job-394733a01129`, raw conditioning E and seed314159 noise from `job-36aca5f049e6`, with the same pinned original model/source/numeric profile and four-clock schedule. Source intake may reuse cycle66's complete byte/custody records only after checking their current identities. Deny all original context trajectories, velocities, intermediate latents and PNGs until candidate seal and producer exit.

Run one candidate, four own scheduler commits and one actual final VAE decode. No new native shadow/reference, second prompt/seed, sweep, fit, target-dependent repair or inherited image by tolerance. Retain signed per-clock velocity, latent and own-write errors, actual final decoder fields/PNG, per-row heterogeneity and the full physically executed work account. Compare against the existing native trajectory and cycle66 descriptively; their timing scopes are not matched speed benchmarks. A poor image or large error constrains this combined fixed-text/physical-execution candidate. It does not by itself isolate evolving text state as the sole cause or exclude all text-state compression. No extra arm or precision campaign is required for this first bounded test.

Before real-model work, use a tiny joint/single toy with frozen text and nontrivial absolute positions to verify the demanded image-row law against the corresponding ordinary block image output at that same supplied text input. Test that text remains fixed and that later blocks consume the candidate image state. These are mechanics checks, not general image-quality gates.

## Bounded execution and custody

One guarded strict-preflight CUDA lease, no automatic retries, following the current full-image measured history and live admission. Cycle66 observed 8,416.4 MB RSS / 9,314 MB device. Keep sliced single-block maps ephemeral or use original source row views; do not materialize twenty extra physical map banks alongside the whole original model. Preserve partial progress/arrays and immutable sources on any failure. Review actual code and measured resource placement before staging. No job is staged or launched by this written plan.

Bind actual own-state boundaries and original map provenance. Keep the existing final-latent unpack/BN/unpatchify source-only qualification unless independently tested; do not upgrade a Frame metadata declaration into a numerical witness. Collect full-byte custody, lossless signed observations and actual candidate PNG, then run the standard evidence sweep with separate mechanics, instrument, trend and terminal labels. No quality threshold may suppress valid exploratory evidence.

The terminal question is bounded: how much of this one image trajectory survives when image computation reads fixed projected text memory instead of evolving text working state? Arbitrary raw-prompt image generation with zero neural evaluations still requires solving the remaining prompt, image-state and pixel-producing computations. Scheduled newest-post review68 and storage69 remain in place, or earlier on material evidence/pressure.
