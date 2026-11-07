# Compile a depth-evolving prompt program without image feedback

Status: **written and unexecuted**. The [fixed-C candidate](../fixed-text-memory.md) removes 5.798 trillion nominal MACs but turns the supplied scene into pale texture. Its first velocity already has 83.27% error. Repeating the original projected conditioning at every layer removes both depth-dependent text processing and image-conditioned text updates, alongside finite geometry changes. It does not identify which omitted dependency matters.

## One source-defined compiler and one image actor

For each of the four original clocks, begin with actual T_0=C(E), then evolve

\[
T_{\ell+1}=G_\ell^{TT}(T_\ell,c_k),
\quad (K_\ell^T,V_\ell^T)=P_\ell(T_\ell,c_k),
\]
\[
I_{\ell+1}=F_\ell^I(I_\ell;K_\ell^T,V_\ell^T,c_k).
\]

G uses the original text-side normalization, projections, gates, residual updates and MLP. Its attention reads **only current text keys and values**, with softmax normalized over the 512 text rows. There are no image operands or image-derived hidden states in the compiler. Preserve original absolute text positions and native sequential joint attention→MLP versus parallel single attention/MLP ordering.

Cache the normalized/rotated text K and original V from the **incoming** T_l before that layer's text update. The image actor must read those caches with its own freshly formed image K/V, image queries, image attention/MLP and residual state. The last block still needs text K/V, but its text query, attention output, MLP and final text update remain dead and must be omitted. There are 100 text K/V formations and 96 live text updates across four clocks.

This restores a depth-conditioned prompt-side program instead of resetting each layer to C. It removes image→text feedback as a source dependency. Text-only attention also changes its normalization by excluding image keys; an adverse outcome constrains this particular program, not image feedback universally. It is an approximate architecture with finite-execution changes, not a real- or finite-arithmetic equivalence to the native model.

## Keep all first-image computation paid

At the pinned widths, the text compiler's prospective matrix account is

\[
C_{compile}=100(2N_TD^2)
+96\left[N_T(2D^2+3DM)+2N_T^2D\right]
=6{,}223{,}407{,}611{,}904.
\]

The BF16 cache of 100 layers/clocks × two banks × 512 rows × 3072 values is **629,145,600 bytes**, excluding receipts and diagnostic states. It is newly produced learned numerical state, not a free original operand.

With common conditioning/clock preprocessing charged once, prospective compiler plus one image/decoder account is **21,062,568,902,656 MACs**. This is only **541,165,879,296 below** the original 21,603,734,781,952 account: 231,928,233,984 from dead terminal work and 309,237,645,312 from removing image keys/values from the 96 live text reads. The fixed-C experiment's much larger saving is relinquished. Derive actual operation counts independently and account for any duplicate preprocessing rather than silently subtracting it.

Compiler-produced clock/modulation outputs may be bound as explicit actor inputs so common work runs once; every learned operation forming them remains paid. Parameter loading, physical slicing, fingerprints, cache I/O, elementwise work and decoding overhead require separate measured scopes. Loading unused original weights does not make their storage free or add fictitious matrix MACs. No matched speedup or amortized benefit is established by one image. Reuse across other image states/seeds is a structural interface possibility whose quality remains unmeasured.

## One candidate against the same existing comparator

Reuse authenticated native case A (`job-394733a01129`) and supplied original E/noise/IDs/schedule, with current custody/byte checks. Use one guarded lease with a fresh compiler child, then a fresh image actor child, then saved-only analysis. The compiler's numerical allowlist excludes initial noise, image IDs/states, native hidden text trajectories, velocities, latents and PNGs. It seals its own caches and common preprocessing outputs and exits before the actor loads them. The actor must consume those own outputs; no native intermediate refresh, fit, new reference, seed/prompt sweep or target-conditioned repair.

The actor advances all four own scheduler states and decodes its actual final latent once. Seal and exit before existing native/fixed-C comparison. Retain actual text-program states/cache lineage, all block consumer joins, per-clock velocity/latent/write errors, actual decoder fields/PNG and signed heterogeneous row outcomes. A bad image remains evidence. Do not use proxy or quality thresholds to suppress valid discovery data.

Use tiny source-defined joint/single fixtures before real-model execution: causal text-only normalization/order, incoming-state K/V cache timing, terminal last-use omission, compiler invariance to absent image/noise inputs, and an image actor consuming its own previous image state. Test the declared program, not parity to native image-conditioned text state.

## Bound memory and execution before staging

Use one strict-preflight CUDA lease, no automatic retries or overlapping full model suites. Model loading is sequential across compiler and actor children, not co-resident duplication. Keep physical projection slices ephemeral. The extra 629 MB cache and retained diagnostics require an actual liveness/memory plan; do not copy twenty fused map banks alongside the full model. Start from current full-image history (8,290.5 MB RSS / 9,116 MB sampled device in cycle67) and fresh intent-first/admission evidence, adjusting for the new working geometry and I/O. Preserve partial caches, stage progress and immutable sources on failure.

The last-latent unpack/BN/unpatchify path remains source-audited unless independently verified. Bind actual VAE input/output and do not upgrade a metadata link into a numerical witness. Complete byte custody, standard evidence sweep and dedicated source/cache/own-state audits precede any bounded result claim. Mechanics, instrument, trend and terminal status stay separate.

This experiment does not solve arbitrary-prompt zero-neural generation: the compiler and image actor still evaluate original learned maps. Its question is whether an image-independent, layer-specific prompt program can replace the current entanglement with image state. Full newest-post survey67, next70 or earlier material evidence; actual storage cleanup66, next69 or pressure. No source is staged or job launched by this plan.
