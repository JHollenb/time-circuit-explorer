# Hold the carrier fixed while changing the receiver

Status: **written and unexecuted**. Selected after the [extension-geometry result](../minimum-response-extension-geometry.md) and [Astra45 review](../results/extension-geometry-evidence/advisor.json). One controlled 2×2 native-consumer assay, not a generator, fit or new reference campaign.

The fixed linear extension fails even when feedback inputs lie almost entirely in its source span. Stop spectral/rank/regularization exploration. The next question is whether an identical attention-payload change has a different effect when the receiving state changes. Unlike observational transfer error, this crossing can test receiver dependence of a declared intervention.

## One pair of complete carriers, two actual receivers

Reuse the exact complete first-joint image-attention outputs from job-1892a00329e9:

- H_A: s60672.base.canonical.
- H_B: s60672.base.far_bottom_right.

Each payload has shape [1,1024,3072], BF16, containing all flattened image-query heads at block0. It is not a single selected head. Bind original bytes, shape, dtype and lineage; reuse the same two logical payloads in both receivers.

R_0 uses the archived actual clock3 input s60672.base.canonical. R_1 uses s60673.base.canonical. Both use the same qualified base conditioning, image/text positions, exact clock/schedule input, model weights and numerical profile. Preserve original input strides using their source metadata. Previous velocity remains part of historical preparation, not an extra native transformer argument. All unpatched image/text/residual operands belong to the receiver.

For the supported native boundary, install a forward-prehook on transformer.transformer_blocks[0].attn.to_out[0]. The pinned transformer_flux2.py ordinary processor presents the flattened, dtype-cast image-head tensor there after text/image separation. Replace that entire first tensor argument with H_A or H_B. Preserve recipient shape, dtype, device, stride and storage offset using a separate backing buffer/view; do not add a delta or mutate/alias the recipient or donor storage. Record the full geometry/allocation ownership, verify one write at precisely this address per call and verify exact consumed BF16 words. The reviewed native source SHA is 0a3c56ed3ca28a416492b50953a433f20b8a89fd26c7b7a49b1b6b45d5236006.

This is a supported module boundary, not a new compiled-clock runtime feature. Use the ordinary qualified native transformer, with no changes to compiled-clock hook admission. The unpatched text to_add_out path remains native. Snapshot the immutable receiver recipe and boundary metadata with Saturn Frames; register the complete carrier replacement as one typed Act. Retain both donor branches and all four outcomes without selection or optimizer updates. Assert that the write surface, input identities and model/buffer identities remain correct.

## Four calls and one interaction

Run the four declared original-transformer calls in one resident CUDA lease:

\[
v_{00}=F(R_0,H_A),\quad v_{01}=F(R_0,H_B),
\qquad
v_{10}=F(R_1,H_A),\quad v_{11}=F(R_1,H_B).
\]

Retain the complete velocities, receiver-specific responses and interaction:

\[
d_0=v_{01}-v_{00},\quad d_1=v_{11}-v_{10},
\qquad I=d_1-d_0.
\]

Seal all four complete velocities before factorial scoring. Cast to FP64 before subtraction and retain finite closure and signed energy/cross terms. Report all per-query/per-coordinate effects, norm ratios, cosine and actual query addresses, both over all rows and the predeclared 944 queries outside the source edit/halo. That subset inherits the original unchanged-query label; all receiver inputs are held fixed across the two carrier arms. The donor pair is identical across R; a response law depending only on that complete pair predicts d_1=d_0. An observed interaction measures receiver dependence for this intervention and qualified finite computation. It does not identify local Z, text state or another specific receiver operand as the sole missing variable, nor exclude a nonlinear/state-conditioned cheaper law or show that natural H0 lacks receiver information.

Do not apply a scientific magnitude gate or claim absence from a small result. Retain the interaction and numerical/profile limitations as measured. There is no payload/gain/rank search, fitting, target-based selection or new natural-response comparison. All four arms are supplied-payload interventions and crossed states may lie outside the natural joint distribution. Their archived carriers were formed with selective image-only SDPA; they are not guaranteed to match ordinary full-attention branch values. Do not label the H_A arm an unpatched native baseline, claim natural-branch bit parity or infer an information-sufficiency result on natural states from these hybrids.

## Charge the actual computation

The simplest trustworthy implementation runs four ordinary full denoisers. Their original first attention still computes the recipient image H0 before the hook discards it. That computation and the native text path are paid. Donor formation is reused, but **new recipient H0 computation is not zero**. This corrects the initial shorthand in the advisor discussion; no executed claim is changed.

Use all 169 original transformer tensors, including the original clock subgraph. The historical same-geometry dynamic final call costs 4,723,927,154,688 nominal affine MACs; its original clock adds 170,655,744 per call. Four ordinary calls therefore have an initial affine estimate of **18,896,391,241,728 MACs**, with 436 affine calls and 100 attention dispatches expected. Attention arithmetic, normalization, activations, I/O and retained fields are additional. This is a source/shape-based estimate, not measured runtime or a zero-neural reduction. Instrument every linear/attention call and record actual counts, wall/CPU/GPU time and sampled RAM/VRAM.

Use one model load, torch.inference_mode/eval, needs={cuda:True}, strict mrun preflight, intent-first placement/sizing and no automatic retry. Start from the existing verified transformer/input/profile qualification; no new encoder, donor-read collection, VAE, scheduler image update, image, noise draw or reference campaign is allowed. All original state/conditioning/weight acquisition remains paid preparation. Seal the four velocities first, then retain/seal the derived interaction before optional retrospective report joins. No old native response target is needed for the primary assay.

First use a tiny independent receiver-dependent nonlinear example, plus hook replacement/alias/write-count mechanics and source/consumed-byte checks. Only integrity, finite, source/profile/hook and resource failures may stop execution. Keep every valid scientific outcome. Inspect actual queue/admission before launch; if the instrument cannot use the declared existing boundary faithfully, preserve the refusal and fix the instrument rather than interpreting it as a null.

A meaningful interaction would motivate identifying a small receiver-conditioned operator. It would not itself build that operator or establish own-state diffusion, general pixels or arbitrary-prompt zero-neural generation. Those requirements remain open.
