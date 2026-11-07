# Preserve readable state while routing fewer residual writes

Status: **written and unexecuted**, selected by [Astra39](../results/coarse-global-evidence/advisor.json) after the [coarse-global result](../coarse-global-innovation.md). This is a reduction of original learned computation on existing development inputs. Arbitrary-prompt zero-neural generation remains open.

## Keep the state, restrict the writes

Do not pool the latent. Keep all 1024 image rows and 512 text rows at their original absolute positions throughout the 25 blocks. For global block index b=0…24, select

\[
S_b=\{\text{all text rows}\}\cup
\{(h,w):2(h\bmod2)+(w\bmod2)=b\bmod4\}.
\]

There are 768 selected queries/writes per block. Every image row receives six or seven updates; every text row updates at every block. Unselected image residual states carry forward unchanged, without a native hidden-state refresh.

The source-fixed hypothesis is that full readable fine state and global K/V access may be retained while fewer learned writes advance that state. Cycle39 instead changed both representation and future computation, leaving a larger error in the retained coarse component. This is a different algorithm, not a resolution or gain sweep.

## Actual native block ordering

For each joint block, form original normalized/modulated image and text inputs from all current own rows. Form K/V from every row. Form Q only for selected image rows and all text rows, with each query's original absolute RoPE position; key positions remain complete. Use ordinary attention mass with rectangular 768-query/1536-key geometry, no multiplicity bias.

Write selected attention outputs into their own residual rows. Selected MLPs must then consume those newly updated attention residuals through the original second normalization/modulation and original gate/up/down computation. Unselected image rows receive neither attention nor MLP residual updates. Their carried bits remain unchanged.

For each single block, original attention and MLP branches share the same normalized/modulated incoming state. Form complete K/V from all1536rows, selected Q and gate/up operands only from S_b. Concatenate selected attention output and selected activated MLP branch, then apply the selected output projection and residual gate. Do not apply the joint block's sequential MLP dependency to this parallel block.

Slice fused input-projection weights to compute full K/V and selected Q/MLP outputs separately. Computing the full fused projection and discarding unwanted outputs is not the declared reduction. Preserve source gate/up ordering and all BF16 arithmetic boundaries; report finite changes caused by altered GEMM geometry rather than assuming native bit parity.

Keep full original image/text input projections, clock constants, positional bindings and final full-resolution normalization/output projection. Emit the routed denoiser's complete velocity directly. There is no pooled state, lifted velocity correction, bias, fitted coefficient or target-selected phase.

## Mechanics before a real model

Use a tiny independent synthetic joint and single block comparison to test rectangular query/key attention, correct selected residual ordering, parallel-branch inputs, full K/V dependence, rotation coverage and bit-unchanged unselected carried rows. Verify sliced affine output counts match the declared operation reduction. A distant source-row edit should remain readable through full K/V even when that row is not selected to write.

Separate mechanics from approximation fidelity. Shape/device/finite/source/release/custody violations can stop execution. Accuracy, gate energy and image distances are observations. Do not require exact BF16 parity after changed projection geometry or interpret a failed fidelity threshold as absence of a useful effect.

## Frozen input and budget

Use the existing two actual Z3H/E/clock3 inputs, source weights, complete IDs, numerical profile and finite scheduler/VAE. Existing HN is the full native velocity on the exact same state; HH is the hold baseline. Deny these targets and all reference images until both candidate images seal. No new full reference, encoder, prefix or noise campaign.

Run two routed candidate calls and two VAE decodes in one guarded finite mrun CUDA lease, serial B1, no retries. Use intent-first preview, measured resident-weight history for any documented envelope exception and a live physical guard-fit check. Four historical full prefix calls and original encoding remain charged.

The authenticated 103-linear source census predicts 2,730,793,893,888 nominal affine MACs per routed call, 57.81% of 4,723,927,154,688 full. Boundary projections cost12,884,901,888; joint blocks543,581,798,400; single blocks2,174,327,193,600. QK/AV geometry is half the full call. Full K/V, all parameters, normalization, softmax, storage, text updates and VAE remain paid. Instrument actual sliced weight/input shapes, physical calls, wall/CPU time and sampled RAM/VRAM; counts do not establish matched speedup.

## Read the result through its actual consumer

Retain complete candidate velocity, final latent, VAE preparation/raw output, float RGB and original PNG fields. Compare endpoint and signed-color changes with same-state HN and hold HH, preserving per-row trends and original images. Retain finite scheduler effects, actual selected-row masks/positions and each block's unchanged-row witness. Report how changed GEMM geometry and carried old state limit interpretation.

This can establish useful routing of dynamic learned computation while retaining fine readable state. It cannot establish a zero-neural velocity producer, fresh arbitrary-prompt coverage or a general non-neural pixel decoder. The full goal remains active.
