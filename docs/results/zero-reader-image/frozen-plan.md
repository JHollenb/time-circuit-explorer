# Does the contextual reader contribute beyond lexical residual input?

Status: cycle28 Astra-selected next cut; not staged or executed. The [raw-reader integration](../raw-rank-one-image.md) preserves the left/right image response at10.72% relative error, even though the packed conditioning response error is125.43%. It has no contextual-reader ablation. Test that dependency before adding rank, precision or more prompts.

Reuse the exact A/B red-cube/blue-sphere strings, complete lexical source,151936-ID manifest, template/tokenizer/positions/512-row mask, source checkpoint, original numerical profiles and seed314159 noise. Use the already sealed full targets from `job-394733a01129` and rank-one candidate evidence from `job-7b206109b072`. These are previously seen targets; this is a retrospective contribution test, not prospective generalization. No new reference images, prompt selection, seed search or parameter fitting.

From each fresh raw string, form its own lexical H0 using the same complete atom path. Supply a fresh **positive-zero BF16 `[1,512,4096]` attention bank** immediately before original block0 output projection. Execute no contextual reader, moment formation, routing, top4 search or same-law fallback. Preserve lexical lookup costs, including any bundled QKV operands it reads; this assay should not introduce a new lexical compiler optimization. Retain and hash the actual zero bank and the original block0 input. Do not zero H0, the residual, the MLP write, any later state or the complete E field.

This contrast is

$$
R_0^{(0)}=H_0+W_O0,\qquad H_1^{(0)}=R_0^{(0)}+\operatorname{MLP}_0(\operatorname{Norm}(R_0^{(0)})),
$$

followed by the unchanged original full512-row blocks1…26, actual H9/H18/H27→E and the same four-step native image consumer/VAE. Preserve original full-row reduction policyTrue and image profile. The inherited exact clock table may remain under its existing identity contract. All52native blocks,8denoisers/2VAE, state handoffs and remaining original learned maps stay paid. No learned prompt-invariant head bank: that would retain a less clean baseline dependency.

Generate only two new zero-reader images with independent clones of the inherited noise and fresh identical schedulers. Seal complete own-state candidates before target-value reads. Compare zero/full and already measured rank-one/full at H1/H2/H9/H18/H27/E, all actual context/velocity/latent ports, decoder and float/PNG RGB. Keep signed B−A response, endpoint and active/padding/per-row observations separate. Use widened subtraction and undefined zero denominators. Retain original PNGs and differences near compute; independently verify complete bytes and publish compact evidence.

Measure the signed contribution difference, rather than gating on an arbitrary accuracy threshold. If zero is similarly good or better at the image response, the current pictures do not demonstrate useful reader necessity for this pair; preserve any local/row advantage the reader still has. If zero loses useful response, report exactly where rank-one information helps and what original consumers recover. Neither outcome proves arbitrary-prompt behavior or permits dropping later contextual work. Large common states and a white background can hide local error, so the paired effect and heterogeneity remain primary observations.

Use one finite guarded mrun transaction, separate lexical and learned-consumer children, no co-resident full models, no retries. Begin with intent-first placement and compatible measured history (107.26s,8074.6MBRSS/8848MBVRAM). Declare actual CUDA and CPU threads, bind all source/assets/seals and inspect queue notes/logs/telemetry. No run is expected to exceed15minutes; if that changes, obtain the required Luna resource/code review first. Keep mechanics, instrument, trend and terminal status separate. A terminal arbitrary-prompt zero-neural claim remains out of scope.
