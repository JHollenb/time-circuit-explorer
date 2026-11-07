# Transport the increment into the recipient's nonlinear baseline

Status: **executed**, job-f21da64e0e39. The [result](../recipient-conditioned-product.md) retains the sealed candidate, controls, quantization and complete queries. The [pre-execution plan](../results/product-transport/frozen-plan.md) remains immutable. [Astra48](../results/first-consumer-evidence/advisor.json) recommended this conditional prediction after the [first-image-consumer result](../first-image-consumer.md). No full suffix, image, fit or rank scan ran.

## The mathematical question

Gate/up response directions remain nearly aligned across the two supplied receivers, while their nonlinear products differ substantially. Test whether the source increments can be reused when the actual recipient supplies the nonlinear baseline. This retains the full SiLU/product law rather than linearizing it.

Use the already authenticated cycle48 fused operands from R0A, R0B and R1A. Write g for the gate half and v for the up half. Freeze, without fitting,

\[
\delta g_0=g_{0B}-g_{0A},\qquad \delta v_0=v_{0B}-v_{0A},
\]

\[
\widehat g_{1B}=Q(g_{1A}+\delta g_0),\qquad
\widehat v_{1B}=Q(v_{1A}+\delta v_0).
\]

Decode the supplied BF16 operands to FP64, perform difference/addition in FP64, then use the explicitly declared **FP64→FP32→BF16** conversion Q. Retain the pre-quantization prediction and conversion discrepancy. Do not assume this conversion is a direct FP64→BF16 rounding operation or matches another unstated arithmetic policy.

Then execute native BF16 SiLU, BF16 multiplication and exactly one original down projection:

\[
\widehat p_{1B}=\operatorname{BF16}(\operatorname{SiLU}(\widehat g_{1B})\odot\widehat v_{1B}),
\qquad \widehat m_{1B}=W_d\widehat p_{1B}.
\]

Reconstruct the B residual in its source order from supplied R1 embedding E1 and the shared B gated carrier aB. Use the qualified original MLP gate gm:

\[
u_{1B}=\operatorname{BF16}(E_1+a_B),\qquad
\widehat y_{1B}=\operatorname{BF16}\!\left(u_{1B}+\operatorname{BF16}(g_m\odot\widehat m_{1B})\right).
\]

Bind the exact four native operand cells, cycle48 source/profile, original down payload, qualified clock gate, query geometry and inherited 944-row source-nonlocal subset. Validate the source files and semantic gate/up views through their existing complete custody and view hashes. Original donor/receiver/clock preparation remains paid.

Implementation clarification before staging: the qualified clock gate is the final 3,072-channel slice of the existing clock3 image modulation parent. Authenticate the complete 36,992-byte parent file and retain only that semantic gate view for the Act; no clock network runs. Form the FP64 transport on CPU, then transfer the predicted contiguous BF16 fused operand to CUDA for the original nonlinear/down path. Both devices and their arithmetic/transfer costs are recorded. A separate primary child exits before the scoring process starts.

## Seal before the unavailable endpoint is admitted

The primary process receives only three fused arrays (R0A, R0B, R1A), E1, aB, gm and Wd. It must not receive R1B fused operands, products, writes or exits. R1A product/write/exit and old control writes are also excluded. The primary must not import an analysis helper that opens those denied paths. Freeze its allowlist and log actual file accesses.

Seal the complete candidate operands, product, down write, gated write and image exit with immutable hashes before a separate scoring process admits any held native endpoint or baseline/control output. No target-dependent choice of dose, basis, gain, rank, rounding policy or seed is allowed.

After sealing, compare the predicted changes against actual R1B−R1A at the product, down write, gated write and image exit. Both the candidate and target responses subtract the same retained native R1A field. Report endpoint error as well: a strong baseline can otherwise hide an inadequate response.

Reuse the raw R0 write-response transfer and zero-response controls only after sealing. At each native field S, the direct transfer control is S1A+(S0B−S0A); zero response predicts S1A. Label these as fieldwise diagnostic controls, not new coherent native rollouts. If a residual-committed control is added, specify its finite addition order beforehand and form it without R1B. No extra down projection is needed for these retained controls.

Keep full signed errors, magnitudes, cosine, energy cross terms, per-query outcomes, all-row and 944-row summaries. Include the difference between transported and actual gate/up increments and the candidate conversion discrepancy. No effect-size threshold may erase an outcome or stop valid evidence collection. A useful local result remains a conditional result; later native consumers would require a separately declared experiment.

## Work and execution contract

The sole new learned affine is one B1 down projection on 1,024 queries:

\[
1024\times9216\times3072=\mathbf{28{,}991{,}029{,}248}\text{ nominal MACs},
\]

with **56,623,104 BF16 parameter bytes**. Native SiLU and product each cover 9,437,184 scalars. Loading three fused arrays supplies 113,246,208 payload bytes; materializing both FP64 increments adds 150,994,944 bytes. These are logical working data, not a measured RSS claim. Captures, transfers, hashing and diagnostic arithmetic remain additional.

No new input projection, carrier projection, LayerNorm, modulation input path, fused gate/up projection, attention, full Transformer, learned clock network, encoder, fit, scheduler image update, suffix, VAE or image is required. Native SiLU and the original down map remain paid learned computation. Supplied native baselines and source increments are not a cheap producer.

Begin with tiny independent quantization/order tests, including a case where the same increment has a different SiLU/product response at another baseline. Use Saturn typed Frames/Acts for the allowed state and candidate handoff. Then use one guarded CUDA mrun lease, strict preflight, no automatic retry and intent-first planning for this actual selected-component geometry. Cycle48's 700 MB device peak and the scheduler's 1,024 MB matching-work recommendation are relevant history; compute the smaller run's real working set before selecting a bounded exception if the planner cannot represent it. Record wall/CPU/device measurements, actual affine calls, parameter reads and retained-array footprint.

Only source/identity, custody, finite/shape/layout/profile, seal/allowlist and resource mechanics may stop this discovery run. Saturn instrument applicability is separate from scientific trend. No serving speedup is assumed without a matched measurement.

## What either outcome would mean

Improvement over raw write transfer would support recipient-conditioned nonlinear transport for this supplied pair. Failure would preserve the measured upstream increment errors, finite-policy effects and their amplification as evidence. Neither outcome establishes whole-future mediation, natural-state sufficiency, arbitrary-prompt coverage, a cheap baseline producer, complete velocity or zero-neural image generation.
