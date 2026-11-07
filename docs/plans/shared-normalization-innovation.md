# Produce both B operands from one shared normalization innovation

Status: **executed**, job-524c9a8e838e. The [result](../shared-normalization-innovation.md) retains both sealed B predictions, native prefix closure, finite defects, inherited controls and query countertrends. The [pre-execution plan](../results/normalization-innovation/frozen-plan.md) remains immutable. [Astra49](../results/product-transport-evidence/advisor.json) selected this upstream cut after the [recipient-conditioned product result](../recipient-conditioned-product.md). Both supplied B fused operands were removed; no fit, rank scan, full suffix or new reference campaign ran.

## The finite-change structure to test

Retain only native A fused baselines F0A/F1A, image embeddings E0/E1, shared gated carriers aA/aB and qualified clock constants. No R0B or R1B fused operand, product, write or exit may enter the producer. Reconstruct actual BF16 residuals in source order:

\[
u_{rA}=\operatorname{BF16}(E_r+a_A),\qquad
u_{rB}=\operatorname{BF16}(E_r+a_B).
\]

Let C center each row and calculate sA/sB from these committed decoded residuals in FP64, with epsilon10⁻⁶. Use the actual native BF16 constant d=BF16(1+scaleMLP), then widen d and shift b to FP64. Bind d to the cycle48 retained onePlusScale constant. Do not replace its original BF16 addition with an FP64 addition.

With the original fused matrix W widened to FP64, calculate once:

\[
h=C(a_B-a_A),\qquad J=W[d\odot h],\qquad \beta=Wb.
\]

J is one complete 1,024-row shared projection. Beta is one clock-shift row, broadcast afterward. For each receiver use

\[
\alpha_r=s_{rA}/s_{rB},\qquad
\widehat F_{rB}=\alpha_r(F_{rA}-\beta)+\beta+J/s_{rB}.
\]

Freeze FP64 for J, beta and the displayed formula; explicitly convert FP64→FP32→BF16 afterward. Feed each complete predicted fused B through native BF16 SiLU/product, one original down, clock gate and original residual commit. No native B operand is refreshed during this computation.

Implementation clarification before staging: both FP64 contractions and the scalar statistics/secant use Torch on CPU with four threads. Original parameters load on CPU; only the BF16 down matrix moves to CUDA. Residual additions and both native nonlinear/down paths use the qualified CUDA BF16 profile. Two typed exit Acts bind the shared J/beta payloads and original allowed operands. The shared projection is counted once outside those Acts; neither Act recomputes it. Candidate and scorer run in separate processes.

## What is exact and what remains unclosed

For the smooth reference T(u)=W[d⊙(Cu)/s(u)+b], this secant formula is exact under a shared additive residual innovation. Native residual addition, LayerNorm, modulation and GEMM have finite policies. Let

\[
\rho_r=(u_{rB}-u_{rA})-(a_B-a_A),\qquad
\eta_{rh}=F_{rh}-T(u_{rh}).
\]

Then the real-arithmetic prequantization defect has the account

\[
\widehat F_{rB}-F_{rB}
=\alpha_r\eta_{rA}-\eta_{rB}
-\frac{W[d\odot C\rho_r]}{s_{rB}}.
\]

FP64 evaluation itself approximates this real identity. Retain rho and combined predicted/native fused errors, along with final conversion differences. Do not add projections of rho or full smooth native endpoints merely to assign causal percentages to these terms. Separate symbolic explanation from measured attribution. This is not a native-exact rewrite.

## Seal and assess

Use an isolated primary process with an explicit numerical-file allowlist. Seal both complete B fused operands, products, writes and exits before the scorer admits either B target or any A/B product/write/exit control. Bind original matrices, clock constant views, model/source/profile, carriers, embeddings and A baselines to their immutable cycle48/49 custody.

Score each receiver's response against its own native B−A at fused operands, product, down, gated write and exit. Reuse cycle49's conditioned result for R1 and raw source-transfer/zero controls after sealing; label inherited controls and their prior paid work. Preserve endpoints, response magnitudes, signed errors/cross terms, all actual 1,024 query rows and the 944-row source-nonlocal panel. No effect-size gate selects a formula or suppresses adverse evidence.

Begin with a tiny independent real-arithmetic LayerNorm-affine example where the identity closes, plus a finite BF16 example retaining its defect. Validate clock slices and target-denial mechanics. Use Saturn Frames/Acts for the source and emitted candidate state. Later own-state closure would require subsequent invocations to consume emitted state without native A-baseline refresh; that is not established by this cut.

## Actual paid geometry

| Work | Affine calls | Nominal MACs |
|---|---:|---:|
| Shared J:1,024×3,072×18,432 | 1 | 57,982,058,496 |
| Clock beta:1×3,072×18,432 | 1 | 56,623,104 |
| Two native downs:2×1,024×9,216×3,072 | 2 | 57,982,058,496 |
| Total | 4 | **116,020,740,096** |

Two original matrices contain **169,869,312 BF16 parameter bytes**. W's FP64 working copy alone is452,984,832bytes; J, beta, native outputs, finite diagnostics and transfers are additional. FP64 contractions and two native down maps are learned arithmetic. No encoder, attention, native LayerNorm module, learned clock network, full Transformer, fit, scheduler image update, VAE or image runs.

Use one guarded mrun CUDA lease with strict preflight and no automatic retries. Declare the actual CPU/CUDA backend of the two FP64 contractions and native downs. Request intent-first planning, then document any selected-component exception using actual live capacity, explicit working-set geometry and cycle48/49 measurements. Do not copy the previous 494 MB peak as proof that a new FP64 working set fits its old reservation. Keep branches resident and stream captures where possible; record physical contractions, wall/CPU/device/RSS/VRAM, parameter accesses and retained bytes. There is no matched speedup claim.

This replaces two B gate/up projections with a shared innovation projection and a clock row while retaining supplied A baselines, embeddings/carriers and original learned maps. It tests a missing operand producer. It does not establish economical first-prompt generation, a general state quotient, later consumer closure or arbitrary-prompt zero-neural images.
