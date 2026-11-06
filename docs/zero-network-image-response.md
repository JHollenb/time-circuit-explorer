# Generating an image without a new neural forward

We eliminated all neural evaluation for a **restricted image response program**. Twelve historical calibration images compile into four context-bound curves. New doses produce 512×512 PNGs using only pixel arithmetic. The broad changes interpolate well; fine contextual effects remain only partly captured. This does not yet produce a new scene from an arbitrary prompt or seed.

The [previous generator](prediction-only-benchmark.md) remains a different result: it reconstructs the entire dense denoiser and advances its own state, with native encoding and decoding. Avoiding a framework's `forward` method does not remove its underlying computation. This experiment actually removes encoder, denoiser, VAE and neural-student evaluations during compilation, serving and scoring, by supplying previously computed RGB anchors instead.

![Network-free output at the predeclared new dose 0.075; context 60666.I0. Native accuracy at this dose is unmeasured.](results/zero-network-response/images/60666.I0.a0.075.piecewise.png)

This output's hash differs from every calibration and evaluation image. Its appearance is inherited from the supplied scene, and its new pixel values come from interpolation. Byte novelty does not establish a new scene, semantic understanding or general model compression.

## What the architecture lets us skip

Write a diffusion pass as

\[
v_k=\mathcal D_\theta(z_k,c,t_k),\qquad
z_{k+1}=B\!\left[z_k+B\!\left(B(\Delta t_k)v_k\right)\right],\qquad
Y=\mathcal V_\psi(z_K),
\]

where \(B\) denotes the qualified BF16 boundary. Internally, the denoiser is an ordered circuit of attention, MLP, normalization, modulation, residual and readout operations. A state must include its working registers, context, clock/address and any memory consumed later. The symbol table identifies computations and their operands; a reduced table is sufficient only if it closes under future execution.

Dependency tracing gives an exact opportunity before it gives a compact approximation. In the qualified late-latent cut, the denoiser has already made its last read of the carried latent. Holding its actual working registers and context fixed, changing that carried latent cannot change the already determined velocity. We can reuse the velocity and perform only the scheduler commit. This is a scoped dependency fact, not an assertion that denoising is globally linear.

The historical source supplied four paths, indexed by known seed and I0/I1 branch:

\[
z_r(\alpha)=B[\operatorname{FP32}(z_{r,0})+
\alpha(\operatorname{FP32}(z_{r,1})-\operatorname{FP32}(z_{r,0}))],
\]
\[
\ell_r(\alpha)=B[z_r(\alpha)+B(B(\Delta t)v_r)],\qquad
Y_r(\alpha)=\mathcal V_\psi(\ell_r(\alpha)).
\]

Historical doses 0 and 1 clone their exact endpoint states; interior doses use the written FP32 interpolation followed by BF16. The state parents, scheduler inputs and velocity are historical supplied quantities. The remaining native consumer is the VAE. Our new program approximates the **composite one-dimensional function** \(\alpha\mapsto Y_r(\alpha)\). It consumes a context key and dose, rather than a latent tensor. Running unused Euler arithmetic would not strengthen this claim.

## Factoring the restricted output function

For each context, take three RGB anchors \(Y_0,Y_q,Y_1\), with \(q=1/4\). Below the middle anchor,

\[
\widehat Y(\alpha)=Y_0+4\alpha(Y_q-Y_0).
\]

Above it,

\[
\widehat Y(\alpha)=Y_q+\frac{\alpha-q}{1-q}(Y_1-Y_q).
\]

The factorization separates a supplied appearance from a context-specific response basis. Rendering clips to [0,255] and rounds once to uint8. It needs no learned weight bank. It also makes the limitation explicit: the basis is calibrated separately for each supplied curve.

A frozen quadratic comparison expands through the same three observations:

\[
\widehat Y_Q(\alpha)=
4(\alpha-q)(\alpha-1)Y_0
-\frac{16}{3}\alpha(\alpha-1)Y_q
+\frac43\alpha(\alpha-q)Y_1.
\]

The primary piecewise rule was selected before scoring; we did not choose a method after looking at the held results. Copy-zero, nearest calibration image, endpoint-only interpolation and another seed's response were fixed controls. The [frozen protocol](plans/zero-network-response.md) records the split and a pre-execution instrumentation amendment.

## What we measured

One guarded mrun CPU job, `job-2e98c6786986`, compiled 12 calibration PNGs at doses 0, .25 and 1. Separate serving generated 96 files including controls: 72 distinct PNG hashes, 16 primary outputs, and four primary outputs at the predeclared new dose .075. The scorer opened 12 historical evaluation PNGs at .05, .1 and .5 only after prediction hashes sealed. The four curves share two known seeds; these are not independent examples of seed generalization.

The primary's mean relative error measures the prediction error against the native change from dose zero, rather than the entire portrait's brightness:

\[
E_{\rm edit}(\alpha)=
\frac{\operatorname{RMS}(\widehat Y_\alpha-Y_\alpha)}
{\operatorname{RMS}(Y_\alpha-Y_0)}.
\]

| Held dose | Primary edit error | Edit cosine | Nearest-image edit error | Quadratic edit error |
|---|---:|---:|---:|---:|
| .05 | 43.2% | .905 | 100% | 46.6% |
| .10 | 29.6% | .957 | 100% | 31.7% |
| .50 | 48.2% | .885 | 60.5% | 30.4% |

These are means of per-curve ratios over four dependent curves, not a pooled error or confidence interval. Primary beats nearest-image error on all twelve held curve/dose rows. At .05/.1, mean absolute RGB error is about .26/.34 on the 0–255 scale. The small effect denominator makes the remaining uncertainty visible. At .5, broad image error is larger and the quadratic comparison is better on that particular observable. Primary background error exceeds the actual background change by roughly 2.6–3.5 times at .5; the interpolation introduces collateral that portrait similarity could conceal. Raw rows, clipping and paired metrics remain in [report.json](results/zero-network-response/report.json); the fixed eye/background coordinates are in [request.json](results/zero-network-response/request.json).

### A stricter contextual contrast

Astra pointed out that the original paired I1−I0 metric includes the branch difference already present at dose zero. We retained that metric unchanged and added a separately labelled retrospective scalar contrast:

\[
D_\alpha=Y_{I1}(\alpha)-Y_{I0}(\alpha),\qquad
J_\alpha=D_\alpha-D_0.
\]

The copy control predicts \(D_0\), so its paired error RMS is exactly \(\operatorname{RMS}(J_\alpha)\). Every method's existing paired error numerator is also its error on \(J\), since subtracting \(D_0\) from both predictions and targets cancels. Dividing those recorded scalars yields interaction-relative error without another pixel fit, decode or model run.

| Dose | Primary interaction error | Nearest control | Primary eye interaction error | Other-seed transfer error |
|---|---:|---:|---:|---:|
| .05 | 94.6% | 100% | 91.2% | 109.6% |
| .10 | 84.1% | 100% | 75.9% | 120.7% |
| .50 | 67.3% | 76.8% | 63.2% | 120.4% |

These are means over two seeds. The fine interaction is mostly unexplained at small dose. Another seed's response is consistently worse than zero interaction. At .5 the quadratic improves broad image accuracy but worsens interaction error to 73.8%, compared with the primary's 67.3%. A visually better image and a better causal-response model are different achievements. [analysis.json](results/zero-network-response/analysis.json) preserves the derivation, individual denominators and all methods.

## Compute and historical cost

| Measurement | Observed value |
|---|---:|
| Primary image arithmetic, median of 16 outputs | 1.075 ms wall / 1.075 ms CPU |
| Arithmetic plus PNG creation and SHA-256, median | 35.43 ms |
| Bundle load in serving | 25.51 ms |
| Serving 96 outputs including controls | 3.800 s |
| Calibration compilation | .328 s |
| Worker, including scorer and child startup | 6.909 s |
| Scheduler process elapsed | 8.38 s |
| Serving process peak RSS | 93.0 MiB |
| Full assay peak RSS, including float64 scorer | 772.1 MiB |
| Compiled RGB bundle | 3,940,997 bytes |
| New neural evaluations / checkpoint bytes read | 0 / 0 |

This ran on Beast's Ryzen 9 7950X with one configured arithmetic thread, under concurrent fleet load. mrun reserved four CPU slots by default; that is different from thread configuration. These are measurements from one run, not a latency distribution, energy/FLOP estimate or a controlled speed comparison with the dense generator. Import/launcher startup, queue wait, staging and collection are outside per-image timing. NumPy 2.5.4, Pillow 13.0.0 and Python 3.14.6 were recorded.

Historical work remains part of the cost. The source neighborhood used 24 native VAE decodes, four native scheduler checks and one model load, taking 61.126 s in its worker and 96.90 s in its runner. Earlier source acquisition adds further cost. The serving chart stores all twelve calibration RGB arrays; its compressed 3.94 MB is **larger** than their original 3.03 MB of PNGs. This is computation elimination by conditional reuse, not established compression of model knowledge.

## What this says about our larger program

Our work now distinguishes four achievements. A recovered symbol table names operations and operands. A full mathematical reconstruction executes them and carries its own state. Dependency analysis proves when a consumer cannot see a change, allowing exact reuse. A restricted response chart predicts outputs within supplied contexts. None alone proves a compact state that supports general future computation.

The simplifying question is whether we can find **consumer-sufficient coordinates that also advance predictably**. If \(s\) is the actual model state, \(c\) its context, \(F\) its next computation and \(\phi\) a reduced representation, the desired closure is

\[
\phi(F(s,c),c)=g(\phi(s,c),c),\qquad
Y=h(\phi(s,c),c).
\]

This condition must survive the relevant histories and quantized arithmetic. A low-rank residual or near-identical full state is useful evidence, but does not establish closure when a downstream consumer amplifies a small omitted component. Existing suffix-context and state-closure work already shows why earlier memory and recipient context can matter even when visible rows agree.

For an MLP \(f(x)=W_2\sigma(W_1x+b_1)+b_2\), a local expansion is

\[
f(x+\delta)=f(x)+W_2\operatorname{diag}(\sigma'(W_1x+b_1))W_1\delta+R_x(\delta).
\]

Fixing a ReLU gate region makes its real-arithmetic map affine. FLUX's smooth gated nonlinearities, attention softmax, normalization and BF16 rounding do not generally provide that exact simplification. The Jacobian depends on context, and attention can change which source contributes. Across a circuit, local Jacobians multiply and curvature terms interact. Interpolating the final image can avoid all these computations locally while concealing why a tiny contextual edit fails.

The most useful working assumption to test next is **bounded, context-conditioned response complexity on an explicitly chosen state family**. In local coordinates \(u\), seek \(Y_c(u)\approx a_c+A_cu\), with measured curvature and a small set of observable context switches. This loosens universal prompt/seed coverage and exactness; it tightens the demanded explanation of context, small effects and actual downstream consumption. This assay supports a local response chart, while its transfer and interaction errors argue against sharing one response basis indiscriminately across contexts.

The next informative experiment would freeze a compact latent/activation response basis on calibration contexts, then predict a genuinely withheld context's output **and its small contextual interaction**, with nearest/copy controls and no native calls during serving. That is a next experiment, not a result established here. General image generation without new neural computation remains open.

Related workspace research reviewed includes [dependency reuse](/Users/jakeholl/domains/saturn-pub/packages/mdb/docs/DEPENDENCY-REUSE.md), [FLUX suffix context](/Users/jakeholl/domains/saturn-pub/packages/mdb/docs/FLUX-SUFFIX-CONTEXT.md), [state closure](/Users/jakeholl/domains/saturn-pub/packages/mdb/docs/FLUX-STATE-CLOSURE.md), [latent direction crosses](/Users/jakeholl/domains/saturn-pub/packages/mdb/docs/FLUX-LATENT-DIRECTION-CROSS.md), and the posts [State, context, and the mathematics of time circuits](/Users/jakeholl/domains/obsidian/blog/2026-11-05-153154-state-context-and-the-mathematics-of-time-circuits.md) and [From replaying writes to predicting fresh edits](/Users/jakeholl/domains/obsidian/blog/2026-11-05-195050-from-replaying-writes-to-predicting-fresh-edits.md). These links are explicitly local workspace references: some documents remain unpublished. They describe distinct experiments and claim scopes. The [explorer's reconstructed pipeline](../viz/README.md) and this assay's immutable source identities, request and provenance are included in this public repository.

## Reproduce a network-free output

The public [chart.npz](results/zero-network-response/chart.npz) contains RGB anchors, not model weights. On a machine with NumPy and Pillow:

```sh
python generator/render_response.py \
  --bundle docs/results/zero-network-response/chart.npz \
  --context 60666.I0 --alpha .075 --output /tmp/response-075.png
```

The supported contexts are `60666.I0`, `60666.I1`, `60667.I0`, `60667.I1`; dose must be between 0 and 1. This CLI renders the primary rule and saves a hash/timing receipt. It is a convenience wrapper added after the assay; its arithmetic is the same executed `response_chart.py` code. The CLI mechanics test uses a synthetic three-pixel bundle. It does not start a GPU, download a model or accept new prompts. The result at .075 has no native accuracy target. Other new doses are also unmeasured until compared.

To rerun the original assay through mrun, use `response_study.py stage|submit|collect` with the source neighborhood directory identified in the frozen protocol, followed by `summarize_response.py RUN DESTINATION`. Staging requires the historical source artifacts; rendering the published chart does not.

## Integrity and limits of the instrument

Guarded mrun declared, sealed and execution-time payload SHA-256 all match `b8b85fbaa6af1aa67e329b9bf4122bf8fa8b94a79d4cf326def00f564bbec9c6`. Collection matched authenticated worker inventories before and after transport to the local files. Source PNGs and metadata were checked against their retained transported inventory. Compiler and serving ran separately, with neural-framework import exclusion, checkpoint-file and network denial, and audit logs showing no evaluation-directory reads. This is source-audited trusted-code execution, not hardware-level forward counting or an adversarial sandbox.

The prediction seal precedes scorer access and binds each PNG, the bundle, request and serving receipt. It is a dedicated manifest, not an MDB transaction certification. All held outcomes existed historically and earlier observations informed the design: this is retrospective held-file isolation, not prospective discovery. No quality threshold was used to suppress outputs or controls. Mechanics completed; a universal terminal claim remains unassessed.

The initial Saturn evidence sweep could not recognize the custom report and returned `evidence-incomplete-empty-target`. A separate scalar observation stream makes the reporting instrument applicable; its subsequent sweep and verification cover only those streams, not semantic accuracy or model-state closure. The raw report is unchanged.
