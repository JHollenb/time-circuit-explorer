# The baseline and its change must support the next computation

A writer can predict the difference between two states while getting both states wrong. The [107 crossover](finite-consumer-crossover.md) found that this hidden baseline error changes the next Q/K/V response. The [108 scalar analysis](results/research-review-cycle108/README.md) showed why the difference looked accurate: the two endpoint errors are large and nearly shared. This assay asks which parts of the finite normalization response expose that error before the learned projections.

The [completed audit](results/coupled-normalization/publication.json) finds a71.42%error in the changed finite normalized operand. The directional term contains a substantial gain mismatch, while baseline-aligned feedback adds a second reinforcing error. The [execution contract](plans/coupled-normalization-assay.md) confines this result to six existing states and six paid normalization/gamma evaluations. Arbitrary raw prompt to image with zero neural evaluations remains unsolved.

## Why a common error changes the response

For native endpoints a,b and our endpoints p,q, write

\[
\delta_n=b-a,\quad\delta_o=q-p,\quad e_0=p-a,\quad e_1=q-b.
\]

Then e1−e0=deltaO−deltaN. A small response error can coexist with a large nearly common endpoint error. An affine consumer would cancel an exactly common error in its response. A nonlinear consumer need not:

\[
F(a+e+\delta)-F(a+e)-F(a+\delta)+F(a)
\]

can remain large even when the two input changes agree. The measured crossover includes finite state admission; this real-arithmetic expression explains the issue without replacing those actual corners.

Let the explanatory normalized operand be Gamma h/s(h), where s(h)=sqrt(mean(h²)+epsilon). Expanding at the actual next endpoint k gives

\[
\Gamma\frac{k}{s(k)}-\Gamma\frac{h}{s(h)}
=\underbrace{\Gamma\frac{k-h}{s(k)}}_{D_{h,k}}
+\underbrace{\Gamma h\left(\frac1{s(k)}-\frac1{s(h)}\right)}_{B_{h,k}}.
\]

The directional term D depends on the write and its outgoing denominator. The feedback term B points along the gamma-weighted baseline. Both depend on the state. Preserving a raw change alone preserves neither term.

The actual original computation casts to FP32, performs square/mean/epsilon/rsqrt/rescale, admits the result to BF16 and multiplies the original BF16 gamma. Denote that finite output by Z. With rho(h)=Z(h)−evaluatedFP64Ideal(h), the measured explanatory equation becomes

\[
Z(k)-Z(h)=D_{h,k}+B_{h,k}+[\rho(k)-\rho(h)]+g_{h,k}.
\]

Here rho includes rounding of the explanatory evaluation as well as the native finite effects. The evaluated regrouping residual g and the final closure are retained separately. Subtracting a computed residual and adding it back does not guarantee exact floating closure.

## What the two crossover paths separate

Use native a→b, own p→q, own change admitted onto the native baseline a→Xno, and native change admitted onto the own baseline p→Xon. These are the same actual six BF16 states used by107. The five contrasts retain their original signs:

| Contrast | Edge difference |
|---|---|
| E | own−native |
| Uno | own−no |
| Vno | no−native |
| Uon | own−on |
| Von | on−native |

For each contrast, subtract D,B and rho differences using the same incidence map, then retain the differences in edge regrouping residuals and the new contrast residual. All pairwise signed inner products remain visible. A large component may cancel another component; its unsigned norm is not a percentage of a causal explanation.

The pre-affine geometry and the previously measured Q/K/V responses answer different questions. A learned projection can amplify one direction and suppress another. No unique projection attribution follows from an operand norm alone, and this assay performs no new projections.

## The error is already present before the projections

At the unchanged active row9 of the discovery vessel00/10pair, the raw H1 response error was4.74%of its native response norm. The [new finite operand](results/coupled-normalization/analysis.json) gives71.42%on its own native-response scale. Moving the own change onto the native baseline gives4.70%; moving the native change onto the own baseline gives76.09%. These are source-policy reconstructions of unavailable107 normalized intermediates, with the original backend, casts, gamma and full512row layout.

| Response error | Finite normalized operand, new | Q, retained107 | K, retained107 | V, retained107 |
|---|---:|---:|---:|---:|
| Own minus native, E | 71.424% | 113.709% | 111.317% | 69.490% |
| Own change on native baseline minus native, Vno | 4.703% | 5.215% | 6.290% | 4.966% |
| Native change on own baseline minus native, Von | 76.091% | 118.902% | 117.090% | 74.345% |

Every column divides by that port's native response norm. The Q/K/V columns reuse [107 observations](results/finite-consumer-crossover/analysis.json); no projection was reevaluated. Different projected errors do not supply a unique attribution from this pre-affine decomposition. All1022unmodified normalization-row controls have zero observed error. The finite gamma-weighted endpoint errors remain83.89%/91.37%of their respective native endpoint norms.

For E, the D,B and R differences have norms50.77%,34.91%and2.18%of the native normalized-response norm. The D:B cosine is+0.3644: the two larger terms reinforce each other. Their norms therefore cannot be added as causal shares. The evaluated final closures are zero; the separately retained edge/contrast residuals are tiny. The Uno Gram reconstruction retains its observed−1.3878e−17scalar residual.

The radius moves in opposite directions: native0.195804→0.186307, own0.121726→0.124085. Those denominators change both D and B. In particular, D is not a pure angular measurement. Its gamma-weighted write is divided by its outgoing radius.

The [additional public-scalar inference](results/coupled-normalization-evidence/public-scalar-direction.json) uses three published D energies to infer the own/native D cosine0.999249 and norm gain1.462656. If A=||Down||²,B=||Dnative||²,C=||Down−Dnative||², polarization gives their inner product(A+B−C)/2. The best nonnegative target-assisted rescaling of Down is0.683174, leaving3.8748%error relative to nativeD. This is inferred geometry from existing scalar energies, without a new coordinate measurement. It neither corrects B nor reconstructs the missing baseline direction, and its target-derived scale cannot be a serving input.

The useful distinction is therefore between a fairly well-aligned write with the wrong normalization gain and baseline-aligned denominator feedback that also disagrees. A direction/radius representation exposes both requirements. It does not supply the initializer or transition that would satisfy them.

## What an owned state must preserve

The [time-circuit account](../../obsidian/blog/2026-11-05-153154-state-context-and-the-mathematics-of-time-circuits.md), [MLP own-state work](../../obsidian/blog/2026-11-06-111533-improving-an-mlp-on-its-own-predicted-states.md) and [state quotient counterexample](../../cmachine/research/state_quotient/FINDINGS.md) point to the same closure requirement. If C is a code for the state, F its native update and R the consumer read, a reusable owned computation needs

\[
C(F(h))=T(C(h)),\qquad R(h)=\widetilde R(C(h)).
\]

It also needs an initializer from its own raw input. These equations state a requirement; they do not construct T or prove that C is small. Matching one contrast is weaker than preserving both the next state and the consumer's read.

The [next selected construction](plans/joint-baseline-response-writer.md) will let both slots of the small input-owned writer preserve baseline geometry and contextual response together, using finite H and offline Z feedback. Its algorithm is not yet specified or implemented. The result cannot repair the separate [nine-atom padding-capacity gap](finite-writer-capacity.md), supply a contextual initializer or close the image pipeline. Learned arithmetic, offline preparation, lookup storage and serving work retain separate accounting.

## Compute and evidence

The single guarded Beast job job-5ad4ac56520d completes in12.49scheduler seconds. The worker reports5.394seconds after imports/request verification, including2.915seconds for full shard authentication and0.260seconds for gamma loading/header handling. It materializes only5120parameter bytes. Six original normalization/gamma calls process3072rows/7,864,320coordinates per pointwise stage, with zero affine MACs or Q/K/V matrices. This is paid neural arithmetic, without a matched timing baseline or speedup claim.

The worker reports3.827CPU seconds,22,287,360bytes peak CUDA allocation and1,136,204KiB high-water RSS. Scheduler samples report1167.2MB RSS/316MB VRAM; its intent-first reservation is4096MB RAM/4096MB VRAM/4CPU slots, with numerical threading fixed at1. Per-Act event timing includes hash work and is not isolated kernel latency.

Seven supplied mechanics cases pass once, in0.07492seconds including candidate imports, after exact source review. The [publisher](../generator/publish_coupled_normalization.py) checks402analysis-derived scalars,36pooled scalars,1024raw comparison records,86retained field ABIs and seven canonical typed transactions. Fresh full-byte checks cover93remote files/5,000,306,048bytes, including the original shard. Vector norms and dot products remain source-qualified producer measurements rather than independent vector replay.

All86new numerical fields,17,361,792file bytes, stay on Beast. The compact publication contains25members/535,264bytes plus its publication manifest. No scientific tensor hydrates locally. The three extra ray witnesses and their scalar inference run separately in0.02731seconds including process startup, with frozen [stdlib source](results/coupled-normalization-evidence/scalar_direction_geometry.py). Full research/storage108 remains current, with the next scheduled review111 or earlier material evidence/pressure. No archive or deletion occurs in this assay.
