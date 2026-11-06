# A source-defined reader works on new strings, but its mixture loses contextual changes

The first source-frozen landmark reader now runs on eight new raw strings. Its positive prefix state is stable, and its computation matches a direct landmark calculation to1.43 ×10⁻¹⁵. The initial smooth mixture is inaccurate. A saved-only follow-up finds both an overly broad routing law and a remaining limitation of mixing fixed reader responses. This is progress toward a reusable contextual computation, not faithful full encoding or arbitrary-prompt zero-neural images.

The [cycle8 bank](common-bank-context.md) had fitted query lookups for known prompts. Here the source-only rule can read a query absent from that bank. The distinction matters: a reusable rule is implemented, while its fidelity remains an observed problem. All results concern Qwen3 text-encoder layer0, head0/KV0, under the existing512-position policy.

## Freeze the source rule, then accept new text

The [protocol](plans/source-landmark-context.md) uses the existing complete151,936-ID lexical store and the previous frozen2,048-ID source sample. Let x=q/d^(1/4), y=k/d^(1/4), d=128. Stable full FP64 covariance over2,048IDs×512positions defines

\[
C_y=\operatorname{Cov}_{\mathrm{source}}(y),\qquad
\delta_a^2(x)=(x-\ell_a)^\top C_y(x-\ell_a).
\]

This distance measures source score-displacement variance. It is not an error bound on a new prefix. The equivalent raw-Q/K formula divides by d. Covariance-root reconstruction differs by at most2.49 ×10⁻¹⁴; no negative eigenvalue repair was needed.

PCG64(seed1909) selects4,096 distinct source token/position candidates; all512 padding-token positions are added. The resulting4,608 identities have no overlap. A50/50 lexical/padding weighted mean selects the first landmark, followed by deterministic farthest-first coverage,64 total. The source nearest-other-landmark squared-distance median fixes τ²=2.0297147165. Routing is

\[
w_a(x)=\operatorname{softmax}_a[-\delta_a^2(x)/(2\tau^2)].
\]

Raw strings and deterministic tokenizer/template identities are bound in the original request. On the worker, exact IDs/masks/rendered-text hashes seal **before** source compilation or target head evaluation. Fresh serving checks that interface seal. This implements the protocol's interface freeze through a worker receipt rather than embedding the resulting token arrays in the original staging request. No string or reader parameter was selected from target-head errors.

## Advance the live prefix

Each fixed landmark maintains

\[
Z_{a,i}=\sum_{j\le i}m_je^{\ell_a^\top y_j},\qquad
N_{a,i}=\sum_{j\le i}m_je^{\ell_a^\top y_j}v_j.
\]

Running per-landmark maxima rescale the numerator and partition together. Repeated occurrences contribute repeatedly; masked keys do not enter the state. Every query, including padding, reads the accumulated prefix. The primary rule is

\[
\widehat h_i(x_i)=\sum_aw_a(x_i)\frac{N_{a,i}}{Z_{a,i}}.
\]

Nearest-landmark, kernel-mixture ΣwN/ΣwZ, and uniform-value controls share the state. Kernel mixing restores the individual log maxima before combining partitions; omitting them would change the rule. All four results stay within the prefix values' coordinate ranges up to5.56 ×10⁻¹⁷ roundoff. That amplitude property does not establish accuracy.

Fresh serving receives raw strings, source assets and the lexical table. It receives no saved target operands or exact-attention fallback. Complete predictions seal before canonical full-prefix assemblies and ideal FP64 attention references. Every canonical operand agrees; selected short/long direct checks close to1.43 ×10⁻¹⁵. Those are mathematical/reference instrument checks, not native Flash parity.

## New prompt changes reveal different failures

Each value below is RMS(predicted change−exact change)/RMS(exact change), over all512 query rows. All four pairs have equal active masks; active lengths are25,24,33,512. The long strings are truncated by the declared tokenizer policy, so their claims concern the resulting512-row inputs.

| New prompt change | Smooth64 primary | Nearest64 | Kernel mixture | Uniform |
|---|---:|---:|---:|---:|
| Lantern above/below bench | 84.2% | 62.9% | 87.4% | 80.8% |
| Green/purple pear-plum binding | 99.4% | 114.6% | 102.7% | 99.4% |
| Japanese bridge left/right | 136.5% | 62.7% | 112.3% | 137.4% |
| Long courtyard order | 91.8% | 110.6% | 93.0% | 96.1% |

The smooth primary's active-query errors are73.7%,81.0%,141.9%,91.8%. Its first two improve over uniform85.4%,90.7%, while Japanese active queries worsen relative to uniform137.2%. Nearest routing instead has active errors82.0%,197.6%,121.0%,110.6%. A method that helps padding can hurt the active binding response.

The nearest control's whole-head errors range26.1–37.8%, whereas the primary's six short-prompt errors range89.1–110.6%. On the long strings the primary improves to18.8–19.0% whole-head error, while its order-change error remains91.8%. Broad endpoint reconstruction is not a substitute for preserving a much smaller change. The ideal saved output-slice consumer retains these substantial failures; no full projection/residual, later encoder or image is evaluated.

## The bandwidth was broad, but sharpening is insufficient

The subsequent [routing diagnostic](results/landmark-routing-diagnostics/report.json) is explicitly retrospective development evidence on these already-scored strings. It freezes τ² multipliers1,1/16,1/256, recomputes candidate reads and seals before opening archived reference heads. The numerical prefix M/z/maxima are identical across scales.

| Change | Original τ² | τ²/16 | τ²/256 |
|---|---:|---:|---:|
| Above/below | 84.2% | 60.5% | 62.9% |
| Color binding | 99.4% | 106.3% | 114.6% |
| Japanese left/right | 136.5% | 58.3% | 62.6% |
| Long order | 91.8% | 84.4% | 108.4% |

For lantern padding, median route entropy falls3.916→.0247 and median maximum mass rises.0772→.9962 at τ²/16. The original bandwidth measured spacing **between selected landmarks**, not the query's interpolation residual. The mixture was averaging many readers despite a useful nearest reader. Sharpening helps three changes here and worsens the binding change, especially active queries81.0%→113.9%→197.4%. Selecting τ²/16 from this diagnostic would not be fresh validation.

## Factor routing from the response bank

For the equal-length padding suffix, corresponding queries are identical and no new keys enter. Write H_A,H_B for the64 normalized landmark reads on each prefix. The source-defined routing weights are the same across the pair:

\[
\Delta\widehat h(x)=w(x)^\top(H_B-H_A).
\]

This identity closes against the original predictions to3.55 ×10⁻¹⁷. It isolates a fixed64-vector **response bank** D=H_B−H_A; changing the bandwidth changes its convex coefficients. More generally the exact midpoint expansion is Δ(wᵀH)=mean(w)ᵀΔH+Δwᵀmean(H). The second term vanishes for these identical padding queries. We have measured that restricted case, not attributed every active-query error.

A target-assisted capacity test then gives each padding query its own best-fit coordinates. The signed affine relaxation minimizes ||wᵀD−target|| with Σw=1 but permits negative weights. The separate feasible convex fit enforces w≥0,Σw=1 with512 projected-gradient steps. The former is a lower bound on convex error; the attained convex fit is an upper bound, not a certified optimum or a serving route.

| Padding-pair response | Original smooth | Nearest | Attained convex fit | Signed affine residual |
|---|---:|---:|---:|---:|
| Above/below | 86.7% | 56.8% | 53.4% | 6.12 ×10⁻¹⁵ |
| Color binding | 100.5% | 106.9% | 48.1% | 1.14 ×10⁻¹⁴ |
| Japanese left/right | 135.9% | 52.1% | 36.6% | 1.51 ×10⁻¹⁴ |

The signed affine response ranks are20/18/25, with maximum negative-weight masses18.7/23.6/16.7. Near-exact affine reconstruction does not reveal a semantic program: finite reads already lie in their supplied value-bank span, and each fitted row reads its own target.

The convex fit retains a numerical objective bound. For f(w)=||wD−t||²/2, g=∇f and feasible w, the Frank–Wolfe gap gᵀw−min_a g_a gives

\[
\max(0,f(w)-\mathrm{gap})\le f_*\le f(w).
\]

Mean lower/attained objectives are.0001331/.0001399,.00006657/.00006839,.000002479/.000003448 for the three pairs. Thus the remaining residual is not merely a failure to iterate the optimizer, under this numerical fixed-bank diagnostic. These are exploratory floating-point bounds, without interval-arithmetic certification.

The restriction matters: this tests **context-independent query routing over normalized landmark responses shared across a pair**. It does not exclude context-dependent routes, kernel mixtures whose effective weights change with Z, richer statistics, other positive kernels, or all compact context states. Native attention is convex in values; forcing it to be a fixed convex mixture of64 different query readers is an additional assumption. We should loosen that assumption while preserving numerical stability.

## Computation, custody and limits

| Measurement | Scope/value |
|---|---:|
| Source covariance/landmark compilation | 24.96s,1,048,576 positioned keys |
| Declared covariance products | 17,179,869,184 MACs, plus positioning/selection |
| Source learned rows/payload read | 2,049 /35,668,992bytes |
| Main worker /scheduler lease | 30.79 /32.91s |
| Sampled process-tree peak RSS /VRAM | 601.5MB /0 |
| Fresh serving wall /CPU after imports | 2.954 /2.055s |
| Serving initialization | .609s |
| Reported serving high-water RSS | 463,892KiB |
| Warm raw-text evaluation | 25.3–34.3ms, all four readers/diagnostics together |
| Landmark core /uniform control /range diagnostics | 66,576 /1,024 /2,048bytes |
| Runtime reader asset /full source asset payload | 196,616 /439,304bytes |
| Routing diagnostic worker /lease | 3.13 /5.51s |
| Routing diagnostic peak RSS /VRAM | 260.4MB /0 |

Warm calls have five repeats and include lexical assembly, routes, all four controls, traces and checkpoint copies. They are not standalone primary latency. One routing-workspace bookkeeping estimate is34.93MB; additional temporaries and outputs remain, so this is not measured total memory. No native speedup, energy or image-generation compute claim follows. Exact landmark score evaluations, routing transforms/exponentials, state updates, cached learned outputs and the2.645GB upstream table remain paid computation/information. Guarded Python imports/source audits are not adversarial hardware attestation; OS memory containment was active, payload sandbox off.

Main job `job-7bafdd5e6846`, request `3ec4511d6b4e40823a96700f9e8621efb862aef541b99bfee60fcfdd136918af`, seals predictions as `1c896980b866d358ec69bd451571776d4005666b92111b1a010b7645be500051`. Diagnostic job `job-0b11cc784689`, request `18dd2f3b60a38be059cd2b7da3494ea937d4441f6217b95e4a28b458b289d146`, seals as `b8c0c3e3bec919e3bb535b3ac633a41a387c7b71e974623d42a39f4ba3502250`. Both declared/sealed/executed payload identities and remote/local collection inventories agree.

Independent byte checks cover31 files/33,311,333bytes for the main result and54files/105,166,718bytes for the diagnostic, with overlapping parents counted separately. Public compact evidence totals2,895,986bytes, retaining24,576 primary and6,144 diagnostic dependent observations losslessly. Full heads, states, ports and oracle fit arrays remain near compute. Saturn run/verify returns0 for both, with target-streams-only scope and no terminal image receipts/address joins. All47 generator mechanics tests pass.

Astra's final review recommends the [live query-transport moment law](plans/query-transport-moments.md) next. Factoring e^(xᵀy) around a source-selected landmark gives an exact exponential change of measure; its first derivative is the prefix-dependent covariance of values and keys. A bounded first-order test would add the missing query–memory interaction without target-fitted routes, at a substantial state cost. It is proposed, not measured. Afterward an actual future consumer must determine whether the representation helps the demanded conditioning.

Sources: [main raw evidence](results/source-landmark-context/provenance.json), [diagnostic raw evidence](results/landmark-routing-diagnostics/provenance.json), [reader/state implementation](../generator/context_landmarks.py), [raw-text assay](../generator/context_landmark_study.py), [routing/capacity implementation](../generator/landmark_routing_diagnostics.py), and the [living blog](/Users/jakeholl/domains/obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md). Full contextual encoding, autonomous reduced diffusion, general non-neural pixels and independent prompt-faithfulness evidence remain open.
