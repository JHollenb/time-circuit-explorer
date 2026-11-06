# Toward arbitrary prompt-to-image computation without neural evaluation

Status: **active research; arbitrary prompt-to-image remains unsolved.** The objective includes prompt interpretation, formation of novel operands, state advancement and pixel production. It is not satisfied by known-scene RGB interpolation, cached native conditioning, native decoding, a smaller neural student, or the original dense forward expressed in another library.

The [frozen research plan](plans/arbitrary-prompt-zero-neural.md) preserves that objective and the experimental boundaries. The [previous response-chart result](zero-network-image-response.md) remains useful restricted serving evidence. It does not generalize its four supplied contexts into an arbitrary generator.

## Why the next cut starts before another image demo

Astra reviewed the newest consumer-response and operational-variable-closure work, the scene compiler, current source and complete prompt-factorial evidence. The useful synthesis is that local computation must retain its products, context and ordered consumers before it can be reduced. OVC10's composed local polynomial reader gives roughly .861/.767-logit axial/joint reader error where the global quadratic H24 prediction gives 4.982/4.982, through a shared native tail. Original dense weights, full states, contextual baselines and that native tail still remain. This strengthens a factorization strategy; it is not zero-neural generation.

The first goal experiment therefore asks whether a compact composition hypothesis already describes a **new combination of two prompt changes**, and where its residual appears. The specimen is the existing full native seed127 prompt square: green/yellow assignment to a cone and cylinder crossed with cone-left/cone-right ordering. The original campaign used four prompt encodings, one common initial noise and four complete native histories. The new work performs no additional neural evaluation.

At the initial review, OVC11 was a prospective writer→memory→reader plan. Its later result now retains61/64 finite memory forecasts and three nonfinite branches; that development strengthens the case for bounded scalar normalization in the state-reader experiment below. The distinction remains: K/V written from a layer's incoming state is not its later hidden-state output, and a good endpoint does not establish that the intervening write and receiving computation have been recovered. The conditioning span result below likewise concerns a representation, not a causal encoder mechanism.

Three native histories calibrate the diagnostic. The fourth corner, yellow cone right of green cylinder, is excluded from the prediction child's file capabilities. All outcomes already exist and previous findings informed the design; the target PNG was viewed before execution. This is retrospective development evidence, not prospective generalization. Each of 34 boundaries is independently supplied its three native calibration states. It is not an own-state rollout or raw-prompt compiler.

## A supplied transport explains some interactions

Let B=X00, C=X10, O=X01 and T=X11. The two fixed laws are

\[
P_A=O+(C-B),\qquad P_H=O+H(C-B).
\]

H is a horizontal image-token-grid permutation derived from the recorded position IDs, or a horizontal pixel permutation at RGB. It is the identity on conditioning and text. It leaves feature channels unchanged and is not a qualified physical latent reflection. The shared noise is unreflected, so this does not test global spatial equivariance of the model.

The finite interaction M, transport correction C_H and residual R_H obey

\[
M=T-O-C+B,\quad C_H=(H-I)(C-B),\quad R_H=M-C_H.
\]

The report retains their raw RMS, signed dot, cosine and the ratio ||R_H||/||M||. It separately measures stored endpoint error against the held color response ||T−O||. Zero reference effects retain undefined ratios. BF16 rounding and RGB clipping remain separate from raw interaction arithmetic.

| Saved boundary | Additive error / held color effect | Transport error / held color effect | Raw transport residual / interaction |
|---|---:|---:|---:|
| Prompt conditioning | 76.6% | 76.6% | 100% |
| Pass0 initial text | 77.2% | 77.2% | 100% |
| Pass0 after joint4, image | 139.5% | 111.3% | 79.8% |
| Pass0 after single19, image | 153.6% | 86.9% | 56.6% |
| Pass0 readout noise | 160.1% | 66.6% | 41.6% |
| Pass2 after joint4, image | 116.9% | 118.8% | 101.7% |
| Final latent | 142.8% | 111.7% | 78.2% |
| Rendered RGB | 111.3% | 39.6% | 28.6% |

“Pass0” is the first denoising pass; local5 is after the fifth joint block and local25 after the twentieth single block. Conditioning's value is not a spatial-transform result: H is the identity there. The initial image projection is invariant in this panel and has zero native color effect, so its relative errors are undefined. The raw horizontal-law decomposition is shared across method rows; `rawMixedErrorRelative` is the actual per-method comparison.

These observations do not support a single transferable color direction in raw coordinates. Transport helps several image boundaries, hurts one measured comparison, and does nothing for the contextual text/conditioning interaction. The same conditioning and initial-text values recur across passes; those repeats do not supply new independent contexts. Localization along saved histories is observational, not causal mediation.

## The stronger image control prevents an overclaim

At RGB, the transported rule's MAE is 8.829 levels out of255; its error RMS is about26.66. Merely mirroring the color-calibration image gives MAE8.822 and RMS25.91, slightly better. The transported output has visible duplicate edges and ghosting. This is a useful failure of a supplied composition rule, not a generator achievement.

![Transported color-response candidate; visible ghosting remains.](results/prompt-composition/images/transported.png)

Clipping also matters. The transported raw residual RMS is about32.08, while clipping/rounding moves the candidate by16.80 RMS levels and reduces stored endpoint error to26.66. Those vectors are not additive scalar budgets. Mirror-only has no clipping displacement. The raw and rendered comparisons remain distinct; the rendered improvement cannot be attributed solely to transport.

## What three calibration corners cannot determine

For any interaction array M, the program

\[
X(c,o)=B+c(C-B)+o(O-B)+coM
\]

matches the three supplied corners. Their values alone cannot identify M at the fourth corner. A successful predictor needs additional structure: a reusable operator, context-dependent interaction law, or qualified computation that determines this term. A lookup of the target would not supply that structure.

This example also shows why an out-of-plane residual does not prove another independent semantic variable. There are two symbolic inputs c,o while four finite outputs may span three linear directions. Nonlinear composition can create that geometry. The question is which factored law predicts the interaction on new inputs, rather than how many labels we can attach after observing it.

## Even target-assisted coefficients leave a contextual residual

The follow-up tests the most permissive two-coefficient affine reconstruction in the supplied calibration plane. With flattened arrays and D=[C−B,O−B], compute

\[
G=D^\top D/N,\quad b=D^\top M/N,\quad
\gamma=G^\dagger b,\quad r_\perp=M-D\gamma.
\]

The best endpoint in that plane is B+D((1,1)+γ). It is explicitly target-assisted: fitting γ reads T, and those coefficients are unavailable to serving. This is a representation-capacity diagnostic, separate from the sealed prediction assay. Its [raw report](results/prompt-composition/span-report.json) retains every boundary, eigenvalue, rank, coefficient and residual dot product.

| Saved boundary | Residual RMS / interaction RMS | Residual RMS / held color effect RMS |
|---|---:|---:|
| Prompt conditioning | 78.4% | 60.0% |
| Pass0 initial text | 78.5% | 60.6% |
| Pass0 after joint4, image | 69.4% | 96.8% |
| Pass0 after joint4, text | 16.7% | 83.7% |
| Pass1 after joint4, text | 15.1% | 24.2% |
| Pass2 after joint4, text | 12.5% | 15.5% |
| Final latent | 63.0% | 89.9% |
| Rendered RGB, before clipping | 42.8% | 71.2% |

These are RMS ratios, not fractions of variance. At conditioning, the two Gram eigenvalues are .6912 and1.6079, condition number2.326. The best endpoint coefficients are approximately(.8212,.4003), yet residual RMS remains .7752. This measured mismatch cannot be removed just by choosing better scalar coefficients for these two complete calibration differences. It appears before the denoiser. That is an observed boundary discrepancy, not localization of the causal operation inside the encoder.

The counter-trend matters: some later text states lie much closer to the plane, with only15.5% of the held color effect left unexplained at pass2 after joint4. Meanwhile final-latent error remains89.9% even with target-assisted coefficients. There is no single “linear versus nonlinear” verdict for the whole computation.

The RGB span residual also exceeds the supplied horizontal-transport residual. This is consistent: mirroring can create directions outside the original two-delta plane, so it is a different hypothesis class. Neither result establishes a new independent semantic variable, excludes a curved two-variable representation, or provides a reusable raw-prompt law. Every boundary still receives native calibration states independently.

## Evidence and continuation

The completed first diagnostic is guarded CPU job `job-3ff4ee580893`, request `1243c7b75de89651798eff248dfa8cc54d7378b8d97230b23092ee025b127164`. It sealed204 concrete candidate-array hashes across34 boundaries/six methods, with zero target-exclusive source files opened by the prediction child. The scorer recomputed those same candidates and verified every hash before reporting comparisons. Source arrays are bound to the original authenticated native campaign; no learned weight math, encoder, denoiser, VAE or neural student ran. Runtime was19.81 scheduler seconds,19.113 worker seconds, with495.4MB peak RSS and0VRAM. These are analysis costs, not image-serving latency or speedup figures.

The second diagnostic is guarded CPU job `job-7667bd48bb25`, request `c01ee02eb75a1a627ce1eb59977abff1e38d1d3c8aa752b02b3c11e673e99e12`. It took21.74 scheduler seconds,16.419 worker seconds and390.7MB peak RSS, with0VRAM and no neural evaluation. Its earlier default-reservation lease `job-a0456ce5fe32` was cancelled while queued, before execution, and remains unexecuted evidence. The replacement used a documented2GiB profile based on the first assay's measured495.4MB peak.

The [published custody subset](results/prompt-composition/provenance.json) retains exact report/request bytes, payload digests, result inventories and all ten original/candidate PNGs. The standard Saturn evidence sweep and verification completed with two observation streams/238 records and `target-streams-only` status. There are no auto-harvested terminal-claim receipts or physical address joins in this bundle; a clean sweep is an instrument check, not scientific certification.

The original native source is `job-95453050cf73`, checkpoint `dd976d67036c4a0a55642f702dcb3740ba7a4f367557596572baac168330904f`. Its complete prompt encodings, states, parameter work, denoising and VAE costs remain historical assistance. The [source plan and limitations](plans/arbitrary-prompt-zero-neural.md), [diagnostic implementation](../generator/prompt_closure.py), [launcher](../generator/prompt_closure_study.py), and [target-assisted span implementation](../generator/prompt_span_study.py) make those boundaries explicit.

The remaining route is a reduced **state algebra with its own continuation**: a raw-prompt compiler must form meaningful operands; a small factored transition must advance its own predicted state; and a validated non-neural decoder must produce pixels on held prompt combinations. Per-clock native anchors must disappear from serving. A proposed eight-prompt shape×color×relation dataset can test whether a color–order interaction transfers to another shape, but even seven-corner reconstruction would remain a diagnostic until that own-state requirement is met.

After reviewing both results and the newest posts, Astra recommended a [compiled MLP write and its residual consumer](plans/compiled-mlp-write.md). That experiment is now [implemented and measured](compiled-mlp-write.md). A supplied32-row input line is compiled offline into five output-coefficient arrays; serving evaluates a degree-four polynomial and lets the residual update consume its prediction, with no original matrices or native output endpoints online. Arithmetic takes .378–.387ms with51.36MiB primary RSS. Offline original-weight arithmetic and4.136s compilation remain explicit.

At the midpoint, pooled rounded-write paired-response error is9.43% versus11.96% for native endpoint interpolation; receiving-state values are12.59% versus15.42%. That advantage is heterogeneous: only7/32 rounded-write rows improve, and median per-row error remains65.1%. Smooth-polynomial versus staged-BF16 differences remain mixed with approximation error. This eliminates original dense projections for one bounded serving computation; supplied operands, context and arbitrary prompt formation remain unresolved.

The [compiled state reader](compiled-state-reader.md) now advances that producer's actual BF16 output into the next joint layer's row-local Q/K/V affines, with no dose input or native refresh. Source-bound job `job-2f0a52357bf1` sealed predictions before six native CPU reference evaluations. A producer-derived five-direction basis and actual state variance/sqrt feed three learned reduced maps, taking .911–.936ms for32rows. Midpoint composite paired errors are18.73% Q,18.61% K,13.90% V. Median row errors remain83–91%; four rows carry about90% of the effect energy. Attention and changing context are outside this edge.

The follow-up saved-array job `job-15dea027e760` separates output rounding, basis omission, combined numerical policy and inherited writer defects, retaining their signed cross terms. Removing basis omission through the same final quantizer still leaves midpoint reader discrepancies of14.92% Q,15.18% K,10.28% V relative to native change. The corresponding omission-through-quantizer terms are14.53%,14.27%,10.47%, with negative cross terms. Both matter. An out-of-basis residual near.166% of full state RMS is10.18% of the midpoint state-change RMS. A large shared baseline can conceal an important change error.

This is compiled learned arithmetic: framework calls0, original dense projections0, learned reduced-map evaluations3 per receiver call, plus coordinate recovery. The raw receiver report's `onlineNeuralEvaluations:0` must be read as an operational native/framework counter. The same bank contractions can be represented as a small linear network. They do not satisfy a representation-independent prohibition on learned maps, and do not complete the arbitrary-prompt objective by renaming computation.

The [next global consumer](compiled-global-head.md) is now measured. Actual emitted states drive joint2head0 through normalization/RoPE and an incremental complete1536-query read, while1504 input rows remain explicitly supplied fixed context. Same-arithmetic direct attention closes near5e-14, with identical BF16 outputs, in a1.8ms primary calculation. Paired forecast error is28.76% raw/35.13% rounded; raw text-region error exceeds100%, and897rounded-reference queries have zero change. The per-query records retain those cases.

Most reference response energy appears in queries outside the32changed input rows, establishing global coupling under this mathematical consumer. Saved-only matched V-delta permutations preserve proposed delta power while increasing raw error from29.84% to115–134%. Their BF16 admission differences remain visible. These results describe one head/context, not semantic addresses, native Flash parity or image quality. The newer [query/bank post](/Users/jakeholl/domains/obsidian/blog/2026-11-05-233846-how-image-queries-and-memory-banks-compute-a-state-change.md) retains much larger interactions at a different clock and consumer horizon; local value dominance cannot be generalized.

Astra's cycle4 review found no material algebra/custody blocker and recommends moving next to **raw-prompt operand formation**. The [source-grounded plan](plans/raw-prompt-encoder-atoms.md) targets the real Qwen3 first-layer token-local embedding/RMS/QKV/QK-RMS prefix. It designs for all151936 token IDs, with a bounded mechanics smoke that cannot redefine vocabulary coverage. New raw strings would gather fixed lexical atoms and apply exact tokenizer/template/position policy, without supplied prompt hidden states or online original projections. No such compiler has run yet. This remains finite learned-function compilation, with contextual encoding through layers9/18/27, denoising and image decoding still open.

The latest [reader-route-future evidence](/Users/jakeholl/domains/saturn-pub/packages/mdb/docs/FLUX-READER-ROUTE-FUTURE.md) reinforces the choice of a receiving operation. It qualifies exact local predictions on four new dose states and the next-block timing of a text-to-image effect, but still executes native attention, projections and MLPs; final images are observed native continuations. Those results do not add zero-neural prompt coverage.

A lower-priority proposed diagnostic uses the existing shape×color×relation cube: freeze the pairwise interaction law from seven corners and predict the eighth. In bit-indexed notation,

\[
\widehat X_{111}=X_{110}+X_{101}+X_{011}
-X_{100}-X_{010}-X_{001}+X_{000}.
\]

Its residual is the signed third mixed finite difference. It asks whether pairwise interactions change with shape, without fitting a coefficient to the eighth target. This archived dataset has a different seed and prompt family from the first assay and only selected saved ports; it cannot be presented as a full112-transition state rollout. Metadata and custody must be bound before any launch. No new neural run is needed for that diagnostic.

The goal remains active. Astra reviewed the receiver and requested the saved-array attribution and same-final-quantizer counterfactual before expanding attention. The newest [image-value route evidence](/Users/jakeholl/domains/obsidian/blog/2026-11-05-230617-a-reader-can-predict-a-change-without-carrying-its-main-effect.md) now directly connects a supplied value payload through the image-output stream to its later native RGB response across four contexts. It still uses native consumers and does not form arbitrary-prompt operands. Related background includes [the receiving-computation post](/Users/jakeholl/domains/obsidian/blog/2026-11-05-220837-predicting-the-change-that-the-next-computation-receives.md), [the simplifying-assumption post with OVC11](/Users/jakeholl/domains/obsidian/blog/2026-11-05-161109-the-one-assumption-that-would-make-model-computation-easier-to-understand.md), and [the earlier scene generator](/Users/jakeholl/domains/research/demos/scene-generator.md). These are workspace links, not claims that unpublished documents are public.
