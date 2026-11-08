# A complete approximate state that does not yet carry context

Cycle 103 constructs all 512 rows of the first Qwen3 contextual state from its own raw prompt, without a neural runtime in the serving process. The result captures a substantial common update, but misses the contextual effect we need: every tested prompt edit produces exactly zero predicted change at unchanged-token positions. Pooled error improves over a position-only writer while padding and most individual transfer rows get worse. Arbitrary-prompt zero-neural image generation remains unsolved.

The specimen is the qualified FLUX.2 Klein Qwen3 conditioner: width 2,560, vocabulary 151,936, sequence length 512, Q width 4,096 and K/V width 1,024 each. This is one approximate state transition, not a complete encoder or image generator. [The frozen implementation contract](plans/contextual-writer-assay.md), [raw report](results/contextual-writer/report.json), [publication inventory](results/contextual-writer/publication.json) and [Astra's actual-result review](results/contextual-writer-evidence/advisor-actual.json) define the evidence scope.

## What the serving program does

The input is an ordinary raw string. Its own tokenizer produces the complete ordered token-ID, mask and position packet. A full-vocabulary lexical table supplies H0; the old E/Q/K/V atom store and native H0/H1 files are inaccessible to the fresh serving child. All learned preparation remains charged offline. Two fixed template rows use the previously qualified cofactor. The remaining 510 rows receive two shared constant updates:

\[
\widehat H_{1,p}=B\!\left(B(H_{0,p}+d_{0,a_0(p)})+d_{1,a_1(\phi_p)}\right),\qquad p\ge2.
\]

Here each B means Python binary64 addition followed by FP32 round-to-nearest-even and then BF16 round-to-nearest-even. The first tree uses position; the second uses causal raw-input features and is fitted to native H1 minus the actual rounded first-slot output. Each tree has depth at most four, at most 16 leaves and at least eight discovery rows per leaf. Its leaf selects a shared 2,560-coordinate FP32 constant. There is no online learned affine projection, gated MLP or dense readout.

The 155 available features include position, mask, causal active count, count/first/last registers for 40 discovery token IDs, 16 fixed current lexical coordinates and their causal active-prefix means. Unknown token IDs retain their full lexical row and generic features; they follow ordinary branches. This is broader input acceptance, not evidence of semantic generalization.

The actual first slot has 14 atoms; the second has 12. The position control uses 14 and nine: both share the same maximum budget, but the selected program actually uses 26 atoms versus 23. The selected dictionaries occupy 266,240 bytes, plus 10,240 bytes of fixed prefix and a 13,859-byte program. The complete lexical/tokenizer assets occupy 794,201,994 bytes, including the 777,912,320-byte H0 table. The small rule program therefore does not imply a small complete deployment.

Discovery uses cube A/B (left/right) and vessel 00/10 (color assignment, both left). Transfer uses vessel 01/11 (color assignment, both right). All six histories were public previously: transfer is reused-history composition, not fresh heldout. A seventh, fixed glass-observatory prompt executes with 11 unique IDs outside the discovery literal set, but has no native target. It establishes execution and ownership only.

## Endpoint gain and its countertrends

Errors below are RMS endpoint discrepancies divided by the RMS native increment from H0. Holding H0 is 100% by construction. The all-row panels include two copied prefix rows; the learned-row panel excludes them.

| Transfer state error | History 01 | History 11 |
|---|---:|---:|
| Accepted writer, all 512 rows | 37.88% | 37.96% |
| Position control, all 512 rows | 41.35% | 41.40% |
| Accepted writer, learned 510 rows | 38.11% | 38.18% |
| Position control, learned 510 rows | 41.59% | 41.64% |
| Accepted writer, active rows | 50.45% | 50.36% |
| Position control, active rows | 71.89% | 71.83% |
| Accepted writer, padding | 35.69% | 35.81% |
| Position control, padding | 34.74% | 34.86% |

The native next input normalization and Q/K/V consumer also show pooled improvement and padding regression. These are paid neural evaluations of the candidate state, separate from non-neural serving.

| Transfer 01 port | Accepted, all rows | Position, all rows | Accepted, padding | Position, padding |
|---|---:|---:|---:|---:|
| Q | 23.37% | 24.86% | 21.38% | 20.42% |
| K | 18.67% | 20.41% | 16.30% | 15.74% |
| V | 32.93% | 34.07% | 30.55% | 29.66% |

Transfer 11 has the same directions at all three ports. State error improves on 154 rows of 01 but worsens on 356, with the two copied rows tied; 11 has 158 improvements, 352 regressions and two ties. Each active panel has 33 improvements and two regressions. Padding has 121 improvements/354 regressions for 01 and 125/350 for 11. Large active-row gains can outweigh many smaller padding regressions in pooled squared error. Neither a pooled win nor a row-count majority alone describes the whole result.

The [report](results/contextual-writer/report.json) retains every case, stratum, port and signed prompt response. Case-panel cosine and prediction/target RMS describe complete endpoints. They are not cosine or RMS of the increment; only the error denominator is the native increment. Each port has a different denominator, so smaller Q/K/V percentages do not by themselves show attenuation of H1 error. Discovery feedback separately compares actual consumer increments, and pair panels compare signed changes between prompts.

## Why the contextual response is exactly zero

For an unchanged token at position p, both prompts supply the same H0 row. In every tested pair, the saved first- and second-slot atom addresses also match at all unchanged-token positions. Deterministic finite arithmetic then gives

\[
H^A_{0,p}=H^B_{0,p},\quad a^A_0=a^B_0,\quad a^A_1=a^B_1
\quad\Longrightarrow\quad
\widehat H^A_{1,p}=\widehat H^B_{1,p}.
\]

The next input normalization and Q/K/V maps are row-local. Equal H1 rows necessarily remain equal at those ports. No later measurement at this immediate consumer can recover a response the writer did not produce.

| Prompt pair | Unchanged-token rows | Native H1 rows with nonzero response | Predicted H1 rows with nonzero response |
|---|---:|---:|---:|
| Cube A/B | 511 | 503 | 0 |
| Vessel 00/10 | 510 | 502 | 0 |
| Vessel 01/11 | 510 | 502 | 0 |

Every observed method/pair panel has zero response on these unchanged-token selections, at H1 and Q/K/V. Half-strength is evaluated on discovery only; no half-strength transfer panel is executed. Pooled relative response error is 100%; cosine is undefined because the predicted response norm is zero. Some pre-edit rows also have zero native response, so this is not a claim of 100% error at each individual row. The pooled native response is nonzero, making the fitted program's failure interpretable.

The second tree uses eight current lexical coordinates, one position predicate and count:198/last:13. It selects no causal-mean feature or edit-token occurrence feature. Its largest leaf contains 1,924 of the 2,040 discovery rows; all 12 leaves contain rows from all four histories. The predicates distinguish local token/template structure but do not route the tested contextual edits differently at unchanged tokens.

This does not show that the permitted features or constant-writer family cannot express context. The dictionary and routes were learned together under endpoint squared error; a common update can dominate that objective. A fixed-dictionary oracle would test only these particular atoms, not every dictionary permitted by the grammar. No such capacity diagnostic was executed here.

## What Saturn feedback selected

The worker retains the initial program and one offline half-strength second-slot dictionary with identical routes. Both pass through the same actual native next consumer on discovery only. The balanced consumer-increment squared-error/reference-energy ratio is 0.1085665455 for initial and 0.1299117689 for half. Initial is selected and restored by exact immutable program bytes; the rejected half remains evidence.

This tests amplitude selection, not consumer-guided predicate discovery. It is immutable branch restoration, not optimizer-state rollback. Complete transfer predictions are sealed before transfer target files are released. Canonical Saturn Frames/Acts bind the actual H1 payload, the four separate native parameter tensors and the retained Q/K/V payloads. Parameter bytes are authenticated before and after all consumers.

## Compute and custody

One guarded Beast lease, job-ac1391fc231a, finishes in 165.09 seconds. Sampled process RSS peaks at 2,403.4 MB and device telemetry at 412.0 MB VRAM; the Torch allocator reports 93,328,384 peak allocated bytes, a different metric. The actual CPU numerical libraries use one thread; scheduler reservation uses four. The worker after imports takes 157.85 seconds.

| Work | Measured or declared cost |
|---|---|
| Authenticate/extract lexical assets | 42.25 s; 2,644,939,904 source bytes read |
| Fit two-slot program and control | 70.04 s; 2,040 rows, 155 features |
| Authenticate/load native consumer parameters | 3.88 s; 31,462,400 selected bytes; 4,967,215,360 source-shard bytes authenticated |
| Fresh final serving, two methods × seven prompts | 18.73 s total child wall time |
| Accepted feature formation per prompt | 2.0–2.65 ms |
| Accepted writer, serialization and hashing per prompt | 0.584–0.855 s |
| Accepted state arithmetic per prompt | 2,611,200 coordinate additions; two ordered rounding stages per learned row |
| Accepted routing per prompt | 4,066–4,069 branch decisions |
| Paid native evaluation | 28 serial B1 inputnorm/QKV units; 84 original linears; 225,485,783,040 nominal MACs plus normalizations |

Tokenization/H0 gathering is separate: the first final prompt takes 0.589 s, while subsequent cases range from 0.00087 to 0.0251 s in this child. These timings include their declared serialization/hash/load scopes; they are not an isolated arithmetic benchmark or a matched speedup comparison. Child high-water RSS can inherit pre-exec history and is not isolated steady serving memory. No energy measurement is made.

Eight supplied tiny mechanics checks pass once in 0.09859 s including import, before the guarded job. The [source review](results/contextual-writer-evidence/advisor-source.json), [mechanics](results/contextual-writer-evidence/mechanics.json) and [preflight](results/contextual-writer-evidence/preflight.json) remain immutable. Preflight checks command/root/reservation; undeclared verifier/geometry hooks are skipped, not passed. Declared, sealed and executed payload SHA agree at 118914e1ba7e2be1d58968258b5ee3e0bd574fffdc035133f08d10b853cf9c47.

Publication verifies 19 compact members totaling 5,600,315 bytes, 67,584 unique observation rows and 2,112 independently rebuilt pooled scalars. Fresh full-SHA remote checks cover 146 files totaling 1,078,982,047 bytes, including raw BF16/NPY payload joins. Complete arrays and compiled tables remain near compute; they are not copied into the development checkout. Import/file canaries test the trusted serving implementation, not an adversarial sandbox.

## The next question

The next [contrast-aware cut](plans/contrast-aware-contextual-writer.md) changes the offline fitting objective while preserving the same raw features, two-slot grammar, depth, dictionary budget and four discovery histories. Its explicit target is the missing signed response at unchanged tokens. For fixed routes, paired loss couples the leaf constants through a small graph Laplacian; independent leaf means no longer optimize that objective. All finite rounding and native-consumer effects still need measurement.

This connects the register-based success of the [ISA state-machine work](../../obsidian/blog/2026-11-06-004500-what-the-c-state-machine-learned-about-carry-and-context.md) to the [own-state MLP results](../../obsidian/blog/2026-11-06-111533-improving-an-mlp-on-its-own-predicted-states.md): a state must carry distinctions that its next consumer needs, and a good endpoint average can erase them. The [living research log](../../obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md) retains the full history. No attention, MLP, full encoder continuation, denoiser, VAE or image is run in cycle 103. Complete contextual conditioning, own-state denoising and general non-neural pixels remain required by the original goal.
