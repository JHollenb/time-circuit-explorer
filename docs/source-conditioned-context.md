# Better attention geometry does not yet preserve prompt changes

One coordinate system compiled from source token atoms improves a first-layer context reader on eight fresh strings. It removes prompt-fitted centers from serving and advances its own prefix statistics. But its important prompt-change errors remain large. Rank64 median whole-head error falls from162.7% unconditioned to92.8%; uniform attention is92.4%. Above/below and inside/outside *all-query* change errors are780.5% and1,743.0%. This is a partial computational improvement, not a faithful encoder or an arbitrary-prompt zero-neural image generator.

The active-query countertrend matters: the kite's above/below change improves to78.0% error, versus84.0% uniform and169.8% unconditioned. Padding rows drive that pair's much larger all-query error. The later FLUX consumer receives all512 rows, so padding cannot be discarded. The [raw paired results](results/source-conditioned-context/report.json) retain both behaviors.

## First establish that coordinates matter

The [cycle6 reader](compiled-context-statistics.md) has a correct streaming implementation but poorly conditioned Gaussian features. Cycle7's [saved-only diagonal experiment](results/context-diagonal-diagnostics/report.json) holds its14 prompts,40 selected queries, Gaussian draws and ranks fixed. For scaled positioned operands x=q/128^(1/4), y=k/128^(1/4), transform

\[
u=D(x-a),\qquad w=D^{-1}(y-b),\qquad \beta(y)=a^\top y.
\]

Because

\[
u^\top w+\beta(y)=x^\top y-x^\top b+a^\top b,
\]

exact softmax is unchanged. D is positive diagonal. With coordinate variances A,B, choosing D_l=(B_l/A_l)^(1/4) minimizes the summed operand-norm terms. It does not minimize normalized attention error.

| Rank64 geometry, retrospective | Median key TV | Median selected-head error | Active selected-head error | Padding selected-head error |
|---|---:|---:|---:|---:|
| Unchanged | .980 | 147.8% | 140.0% | 194.0% |
| Centered, scalar balance | .756 | 109.4% | 108.9% | 127.0% |
| Diagonal balance only | .620 | 94.3% | 76.5% | 104.5% |
| Centered, diagonal balance | .556 | 80.4% | 88.0% | 65.3% |

Centered diagonal geometry improves head RMS on35/40 selected queries and TV on33/40 compared with centered scalar geometry. Median per-prompt pair-sum norm squared falls from1,475 unconditioned to18.71 with centered scalar balance and8.08 with centered diagonal balance. Exact probability-gauge closure is6.11 ×10⁻¹⁶; unchanged-feature heads recover sealed results within3.19 ×10⁻¹⁵. No coordinate moment floors activate at either1e-12 or1e-10; the two floor arms coincide. BF16 reference words match at all selected queries.

These coordinates use whole held arrays with future/padding-query lookahead. They establish a useful conditioning effect, not reusable serving parameters or an optimized oracle bound. The finite mrun job `job-f10e795a9d2c` took3.25s with99.5MB sampled RSS and no VRAM, native inference or original checkpoint reads.

## Compile one geometry without test prompts

The [source-frozen protocol](plans/source-conditioned-context.md) fixes2,048 vocabulary IDs sampled without replacement, all512 positions, a query distribution, the same Gaussian array and eight new strings before execution. The [exact ID list](results/source-conditioned-context/source-ids.npy) has SHA-256 `887ef22f692c04965643663162f1026a31d47bc02b2e31ccc07011b4d8d7efaa`. Sampling seed1707 is explanatory; the frozen bytes are authoritative.

The key pool is uniform over sampled token×position pairs after the real eager BF16 RoPE program. Query moments mix50% of that pool with50% padding-token×position pairs. This is a source-distribution heuristic. It is neither natural-language frequency nor an estimated deployment distribution. Uniform vocabulary sampling includes unusual/special IDs.

The compiler streams64 tokens through all512 positions at a time and merges FP64 means/M2. For uniform-pool U and padding P, it includes the between-component variance:

\[
a=\tfrac12\mu_U+\tfrac12\mu_P,\qquad
A=\tfrac12\operatorname{Var}_U+\tfrac12\operatorname{Var}_P
+\tfrac14(\mu_U-\mu_P)^2.
\]

Key center b and variance B come from the key pool. The compiler freezes a,b,D once; no test prompt or contextual target enters this fit. The resulting3,072-byte geometry payload is [public](results/source-conditioned-context/geometry.npz), bound to hash `cd20f10d105c8ea83cad38c940cb15a33b38b54abb59421affb0c7a47f641294`. No moment floor activates.

The fresh serving process accepts raw strings, compiled lexical atoms and those frozen constants. It forms positive feature logs, adds aᵀy to each key, and advances actual z/M/maxima/count/clock statistics. It never refits on a prompt, receives no contextual hidden target, and has no original-projection or exact-attention fallback. Selected state checkpoints and complete heads seal before canonical full-prefix/reference attention access.

The transformed log partition is retained separately. Original-gauge comparison adds xᵀb−aᵀb. This accounts for a row factor; it does not make the approximate kernel exact. Synthetic tests verify this correction, mixture moments, masks and continuation from the candidate's own saved state.

## Broad reconstruction and prompt changes disagree

Relative head error is RMS(candidate−reference)/RMS(reference). Entries below are medians across eight fresh strings, or six for padding. Two long strings use all512 active positions. References use canonical BF16 operands with FP64 mathematical attention, not native Flash/SDPA.

| Method | All queries | Active queries | Padding queries | Ideal W_o-slice error, all queries |
|---|---:|---:|---:|---:|
| Conditioned16 | 133.3% | 69.5% | 149.7% | 124.2% |
| Conditioned32 | 127.6% | 98.4% | 139.3% | 118.2% |
| Conditioned64 | 92.8% | 85.1% | 105.9% | 82.8% |
| Unconditioned64 | 162.7% | 132.0% | 173.3% | 149.0% |
| Uniform | 92.4% | 58.0% | 97.0% | 82.5% |

Conditioned64 beats uniform whole-head error on only2/8 strings. Japanese-color strings each remain around164% head error. Rank trajectories are not monotone on active queries. All canonical port comparisons match, and six direct conditioned-kernel checks close within4.72 ×10⁻¹⁶. This excludes a large streaming implementation discrepancy as the explanation for these observed errors; it does not certify native numerical parity.

Prompt-change error divides RMS(predicted change−exact change) by RMS(exact change), retaining small denominators:

| Pair, all queries | Conditioned64 | Unconditioned64 | Uniform |
|---|---:|---:|---:|
| Kite above/below boat | 780.5% | 114.6% | 80.3% |
| Key inside/outside box | 1,743.0% | 100.0% | 78.3% |
| Japanese umbrella/bicycle colors | 117.0% | 100.0% | 64.1% |
| Long repeated scene order | 217.6% | 550.3% | 95.0% |

| Same conditioned64 pair | Active-both error | Padding-both error |
|---|---:|---:|
| Kite above/below | 78.0% | 854.1% |
| Key inside/outside | 142.3% | 2,361.7% |
| Japanese colors | 110.0% | 451.4% |
| Long order | 217.6% | No padding |

All four pairs have matching active-token counts, so their masks match at every position. Content/positioned operands still change. Japanese padding's exact change RMS is only8.32 ×10⁻⁵; its large ratio retains that scale explicitly. The long-order result improves substantially over the unconditioned feature reader but remains worse than uniform. The kite active-query advantage persists through the ideal output slice (78.5% versus83.7% uniform), while its padding error remains severe. A single pooled verdict would erase these distinctions.

## Measured cost and evidence

| Measurement | Value and scope |
|---|---:|
| Source geometry compilation | 26.234s |
| Source atom payload read | 35,668,992bytes,2,049 unique rows including pad |
| Uniform source pool | 1,048,576 Q and1,048,576 K token-position vectors |
| Additional padding pool | 512 Q vectors |
| Source RoPE arithmetic | 537,001,984 multiplies;268,500,992 adds;805,502,976 scalar BF16 boundaries |
| Warm raw-text median range, conditioned64 | 7.74–15.11ms |
| Warm raw-text median range, unconditioned64 | 10.16–19.22ms |
| Warm raw-text median range, uniform | 3.54–5.63ms |
| First prefix assemblies in observed sequence | 1.10–94.87ms |
| Serving initialization after imports | .317s |
| Serving wall /CPU after imports | 2.068 /1.731s |
| Reported serving-process high-water RSS | 470,732KiB, about459.7MiB |
| Complete worker /scheduler lease | 32.022 /35.06s |
| Sampled full-process-tree peak RSS /VRAM | 542.1MB /0 |

At rank64, core state remains66,576bytes; feature constants add65,536, geometry3,072 and diagnostic ranges2,048. The implementation also preforms524,288bytes of log-feature arrays, keeps current Q/K/V/coordinate workspaces, saves output/trace arrays and fetches1,340,416bytes of unique lexical payload in serving. The full2.645GB learned atom store remains near compute. Core state is not total memory or compressed learned knowledge.

Coordinate shifts/scales, key bias, query partition correction, norms, projections, state contractions, exponentials and diagnostic logarithms are counted in the seal. This implementation batches feature formation while the unchanged control forms features incrementally; timing differences cannot all be assigned to coordinate geometry. No controlled native speedup, FLOP total, energy estimate or full-image latency is claimed. Warm medians have three repeats per prompt/method; first-use rows are not independent cold-cache trials. High-water RSS and sampled process-tree RSS have different scopes.

Job `job-b372c2f27c66`, request `abff7fa2ba69da4d93c59bfc6023611237545bbd1f9bb0404bed4f0e4bae6220`, has matching guarded declared/sealed/executed payload identities and authenticated collected inventories. Its [prediction seal](results/source-conditioned-context/prediction-seal.json) is `4190d74cc14397a2b6940ebb31f513f038426e807894e165ecd7af83f9f335f0`. Independent SSH byte verification checks63 current/parent-root files totaling42,438,417bytes. Original checkpoint bytes/new native inference are zero; post-seal references still evaluate eight full mathematical attention heads and six direct feature kernels and read the saved original output slice. Those are paid model-derived reference computations.

The [public evidence](results/source-conditioned-context/provenance.json) preserves every compact result, source IDs, geometry and feature bytes, with byte-lossless gzip observation streams. There are20,480 head-position and10,240 pair-position rows, all dependent. Saturn's primary run/verify returns0, `target-streams-only`. The600-row retrospective diagonal sweep emits40 automatic entropy-change fires and verifies its ledger with exit0; interleaved ranks/variants make these uncalibrated panel-order diagnostics. No terminal semantic/image claim or physical-address join is supplied. Source guards and audited code are not adversarial hardware attestations; memory containment was active and payload sandboxing off.

The [raw diagonal watch ledger and separate interpretation](results/context-geometry-watch/interpretation.json) retain those automatic alerts without changing the original mrun report.

## The next representation question

We now have a reusable source-compiled geometry and real fresh-string execution, with a clear limit: broad head reconstruction can improve while small prompt changes are badly distorted. A useful replacement must preserve those changes through actual consumers. The [next saved-only diagnostic](plans/common-bank-context-rank.md) will ask whether one shared finite-bank low-rank kernel can retain the paired reads, separating this Gaussian feature family from rank capacity. It will be explicitly oracle/transductive, not a new serving or generalization claim.

Code: [geometry compiler](../generator/context_source_geometry.py), [actual streaming server](../generator/context_conditioned.py), [source-frozen assay](../generator/context_source_study.py), [retrospective diagonal assay](../generator/context_diagonal_diagnostics.py), and [publication/byte verification](../generator/publish_source_context.py). See the [active research record](zero-neural-research.md) and local [living findings blog](/Users/jakeholl/domains/obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md) for the unchanged full goal.
