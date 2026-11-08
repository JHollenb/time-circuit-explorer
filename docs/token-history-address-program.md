# What a small token-history program captures—and misses

An integer program can produce attention addresses in roughly 31–37 milliseconds per recorded prompt, without evaluating Q/K scores. On four retrospective transfer prompts, however, it covers 68.43% of our old reader’s selected weight, while recent-token selection covers 69.09% and a token-null control covers 69.33%. Its addresses ignore every controlled color and relation edit. This is an executed address producer with weak transfer, not a replacement contextual encoder or a new image generator.

The result connects the [ISA register-program method](../../isa/docs/executable-symbol-scan-lowering.md) to the missing raw-prompt computation. ISA has a useful native input interface: instruction bytes, registers and flags. A compact Boolean/copy program can advance decoded feedback within its measured scope. FLUX’s contextual operands still have to be produced. The [behavioral inference analysis](behavioral-authority.md) explains why a compact program and its evidence boundary must both be explicit; it does not require a global proof before this kind of exploration.

## The address port

This experiment predicts the ordered support of [our approximate top-four first encoder reader](uncentered-reader.md). Native FLUX does not expose a semantic top-k instruction. The approximate reader admits the masked causal key prefix, evaluates scores and retains four addresses:

\[
S_{h,i}=\operatorname{top4}_{j\le i,\ m_j=1}\ s_{h,i,j},\qquad
s_{h,i,j}=\frac{q_h(x_i,i)^\top k_{g(h)}(x_j,j)}{\sqrt{128}}.
\]

Ties use increasing key position. Here q and k denote the qualified represented operands, with their normalization, rotary position and BF16 boundaries. Their token-local source is already understood and [compiled](raw-prompt-encoder-atoms.md); the contextual read is additional computation.

The new program sees IDs, masks, positions and head identity. Its initially empty state records admitted token positions and occurrence lists. Each head has four selectors: first/recent positions, occurrences of the query token, or occurrences of literal token IDs drawn from discovery inputs. Missing or duplicate addresses fall back to recent unused positions. It has no hidden vectors, Q/K/V, scores, weights, prompt identifiers, complete-prefix keys or query-ID dispatch table.

Two historical cube/sphere prompts supply discovery supports. Greedy fitting uses 62 nontrivial active queries per head, a fixed inventory of 73 expressions and the actual duplicate/fallback behavior. The emitted program is frozen before serving the four historical pitcher/watering-can prompts. These are retrospective transfer cases, not newly quarantined semantic examples. There is no post-transfer refit or rule-budget expansion. The [frozen plan](plans/token-history-address-program.md), [source](../generator/token_history_address.py), [fresh-process runner](../generator/token_history_process.py) and [driver](../generator/token_history_study.py) specify the complete experiment.

## What transferred

The principal denominator is 4,224 active head/query rows with more than four admitted keys, across four dependent prompts. Each fraction below is an average over those rows.

| Method | Ordered support exact | Set overlap | Old selected-weight coverage | Top-one exact |
|---|---:|---:|---:|---:|
| Learned token-history program | 3.10% | 56.17% | 68.43% | 48.39% |
| Position-only fitted program | 3.24% | 54.23% | 66.65% | 48.67% |
| Recent four, descending position | 2.51% | 55.91% | 69.09% | 38.12% |
| Earliest four, ascending position | 0.00% | 12.99% | 8.94% | 0.00% |
| Fixed one-head rotation | 0.99% | 49.79% | 62.09% | 25.99% |
| Learned program with unique token sentinels | 2.91% | 55.92% | 69.33% | 47.16% |

The token-null control uses synthetic negative identities, preserving positions and masks while removing both repeated-token and literal-anchor information. It is not a valid raw prompt. The position-only family has fewer candidates, so its comparison does not isolate token information from search capacity.

Coverage means precisely

\[
\operatorname{coverage}(P,S)=\sum_{j\in P\cap S}w_j^{\mathrm{old}},
\qquad \sum_{j\in S}w_j^{\mathrm{old}}=1.
\]

These are weights normalized within the old retained top four. Coverage is not full-attention mass, candidate read accuracy or image quality. On discovery rows the program covers 71.75%, versus recent-four’s 68.88%; that advantage does not transfer. On 1,280 transfer rows with previously unseen query IDs, coverage is 68.67%, versus recent-four’s 71.19%. Head outcomes vary substantially: transfer coverage ranges from 22.38% to 99.99%. Fifteen heads improve mean coverage over recent-four, fifteen worsen and two tie. There are 891 improving rows, 874 worsening and 2,459 ties; the larger losses leave negative aggregate coverage despite the slightly larger improvement count. The [complete panels](results/token-history-address/panels.json) retain every case/head and adverse outcome.

Fallback supplies 7,592 of 16,896 primary active transfer slots, or 44.93%. The program differs from recent-four on 3,204 rows and from token-null on 2,440, so it is doing additional work; those differences do not establish a useful token-sensitive routing law. Its 128 slots contain 74 literal-occurrence selectors, 42 recent selectors, eight query-occurrence selectors and four first-position selectors.

Padding has a separate 60,800-row denominator. Primary coverage falls to 12.14%, with zero exact ordered supports, versus recent-four’s 17.49% coverage. Prefixes with at most four keys remain separate because their complete set coverage can be automatic. None of these dependent row counts creates independent prompt replication or an iid confidence interval.

## The controlled edits expose the gap

The primary prediction and selector-trace bytes are identical between the two discovery prompts. All four transfer prompts likewise have identical primary prediction/trace bytes. The reference routes change:

| Historical edit | Active nontrivial reference routes changed | Program routes changed |
|---|---:|---:|
| Cube/sphere left → right | 89 / 992 | 0 |
| Square 00 → 10: color swap | 150 / 1,056 | 0 |
| Square 00 → 01: left → right | 80 / 1,056 | 0 |
| Square 10 → 11: left → right | 83 / 1,056 | 0 |
| Square 01 → 11: color swap | 150 / 1,056 | 0 |

This post-hoc scalar contrast is retained in the [independent audit](results/token-history-address/scalar-audit.json). Identical addresses alone do not imply identical images: values, weights and later computations could still carry changing content. No such continuation was run here.

There is also a structural limitation. Once admission ends, these programs have a fixed token history and a fixed padding query ID. Every selector in this language becomes constant. The actual reader continues to change rotary query phase with position. Merely naming the clock in the input schema did not give the program a computation that uses it adequately.

Ignoring quantization for this algebraic illustration, ideal rotary attention has

\[
s_h(a,b;i,j)=\frac{q_h(a)^\top R(j-i)k_{g(h)}(b)}{\sqrt d}
=\sum_\ell\left[A_{h,a,b,\ell}\cos(\omega_\ell(j-i))
+B_{h,a,b,\ell}\sin(\omega_\ell(j-i))\right].
\]

The coefficients contain learned lexical geometry; the phase supplies a changing read coordinate. The qualified BF16 implementation rounds rotary products and sums at the absolute positions, so a source-faithful comparison predicate must initially retain both absolute positions rather than assume relative phase alone is exact. The ideal expression is not a replacement certificate. It nevertheless identifies a source variable that our selector family cannot express. Weak transfer here does not exclude symbolic contextual computation in a more appropriate family.

For unchanged tokenization, masks and positions, a token substitution has a second useful source constraint: it changes that token’s query row and key column. At an unchanged query, all unaffected-key scores keep their relative order. Removing edited-key addresses from the two top-four lists must therefore leave compatible prefixes of one common unaffected-key ordering. An edited key can displace another key into or out of top four; claiming that only edited addresses change would be wrong. The saved scalar audit retains this narrower check separately from prediction accuracy.

This points toward a reusable comparison interface,

\[
C_h(a,b,c;i,j,k)=\mathbf1\!\left[(s_{h,i,j}>s_{h,i,k})
\lor(s_{h,i,j}=s_{h,i,k}\ \land\ j<k)\right].
\]

The next mathematical question is whether source-derived lexical and positional structure admits a compact program for such comparisons. Defining C, tabulating complete prompts, or evaluating the original dot products under another name does not solve it. Astra recommends one [source-ordering feasibility cut](plans/source-ordering-program-feasibility.md): fixed padding query/head and discovery alphabet, retain absolute positions, census a lossless integer ordering representation, and charge source compilation/storage. It is written and unexecuted. No comparison compiler, further assay or image continuation is queued by this result.

## The measured computation and its boundary

One guarded mrun CPU job, `job-1ef4d6dc284b`, completed in 10.39 seconds with reported peak RSS 121 MB and zero VRAM. Intent-first admission chose a 4,096 MB/four-thread scheduler reservation; numerical-library thread settings were one. This is a reservation, not measured memory consumption.

| Work | Measured scope |
|---|---|
| Discovery | 228.25 ms fitting; 271.37 ms child process including startup/IO |
| Primary addresses | 31.38–37.39 ms per case; 16,384 head queries / 65,536 slots |
| All six methods / six cases | 1.573 s serving child process, including serialization/guards |
| Scoring and observation writing | 6.943 s; 98,304 dependent rows / 589,824 method rows |
| Emitted program | 13,235 JSON bytes |
| Token-history list payload | 840 bytes for 35 tokens; 888 bytes for 37 tokens |
| New learned matrix products, Q/K score products, neural evaluations, images | Zero |

Each primary case evaluates 15,872 selector expressions shared across heads and emits 65,536 slots. The named counters omit other comparisons, hashing, indexing, JSON, Python objects and allocator overhead. State grows with admitted history; membership and some selector operations scan it. This is neither a constant-memory machine nor an exhaustive instruction count. Program parse time, output construction and list payload are not total serving RAM.

The old reader paid 70.963 million Q/K products per discovery prompt and 74.867 million per transfer prompt. Those describe the operation targeted by this address-only replacement, not a matched runtime or parity saving. Historical learned preparation remains paid. Selected logits, soft weights, values, later contextual layers, denoising and pixels are still missing from this program. The existing live generator accepts new prompts while evaluating the reconstructed learned equations and retaining the native encoder/VAE. The hosted site displays recorded examples. Arbitrary-prompt zero-neural generation remains open.

## Evidence

[Receipts](results/token-history-address/publication.json) preserve all 68 original output files byte-exact, source/request/payload custody, prediction seals, every address/trace/metric row and eight targeted synthetic checks. Compiler and serving children block numerical/model imports and demonstrate 17/15 file-canary denials. These are trusted-source file/import exclusions, not adversarial OS isolation. The source review overlapped submission/execution; it was not a pre-submission admission gate. A separate actual-result review follows the output.

The first publication verifier required exact floating sum equality and rejected a one-ulp cross-Python difference. The [v2 verifier](../generator/publish_token_history_v2.py) preserves that failed version, audits the unchanged results with calibrated scalar tolerances, and keeps indicator counts exact. It does not rerun or alter the scientific assay. Full source and storage review 93 remains current, next due 96; no archive or deletion was needed for this compact publication.
