# Compiling an attention order into an integer program

One attention head now chooses the same four addresses as our historical reader using only token IDs, positions, an integer table and its own admitted history. On the two recorded discovery prompts it matches all 954 padding-query supports and reproduces all 426 changed supports across 477 paired padding queries after replacing “left” with “right.” Fresh serving takes 5.62–5.70 milliseconds per packet, with a separate 24.5 milliseconds to load, decompress and verify the table.

This is a bounded component result. Compilation pays 502,333,440 floating coordinate products and emits a 7.14 MB compressed program. Its domain is one padding query token, one head, 30 admitted token IDs and 512 absolute positions. All four retrospective transfer prompts remain unsupported. No image is generated. The local reconstructed learned generator accepts new prompts; arbitrary-prompt image generation with zero neural evaluations remains open.

## What the previous program left out

The [token-history selector experiment](token-history-address-program.md) discovered a small rule language with weak transfer. After the final active token, its selectors became constant. The actual first reader continued changing its rotary query phase with position. Merely listing the clock among the inputs had not supplied a computation that used it.

The [ISA register programs](../../isa/docs/executable-symbol-scan-lowering.md) suggest a constructive target: identify the input interface, advance an explicit state, and emit an executable program. The [behavioral-authority analysis](behavioral-authority.md) keeps the inference boundary visible. Here we have stronger source information than a few input/output examples: [compiled token-local Q/K operands](raw-prompt-encoder-atoms.md) for the first encoder layer. The question is whether their demanded ordering can be compiled into another representation.

Our target is the address port of [our approximate top-four first encoder reader](uncentered-reader.md). It is not a semantic top-k instruction exposed by native FLUX. A complete attention computation would additionally require scores, soft weights, values and an output write. Later encoder layers depend on contextual state rather than token-local operands alone.

## Fixing the numerical function

Let the query token be padding ID 151643, the head be zero, the allowed key-token alphabet be \(V\), and the sequence length be \(L=512\). We retain both absolute query position \(i\) and key position \(j<i\). BF16 rotary products and additions round at those positions; assuming that relative phase alone reproduces them would discard part of the actual numerical interface.

The old reader used a NumPy matrix/vector reduction whose finite result can depend on bank shape and layout. A universal subset-order argument needs a score that does not depend on which other keys are present. The [frozen feasibility plan](plans/source-ordering-program-feasibility.md) therefore declares a new, pinned numerical contract:

\[
u_{i,\ell}=\operatorname{FP64}(q^{\mathrm{BF16}}_{i,\ell})/\alpha,\qquad
v_{b,j,\ell}=\operatorname{FP64}(k^{\mathrm{BF16}}_{b,j,\ell})/\alpha,
\quad \alpha=128^{1/4},
\]

\[
\sigma_i(b,j)=\operatorname{math.fsum}_{\ell=1}^{128}
\left[\operatorname{FP64mul}(u_{i,\ell},v_{b,j,\ell})\right].
\]

The implementation pins the scale value, Python/platform profile, BF16 source words, elementwise FP64 multiplication and row-wise `math.fsum`. Batching prepares products without changing that reduction. No BLAS dot product or fused multiply/add substitutes for this function. Each candidate has the same score irrespective of its history bank.

This changes the numerical contract. The all-history construction below concerns this declared function. Agreement with the old reader on the two observed prompts is separate finite evidence; it is not a theorem about universal old NumPy parity. Per-history diagnostic references recalculate separate banks and sort them independently of the table, but share the compiler’s score helper. They are not an independent numerical implementation.

## A total order can answer every legal subset

For each query clock form the finite candidate universe

\[
\mathcal U_i=\{(b,j):b\in V,\ 0\le j<i\}.
\]

Sort it by decreasing score, then increasing key position, then token ID. The third tie coordinate only orders mutually exclusive token alternatives at the same position. Let \(\rho_i(b,j)\) be the resulting unique integer rank, with zero best.

A live history \(H_i\) contains only the token actually admitted at each active position. Restricting one total order to a subset preserves all comparisons. Consequently,

\[
\operatorname{top4}_{(b,j)\in H_i}\sigma_i(b,j)
=\operatorname{first4}\!\left(\operatorname{sort}_{\rho_i}H_i\right).
\]

This is the useful factorization: learned lexical/rotary geometry determines a fixed ordering offline; a history chooses a subset online. The table does not contain whole prompts, complete prefixes or contextual hidden states. Within this finite contract, it can answer histories that were never present during compilation.

The initial state is empty. Active tokens append their ID and position. At an inactive padding query the program scans its own history, looks up ranks and maintains four best entries through integer comparisons. An unknown active ID makes later queries unsupported; silently deleting that key would change the specified function. Active queries, other query-token types and clocks outside the domain have explicit statuses. The tiny mechanics tests also cover empty history and masked unknown tokens, although none of the production histories has a supported empty prefix.

The [source](../generator/source_order_program.py), [compiler](../generator/source_order_compile.py), [driver](../generator/source_order_study.py) and [implementation notes](plans/source-ordering-implementation.md) preserve this interface. Fresh standard-library serving excludes Q/K arrays, scores, old labels and numerical/model imports. Its file/import guards audit trusted code; they are not adversarial OS isolation.

## What ran

Compilation uses the 30 active token IDs from the two historical discovery prompts. It loads 31 compiled lexical rows, including padding, and qualifies the consumed rotary Q/K words against both historical lexical records: 131,072 BF16 words compared, zero mismatches. Fresh full-byte authentication covers 22 retained source files, 174,515,131 bytes. It preserves their original historical provenance.

Serving seals before per-history direct-score references or old support/weight interpretation. The assay includes the two historical prompts, four historical transfer prompts, and seven fixed synthetic histories with long, short, sparse and edited admission patterns. All rows and controls remain in the [publication](results/source-order-program/publication.json).

| Interface and denominator | Primary result |
|---|---:|
| Historical A/B head-0 padding queries | 954 / 954 ordered supports match the old reader |
| Historical A→B edit | 426 / 477 supports change, matching the old reader |
| Distinct ordered padding supports | A: 319; B: 280 |
| Synthetic nontrivial padding queries | 2,311 / 2,311 match the declared comparator |
| Synthetic short-prefix queries | 523 / 523 match, reported separately |
| Total supported primary queries | 3,788; zero declared-order mismatches |
| Four transfer histories | 1,900 padding queries unsupported; ten unknown active IDs per history |
| Fixed synthetic key substitution | 0 / 477 reference supports change |

The historical edit is a positive context-response observation. The synthetic edit is weak and remains so; we did not search for a replacement contrast. These are dependent query rows from a small fixed case set, not thousands of independent semantic replications.

Clock+1 reads the next absolute-clock table while keeping the actual history. On A/B it matches only 107 of 952 applicable ordered supports, despite 88.01% coverage of the old selected weight and 84.40% set overlap. The final query has no next clock and is inapplicable. Recent-position and reversed-rank controls have zero ordered matches and zero selected-weight coverage on these 954 head-0 rows. Those head-specific results differ from the previous experiment’s 32-head averages.

Old-weight coverage sums weights normalized within the old selected four. It measures neither full-attention mass nor image quality. Correct addresses do not supply the missing weights. The [complete panels](results/source-order-program/panels.json) retain every method, scope and short-prefix denominator.

## Compute is relocated and counted

The table has a triangular layout:

\[
N=|V|\sum_{i=0}^{L-1}i
=30\cdot\frac{512\cdot511}{2}=3{,}924{,}480.
\]

At most 15,330 candidates occur at one clock, so unsigned 16-bit ordinals suffice. Serialization uses clock, sorted token ID, then key position, in little-endian form. Deterministic gzip reduces 7,848,960 raw bytes to 7,139,604 bytes: it retains 90.96% of the raw size. This census supplies an executable representation; it has not discovered a compact semantic law.

| Work | Actual measurement |
|---|---|
| Offline score preparation | 502,333,440 FP64 coordinate products; 3,924,480 `fsum` calls |
| Score, sort and permutation compilation | 12.324 s |
| Serialization | 0.171 s |
| Post-seal direct references | 14,310,400 additional products; 111,800 `fsum` calls |
| Fresh program load, decompression and hashing | 24.505 ms |
| A / B primary serving, excluding startup and output IO | 5.700 / 5.620 ms |
| A / B integer work | 16,695 rank lookups each; 57,838 / 57,398 comparisons |
| All 13 packets × four methods, child wall time | 254.789 ms |
| Entire guarded CPU job | 14.48 s; reported peak RSS 208.1 MB; zero VRAM |
| Original model / encoder / denoiser / VAE evaluations; new images | Zero |

Job `job-c22fba1e07cd` uses intent-first mrun admission, a 4,096 MB/four-thread scheduler reservation, one numerical-library thread and a 600-second cap. Reservation is not measured consumption. The child reports 134,576 KiB high-water RSS, which can include pre-exec process history inherited from the numerical parent; it does not establish isolated steady integer-serving memory. The decompressed rank payload alone is 7.85 MB. A/B’s 560 logical history bytes exclude Python objects, indices, outputs, table, temporary storage and allocator overhead.

Coordinate-product counters omit rotary/scaling preparation, sorting, comparisons, hashes, compression and Python work. These measurements do not establish a matched runtime speedup. Offline learned arithmetic and prior lexical compilation remain paid, even though fresh serving evaluates no learned score or vector operation.

## The next state-machine question

Blindly extending the same table is unattractive. For this representation at 151,936 key IDs, the largest clock would need 77,639,296 ranks and hence 32-bit entries. One padding head would occupy 79,502,639,104 raw bytes; 32 such heads, 2,544,084,451,328 bytes. A naive additional factor for every query ID reaches 386,538,015,196,971,008 bytes. These are costs of this chosen representation, not lower bounds or an impossibility result. Shared structure could change them substantially.

Astra’s next question, subject to the scheduled full research/storage review 96, is whether the existing order has an economical exact clock transition. Let \(\pi_i\) be the list sorted by \(\rho_i\). A fixed copy/insert grammar could describe \(\pi_{i+1}\) as maximal contiguous runs copied from \(\pi_i\), interleaved with the 30 new candidate identities \((b,i)\). Such a program would advance its own ordering state from empty and serve its own token history.

That is a proposed constructive discrete recurrence test on the same domain. It would reuse saved ordinals, charge compiler sorting, program bytes, current-order/inverse-rank memory and update work, and verify reconstruction of every ordinal. It could be larger or slower; that outcome would remain useful evidence. No vocabulary/head expansion, experiment or image continuation is queued by this result.

The missing full computation still includes arbitrary-token query/key comparisons, soft weights, value formation and mixing, contextual MLP writes, later encoder layers, denoising and pixel decoding. The [own-state continuation work](own-state-successor.md) and [gate-first trajectory](gate-first-trajectory.md) measure parts of that larger path while retaining learned arithmetic. A reusable ordering component is progress toward understanding it; it does not close it.

## Evidence and review

The [evidence index](results/source-order-program-evidence/README.md) links exact source/request/payload custody, eight targeted synthetic tests, every prediction, every status and all 6,656 head/query / 26,624 method records. The [independent public audit](results/source-order-program/scalar-audit.json) validates all 512 ordinal permutations and replays every output with standard-library integer sorting, without evaluating Q/K scores.

The first publisher incorrectly treated output-file hash reads as forbidden inputs. The preserved [v1 source](../generator/publish_source_order.py) refused publication before copying. A separate [v2](../generator/publish_source_order_v2.py) admits the exact 52 declared prediction outputs for hashing, consistent with the executed guard; scientific sources, raw outputs and the assay remain unchanged. Generic empty-prefix overlap aggregation is an acknowledged instrument limitation, inapplicable to these production histories.

Astra’s source review overlapped execution rather than acting as a pre-admission gate. Separate actual-result review confirms both the useful historical response and the unsupported/weak cases. Full [research review 93](results/research-review-cycle93/README.md) and [storage audit 93](results/storage-audit-cycle93/README.md) remain current, next due 96. This cycle publishes compact integers and scalar evidence; no neural arrays or model weights move to the Mac, and no archive/deletion is claimed.
