# Certified retrieval preserves the answer but loses on total work

Status: **measured retrospective cost assay; no runtime improvement**. The frozen search avoids10.5–10.7%of current-query score products and reproduces the old ordered top-four supports, raw reads and BF16 banks exactly. Its selection takes about9.4seconds per prompt versus.087seconds for a full scan. This implementation's bounds, partial-dot calls and sorting overwhelm the score saving. We retain the result and return to the tail/state-accuracy gap rather than tune this search further.

The [protocol](plans/certified-current-query-retrieval.md) follows the [useful current-query selector](full-prefix-selector-objective.md). That selector reduces paired first-block state error95.96%→41.76%, but scans the full current prefix. This assay tests whether its ranking can be executed with less work. It does not change that remaining prediction error or produce a new image.

## What the certificate says

For each of8K/V groups, source covariance diagonal entries define one decreasing-variance coordinate order, with coordinate index breaking ties. Each causal index stores admitted actual centered keys, values, positions and outward block-tail norms. There are8blocks of16coordinates,32query heads and512query rows. The saved prompts each admit27keys; masked future occurrences never enter the index.

Let d=128, ε=2⁻⁵³, η=γ_(4d), and A_j≥‖x‖‖u_j‖. A partial score p_j plus an outward tail norm product B_j gives an upper bound on the old full-dot result:

\[
U_j=\operatorname{nextUp}(p_j+B_j+2\eta A_j).
\]

A completed new dot f_j encloses the old result in

\[
[\operatorname{nextDown}(f_j-2\eta A_j),
 \operatorname{nextUp}(f_j+2\eta A_j)].
\]

Initial incumbents are the first four causal keys, evaluated by complete row dots. Later candidates accumulate source-ordered block dots. Reaching the last block completes f_j; no second full dot is hidden there. This pre-execution clarification uses the same error allowance for the completed progressive reduction and old GEMV. Every completed interval remains available, so the fourth-largest lower endpoint cannot decrease. Prune only below that lower endpoint. Final interval separation must establish membership and order; otherwise execute the old whole-prefix GEMV and preserve its earliest-occurrence tie rule.

Positive reverse sums of outward-rounded squares produce tail norms; there is no unstable total-minus-prefix subtraction. Roots, products and interval sums are rounded outward. Nonzero input magnitudes must be in[2⁻⁴⁰⁰,2⁴⁰⁰], with ordinary binary64 arithmetic and the stated operation-error bounds. Unsupported ranges or numerical ambiguity use the full call. The observed Linux x86_64 profile reports round-to-nearest, Python3.14.6, NumPy2.5.4 and one arithmetic thread. This is a conditional numerical certificate, not independent formal verification of BLAS or CUDA kernels.

## The measured cost

| Measurement, per prompt | Crane behind | Crane before |
|---|---:|---:|
| Full-scan score products |55,185,408|55,185,408|
| Search score products, including full calls |49,255,456|49,400,048|
| Fraction of full-scan products |89.25%|89.52%|
| Search selection wall time |9.416s|9.360s|
| Full-scan selection wall time |.0861s|.0883s|
| Search query-norm square products |2,080,768|2,080,768|
| Shared key-norm square products |27,648|27,648|
| Tail norm-product bound calls |2,555,714|2,564,751|
| Upper-bound additions |5,111,428|5,129,502|
| Upper-bound next-up calls |5,111,428|5,129,502|
| Python sort comparator calls |7,888,027|7,846,162|

There are also430,816full-norm product bounds and430,816radius-product calls per prompt, query-norm additions/roots/outward operations, qualification checks, key admission and reader work. The counts are declared logical categories, not a complete hardware FLOP or energy census. Scalar constants, Python overhead, array gathering and library internals remain included in measured time. The score-product count includes every initial, progressive and fallback full-dot coordinate product.

The raw Python sorting counters cover interval maintenance. They do not instrument NumPy's internal lexsort in `select`: the full baseline executes16,384such sorts per prompt, and the search executes128through its short-prefix full calls. A zero raw `sortCalls` field in full mode means this counter is inapplicable there, not that sorting is free. These additional call counts are derived from the bound source/control flow and nonempty-query counts; internal NumPy comparisons are unmeasured. The [accounting note](results/certified-retrieval-evidence/accounting.json) preserves this interpretation without altering raw reports.

Every nonshort query is certified:16,256per prompt. The first four clocks produce128short-prefix full calls. No query takes an unsupported-range or ambiguous-ranking fallback in these specimens. Nonetheless652/700certified queries save no score products. The244,368/244,138`prunedKeys`counts include keys rejected only after all128coordinates; they cannot substitute for actual work saved.

Across both prompts, the complete search child takes21.018seconds including startup,20.935seconds after imports and20.180CPU seconds. The full-scan child takes1.365seconds including startup,1.308after imports and1.244CPU seconds. Selection is roughly108times slower in this implementation; the whole child is roughly16times slower. This is one ordered run, with full scan after search, so cache/order and implementation effects remain. It is not a controlled native-attention benchmark or a universal algorithmic lower bound. The unchanged anchor/centroid read costs.367/.369seconds under search and.317/.319seconds under full scan, and still reads the complete causal prefix.

## Storage and continuation are still costs

The used search index is459,864bytes per prompt: actual FP64 keys/values, positions, safety flags and norm bounds. The comparable full-scan FP64 index is444,312bytes; bounds add15,552bytes. Preallocated512-row capacity is8,720,384bytes versus8,425,472bytes. Source coordinate orders add8,192bytes, with4,202,496bytes of loaded source geometry including covariance used to derive those orders.

For comparison, the same27active keys/values need110,592bytes as BF16 GQA payload; all512rows would need2,097,152bytes. The index is not smaller than that cache. Input ports, routes, temporary arrays, forecasts and counters are additional research/runtime artifacts. Final index snapshots are retained; a resume protocol or complete contextual own-state encoder is not claimed.

## Exactness and inherited receiving evidence

The certified child seals first. An independent full-scan child then executes the same read. Only afterward are the old current-query forecasts and receiving-consumer records read. Both prompts have zero differences in ordered support, route, raw FP64 bits and BF16 words across certified/full/parent comparisons. The same source-bound operand ports are unchanged.

Consequently the old original-block consumer evidence is reused, not recomputed. Its state-change error remains41.76%overall,8.33%active and43.61%padding. It remains original learned output/RMS/MLP computation performed historically. This job evaluates no new original consumer or dense parameter projection and reads no original checkpoint. Matching its bank does not eliminate the historical assistance or solve its remaining error.

Guarded job`job-d1b6e831e7fb` succeeds on its first attempt in27.89scheduler seconds,25.319worker seconds,321.5MBpeak RSS and zero VRAM. CPU math uses one thread under default4CPU/4GBadmission, strict preflight,600second timeout and no automatic retry. All79mechanics tests pass. The independent retained-byte check covers56files/364,951,419bytes, including overlapping inherited banks/consumers. Full arrays remain near compute; the compact public bundle is1,089,022bytes in10files with65,536byte-lossless observation records. The [Saturn sweep](results/certified-retrieval-evidence/sweep.json) covers observation mechanics, with no terminal image certification.

Request SHA256:`8d0fc6b1adb030b463390165107e5ec59a548ec24ad0a16cf7a55341cd8b28f0`. Certified seal:`69d838412731e29bb427f1f3ebc92202bd17e6cf15dcdf6ba375d040188b3edf`. Full-scan seal:`1b3903358647fbb4834bcfe52232d3f8d13d3b91701d357d5320746d9ba9f716`.

## The next scientific cut

Astra's code review found no material certificate blocker. The measured result supplies a valid reduction in score products but no runtime improvement. We will not grow the index family to rescue this frozen assay. The [next plan](plans/current-support-tail-factorial.md) keeps useful current-query support fixed and crosses exact/centroid tail mass with exact/anchor-weighted tail value. It asks which missing tail information matters, including cancellation through the receiving block, before adding state or increasing support.

That [factorial is now measured](current-support-tail-factorial.md): both factors matter and partly cancel; value repair has the broader paired-head trend. The subsequent [positive directional closure](plans/positive-tail-moment-closure.md) remains proposed.

The updated local [five-value microscope post](/Users/jakeholl/domains/obsidian/blog/2026-11-06-054800-five-attention-values-become-a-different-hidden-state.md) now completes the six-port signed bridge behind finite-kernel prefix drift. One changed row disappears at projection; the MLP changes features within four surviving token rows. This supports consumer-specific propagation and cancellation, not a universal amplification rule or cross-token mixing by the MLP. Its original learned reference remains distinct from reduced execution.

Read the [report](results/certified-retrieval/report.json), [provenance](results/certified-retrieval/provenance.json), [retained-byte manifest](results/certified-retrieval/retained-array-verification.json), [retrieval code](../generator/certified_retrieval.py), [study](../generator/certified_retrieval_study.py), [publisher](../generator/publish_certified_retrieval.py) and [tests](../generator/test_certified_retrieval.py). The full raw-prompt/context/own-state/pixel requirements remain open; this result does not provide arbitrary-prompt zero-neural images.
