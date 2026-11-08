# A concrete weight producer and its scaling cost

The next bounded construction is to compile a scalar score function of token identity and absolute positions, then normalize its selected scores. This would remove online Q/K vector products while retaining explicit scalar attention normalization. It is a partial evaluation of one learned primitive, not an arbitrary-prompt zero-neural generator. No score table or weight producer was executed this cycle.

The new [source-bound feasibility report](results/relative-mass-feasibility/report.json) supplies an exact size census, an exact small rounding counterexample and a concrete finite contract for the next implementation. Three tiny mechanics tests pass. This resolves the [relative-mass design question](plans/first-read-relative-mass-producer.md) sufficiently to choose a bounded candidate; full research/storage review 99 comes before implementation.

## Magnitudes are another input

The [first-read experiment](first-read-route-payload.md) showed that knowing the selected addresses leaves their weights unresolved. For example, positive masses (8,4,2,1) and (64,16,4,1) have the same ordering, but normalization gives (8/15,4/15,2/15,1/15) and (64/85,16/85,4/85,1/85). This supplied rational example proves that ordering alone does not identify weights. It says nothing about whether another cheap input-defined producer exists.

For selected scores \(s_0,\ldots,s_3\), three relative gaps suffice mathematically:

\[
g_i=s_i-s_0,\qquad p_i=\frac{\exp(g_i)}{\sum_k\exp(g_k)}.
\]

The missing computation is how to form the gaps or equivalent masses from the input. Supplying the edited-address weight or old conditional mixture would retain the missing input rather than produce it.

## Why ideal RoPE is not an exact finite shortcut

In real arithmetic, rotations give \((R_iq)^T(R_jk)=q^TR_{j-i}k\). A 128-coordinate score can therefore be written as a 64-frequency cosine/sine sum in the relative offset. Directly evaluating those learned coefficients still pays 128 coefficient products per score. That is a count for the direct rewrite, not a lower bound on all possible programs. A phase recurrence can reuse positional work without automatically eliminating those coefficient computations.

The [actual lexical frontend](../generator/encoder_atoms.py) separately rounds each rotary product and the subsequent sum to BF16. It does not implement the ideal rotation group exactly. A supplied two-dimensional example makes the consequence explicit. Use \(q_0=k_0=(1,0)\), angle step \(\pi/8\), and the rounded operands

\[
c=237/256,\qquad s=98/256,\qquad a=181/256=\operatorname{BF16}(\cos(\pi/4)).
\]

The unscaled score at query/key positions (1,0) is \(c=60672/65536\). Translate both positions to (2,1), preserving their offset, and the score is \(a(c+s)=60635/65536\). The exact difference is \(-37/65536\).

This is a tiny supplied rational witness, not native Qwen evidence or a measured model error. It excludes assuming universal exact relative-offset reuse under this rounded policy. It does not exclude useful approximations, another compact evaluator, or a representation with explicit finite corrections. The prospective compiler will reuse the source's represented absolute-position operands.

## Compile a function, then select its arguments

For fixed padding query ID 151643 and head 0, define

\[
S(i,u,j)=\operatorname{fsum}_r\!\left[\operatorname{FP64}(q_{i,r}k_{u,j,r})\right],\qquad j<i.
\]

Here the query/key coordinates are the qualified rotated BF16 words widened and scaled under [cycle 95's pinned contract](source-ordering-program.md). The parentheses denote individual FP64 products followed by the declared reduction, not ideal real products. The scalar function depends on key token \(u\), key position \(j\) and query clock \(i\); it does not depend on the other keys in the history.

Compile those values offline for the same 30 IDs and 512 clocks. Fresh serving reuses its own integer rank program to select addresses, retrieves their scalar scores, and computes all selected weights. It receives no historical weights, supplied mass, Q/K vectors or complete-prefix cache. This is not a whole-prompt lookup: a new legal history chooses a new subset of independently compiled entries.

Cycle 95 retained each clock's score hash but discarded its score values. Those hashes can bind a new table to the same scalar computation; ranks and hashes do not provide the magnitudes themselves. Compilation would therefore pay another 502,333,440 coordinate products and 3,924,480 reductions, plus operand formation and serialization. No such compilation occurred in this feasibility study.

## The literal cache scales poorly

The strict-past position count is \(L(L-1)/2=130816\). For \(A\) key IDs, one fixed query token and one head, the literal FP64 table has

\[
N=A L(L-1)/2,\qquad B=8N.
\]

| Literal strict-past domain | Raw FP64 bytes |
|---|---:|
| Current 30 IDs, one pad query/head |31,395,840|
| Full 151,936 key IDs, one pad query/head |159,005,278,208|
| Full key vocabulary, pad query, 32 heads |5,088,168,902,656|
| All query/key IDs, 32 heads |773,076,030,393,942,016|

These are uncompressed layout counts, not measured compression, minimum description lengths or impossibility bounds. The last row is about 773 PB in decimal units. Active queries also admit their current key: legal same-token diagonals add 77,791,232 scores per head, bringing that last literal layout to 773,076,050,308,497,408 bytes. GQA shares K/V groups but does not remove the distinct query-head score functions.

The current score table plus existing raw ranks is 39,244,800 bytes before metadata, indexing, values and Python/runtime overhead. An ideal relative-offset table would need only 122,640 score bytes in this domain, a 256-fold reduction, but the rounding witness prevents assuming its exact equivalence. No compression ratio or runtime speedup has been measured.

## A declared finite normalizer

The chosen prospective serving rule is the new pure-standard-library contract

\[
m=\max_i s_i,\quad e_i=\operatorname{math.exp}(s_i-m),\quad Z=\operatorname{math.fsum}(e_i),\quad w_i=e_i/Z.
\]

Preserve selected-address order and pin the runtime/platform profile. Four keys require four score lookups, four subtractions, four exponentials, one four-term sum, four divisions and maximum selection. Ranking/history maintenance, loading and authentication are additional work. Empty prefixes emit empty weights with zero padded slots; short prefixes normalize their actual selected keys and pad unused slots with zeros. No second renormalization follows.

This changes the old NumPy log-sum-exp operation order. Historical weight differences must be measured separately; matching ranks never established universal old-reader finite parity. A weighted read would additionally need values and up to 512 scalar value products per query. Until supplied and charged, the candidate is a weight producer.

The [next implementation plan](plans/compiled-selected-mass.md) fixes these boundaries. It permits one same-domain producer transaction after full review 99, with no precision search or vocabulary/head expansion. Later encoder layers replace token-local inputs with actual contextual states; this table does not supply that state transition, an MLP update, or pixels. The original arbitrary-prompt zero-neural goal remains active.

The [helper](../generator/relative_mass_feasibility.py), [three-test source](../generator/test_relative_mass_feasibility.py), [mechanics receipt](results/relative-mass-feasibility/mechanics.json), [execution receipt](results/relative-mass-feasibility/execution.json) and [preexecution Astra review](results/relative-mass-feasibility/advisor-source.json) preserve the evidence. The report takes 0.001849 seconds before writing, with no model, scientific-array read, learned score product or image. That timing describes tiny bookkeeping and rational arithmetic, not a measured compiler or generator. Full research/storage 96 remains current, next 99; no large artifacts or cleanup were added this cycle.
