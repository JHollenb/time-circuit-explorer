# Query transport through live context, then its receiving MLP

The first-order reader improves all four tested raw-prompt changes over the same source-selected nearest reader. Its advantage also survives the original first-block output, normalization, MLP and state update at predeclared probes. This is evidence for retaining query–memory interaction in the reduced representation. It remains one predicted attention head surrounded by 31 exact reference heads in the consumer experiment, not a complete encoder or arbitrary-prompt zero-neural image generator.

The [cycle9 reader](source-landmark-context.md) established a reusable source-only route but lost contextual changes through its fixed convex response mixture. The [transport protocol](plans/query-transport-moments.md) was frozen before these results. No new landmark fitting or target-selected correction was needed.

## Factor the exponential before expanding

With x=q/d^(1/4), y=k/d^(1/4), d=128 and a source-selected landmark ℓ, the actual causal prefix defines

\[
p_\ell(j)=\frac{m_j e^{\ell^Ty_j}}{\sum_km_k e^{\ell^Ty_k}},\qquad
h(x)=\frac{E_\ell[v e^{(x-\ell)^Ty}]}{E_\ell[e^{(x-\ell)^Ty}]}.
\]

Differentiating the ratio at ℓ gives

\[
\widehat h(x)=h_\ell+J_\ell(x-\ell),\qquad
h_\ell=E_\ell[v],\quad J_\ell=\operatorname{Cov}_\ell(v,y).
\]

The current query supplies its displacement. Its actual admitted keys and values supply the covariance. This is a source-defined law on fresh strings; it is not a fit to the desired head. The missing interaction has a concrete form: how value and key vary together in the memory this query reads.

The [implementation](../generator/context_transport.py) advances stable weighted centered cross moments. A fixed source key center changes all landmark scores by the same prefix-independent constant and therefore preserves the normalized read and covariance. For running-maximum rescale ρ, old rescaled mass z, new mass α, old value/key means h/μ and old cross-sum C,

\[
C'=\rho C+\frac{z\alpha}{z+\alpha}(v-h)(y-\mu)^T.
\]

Value sums, centered key sums and the partition use the same scale. Masked keys do not enter; repeated occurrences do. A read contracts only the selected landmark's cross-sum with the actual displacement. Resume checks bind source identity and sequential clock. There is no clipping or exact fallback.

The equivalent first-order weights p_ℓ(j)[1+(x−ℓ)ᵀ(y_j−μ_ℓ)] sum to one but can be negative. This loosens the previous fixed positive response-mixture constraint. It also admits extrapolation error; positivity is a diagnostic, not an accuracy guarantee.

## Fresh raw strings and paired effects

Eight frozen strings test compass inside/outside, balloon/suitcase color binding, Japanese tower above/below and long reversed word order. All 512 query positions remain scored, including padding. English short pairs have 26 active tokens each; long strings fill 512. The Japanese pair has **33 versus 34 active tokens**, so it changes tokenization and the mask as well as text. The changed-mask row is retained separately; this pair is not a clean equal-length semantic isolation.

The complete head forecasts seal before canonical full-prefix references, exact mathematical heads, signed-weight diagnostics and scoring. All canonical port comparisons match. The seal is [`b47afaea…293e`](results/query-transport/prediction-seal.json).

| Prompt change | Nearest head error | Transport head error | Transport cosine |
|---|---:|---:|---:|
| Compass inside/outside | 73.27% | 44.07% | .902 |
| Balloon/suitcase binding | 132.37% | 81.91% | .752 |
| Japanese tower above/below | 101.74% | 57.65% | .888 |
| Long order reversal | 133.14% | 71.22% | .781 |

Errors compare the predicted paired head difference with the exact paired difference over all queries. Target change RMS values are .001060, .001055, .004041 and .005909 respectively. These are dependent component observations, not image quality scores. Transport also improves pooled active-both and padding-both regions in these comparisons; that does not assert every row improves. Full endpoint head errors fall from 27–38% to 14–19% across the eight strings.

Selected post-seal direct checks close the degree-zero read to 9.03e−17, first-order read to 1.39e−16 and exact exponential change of measure to 9.72e−17. These validate the moment computation under shared arithmetic, not the Taylor approximation's accuracy. All actual reads are finite. Sampled negative effective mass reaches .21382 and minimum weight −.11144. Maximum value-hull excursion reaches .04030. Signed behavior remains visible and uncorrected.

## Does the next computation retain the improvement?

The [consumer](../generator/transport_consumer.py) evaluates the actual original equations:

\[
o=\operatorname{BF16}(W_o[H_{other},\widehat h_0]),\quad
r=\operatorname{BF16}(E+o),\quad u=\operatorname{RMSNorm}(r),
\]
\[
M(u)=W_{down}[\operatorname{SiLU}(W_{gate}u)\odot W_{up}u],\qquad
s=\operatorname{BF16}(r+M(u)).
\]

Every arm recomputes this path from its own head0 prediction, with explicit BF16 boundaries. The other **31 attention heads are exact mathematical reference context**, separately evaluated for each prompt. Their presence is useful for isolating the contribution of head0 but prevents a whole-encoder claim.

The primary score subtracts the head0-deleted arm before measuring the change: compare Δ(F_pred−F_zero) with Δ(F_exact−F_zero). Normalizing by the whole state would let supplied embeddings and other heads hide a small but important error. The final-state head0-effect target RMS values are .001507, .001154, .013599 and .012385.

| Prompt change | Nearest final-state effect error | Transport final-state effect error | Probe rows per prompt |
|---|---:|---:|---:|
| Compass inside/outside | 57.66% | 42.69% | 0,25,26,511 |
| Balloon/suitcase binding | 93.23% | 52.02% | 0,25,26,511 |
| Japanese tower above/below | 56.20% | 25.66% | 0,32,33,34,511 |
| Long order reversal | 37.30% | 16.28% | 0,511 |

Output projection, residual and RMS are measured on all 512 rows. MLP and final state are measured **only at these source-clock probes**, declared before results; no error-selected probes are added. All four paired effects improve at each measured stage. Final-state transport cosines are .908, .860, .979 and .987. This supports the value of the interaction at the measured consumer; it is not evidence about unmeasured layers or pixels.

Contractions use FP64 on BF16 operands, with explicit BF16 boundaries, FP32 RMS variance and FP32 SiLU. The reference uses the same equations/arithmetic. Native CUDA, Flash Attention and framework numerical parity are unmeasured. The consumer executes original learned projections and MLPs despite making zero framework calls.

## Cost and custody

Core state is **8,520,720 bytes per head**, versus 66,576 for the previous reader: 8,388,608 bytes of cross moments and 65,536 bytes of key sums account for the increase. Runtime transport assets add 197,632 bytes, range diagnostics 2,048, plus outputs, checkpoints, workspaces and the external 2.645 GB learned lexical store.

Warm short-string calls take **40.2–47.3 ms**; long calls take **391–400 ms**. These jointly include raw-prefix gathering, nearest and transported reads, all 64 moment updates, diagnostics and fixed checkpoint copies. They are not isolated read latency or a speedup comparison. A 512-active-key run counts 1,077,936,128 mixed-moment scalar products, 536,870,912 additions and at least 17,179,869,184 bytes of cross-moment update traffic. Finite-state scans add at least 4,362,600,448 bytes. The richer state is substantially more expensive.

| Job | Lease wall | Worker wall | Peak sampled process-tree RSS | VRAM |
|---|---:|---:|---:|---:|
| Fresh transport `job-1149830281fd` | 13.56 s | 10.73 s | 550.9 MB | 0 |
| Original consumer `job-d8421ee1f182` | 22.68 s | 20.67 s | 1,280.2 MB | 0 |

Both are finite guarded CPU mrun jobs, one configured math thread, strict preflight, no retry, near the immutable source store. Serving itself reports 7.456 s after imports and 288,828 KiB high-water RSS for the complete cohort and repeats. The consumer loads 170,398,720 BF16 parameter bytes, widened to 681,579,520 resident matrix bytes; it explicitly pays for 256 reference head evaluations, 8 shared other-head projections, 32 head0 slice projections and 96 MLP projections. Those are diagnostic costs, not reduced-serving or image latency.

The [main provenance](results/query-transport/provenance.json) and [consumer provenance](results/query-transport-consumer/provenance.json) retain declared/sealed/executed payload identities. Independent retained-byte checks cover 33 files/276,595,850 bytes and 40 files/5,628,721,287 bytes respectively. The latter includes the original 4,967,215,360-byte checkpoint shard and overlapping parents; it is not 5.6 GB of new predictions. Full arrays remain near compute. Compact publications contain 1,222,520 and 740,278 bytes, with all observation streams preserved as byte-lossless gzip.

All **52 generator mechanics tests pass**. The [Saturn run and verify record](results/query-transport-evidence/sweep.json) returns 0 for both jobs, with 12,288 and 24,696 observation records, no watch fires, and `target-streams-only` scope. No terminal-claim receipts or physical address joins are harvested. Historical receipts remain immutable.

Astra's holistic review recommends the [complete attention-bank test](plans/complete-attention-bank.md) next: remove the supplied other-head dependency, then consume all predicted errors together. The [raw-prompt reconstruction post](/Users/jakeholl/domains/obsidian/blog/2026-11-06-032000-predicting-conditioning-from-the-raw-prompt.md) strengthens the original-equation reference, but its exact contextual recurrence still performs learned computation. The [living research record](zero-neural-research.md) keeps full conditioning, later state continuation, general decoding and arbitrary-prompt image quality open.
