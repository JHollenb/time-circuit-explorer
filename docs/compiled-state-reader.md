# A compiled writer feeds the next reader from its actual state

The [compiled MLP write](compiled-mlp-write.md) now has a measured downstream edge. A fresh process consumed its **actual BF16 state bytes**, recovered coordinates from those bytes, and computed the next joint block's image Q/K/V affines. The receiver received no intervention dose, native replacement state or reference output. At the midpoint, paired Q/K/V response errors were **18.73%, 18.61% and 13.90%**. These pooled figures hide substantial row heterogeneity.

This qualifies a bounded piece of compiled execution. It stops before Q/K head normalization, RoPE and attention. The producer family, 32 image rows and modulation context are supplied. Arbitrary-prompt interpretation, connected denoising and general pixel decoding remain open.

The computation uses original-weight-derived reduced banks. “No original dense projections online” and “no neural framework calls” describe this implementation precisely. A reduced contraction is still learned-weight-derived arithmetic, and can itself be represented as a small linear network. Renaming such arithmetic does not establish a representation-independent absence of neural computation. The raw report's `onlineNeuralEvaluations: 0` is an operational counter for native/neural module evaluation, subject to this qualification.

## The actual edge and its custody

The parent Saturn StateCut identifies diffusion step 2, before `joint_block.2`, immediately following the selected joint-1 writer. Its fingerprint is `2f1dbab51da6022302634c5c7d26253cd759fe334341ed8eaa8a23b244599b8a`. Its captured `img_mod` supplies the fixed modulation for this receiver. The original parameter recipes select only joint-2 image `attn.to_q`, `attn.to_k` and `attn.to_v`, each a bias-free 3072 × 3072 projection.

The row-local native preparation is

\[
n=\operatorname{BF16}(\operatorname{LayerNorm}(u)),\quad
d=\operatorname{BF16}(1+\mathrm{scale}),\quad
x=\operatorname{BF16}(\operatorname{BF16}(d\odot n)+b).
\]

The native projections then produce BF16 Q/K/V. This assay uses the inspected Diffusers expression and the original `torch.nn.LayerNorm` and `torch.nn.Linear` primitives on CPU. It does not instantiate the complete block. The source-file hash, parameters, dtype, shape and captured context are bound in the [request](results/compiled-state-reader/request.json) and [compiler specification](results/compiled-state-reader/compile.json).

Job `job-2f0a52357bf1` succeeded under one finite guarded mrun lease. Declared, sealed and executed payload hashes match. Collection verified all **31 files / 53,251,417 bytes** against remote inventories before and after transfer. [Provenance](results/compiled-state-reader/provenance.json) contains that complete inventory; the checked-in bundle is a clearly identified subset.

## Factor the computation around the emitted state

The producer has five smooth output coefficient arrays \(c_0,\ldots,c_4\), a fixed residual parent \(u_0\) and gate \(g\). Construct centered state directions

\[
h_0=\operatorname{center}(u_0+g\odot c_0),\qquad
h_j=\operatorname{center}(g\odot c_j)\quad(j>0).
\]

Normalize the five columns separately, take an SVD, and retain the resulting orthonormal basis \(B\) for each image row. Column normalization prevents small high-order coefficients from disappearing merely because their magnitudes differ. It does **not** establish equal causal importance. All 32 rows have numerical rank 5 in this supplied span. Original column norms, scaled singular values and numerical cutoffs remain in the [compiler receipt](results/compiled-state-reader/receiver-compile-receipt.json); these are not semantic-dimension estimates.

Serving gets an opaque packet ID plus \(u\)'s path, hash and shape. It computes

\[
v=u-\operatorname{mean}(u),\qquad
s=\sqrt{\operatorname{mean}(v^2)+10^{-6}},\qquad
z=B^\top v,\qquad r=v-Bz.
\]

The actual state determines its centering, scale and coordinates. There is no dose input and no replacement by an unrounded polynomial state. Exact scalar square-root normalization preserves the amplitude bound that a polynomial inverse-root approximation can lose.

For each original projection \(W\), compile

\[
K=WDB,\qquad o=Wb,
\]

where \(D=\operatorname{diag}(d)\). Serve

\[
\widehat q_{\rm raw}=Kz/s+o,\qquad
\widehat q=\operatorname{BF16}(\widehat q_{\rm raw}).
\]

The exact smooth identity is

\[
W(Dv/s+b)-\widehat q_{\rm raw}=WDr/s.
\]

Post-seal diagnostics measured this identity's maximum discrepancy between **2.3 × 10⁻¹⁴ and 3.1 × 10⁻¹⁴**. That verifies the algebra and implementation on these inputs. It does not make the omitted term zero or establish native BF16 parity.

The centered-state residual is about **0.166–0.168%** of state RMS. Its projected Q/K/V contribution has RMS **0.0043–0.0047**. The full smooth projection differs from the native CPU result by RMS **0.0068–0.0099**. Candidate output rounding contributes a displacement of RMS **0.0044–0.0084**. These vectors overlap; their magnitudes cannot be added as independent error percentages. BF16 LayerNorm, intermediate modulation rounding, projection accumulation and output rounding remain part of the numerical-policy difference.

## Separate writer error from reader error

Let \(U_P\) be the producer's saved emitted state. Let \(U_T\) be the archived native isolated MLP write put through the **same fixed-parent residual update**. The latter is a reconstructed native-write state, not a newly observed complete-block midpoint history.

Primary predictions sealed before the worker opened the archived native MLP bundle or ran any native receiver. Then six native receivers evaluated \(R(U_P)\) and \(R(U_T)\) at doses 0, .5 and 1, preserving shape [1,32,3072] and one CPU thread. No target-dependent basis fitting occurred.

For \(P=\widehat R(U_P)\), \(C=R(U_P)\), and \(T=R(U_T)\), retain

\[
E=P-T=\underbrace{P-C}_{A:\ \text{reader}}+
\underbrace{C-T}_{H:\ \text{inherited writer}},
\qquad
\langle E^2\rangle=\langle A^2\rangle+\langle H^2\rangle+2\langle AH\rangle.
\]

Paired changes use each arm's **own** dose-0 baseline. The table normalizes paired error RMS by the corresponding native Q/K/V change RMS.

| Dose | Field | Composite error | Reader error | Inherited error | Reader/inherited cosine |
|---|---|---:|---:|---:|---:|
| .5 | Q | 18.73% | 17.49% | 17.83% | −.438 |
| .5 | K | 18.61% | 17.40% | 17.64% | −.436 |
| .5 | V | 13.90% | 12.42% | 13.54% | −.429 |
| 1 | Q | 10.07% | 9.47% | 9.09% | −.412 |
| 1 | K | 10.04% | 9.46% | 8.98% | −.407 |
| 1 | V | 7.40% | 6.72% | 6.84% | −.406 |

The negative cross terms are material. A component norm is not a positive causal share of total error. All rounded-output decomposition identities close exactly in the retained arithmetic. Raw candidate forecasts, rounded outputs and both absolute and paired comparisons are preserved separately.

Row results are less favorable than the pooled numbers. Midpoint median row errors are **91.4% Q, 90.3% K and 82.6% V**. Only 21/32 Q and K rows and 28/32 V rows improve over a zero-response scoring control. Worst row errors exceed 100% for every field. Four rows—436, 600, 533 and 531—carry approximately **90.3–91.0%** of native midpoint response energy. The control uses the post-seal native baseline for scoring and is not a presealed serving candidate. Null denominators at dose 0 remain null.

[The report](results/compiled-state-reader/report.json) retains aggregate metrics and residual diagnostics. [The 1,152 per-row observations](results/compiled-state-reader/observations.jsonl) retain counter-trends and signed powers. These are dependent rows from one supplied family, not independent prompt replications.

## Attribute the discrepancy before changing the receiver

Astra recommended a saved-array follow-up rather than immediately advancing attention. Job `job-15dea027e760` completed that analysis without any new native evaluations, dense projections or weight reads. It kept the original run immutable and formed

\[
E=\underbrace{P-P_{\rm raw}}_{O:\ \text{output rounding}}
+\underbrace{P_{\rm raw}-S}_{G:\ \text{basis omission}}
+\underbrace{S-C}_{N:\ \text{numerical policy}}
+\underbrace{C-T}_{H:\ \text{inherited writer}},
\]

where \(S\) is the saved full smooth original projection. The native-policy term combines LayerNorm, modulation and projection differences; available arrays do not isolate those individual boundaries. Paired vectors subtract each term's own unchanged baseline. Every diagonal power and signed cross term remains in the [attribution report](results/state-reader-attribution/report.json) and [per-row observations](results/state-reader-attribution/observations.jsonl).

| Midpoint paired term / native change RMS | Q | K | V |
|---|---:|---:|---:|
| Output rounding \(O\) | 11.72% | 12.35% | 6.70% |
| Basis omission \(G\) before final rounding | 7.28% | 6.66% | 6.39% |
| Combined numerical policy \(N\) | 13.97% | 14.44% | 9.03% |
| Inherited writer \(H\) | 17.83% | 17.64% | 13.54% |

These are correlated vector norms, not additive shares. In particular, numerical-policy and inherited-writer errors have negative cosine near −.43 to −.46.

The more informative counterfactual removes basis omission **through the same final quantizer**. Form \(Q_{\rm full}=\operatorname{BF16}(S)\), then retain

\[
P-C=(P-Q_{\rm full})+(Q_{\rm full}-C).
\]

At the midpoint, the first term is **14.53%, 14.27%, 10.47%** of native change RMS for Q/K/V. The second is **14.92%, 15.18%, 10.28%**. Their cosines are −.28 to −.30. Removing smooth basis omission therefore changes which BF16 rounding cells the output enters; its effect cannot be inferred from the smaller unrounded omission norm. Both basis omission and numerical policy remain material. This full-projection counterfactual is an offline diagnostic, not a deployable reduced receiver.

The state residual tells the same cautionary story. Although \(r\) is only about .166% of the full centered state's RMS, \(r(.5)-r(0)\) is **10.18%** of the actual midpoint centered-state change. At full dose it is **5.57%**. The relevant consumer-scale denominator is a change, not the much larger shared baseline.

All telescoping identities close within 8.7 × 10⁻¹⁹ maximum absolute error, with power closure within 5.5 × 10⁻²⁰. The follow-up took **.212 s** after imports and used **111.7 MiB** process high-water RSS. The scheduler's sparse 13.5 MB sampled peak missed that short-lived process maximum; both measurements are retained. Its total finite lease was 1.19 s. The follow-up adds 1,152 dependent per-row records, half for the four-term composite and half for the rounded full-projection reader counterfactual.

## Compute and storage

| Stage | Measured work/cost |
|---|---|
| Offline compiler | 5.565 s wall, 4.570 s CPU; 260.0 MiB process high-water RSS |
| Original QKV parameters | 28,311,552 parameters / 56,623,104 BF16 payload bytes |
| Offline parameter integrity | Full 7,751,109,744-byte checkpoint hashed; three selected ranges validated |
| Basis | 3,932,160 FP64 payload bytes |
| Three reduced banks and offsets | 11,870,208 FP64 payload bytes |
| Warm serving arithmetic | 0.911–0.936 ms per 32-row packet, including centering, variance, coordinate recovery, residual diagnostics, three reduced contractions and BF16 output rounding |
| Fresh serving process | 0.300 s for imports, loads, three packets × (one warmup + 31 timed evaluations), seals and I/O; **not single-query cold latency** |
| Serving process high-water RSS | 70.8 MiB |
| Post-seal reference work | 6 native LayerNorm + 18 native Linear calls; 18 additional smooth/residual dense diagnostic matmuls |
| Native reference process | 13.683 s fresh process; 9.050 s after imports, including checkpoint verification, parameter preparation, references and diagnostics |
| Whole mrun lease | 23.94 s, sampled job peak RSS 1,195.6 MB; zero measured VRAM |

Compiler and native-reference processes each hashed the complete checkpoint once. All dense learned-weight math belongs to research preparation or post-seal diagnosis. Serving reads the compiled banks and state, with zero original checkpoint bytes and zero original dense projections. Three reduced bank contractions occur on every evaluation.

For these 32 rows, reduced-bank projection work is 1,474,560 multiply-accumulates, versus 905,969,664 for three original dense QKV projections. Coordinate recovery and the retained residual reconstruction add 983,040 more multiply-accumulates, plus scalar normalization and rounding. This arithmetic count is **not a controlled end-to-end speedup measurement**. Native reference timings are single evaluations with substantial first-call setup.

The 15,802,368-byte basis/bank payload covers 32 selected row paths. Extending that row-specific representation to 1,024 rows while sharing the three offsets once would take **503,390,208 bytes**, before additional contexts, compared with 56,623,104 bytes of shared BF16 QKV weights. (Thirty-two independent copies including repeated offsets would take505,675,776bytes.) No whole-model compression claim follows.

The CPU reference profile is Torch 2.13.0+cu130, NumPy 2.5.4, Python 3.14.6, AVX512, MKLDNN enabled, BF16, deterministic algorithms, one intra/inter-op thread. It used no GPU. Native CUDA parity remains unmeasured. mrun reserved four CPU slots by its probe default; the actual worker libraries and native references used one thread.

## Reproduce the bounded reader

The reader requires NumPy and the inherited import guard's Pillow utility dependency. The public bundle includes the exact emitted midpoint state for convenience; its explanatory dose label is not passed to serving.

```bash
python generator/render_state_receiver.py \
  --bundle docs/results/compiled-state-reader \
  --state docs/results/compiled-state-reader/state-22185a19edaa2661.npy \
  --output artifacts/my-reader-output
```

The output contains raw and BF16 Q/K/V, recovered coordinates, exact scale, out-of-basis residual and a file-audited seal. Other [1,32,3072] BF16 states can be supplied mechanically, but accuracy outside this producer family is unmeasured.

The research launcher uses `state_receiver_study.py stage|submit|collect`, authenticated prior producer/context sources, a finite mrun CPU lease, and a separately provisioned post-seal native process. The four new synthetic mechanics checks cover the residual identity, rank degeneracy, state-only API, file isolation, hash binding, full compiler receipt and signed defect algebra. All **26** generator mechanics tests pass. Saturn's evidence sweep and verification return 0 over one stream / 1,152 records; that is a `target-streams-only` mechanics check, with no automatic terminal claim receipts or address joins.

## What this changes

The next computation can now read the actual emitted state rather than receiving a second copy of the intervention coordinate. Exact scalar normalization and retained residuals make that claim inspectable. The remaining reader discrepancy is not simply a missing fifth direction: native arithmetic boundaries matter, and the residual's effect must be measured through its consumer.

The [fresh image-futures work](/Users/jakeholl/domains/obsidian/blog/2026-11-05-225618-testing-state-equations-against-fresh-image-futures.md) motivated this writer/reader defect split. The [reader and stream work](/Users/jakeholl/domains/obsidian/blog/2026-11-05-230617-a-reader-can-predict-a-change-without-carrying-its-main-effect.md) cautions that local prediction accuracy does not establish dominant image causality. Astra's review also emphasized retaining actual variance/sqrt after nonfinite polynomial-normalization branches in the [simplifying-assumption experiment](/Users/jakeholl/domains/obsidian/blog/2026-11-05-161109-the-one-assumption-that-would-make-model-computation-easier-to-understand.md).

The full [arbitrary-prompt objective](zero-neural-research.md) remains active. A useful local edge must ultimately connect raw prompt operands, evolving context, global attention, repeated own-state transitions and a general image decoder. This result establishes none of those missing capabilities by implication.

Astra's result review recommends [one global attention-head continuation](plans/compiled-global-attention.md) as the next cut. It must derive every changed Q/K/V from the actual state packets, keep the frozen remainder explicit, and recompute changed queries against all keys. The measured rounding/basis differences remain diagnostics as that global interaction is tested.
