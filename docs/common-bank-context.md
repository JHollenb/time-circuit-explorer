# A shared state preserves three prompt changes much better, with a signed-read exception

One shared factorization of eight saved prompts substantially improves three contextual changes over the source-frozen Gaussian features. At rank128, above/below, inside/outside and long-order errors are **20.0%,30.0%,36.6%** of the exact change RMS. The Japanese-color change remains117.8%, with much worse padding-relative error. The factorization uses all eight prompts and has no feature rule for unseen queries. It is a transductive capacity result, not arbitrary-prompt serving or zero-neural image generation.

The result answers a useful narrower question from [cycle7](source-conditioned-context.md): this finite query/key bank admits a substantially better shared representation than the tested Gaussian features. It does not establish the optimal rank for contextual reads, because the SVD optimizes a global matrix norm and permits signed weights. The complete encoder, own-state diffusion, general non-neural pixels and independent prompt-faithfulness evaluation remain open.

## Build one operator across all prompts

The [frozen protocol](plans/common-bank-context-rank.md) deduplicates queries and active keys by `(token ID, absolute position)`. This is valid for the source's first token-local prefix; it cannot silently be applied to contextual states in later layers. Repeated query operands and key/value operands must match BF16 bytes.

The bank contains1,593 unique queries, including padding queries, and1,108 unique active keys. Its original prompts contain4,096 query occurrences and1,212 active-key occurrences. All2,503 repeated queries and104 repeated keys verify exactly. Each prompt retains its own occurrence maps, masks and values; deduplication does not erase multiplicity in a read.

Construct one unmasked matrix

\[
K_{uv}=\exp(q_u^\top k_v/\sqrt{128}),\qquad
P_{uv}=K_{uv}/\sum_w K_{uw}.
\]

Stable FP64 row normalization avoids overflow. Global row scaling cancels when the row is restricted to a prompt's actual causal key occurrences and renormalized. This exact restriction reproduces all archived mathematical heads within9.72 ×10⁻¹⁷, with no underflow or invalid restricted reads.

One actual thin SVD fits unique bank rows equally:

\[
P=U\Sigma V^\top,\qquad
A_r=U_r\sqrt{\Sigma_r},\quad B_r=V_r\sqrt{\Sigma_r}.
\]

The four prespecified ranks are16/32/64/128. There is no per-prompt factorization, target-selected rank or silent clipping. Full thin-SVD reconstruction has relative Frobenius error6.00 ×10⁻¹⁵. The unmasked matrix is14,120,352bytes; its rank128 approximation has17.49% relative Frobenius error. That matrix observable is not the primary consumer metric.

## Consume an advancing state

A fresh guarded process receives only factors, query/key occurrence maps and BF16 values. It does not receive the full kernel, original Q/K, archived head targets or a native fallback. For actual allowed key occurrences it advances

\[
z_i=z_{i-1}+m_iB_r[k_i],\qquad
M_i=M_{i-1}+m_iB_r[k_i]v_i^\top,
\]
\[
\widehat h_i=\frac{A_r[q_i]M_i}{A_r[q_i]z_i}.
\]

The query coordinate is a fitted-bank lookup, not a function for an unseen atom. Complete heads and state checkpoints seal before archived-head and output-slice scoring. Streaming agrees with direct restriction of the same rank-r matrix to1.54 ×10⁻¹² maximum absolute difference; the worst per-case RMS difference is6.41 ×10⁻¹⁴ relative to the direct head. Numerical reduction order is small compared with the observed forecast errors.

## Prompt changes first

Each number divides RMS(predicted change−exact change) by RMS(exact change), over all512 query positions. These four pairs fit the bank and therefore supply capacity evidence, not held-out semantic replications. Historical controls are unmodified rows from the separately sealed cycle7 assay.

| Prompt change | Rank16 | Rank32 | Rank64 | Rank128 | Source-frozen Gaussian64 | Uniform |
|---|---:|---:|---:|---:|---:|---:|
| Kite above/below | 49.2% | 65.7% | 28.0% | 20.0% | 780.5% | 80.3% |
| Key inside/outside | 52.1% | 50.6% | 36.6% | 30.0% | 1,743.0% | 78.3% |
| Japanese colors | 7,795.4% | 135.3% | 134.1% | 117.8% | 117.0% | 64.1% |
| Long repeated order | 76.1% | 70.2% | 57.5% | 36.6% | 217.6% | 95.0% |

At rank128, signed change cosines are.980,.954,.654,.931 in this order. Ideal saved W_o-head-slice errors are19.6%,32.3%,120.5%,38.3%. Thus the three advantages survive this linear consumer, while the Japanese failure remains. This is not the full fused native projection, residual, later encoder or image.

The split reveals distinct behavior:

| Rank128 prompt change | Active-both error | Padding-both error |
|---|---:|---:|
| Kite above/below | 30.8% | 17.0% |
| Key inside/outside | 37.0% | 22.4% |
| Japanese colors | 88.1% | 865.1% |
| Long order | 36.6% | No padding |

Japanese padding's exact change RMS remains8.32 ×10⁻⁵. Rank64 improves its active error to76.1%, better than rank128's88.1%, while both have large padding errors. Rank32 also worsens the kite active effect to141.0% despite improving its padding effect to34.9%. These nonmonotone and heterogeneous trajectories remain in the raw rows. Padding cannot be dropped because the later FLUX conditioner consumer receives every row.

## Why a signed kernel can fail sharply

SVD weights need not be nonnegative. At rank16, each Japanese string has26 negative prefix denominators:52 total. There are no zero denominators or nonfinite actual reads in the assay, so every row remains scored. The smallest cancellation ratio is0.00214; the largest normalized negative mass is233.2. A signed read can extrapolate far beyond its values even though the matrix is a good low-rank approximation in another norm. Higher ranks have positive denominators on this panel but still contain negative weights; rank128's Japanese normalized negative mass reaches1.53.

For a fixed allowed prefix, write the exact row as p, the kernel approximation as p+e, D=Σp and h=Σpv/D. Expanding and factoring gives the exact real-arithmetic error identity

\[
\widehat h-h=\frac{\sum_j e_j(v_j-h)}{D+\sum_j e_j}.
\]

It shows why matrix reconstruction alone cannot certify a read: the kernel defect is weighted by the values the consumer distinguishes and divided by the actual approximate partition. For a paired prompt change, the error is the difference of these two head-error vectors. Even small head errors can overwhelm a much smaller contextual change. The identity is explanatory algebra here, not an additional measured causal attribution or a native Flash parity claim. BF16 rounding remains separately scored.

This is not evidence against every rank-r state. SVD minimizes global Frobenius error on P, not normalized prefix reads, small finite differences or downstream writes. Signed rank capacity also differs from positive-feature capacity.

## Cost, storage and evidence

| Measurement | Observed scope/value |
|---|---:|
| Unmasked kernel construction | .0444s |
| Actual thin SVD | .5775s |
| Worker /scheduler lease | 5.922 /7.35s |
| Full sampled process-tree peak RSS /VRAM | 247.9MB /0 |
| Serving wall /CPU after input load | .603 /.561s |
| Reported serving high-water RSS | 158,592KiB, about154.9MiB |
| Rank128 warm statistic evaluation | 2.39–7.27ms |
| Core state, ranks16/32/64/128 | 16,528 /33,040 /66,064 /132,112bytes |
| Logical factor payload, ranks16/32/64/128 | 345,728 /691,456 /1,382,912 /2,765,824bytes |
| Resident max-rank factors | 2,765,824bytes for every tested rank |
| Resident saved occurrence/value packets | 1,179,648bytes |

Warm measurements have five repeats per prompt/rank and exclude factor/input loading, compiler, queue and collection. They evaluate known-bank factors and supplied values, not raw text or images. State updates charge activeKeys×r×128 value products and activeKeys×r×129 additions; reads charge512×r×128 numerator MACs and512×r partition MACs. No exponentials run in this finite-bank serving process. Offline exact kernel formation and SVD, model-derived factors and supplied values remain paid inputs. No controlled native speedup, energy or total image-compute claim follows.

The resident factor/packet payload is larger than directly caching all eight final BF16 head fields, which would take1,048,576bytes. This experiment investigates a shared advancing operator, not compression versus memorizing these eight outputs. It leaves the2.645GB upstream learned lexical table in the broader research stack.

Guarded job `job-eea183e994b5`, request `87617bc79562329e9a58f56866c0570b0d9d4e63f1dd6b7405002520af620965`, has matching declared/sealed/executed payload identities and authenticated collected inventories. The [prediction seal](results/common-bank-context/prediction-seal.json) is `ecbb11e693b4ef9d506f6e411cc6c0b3d990a8e8bb6b1cb6d200aa1f554dffe1`. Independent byte verification checks47 current/ancestor files totaling53,894,674bytes. The [public evidence](results/common-bank-context/provenance.json) contains the portable factors/occurrence packets and every compact collected result, with byte-lossless gzip observation streams. Full unmasked kernel/atoms and sealed heads stay near compute.

There are16,384 head-position and8,192 pair-position observations, all dependent. Saturn run/verify returns0 with two recognized streams and `target-streams-only` scope. No terminal semantic/image receipts or physical-address joins are supplied. The43 generator mechanics tests include duplicate identity, exact causal restriction, occurrence multiplicity, signed/zero-denominator handling and state resume. Source guards and audited code are not adversarial hardware attestations; memory containment was active and the payload sandbox off.

## Continue toward a rule for unseen queries

The next question is how to obtain a reusable query feature rule while retaining stable reads. Astra's [proposed64-landmark experiment](plans/source-landmark-context.md) uses source-frozen query landmarks with live prefix numerators/partitions and nonnegative query interpolation. It distinguishes interpolating *normalized readers* from mixing their kernels, because the latter changes route weights according to each reader's partition. Its source-distance metric measures score-displacement variance with consistently scaled operands. No such result is established here. Later encoder layers, autonomous image-state evolution and non-neural pixel production remain necessary for the original goal.

Sources: [factor/state implementation](../generator/context_common_bank.py), [single-bank assay](../generator/context_common_study.py), [publication and retained-byte verifier](../generator/publish_common_context.py), [source-frozen comparison](source-conditioned-context.md), [active research record](zero-neural-research.md), and the local [living blog](/Users/jakeholl/domains/obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md).
