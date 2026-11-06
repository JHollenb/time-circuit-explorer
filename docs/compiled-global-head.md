# An emitted state changes a complete attention head

The [compiled state reader](compiled-state-reader.md) now drives one global consumer. At diffusion step2/joint2, prespecified head0 reads all **1,536 text/image positions**. Its32 changed image rows derive Q/K/V from the producer's actual BF16 state, without a dose input or native refresh. The remaining512 text and992 image rows are an explicitly supplied frozen context.

The incremental attention calculation agrees with streamed full attention under the same mathematical policy to **5 × 10⁻¹⁴**, and both packets' BF16 head outputs match exactly. Recovering Q/K/V from the state and computing the complete head field takes **1.80–1.81ms** on one CPU thread. These are bounded compiled-execution results, not arbitrary-prompt generation.

Prediction error remains substantial: the midpoint paired response differs from the native-QKV-derived reference by **28.76% before head-output rounding and35.13% afterward**. The reference is a NumPy mathematical head consuming archived native CPU Q/K/V at the changed rows. **No native attention or CUDA Flash parity is claimed.** The experiment stops before the learned output projection, residual write and later blocks.

## What was supplied, and what advanced

The preceding [MLP write](compiled-mlp-write.md) emits an actual [1,32,3072] BF16 state array. This experiment uses its baseline and midpoint packets, whose bytes and opaque IDs are unchanged. The already compiled five-direction reader recovers coordinates from those arrays. Its head0 Q/K/V affines match the preceding reader's corresponding outputs exactly after the declared head preparation.

The authenticated Saturn cut remains `2f1dbab51da6022302634c5c7d26253cd759fe334341ed8eaa8a23b244599b8a`, before `joint_block.2`. Offline preparation reads that cut's full image/text registers, modulation and RoPE, plus original joint2 parameters. It projects **only the fixed992 image rows and512 text rows**, selecting head0's128 output channels from each Q/K/V matrix. All32 changed positions are explicitly zero in the fixed bank. They can only be filled by the supplied current state packets.

For each stream, preparation uses population LayerNorm, staged BF16 modulation and selected original-weight projections. Q/K then receive learned head RMS normalization and captured interleaved RoPE. The declared policy follows the pinned [reference equations](../generator/kernels/flux_reentry_reference.py): FP64 reductions, explicit BF16 boundaries, separate FP32 RoPE multiplies/addition, and FP64 unmasked noncausal attention. It does not reproduce Flash's internal reduction schedule.

The new head0 reader bundle contains the existing basis, only head0's reduced Q/K/V banks and offsets, two image head-normalization weights, and selected positional factors. Three learned reduced maps and two learned head-scale applications still run online. “No neural-framework calls” and “no original dense projections online” describe the implementation; they do **not** establish a representation-independent absence of learned computation.

## Factor the global read by changed addresses

For a query row i and head width d=128,

\[
h_i=\frac{\sum_j e^{s_{ij}}v_j}{\sum_j e^{s_{ij}}},\qquad
s_{ij}=q_i^\top k_j/\sqrt d.
\]

Let S be the32 changed image key/value positions, and I the same positions as queries. For unchanged queries, only columns in S change. Compile the fixed sources' contribution once:

\[
m_i=\max_{j\notin S}s_{ij},\quad
Z_i^{\bar S}=\sum_{j\notin S}e^{s_{ij}-m_i},\quad
N_i^{\bar S}=\sum_{j\notin S}e^{s_{ij}-m_i}v_j.
\]

For current packet keys and values, set

\[
m_i'=\max\{m_i,\max_{j\in S}s'_{ij}\},\qquad c_i=e^{m_i-m_i'},
\]

then compute

\[
h_i'=\frac{c_iN_i^{\bar S}+\sum_{j\in S}e^{s'_{ij}-m_i'}v_j'}
{c_iZ_i^{\bar S}+\sum_{j\in S}e^{s'_{ij}-m_i'}}.
\]

This cache contains only unchanged sources. It neither subtracts a nearly complete selected-source mass from a full partition nor mixes a native selected-row baseline with a candidate update. Probabilities, numerator and denominator use the same declared arithmetic.

For i in I, the query itself changes. Those32 queries recompute their scores against **all1,536 current keys**, including changed sources, then recompute their weighted values. Restricting them to32 columns would omit the changed-query/fixed-key interaction.

The two primary forecasts and six computational controls seal before the worker opens archived native-derived Q/K/V or the prior reader's forecasts. Post-seal direct evaluations measure the incremental/full arithmetic discrepancy. Both baseline and midpoint have raw RMS discrepancies about2 × 10⁻¹⁵; their maximum absolute discrepancies are4.97 × 10⁻¹⁴ and4.26 × 10⁻¹⁴. Every rounded output value agrees. These are instrument checks under the shared mathematical policy, not native-kernel certification.

## A global effect with heterogeneous errors

Use the same frozen remainder and the same mathematical attention consumer for all arms. P consumes reduced-reader Q/K/V, C consumes the archived native CPU receiver's Q/K/V on the predicted state, and T consumes that receiver's Q/K/V on the fixed-parent native-write-derived state. T is not a newly observed full native block history.

Own-baseline paired errors retain

\[
\Delta P-\Delta T=(\Delta P-\Delta C)+(\Delta C-\Delta T),
\]

including signed cross powers. The first term contains the propagated Q/K/V reader discrepancy, while the second contains inherited writer discrepancy. The incremental attention algorithm itself contributes only the measured floating-point closure difference under this policy.

| Midpoint query region | Queries | Raw response error | BF16 response error |
|---|---:|---:|---:|
| Complete head | 1536 | 28.76% | 35.13% |
| Fixed text queries | 512 | 122.11% | 93.78% |
| Changed image queries | 32 | 79.05% | 90.26% |
| Other image queries | 992 | 27.13% | 32.76% |

All ratios divide error RMS by the corresponding T own-baseline change RMS. Global raw reader/inherited component norms are32.22%/25.98%, with cosine−.529. Rounded values are37.34%/32.66%, with cosine−.503. They partly cancel; those norms are not additive causal percentages.

About **98.36% of raw reference response energy appears in the992 image queries whose input rows were held fixed**. Changed sources therefore alter reads elsewhere in the register. The512 text queries carry only.00393% of that response energy; their small absolute response can still be poorly predicted. The direct native image or RGB meaning of this head has not been tested.

Per-query records remain less favorable than the pooled error. Median raw row errors are34.1% for text,91.4% for changed image queries and77.9% for other image queries. Worst raw ratios exceed100% in every region. After rounding,897/1536 reference queries have zero change; their ratios stay undefined. The candidate changes52 of those rows. Their absolute errors remain in the record, contributing about.0725% of total squared error. Among nonzero rounded-reference rows, some ratios are extremely large, including a maximum near4096; these were retained rather than filtered.

The [report](results/compiled-global-head/report.json) and [3,072 per-query observations](results/compiled-global-head/observations.jsonl) retain the full region, raw/rounded, absolute/paired and inherited-error panels. These are dependent query rows in one supplied context and one mechanically chosen head, not independent prompt replications.

## Computational roles and interaction

The controls keep the candidate baseline fixed while changing only selected prepared operands. Values-only, keys-only, queries-only and keys+values all use the same complete context. Unchanged replay reproduces the baseline forecast bytes exactly.

| Raw head control | Error / T response | Cosine to T response |
|---|---:|---:|
| Complete Q/K/V change | 28.76% | not separately recorded |
| Values only | 29.84% | .9544 |
| Keys only | 99.91% | .0977 |
| Queries only | 99.83% | .0724 |
| Keys and values | 29.45% | .9558 |
| Zero change | 100% | undefined |

Under this head's fixed addresses, the value change carries most of the measured raw response. This is a numerical role at this cut, not an object/color symbol or a universal property of attention.

The signed K/V interaction has raw norm.882% of the complete candidate response; the query-versus-source interaction has norm.0327%. After BF16 output rounding, the same finite-difference norms are15.35% and5.27%. Quantization changes the response geometry; these terms cannot be read as causal shares.

The newest [query–bank experiment](/Users/jakeholl/domains/obsidian/blog/2026-11-05-233846-how-image-queries-and-memory-banks-compute-a-state-change.md) reports much stronger interactions through a different complete native block and recipient. That is a useful counterpoint, not a contradiction: our cut changes32 rows in one head at step2/joint2 and stops before projection/MLP; that experiment changes complete banks/queries at step1/joint0 and follows a native image tail.

## A fairer address control

The initial wrong-address control permutes **whole raw Q/K/V payloads**, including their baselines, before normalization and destination RoPE. It produces37.5 times the raw T response error. That is a large destructive perturbation, not a norm-matched semantic null. Its outcome remains in the original report.

A separately sealed saved-array experiment instead keeps baseline Q/K/V and permutes only the value **change**. With fixed Q/K, define the baseline selected-source probabilities P_S and \(\delta V=V_1-V_0\). The exact value-read law is

\[
\delta H=P_S\delta V.
\]

The correct placement reproduces the earlier values-only head to4.80 × 10⁻¹⁴. Cyclic, reversed and seeded-shuffle placements retain exactly the same proposed squared delta norm. Two profiles keep both algebra and native-format admission visible:

\[
V'=V_0+\Pi\delta V\quad\text{(smooth)},\qquad
V'=\operatorname{BF16}(V_0+\Pi\delta V)\quad\text{(admitted)}.
\]

| Smooth placement | Raw error / T response | Cosine to T response |
|---|---:|---:|
| Correct | 29.84% | .9544 |
| Cyclic | 132.93% | .0789 |
| Reversed | 115.10% | .2347 |
| Seeded shuffle | 133.57% | −.0117 |
| Zero delta | 100% | undefined |

The admitted profiles show similar outcomes. All nonzero proposed delta powers are exactly.00296114256. Realized admitted powers range.00296282134–.00296809443, with admission displacements of RMS.002756–.002810 for the wrong placements. These are address-sensitivity diagnostics under a frozen mathematical consumer, not proof of a semantic location, and not held-out prompt generalization. The [control report](results/global-value-address/report.json), [selected probabilities](results/global-value-address/probabilities.npz) and [custody](results/global-value-address/provenance.json) preserve the distinction.

## Compute, storage and execution

| Stage | Measured cost or explicit work |
|---|---|
| Fixed-context compiler | 5.710s wall,4.937s CPU;265.2MiB process high-water RSS |
| Original parameter verification | 113,247,232 payload bytes across six Q/K/V matrices and four head norms; full7,751,109,744-byte checkpoint hashed once |
| Actual original projection work | 1,774,190,592 MACs, selecting128 head0 channels for1504 fixed rows |
| Fixed-source attention-cache preparation | 579,076,096 MACs |
| Online reader/fixed/cache payload | 4,460,544 +2,359,296 +1,576,448 = **8,396,288 bytes** |
| Primary arithmetic | 1.801–1.806ms: state coordinates/residual, three reduced QKV maps, two learned head scales, RoPE, complete-query attention and BF16 head rounding |
| Variable attention work | 24,903,680 MACs versus603,979,776 for a full one-head read |
| Reader contraction work | 61,440 reduced-map MACs plus983,040 coordinate/residual-reconstruction MACs; scalar normalization/rotation/exponentials additional |
| Serving high-water RSS | 64.6MiB |
| Fresh serving process | .270s for imports, bundle loads,2packets × (1warmup+31timed evaluations),6single-shot controls, seals and I/O; not single-query cold latency |
| Post-seal references | .291s; six full mathematical head evaluations: two same-candidate closure checks and four archived-native-QKV-derived references |
| Complete head job | 8.35s scheduler wall;6.573s worker wall;351.8MB sampled job peak RSS;0VRAM |
| Matched value-address job | 1.23s scheduler wall;.375s worker wall;111.8MiB parent process high-water RSS;0new weight reads or neural calls |

The variable attention MAC ratio is about24.25, **not a measured end-to-end speedup**. Compilation, bank materialization, coordinate work, copying and nonlinear scalar work remain real costs. The entire full context remains present. Extra heads, changed rows, contexts, denoising steps and prompt formation are outside these timing/storage figures.

No neural framework runs in either new experiment, but fixed-context preparation evaluates original learned projections, while serving evaluates reduced learned maps and learned head scales. Historical native encoder, MLP/QKV and captured-context costs remain assistance. The strict arbitrary-prompt zero-neural objective is not met by this arithmetic refactoring.

The head job is `job-13617045e079`, request`766a85ffa1f712024f01d4a625b4f932702ba3bbac2a40c156ba21a885af3618`. Its complete20-file collection is38,909,376bytes. The matched-control job is `job-4cad9f4cb662`, request`8240cc95b929dc23f58324352b6d5dfa8046f2a516acfca03c1afb246eed26df`; its complete4-file collection is26,004,098bytes. Guarded payload digests and remote-before/after/local collection inventories match. Public bundles are explicitly identified subsets, with omitted arrays retained in the complete collections.

## Run the bounded head

The portable utility requires NumPy and Pillow (the inherited import guard's image utility dependency). It takes a state array and returns a complete[1536,128] head field, not an image or a prompt interpreter:

```bash
python generator/render_global_head.py \
  --bundle docs/results/compiled-global-head \
  --state docs/results/compiled-global-head/state-22185a19edaa2661.npy \
  --output artifacts/my-global-head
```

The baseline packet supplies the computational-control baseline. The current packet drives all changed fields. Other states must retain the same[1,32,3072] exact widened-BF16 geometry; accuracy beyond this supplied family is unmeasured.

Four synthetic mechanics checks cover changed-query/global-source coupling, stable exponential rescaling, native-refresh exclusion, fresh state-only serving, portable-bundle execution, hash/capability isolation, and the complete matched-delta transaction. The full generator mechanics suite now has30tests. The [implementation](../generator/global_attention.py), [launcher/scorer](../generator/global_attention_study.py) and [matched-control implementation](../generator/global_value_address.py) retain the computation and work census.

This result extends one supplied state algebra through a global read. The full [goal](zero-neural-research.md) still requires raw-prompt operand formation, reusable context selection, connected state writes and later transitions, a general non-neural pixel decoder, and independent prompt-faithfulness evidence.

Astra's completed cycle4 review found no material algebra/custody blocker and recommends the [raw-prompt encoder-atom cut](plans/raw-prompt-encoder-atoms.md) next. The latest [twenty-block continuation post](/Users/jakeholl/domains/obsidian/blog/2026-11-05-235313-following-a-state-change-through-twenty-model-blocks.md) provides native own-state continuation at another supplied boundary, with differing absolute/relative response trends. The [simplifying-assumption work](/Users/jakeholl/domains/obsidian/blog/2026-11-05-161109-the-one-assumption-that-would-make-model-computation-easier-to-understand.md) now retains OVC13's recovered yet inaccurate forecasts. Those developments support keeping numerical defects and future behavior visible; they do not supply the missing raw-prompt operands or image decoder.
