# Let the emitted state change a global attention read

Status: proposed after the [state-reader and signed-attribution results](../compiled-state-reader.md). No global-consumer run is claimed. The arbitrary-prompt objective remains unchanged.

The smallest first organism is one prespecified head0, baseline and midpoint, with its complete1536-query head field. Head0 is a mechanical choice, not a claim that this head carries the late color effect observed at another clock. Additional heads/doses can follow once this instrument's geometry and arithmetic are qualified.

The local reader can now consume actual emitted BF16 states without receiving the intervention dose. Its basis omission and native numerical-policy difference are both material. Keep those as diagnostics while testing the next structural question: can changed rows' predicted Q/K/V alter reads across a complete fixed context, without a dose-specific fit or a native state refresh?

## One coherent supplied parent

Use the same diffusion-step2/joint2 cut. Embed the selected32 emitted image rows into a frozen, explicitly supplied parent for all other image rows and512 text rows. Preserve original image-row addresses, stream ordering, captured modulation and positional coordinates. These supplied fixed rows are retained native assistance, not an arbitrary-prompt context compiler.

Image Q/K/V for the changed rows must come from their actual current state packets through the compiled reader. The unchanged image/text banks can be prepared once offline from the common parent, with every original dense projection and parameter byte counted. Inspect native Q/K head normalization and RoPE first; preserve learned head-normalization weights and their arithmetic as separately counted work. Do not imply that these operations stop being learned computation because they use ordinary arithmetic.

Any baseline cache must use the candidate's own dose0-derived Q/K/V at the selected rows. Do not combine a native selected-row cache with a candidate selected-row update. Its probabilities, denominator and numerator must come from the same declared attention arithmetic: ideal softmax probabilities paired with a captured Flash head output would invalidate the identity. Serving receives current state, a fixed bank/cache identity and address metadata; no dose, target or per-dose native refresh is admitted.

## Factor attention by changed addresses

For one head and a query row i, write

\[
a_i=\frac{\sum_j e^{s_{ij}}v_j}{\sum_j e^{s_{ij}}},\qquad
s_{ij}=q_i^\top k_j/\sqrt{d_h}.
\]

Let S contain the32 changed image key/value addresses, and I the corresponding changed query rows. For i outside I, its query is fixed, so only columns in S can change. Precompute an **unchanged-source** cache

\[
Z_i^{\bar S}=\sum_{j\notin S}e^{s^0_{ij}-m_i} ,\qquad
N_i^{\bar S}=\sum_{j\notin S}e^{s^0_{ij}-m_i}v_j^0,
\]

with \(m_i=\max_{j\notin S}s^0_{ij}\). At runtime, choose

\[
m_i'=\max\{m_i,\max_{j\in S}s'_{ij}\},\quad c_i=e^{m_i-m_i'},
\]

then compute

\[
a_i'=\frac{c_iN_i^{\bar S}+\sum_{j\in S}e^{s'_{ij}-m_i'}v_j'}
{c_iZ_i^{\bar S}+\sum_{j\in S}e^{s'_{ij}-m_i'}}.
\]

This directly combines fixed and changed sources; it avoids subtracting a nearly complete selected-source mass from a full cached partition. Retain numerical closure against a streamed full-bank evaluation under the same arithmetic.

For i in I, its query changes. Recompute its scores and weighted values against **every** key/value row, including changed rows. Reusing only the changed columns for such queries would miss all the changed-query/fixed-key interactions. Both image and text queries outside I can receive changes through the selected image values and keys. This is the desired global coupling.

With N=1536, m=32 and head dimension128, the variable score work has \(((N-m)m+mN)hd_h\) multiply-accumulates, with corresponding weighted-value work. The first organism has h=1; extending to all24 heads multiplies that work by24. Report cache compilation, cache/bank bytes, row-normalization/RoPE work and actual timing separately. The full fixed context still exists; fewer variable contractions do not establish context compression or end-to-end generation speedup.

## References and controls

Seal complete predicted head outputs before new native comparisons. Reuse existing sealed Q/K/V forecasts and same-input/native-write reference outputs where appropriate; keep any newly generated reference or original-weight preparation costs explicit.

First compare incremental attention with streamed full attention on the **same supplied Q/K/V and same arithmetic**. This validates the partition algebra, stable exponent rescaling, source order and changed-query handling. Shape/finite/custody and actual-instrument execution are mechanics requirements. Accuracy against native BF16/Flash is an observation and should not block collecting the remaining valid arms.

Retain the composite, same-input reader and inherited-writer paths through the same declared consumer. Where useful, include the saved full-projection/quantizer counterfactual, rather than fitting a new basis to native targets. Preserve own-baseline paired effects, per-query/per-head error distributions and signed cross terms. A first primary scope may stop at pre-output-projection attention head values; original dense output projection must not silently return online.

Controls should include unchanged packet replay, changed-values-only, changed-keys-only, changed-queries-only, and a declared wrong-address permutation. Keep the key/value interaction and query/fixed-bank interaction visible. These are computational controls, not independent semantic prompt examples. A native full attention comparison must declare backend and numerical contract; CPU softmax is not CUDA Flash parity.

## What would change the larger account

A successful result would connect a producer-fed local state to a global reader, with all changed operands derived from that state. It would still have a frozen supplied context and learned reduced maps. It would not establish a raw-prompt compiler, independent prompt generalization, denoising closure, non-neural image decoding or representation-independent zero learned computation.

The newest [value-to-image-output-to-RGB work](/Users/jakeholl/domains/obsidian/blog/2026-11-05-230617-a-reader-can-predict-a-change-without-carrying-its-main-effect.md) makes values a useful controlled first branch. It also shows that keys and queries condition the read, and that local prediction and semantic causal importance need separate evidence. The goal remains a reusable computation over new prompt-formed operands; another independently fitted dose chart would not supply it.
