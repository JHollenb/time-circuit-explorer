# Project the demanded value read after averaging its inputs

Status: **written and unexecuted**. After the [fixed spatial norm-capacity result](../spatial-key-norm-capacity.md), stop further absolute norm approximations and test a different source operand. Defer the original value projection until the live attention allocation has selected its weighted input. This is a demand-specific algebraic rearrangement, with original learned weights and source states still paid.

## The identity and its boundary

The pinned FLUX.2 joint attention uses bias-free linear image/text value projections. Values receive neither Q/K RMSNorm nor RoPE. Native ModLN still runs separately on image and text inputs. For each head h and demanded query a, with the existing joint text/image probabilities p,

\[
H_{ha}=\sum_{j\in t}p_{haj}X^{t}_jW_{V,t,h}^{T}
+\sum_{j\in i}p_{haj}X^{i}_jW_{V,i,h}^{T}
=\Big(\sum_{j\in t}p_{haj}X^{t}_j\Big)W_{V,t,h}^{T}
+\Big(\sum_{j\in i}p_{haj}X^{i}_j\Big)W_{V,i,h}^{T}.
\]

This is exact in real arithmetic. It does not commute ModLN, softmax or a nonlinear value function. The original BF16 per-row value projection rounds before summation; the proposed FP64 deferred projection rounds differently. Those combined finite discrepancies are observations, not an asserted native parity identity.

The identity applies at fixed allocation even though p depends on the current Q/K state: V does not enter that attention call's probability calculation. Each head generally has different weights, so weighted inputs must remain head-specific. Text and image must not be merged before their different learned maps.

## Freeze the existing allocation and change only value production

Reuse the exact cycle58 support **[4,11,12,13]**, compiler duals/group IDs, retained P smooth norm and selected P Q/K. Recompute image numerator and text logits from the actor's actual current state with the unchanged cycle58 law; do not supply old probabilities, logits or an archived H. This keeps the previous useful allocation approximation fixed while testing a new value producer.

Form actual native BF16 ModLN fields from Q's image/text StateCut. Form P's selected normalized text inputs and splice those rows into Q's normalized text-input field **before** weighted sums. This matches the original selected-P value splice without reading or projecting selected V. P/Q source state preparation, native text-key prefix, dual compilation and retained P norm remain paid.

The actor loads only the original `transformer_blocks.1.attn.to_v.weight`, `add_v_proj.weight`, and the five unchanged receiving/next-query parameters. Authenticate the actual bias-free constructor and tensor inventory, source pages, shape/dtype, input/RNG closure and requested numerical backend. Deny numerical access to **all** full P/Q V banks, selected V, prior hybrid V, historical H/states/queries, native target outputs and previous score/probability arrays before the candidate seals and exits. Existing full key arrays remain denied; the already-qualified text-only key prefix is the declared exception.

The actor forms two per-head weighted input fields, projects each through its own head slice of the original V map, sums their outputs, and admits FP64→FP32→BF16 H through the unchanged original receiving output projection/MLP→own T→own next query. Retain the actual weighted inputs, separate text/image head contributions, H in both finite representations, own receiving states/query, numerical file closure, explicit typed output ABIs and seals. No other candidate or MLP arm, fit, gain choice, new norm law, suffix or image.

## Post-seal evidence

Only after candidate exit, authenticate and open historical cycle58 allocation/output fields and the paid original native V bank with the same selected-P splice. Check actual input/numerator/text-logit/allocation identity as applicability evidence. Preserve any mismatches and separate their interpretation from finite value effects.

With the candidate's actual probabilities, perform one historical-bank AV to measure the deferred head's difference from the original BF16 V route. This adds **18,874,368 matrix MACs**, explicitly charged. It is a post-seal diagnostic; no extra learned consumer follows it. Do not recompute a new full smooth V bank merely to validate the real-arithmetic identity. Independent tiny fixtures can test the algebra, separate-stream/head handling and selected-input splice.

Compare actual own H/A/MLP/T/nextQ endpoints and P→Q responses with unchanged cycle58, stale and zero controls, retaining all rows/heads and signed effects. Distinguish matched-allocation finite discrepancies from norm/allocation errors, which this experiment does not repair. No fidelity threshold blocks exploration; source/byte/typed/seal/resource integrity remains strict.

## Prospective compute and scaling

For D=3072, H=24, d_h=128, four demanded queries and N_t+N_i=512+1024,

- Per-head weighted inputs: **24×4×1536×3072 =452,984,832 MACs**.
- Two separate per-head stream projections: **2×24×4×3072×128 =75,497,472 MACs**.
- Total demanded value formation: **528,482,304 MACs**, replacing full original value formation **1536×3072² =14,495,514,624 MACs** and its old **18,874,368-MAC AV**.
- Complete actor, including the unchanged cycle58 numerator/text-score/receiving work: **1,855,979,520 MACs** =1,346,371,584−18,874,368+528,482,304. The post-seal bank AV brings new run work to **1,874,853,888 MACs**, before nonmatrix state/softmax work. There is no new dual compiler or full-bank projection in this proposed run.

These are nominal counts, not measured speedups. Head-specific weighted-input work grows with the number of demanded queries. With all 1,536 queries it would cost more than projecting the full bank once. This is a candidate for a reusable few-read producer, not a whole-denoiser saving or a strict zero-neural result.

Use one guarded strict-preflight mrun lease, current measured selected-map resource basis, actual CPU-FP64/CUDA-BF16 backend, finite timeout and no retries. Preserve full arrays near compute, compact/lossless publication and complete byte custody. At the next cycle62 review, have Astra inspect the newest working blog posts and this implementation before launch. Storage review remains63 or earlier pressure. The arbitrary raw-prompt zero-neural goal remains unsolved.
