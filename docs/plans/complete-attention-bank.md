# Predict the complete first-layer attention bank

Status: **proposed after cycle10 and Astra review; not staged or measured**. The [live query-transport result](../query-transport-context.md) improves four paired changes and their measured MLP/state effects. Its largest remaining supplied dependency is the other 31 attention heads. The next experiment asks whether errors from all predicted heads remain useful when they meet at the actual common consumer.

Freeze the existing source-only recipe for every query head: complete lexical atoms, source covariance of each of the eight KV groups, and 64 landmarks selected separately for each of 32 query distributions. Reuse source operand gathering and eight covariance computations where exact. Initially preserve 64 landmarks per query head; do not pool landmarks, change routing and compress coordinates simultaneously.

GQA shares K/V operands, not query distributions. A live moment bank can be shared exactly only for identical landmark queries. Eight KV groups alone do not establish a fourfold reduction in the 32 query heads' distinct moment banks. Report about 273 MB of logical FP64 core state for 32 independent banks, plus assets/table/workspaces. Sequential head execution may reduce resident memory but must not erase the logical total or repeated update work.

Start with one frozen fresh prompt pair, declared before reference computation. Form all 512×4096 attention operands from the actual raw strings and prefix histories. Seal complete nearest and transported candidate banks before exact references. No exact replacement heads, reference refresh, target-selected repair or output-fitted route is admitted in either prediction arm.

Then evaluate original W_o, residual, RMS and gated MLP on **all 512 rows** as a separately paid offline consumer. Exact attention and attention-deleted banks are diagnostic arms. This still evaluates original learned equations and is not a reduced full encoder or zero-neural image result.

Lead with paired first-block-state error and cosine, absolute target/error RMS, active-both/padding-both/changed-mask splits and error relative to the attention-deletion effect. Retain full endpoint errors, each head's output contribution and signed cancellation so a good pooled value cannot hide opposing errors. Preserve all completed rows and signed-weight/range diagnostics; accuracy and positivity do not abort exploratory evidence collection.

Mechanical checks cover GQA mapping, per-head source/query identity, masks and prefix clock, complete-bank sealing, original consumer dependence, finite-value contracts and artifact custody. Scientific execution is one finite guarded CPU mrun lease near the store, with strict preflight and no automatic retry. Measure source gathering, compilation, prediction, original reference/consumer, RAM and actual byte/work counts separately. Start with the smallest informative pair; if expected runtime exceeds 15 minutes, obtain the workspace-required execution review first.

Defer reduced displacement coordinates until the joint consumer error is visible. A subsequent source-frozen basis B could test

\[
J_\ell(x-\ell)\;\longrightarrow\;(J_\ell B)B^T(x-\ell),
\]

with omitted correction measured separately. That changes approximation capacity and should not be bundled into this first complete-bank test.

The original MLP, later contextual projections, own-state diffusion and general pixel decoder remain substantive dependencies. A useful whole attention bank would establish a stronger component representation, not completion of the [arbitrary-prompt goal](../zero-neural-research.md).
