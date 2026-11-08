# Fit the contextual distinction under the same writer budget

Status: proposed next cut after the executed [cycle 103 state-writer assay](../contextual-writer-program.md). Not implemented, fitted or executed. This proposal does not authorize an automatic sweep, larger dictionary, new features, more histories or a claim of arbitrary-prompt image generation. Implementation and source/resource review must precede any execution.

Cycle 103 improves pooled state/consumer endpoints but predicts zero response at every unchanged-token position in all three tested prompt pairs. Most transfer rows regress against position control, and padding error is worse at H1 and all Q/K/V ports. The next question is whether an objective that preserves contextual differences can change routing within the existing grammar.

Keep the qualified specimen, own tokenizer/full-vocabulary H0, fixed rows 0–1, existing first-slot program, two slots, depth four, at most 16 second-slot leaves, minimum eight source rows per leaf, 155 features, fixed threshold enumeration and finite rounding policy. Discovery remains A/B and 00/10; transfer remains 01/11 with target release after final serving seal. Keep the same target-free observatory prompt. All histories are reused public evidence, not fresh heldout.

Retain the executed endpoint-trained program and position control unchanged. Fit exactly one new second-slot tree with a contrast-aware objective. No half-strength or lambda sweep. Use lambda=1 as an explicit choice to give separately normalized endpoint and contextual-response losses equal weight, not as a calibrated threshold. Any imperfect result remains evidence.

## The mathematical change

Index discovery rows p≥2 by i. Let S_i be the actual finite output after the unchanged first slot, Y_i the native H1 target and a_i the second-slot leaf address. Let D contain one unrestricted vector constant per leaf. Before finite admission, the relaxation predicts S_i+D[a_i]. Define R_i=Y_i−S_i. For each unchanged-token discovery pair e=(i,j), oriented A→B or 00→10, define

\[
q_e=(Y_j-Y_i)-(S_j-S_i).
\]

Choose positive endpoint weights w_i with equal history and active/padding weight, and pair weights v_e with equal discovery-pair and active/padding weight over eligible rows. Normalize each set to sum to one. Empty strata are explicitly not applicable and their nonempty weights are renormalized. Fix the endpoint energy E=Σ_i w_i‖Y_i−H0_i‖² and contextual energy T=Σ_e v_e‖Y_j−Y_i‖² from discovery only. Preserve absolute losses when either energy is zero; do not divide by zero or label an empty response a scientific failure.

For the actual nonzero-energy case, minimize the relaxation

\[
L(D)=\frac{\sum_i w_i\|D[a_i]-R_i\|^2}{E}
+\lambda\frac{\sum_e v_e\|D[a_j]-D[a_i]-q_e\|^2}{T},\quad\lambda=1.
\]

Let A be the row-to-leaf one-hot matrix and B have one row per pair, with −1 at a_i and +1 at a_j. A pair routed to the same leaf has an all-zero incidence row. With W=diag(w/E), V=diag(lambda·v/T), the fixed-route normal equations are

\[
\big(A^TWA+B^TVB\big)D=A^TWR+B^TVq.
\]

AᵀWA is a diagonal matrix of positive leaf weights; BᵀVB is a graph Laplacian. There are at most 16 leaves, hence a small system with 2,560 right-hand sides. Every occupied leaf has positive endpoint mass, so the fixed-route system is positive definite in exact arithmetic. This is an offline learned constant fit, not a serving matrix multiplication. Count its numerical work and memory separately.

The identity exposes the routing failure directly. If a_i=a_j, the pair term is ‖q_e‖² independent of D; changing that dictionary cannot express the pair at this fixed first-slot state. A split that makes the routes differ can make the term adjustable. This statement is about fixed routes and the unquantized relaxation, not a global capacity bound for the grammar or exact BF16 serving.

Greedy tree construction should rank a candidate split by the decrease in the jointly minimized global relaxation, refitting all currently occupied second-slot constants when the pair graph changes. Independent CART leaf means are not the optimizer of this coupled objective. Keep the original feature/cut order for ties, depth/leaf/min-row restrictions and stop when no positive relaxation improvement exists. This is a greedy search, not a proof of the global best tree. Efficient sufficient-statistic evaluation must be designed and source-reviewed before freezing a runtime budget; do not silently implement a heuristic while calling it exact paired-loss minimization.

Admit the fitted constants to FP32 and run the original ordered FP32/BF16 serving steps. Report the unquantized fit and actual finite state/response losses separately. Quantization may reverse a split's apparent benefit. Preserve this discrepancy; no tiny proxy threshold may erase an otherwise mechanically valid experiment.

## Consumer and continuation authority

Use canonical Saturn Frames/Acts to evaluate the new candidate's actual H1 through the same native next inputnorm/QKV consumer. Reuse authenticated cycle 103 native/H0/control output fields when identities and numerical profile match; do not reload or recompute identical neural branches by default. Reusing a recorded output is an evidence comparison, not a new neural evaluation.

With one new candidate on six target-bearing histories, the prospective new neural panel is six serial B1 units: six normalizations, 18 original Q/K/V linears and 48,318,382,080 nominal MACs, plus fitting, serving, authentication and evidence costs. The observatory prompt has no native target or consumer call. If discovery chooses the historical program, transfer candidate evaluation still remains diagnostic evidence rather than being silently omitted.

Use discovery-only separately normalized endpoint and signed-response losses at the actual consumer to choose whether the new candidate or the historical endpoint program continues, preserving both scalar components, every row, all strata and all rejected branches. Do not select on transfer. Restore the chosen immutable program bytes exactly and seal fresh final own-input predictions before opening transfer target fields. Do not claim optimizer rollback or native consumer-free fitting.

The final report must show all512 and learned510 endpoints, active/padding/all strata, unchanged-token signed responses, route-change counts, atom occupancy, per-row improvement/regression counts, Q/K/V effects and actual cost. Successful edit propagation would establish one missing local behavior on reused histories, not full contextual conditioning or pixels. A zero response or tradeoff diagnoses this objective/grammar/search combination; it does not prove no permitted non-neural program exists.

Use one guarded mrun lease and one resident selected parameter set, with CUDA requested, CPU numerical threads declared, no retries and no parallel full suite. Determine expected fit cost from the implementation before submission; if over 15 minutes, follow the workspace's additional code/resource review requirement. Keep large lexical assets and states near compute. No image continuation or new teacher acquisition is part of this cut.
