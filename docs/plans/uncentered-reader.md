# Remove the common key offset from the direct reader

Status: executed in cycle33, guarded CPUjob-2ed557dbb033. See the [result](../uncentered-reader.md): no centers or sampled geometry are read, and all actual-consumer inputs match. Two images are inherited; no new neural/image execution occurs. Astra reviewed source, results and newest posts in cycle33. The exact pre-run version remains in the [frozen plan](../results/uncentered-reader/frozen-plan.md).

Additive cost correction: the35,840 figure below counts admitted-key coordinates. The executed centered parent subtracts centers on all512packet rows before masking, so the actual removed arithmetic is524,288scalar subtractions per prompt. See the [correction receipt](../results/uncentered-reader-evidence/cost-correction.json). The original numerical-state and centers-payload counts are unchanged.

The direct reader stores source-centered keys U_j=K_j−c and computes x^TU_j. For each fixed query, −x^Tc is common to every allowed key. Therefore, in real arithmetic,

$$
\operatorname{top4}(x^T(K_j-c))=\operatorname{top4}(x^TK_j),
\qquad\operatorname{softmax}_S(x^T(K_j-c))=\operatorname{softmax}_S(x^TK_j).
$$

Test one fixed uncentered read. Keep raw strings, complete lexical atoms/template/tokenizer, masks/positions, original BF16 positioned Q/K/V, quarter-power scaling, causal key admission, cache growth/layout, ordered top-four tie policy, full score-call shape and FP64→FP32→BF16 conversion. Remove only the source-center subtraction and the centers asset. Preserve full causal K/V and the current-query score scan. The numerical state remains1,052,800B at the tested capacity;8,192B of runtime centers and35,840 key-centering scalar subtractions per prompt disappear. This is a source-dependency reduction, not compact-KV closure or a whole-pipeline speed claim.

Neither compilation nor serving may read the original source geometry, basis, landmarks, metrics or centers. A fresh child must enforce those exclusions. The only learned input store is the already compiled complete lexical table; its large offline learned computation and stored bytes remain paid. No new source fit, precision choice, support budget, native reference or noise draw.

Seal both complete raw/BF16 candidate head banks and own H0 before target values open. Compare all supports, selected scores/weights and raw/rounded fields against the current direct reader. Preserve finite reassociation and near-tie outcomes; the real-arithmetic gauge identity does not guarantee finite support or BF16 identity.

For downstream reuse, bind the ultimate actual encoder/image consumer from cycle31 explicitly. Its actual head/H0 and lexical controls must match the new candidates, with unchanged checkpoint, numerical profile, source, clock, schedule and noise. Use cycle32 as the arithmetic comparator and retained evidence chain; do not confuse its newly written lexical ZIP receipt hash with the actual original consumer's receipt. Verify complete demanded values and metadata at the real consumer boundary. Exact input identity permits inherited outputs; changed inputs require the candidate's own original suffix and images.

Begin with one finite guarded mrun CPU head/port-proof transaction, using current measured sizing and actual CPU1 environment. Reserve CUDA only for a changed input requiring a new consumer. A nonmatching input is an exploratory observation and continuation request, not a failed scientific gate. Keep new work, compilation, diagnostics, inherited work and field custody separate; complete arrays remain near compute with full-byte verification and compact publication.

This would leave a plain causal top-four attention program over raw-text lexical operands, with no sampled source geometry. It still does not replace later contextual operand formation, denoising or the pixel decoder. The arbitrary-prompt zero-neural objective remains active and unsolved.
