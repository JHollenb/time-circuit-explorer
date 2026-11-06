# Advance the emitted state into the next QKV reader

Status: implemented and measured in job `job-2f0a52357bf1`, followed by saved-array attribution job `job-15dea027e760`. See [results and limitations](../compiled-state-reader.md). The design below records the experiment's intended scope; the arbitrary-prompt goal remains unchanged.

## Actual computational edge

The producer emits the selected image rows after pass 2/joint block 1. The next joint block's image normalization/modulation and QKV projection are row-local until attention mixes positions. Use that actual edge rather than a second independent dose chart. Verify the native implementation, parameter names, numeric boundaries and available context before staging.

The receiver's public input is the producer's actual BF16 state plus its frozen context/bank identity. It receives no α, target state, native refresh or projected smooth substitute. Its small coordinates must be recovered from the emitted state itself.

## Compile a reader of a state basis

Let u be an emitted row, v=u−mean(u), and s its actual normalization scale. Choose a producer-derived centered basis B offline, retaining its construction, dimension, Gram spectrum and pseudoinverse cutoff. Recover coordinates from v and keep

\[
v=Bz+r.
\]

For a smooth row-local modulation D,b and dense projection W, the exact algebra is

\[
q=\frac{WDBz}{s}+Wb+\frac{WDr}{s}.
\]

Compile WDB and Wb offline; serve the reduced projection from recovered z and actual s. The last term is the omitted contribution, not permission to call r zero. Retain r's full diagnostics and measure its projected effect offline against the original operator. Declare all differences introduced by native LayerNorm, modulation, projection and BF16 boundaries. Do not silently replace the actual emitted state by an unrounded polynomial evaluation.

Basis construction must not use native target outputs. Start with the supplied producer coefficient directions, parent and modulation, with numerical rank disclosed. A basis augmented after viewing target residuals belongs to a separate development experiment. Do not fit new target-dependent coordinates or quietly refresh the state between stages.

## Separate two defects through the same receiver

For existing doses 0,.5,1, serve and seal the composite prediction P=Rhat(UP). Afterward, run six bounded native receiver references: R(UP) on the predicted states and R(UT) on the native-write-derived states. This gives

\[
P-R(U_T)=[P-R(U_P)]+[R(U_P)-R(U_T)].
\]

The first term measures receiver approximation on the same supplied input; the second is inherited writer error. Apply the identity to endpoints and paired changes using each arm's own baseline. Retain signed cross terms and per-row results; neither component norm is a positive causal percentage. UT remains the fixed-parent native-write reconstruction from the first assay, not an observed complete-block midpoint history.

Candidate serving must not execute original dense projections or neural modules. Offline compilation and post-seal native diagnostic work are real research costs and must be counted. Acquire the selected component once, batch equal-shape rows where numerical qualification permits, use one finite guarded mrun lease, strict preflight and no automatic retries. Keep all valid outcomes and null mechanics failures; do not gate exploration on accuracy.

## Evidence that would count as progress

The receiver consumes the exact emitted state bytes; no dose or reference-state field is admitted by its API. Complete forecasts seal before native receiver calls. Original parameter recipes and state custody remain bound, and numerical-policy differences are explicit. Report basis rank, coefficient/consumer bytes, compiler work, fresh serving wall/CPU/RSS, residual/error distributions and the six native diagnostic costs separately.

This would qualify one bounded own-state edge. Attention, text/image mixing, evolving context, raw-prompt operand formation and pixel decoding would still be open. The newest [image-futures post](/Users/jakeholl/domains/obsidian/blog/2026-11-05-225618-testing-state-equations-against-fresh-image-futures.md) motivates separating inherited and same-input defects; [the reader/stream post](/Users/jakeholl/domains/obsidian/blog/2026-11-05-230617-a-reader-can-predict-a-change-without-carrying-its-main-effect.md) cautions that exact local computation alone does not identify the dominant behavioral carrier.
