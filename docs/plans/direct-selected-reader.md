# Reuse the current-query scores and remove the anchor machinery

Status: cycle31 Astra-selected next cut; not staged or executed. The [matched tail deletion](../selected-reader-image.md) retains useful images with no tail state, at19.37% signed PNG response error. Its supports, routes, logits, weights and selected read match the frozen rank-one parent exactly. Now remove machinery that is algebraically redundant once only the selected read remains.

Keep exactly the same raw A/B strings, complete lexical source/template/positions/mask, source-centered/scaled keys U, scaled query x, value cache V and causal key order. Preserve the existing full-prefix GEMV and ordered top-four selector:

$$
g=U_{:n}x,\qquad S=\operatorname{top4}(g),\qquad
H_{\mathrm{direct}}=\operatorname{softmax}(g_S)^T V_S.
$$

Reuse the very same g vector used for selection; do not introduce another selected-score dot product or change key centering, scaling, layout or tie arithmetic. The current reader uses a_j=ell^T u_j+(x−ell)^T u_j, which equals g_j in real arithmetic. FP64 evaluation may differ, so this is a new fixed numerical contrast, not assumed byte equivalence.

Export an8,192-byte centers-only asset from the frozen source geometry, with original asset hash and extraction provenance. The serving child must refuse the original landmark/metric asset; loading it and ignoring most arrays does not establish this exclusion. Remove all landmarks, metrics, nearest routing, anchor admission GEMVs, anchor-score cache and selected delta-dot work. Retain full actual K/V, positions/count/clock, the current-query full-prefix score scan, selected softmax/value read and inherited FP64→FP32→BF16 conversion. At the tested capacity, the comparable numerical index plus count/clock becomes1,052,800 bytes; charge source identities/counters, input/output arrays, full lexical compilation, scratch and diagnostics separately. No compact-KV claim.

Seal both complete raw/BF16 candidate head banks and own raw H0 before opening current selected-parent inputs. Then compare their complete BF16 bits, H0 bits, shapes, identities and numerical/source contracts. Keep raw finite score/weight/head changes as observations, including changes erased by FP32/BF16 rounding.

If complete demanded BF16 head banks and H0 match, bind the existing qualified downstream computation through those exact input ports and inherit its already retained encoder/image evidence. Do not rerun dense consumers merely to rediscover deterministic equality. This is exact input-bound result reuse on known prompts/noise, not fresh arbitrary-prompt zero-neural image synthesis. Bind checkpoint/profile, consumer source, clock, schedule, noise and all reused fields; report inherited work separately from new work.

If either candidate input differs, advance only this frozen arm through the unchanged original block0 tail, own full512-row encoder conditioning and original four-step image/VAE consumer. Seal resulting outputs before full/rank/zero/uniform image scoring. Preserve per-case outcomes, support/score/weight differences, endpoint and signed responses, and q-power/cross terms. No refit, support/rank/precision sweep, new references, noise draws, exact-head replacement or native-state refresh. Accuracy is an observation, not an exploratory abort condition.

Run the smallest source-bound transaction through mrun. Begin with the CPU head/port-proof work; reserve CUDA only if a changed input needs new learned consumers. Reuse a compatible measured envelope only with its stated geometry and scope, retain complete fields near compute, verify full bytes and publish compact evidence. Runs expected above15minutes require the workspace's Luna code/resource review.

Future derived producer text must use its actual direct-score K/V state and cost account. Do not copy the historical uniform-mean/8192-byte limitation prose identified by the [cycle31 correction](../results/selected-reader-image-evidence/report-correction.json). The executed cycle31 worker and raw report remain frozen.

This tests whether source routing and anchor caches can be eliminated from a useful reader. Later encoder blocks, denoising and pixel decoding remain unresolved parts of the arbitrary-prompt zero-neural goal.
