# Prediction-only FLUX generator

This generator makes a fresh 512 × 512 image from a prompt and seed. It evaluates the recovered FLUX.2 Klein 4B computation through all four denoising passes, feeding each predicted state into the next computation. The native transformer is never constructed; calling its forward method raises an error.

The native text encoder supplies conditioning and the native VAE renders the final predicted latent. This remains a full dense learned-weight calculation, using the original parameter banks and pinned NumPy equations with FP64 reductions and explicit BF16 boundaries. It is an executable reconstruction, with no compact student or claimed speedup.

The separate [zero-network response chart](../docs/zero-network-image-response.md) eliminates all neural evaluation for a supplied context/dose family. Its public RGB bundle and `render_response.py` need only NumPy and Pillow. It interpolates a known scene, with measured limitations on small contextual effects; it does not replace this prompt/seed generator.

Arbitrary prompts are supported by the full reconstructed generator, with dense learned-weight computation and native encoding/decoding. Arbitrary prompts with **zero neural evaluations** remain open. The [research record](../docs/zero-neural-research.md) documents that distinction and the first prompt-composition diagnostics.

The [sign-composition assay](../docs/sign-composition-boundary.md) now measures whether a qualified conditional shortcut can feed another shortcut. Exact one-block outputs leave the row-scaling family; the next consumer needs the actual emitted state. `sign_composition_study.py stage|submit|collect` runs the saved-only CPU assay, and `publish_sign_composition.py` verifies retained arrays and publishes compact reports. It performs no new learned computation and supplies no arbitrary-prompt image backend.

The [complete-vocabulary lexical compiler](../docs/raw-prompt-encoder-atoms.md) now accepts raw strings and serves the encoder's first embedding/QKV/position operands without original projections or a neural runtime. It memoizes learned outputs for all 151,936 IDs, with 2.645 GB of atoms retained near compute. Eight frozen strings exactly matched the isolated native prefix; contextual attention, later encoder layers and images are outside that result.

The [context-state experiment](../docs/compiled-context-statistics.md) tests the first contextual read after that prefix. A positive-feature streaming state closes against its own kernel, but not against exact attention: rank64 median head error is183%. `context_statistics_study.py stage|submit|collect` runs the source-bound assay; `context_kernel_diagnostics.py` and `context_key_diagnostics.py` analyze sealed saved arrays. Diagnostics retain key collapse and lookahead-fitted conditioning separately from serving. They are not an image-generation backend.

The [source-conditioned follow-up](../docs/source-conditioned-context.md) compiles one coordinate geometry from token/position atoms and tests eight fresh strings through actual prefix statistics. `context_diagonal_diagnostics.py` runs the preceding saved-only geometry assay; `context_source_study.py stage|submit|collect` runs compilation, fresh serving and post-seal references. The compiler includes mixture between-mean variance; serving retains both gauge and original-gauge approximate log partitions. Broad reconstruction improves, but prompt-change errors remain large. This is not a faithful full encoder or image-generation backend.

The [common-bank capacity assay](../docs/common-bank-context.md) fits one globally normalized kernel/SVD across all eight saved prompts. `context_common_study.py stage|submit|collect` runs the finite CPU job; `context_common_bank.py` advances signed prefix statistics in a fresh process. It improves three paired effects but has signed-read failures and no unseen-query rule. Public factors are model-derived fitted information, not a zero-neural generator.

The [source-frozen landmark reader](../docs/source-landmark-context.md) replaces fitted query coordinates with a source-only covariance/routing law. `context_landmark_study.py stage|submit|collect` compiles64landmarks and seals eight new raw-string forecasts; `context_landmarks.py` advances the positive prefix state. `landmark_routing_diagnostics.py` separately tests bandwidth and target-assisted padding-response capacity on saved arrays. Those oracle fits do not serve images or new prompts. Warm timings charge all four reader controls together.

The [live query-transport experiment](../docs/query-transport-context.md) adds stable causal value–key cross moments to those source-selected readers. `context_transport_study.py stage|submit|collect` seals fresh raw-string head forecasts; `transport_consumer.py stage|submit|collect` separately measures the original output/RMS/MLP consumer with 31 exact other heads. Both improve the four paired effects at their measured scopes. This is component research, not the website's arbitrary-prompt image backend. The next proposed cut predicts the [complete attention bank](../docs/plans/complete-attention-bank.md).

The [complete attention-bank study](../docs/complete-attention-bank.md) extends prediction to all 32 first-layer heads, with no exact replacement heads. `attention_bank_study.py stage|submit|collect` freezes one fresh raw-string pair, compiles source-only banks, seals all candidate heads, then evaluates the original block consumer on all 512 rows. `attention_remainder.py stage|submit|collect` separately attributes its already-scored error to exact partition/value remainders and BF16 defects, with baseline-assisted one-address controls. Neither is a new zero-neural image backend.

The [sparse joint-tilt study](../docs/sparse-joint-tilt.md) tests a positive top-m/tail state without the covariance tensor on that retained pair. `sparse_tilt_study.py stage|submit|collect` seals all32head banks before its separate current-prefix oracle and original fullblock consumer; `publish_sparse_tilt.py verify|publish` authenticates retained bytes. Four-key core state is8.55MB, but padding response remains poor. The [next plan](../docs/plans/tail-centroid-tilt.md) compares tail laws with support frozen. These are component experiments on supplied ports, not the site's arbitrary-prompt image backend.

The [matched tail-centroid study](../docs/tail-centroid-tilt.md) reuses the previous eligible/oracle support sets without reranking. `centroid_tilt_study.py stage|submit|collect` seals its own eligible unit/centroid banks before frozen-oracle/original-consumer work. `publish_centroid_tilt.py verify|publish` checks retained bytes. It improves eligible state response at unchanged core bytes but remains retrospective component research, not an arbitrary-prompt image backend.

The [source-witness study](../docs/source-witness-support.md) compiles source-cell witnesses from retained lexical atoms. `source_witness_study.py stage|submit|collect` compares own-state anchor/witness selectors under the same centroid read, sealing both before reference/consumer work. `publish_source_witness.py verify|publish` also authenticates touched atom shards. The result is heterogeneous and overall worse; this is retrospective component research, not an arbitrary-prompt backend.

The [full-prefix selector diagnostic](../docs/full-prefix-selector-objective.md) supplies the actual query to four-key selection. `selector_objective_study.py stage|submit|collect` seals current-query/envelope supports and raw/BF16 banks before signed correction diagnostics and the original receiving consumer. `publish_selector_objective.py verify|publish` authenticates complete near-compute arrays and publishes lossless diagnostic/observation compression. Current-query selection improves the retained paired state response substantially, but full-prefix scans are diagnostic work; the subsequent [retrieval-cost assay](../docs/certified-current-query-retrieval.md) preserves its support but is slower. This is not the website's arbitrary-prompt image backend.

The [certified retrieval assay](../docs/certified-current-query-retrieval.md) uses source-fixed coordinate blocks and outward score intervals, with the original full-scan fallback. `certified_retrieval_study.py stage|submit|collect` seals a causal index/read bank before fresh full-scan and historical comparisons; `publish_certified_retrieval.py verify|publish` authenticates complete retained bytes. Identical banks inherit the existing original-consumer result. It avoids about10.6%of score products but is slower in this implementation. Full KV storage and bound/read costs remain; no strictzero-neural image claim.

The [fixed-support tail factorial](../docs/current-support-tail-factorial.md) crosses exact/centroid tail mass with exact/anchor tail value on the useful ordered current-query top4. `tail_factorial_study.py stage|submit|collect` seals complete raw/BF16 banks and signed factors before old outputs or original consumer parameters; `publish_tail_factorial.py verify|publish` authenticates retained evidence. Mass and value repairs both improve pooled paired state response, with broader head improvements for value and signed cancellation retained. Four new learned receiving consumers are offline diagnostic work, not the site's arbitrary-prompt image backend.

The [positive-quadratic follow-up](../docs/positive-tail-moment-closure.md) predicts tail mass/value with one positive measure, reducing paired receiving-state error to 24.15%. `positive_tail_study.py stage|submit|collect` seals the sole new bank before exact/negative-tail/Gaussian diagnostics and two original receiving consumers; `publish_positive_tail.py verify|publish` authenticates retained arrays and lossless descriptions. `plot_positive_tail.py` renders published scalar records only. The [next plan](../docs/plans/rank-one-tail-state.md) tests a source-defined streaming mixed-moment state; current full-prefix operands remain supplied and this is not a zero-neural image backend.

The [rank-one state follow-up](../docs/rank-one-tail-state.md) implements source-only coordinate compilation and actual causal mixed moments in `rank_one_tail.py`. `rank_one_tail_study.py stage|submit|collect` seals complete banks/states before independent direct-law references and original consumers; `publish_rank_one_tail.py verify|publish` authenticates source shards and retained arrays. Serving/direct BF16 outputs match with no fallbacks, while paired state error rises to39.02% from the full-direction law's24.15%. `plot_rank_one_tail.py` renders published scalars only. This is component research, not the website's arbitrary-prompt image backend.

The [omitted-score diagnostic](../docs/rank-one-omission-diagnostic.md) expands the fixed rank-one/full-quadratic difference into signed mass and value terms. `rank_one_omission_study.py stage|submit|collect` runs the saved-array CPU assay; `publish_rank_one_omission.py verify|publish` authenticates complete arrays and lossless observations. No original parameters, consumers or model calls run. `plot_rank_one_omission.py` renders published scalars. The useful mixed-value term is diagnostic, with full-direction inputs still supplied; it is not a serving correction.

## Generate from the website

From this repository in the domains workspace, with the configured mrun client and reachable fleet:

```sh
../common/bin/common exec python generator/server.py
```

Open `http://127.0.0.1:4177/?model=flux2-klein-4b#generator`. Enter a prompt and seed, then generate. The loopback server submits a finite mrun lease, streams transition progress, collects the original result, and serves its PNG and measurement report. Credentials stay in the server process. Only one generation may be active through this server. Request IDs and job ownership survive a server restart; ambiguous launches require inspection rather than automatic resubmission.

The hosted static Site displays saved measured examples. It cannot execute the model; its form explains how to start the local live server. The live server does not expose a remote inference endpoint.

## Generate from the command line

```sh
../common/bin/common exec python generator/run.py stage artifacts/my-prediction \
  --prompt 'A blue glass sphere beside an orange cube on a white background.' --seed 2718
../common/bin/common exec python generator/run.py submit artifacts/my-prediction
../common/bin/common exec python generator/run.py status artifacts/my-prediction
../common/bin/common exec python generator/run.py collect artifacts/my-prediction
```

Each run directory is unique. Submission is separate from staging so its exact source hashes, request identity, physical geometry, resource calibration and execution contract are reviewable. Existing jobs are inspected, never resubmitted. Collection verifies scheduler payload custody and authenticated remote-before, remote-after, and local file inventories. Model work runs through mrun on CUDA-capable Beast; no real model runs on the development machine.

New submissions use mrun's guarded payload path and strict preflight. The hard timeout is 25 minutes for one image or 30 minutes for two; planning estimates are 1,000/1,500 seconds. These budgets include the measured cold runtime cost. The first benchmark used ordinary admission: its declared/sealed payload matches and its input files were independently hashed on the authenticated worker before and after collection, but mrun did not report an execution-time payload seal. That absence remains explicit in the published provenance; post-run hashes do not create a retroactive attestation.

Use `stage ... --benchmark` for the two serial benchmark specimens. Export only a successfully collected result with:

```sh
python3 generator/publish_benchmark.py artifacts/my-benchmark
npm run build
```

## What is computed

For fixed conditioning `c`, each pass computes `v = D(z, c, t)` through projection, 5 joint blocks, 20 single blocks and readout. Then the scheduler commits `z_next = BF16(z + BF16(dt × v))`, with the qualified arithmetic boundaries. Four passes contain 112 explicit transitions. Attention, gated MLPs, normalization, residual paths, modulation and parameter projections all run from the reconstructed equations. No intermediate native activations or reference image are supplied.

Only current working registers remain resident. The worker saves final BF16 latent bits, original PNG, per-pass latent hashes, progress events and resource measurements. It releases the encoder after conditioning; a subsequent sample reloads the same encoder, charged to that sample's encoding stage.

## Measurement meanings

- **Prediction wall time:** complete CPU denoiser, including bank reads, hash checks and own-state advancement; excludes encoder, VAE, startup and queue wait.
- **Image wall time:** encoding/transfers, initial noise/schedule, prediction, VAE/transfers and file save after initial setup.
- **Startup:** package imports, pinned-checkpoint verification and initial encoder/VAE loading. The first image including startup is reported separately.
- **CPU seconds:** process CPU time summed across all threads. Four configured threads can consume several CPU seconds per elapsed second. This is not FLOPs or energy.
- **RAM:** process RSS sampled every 100 ms for stages; the report also includes the process high-water mark. Image cards exclude setup-only peaks.
- **CUDA allocation/reservation:** PyTorch allocator counters at synchronized stage boundaries. These exclude driver/context allocations and are distinct from scheduler device telemetry. The CPU denoiser does not perform CUDA matrix computation.
- **Parameter bytes:** logical BF16 bytes read from parameter files, which may be served by the OS page cache. They are not a measurement of physical disk traffic.

Timings depend on cache state and other fleet work. Queue wait, mrun staging and collection are excluded from worker timing. No energy measurement, FLOP count or controlled native speed comparison is claimed.

The qualified prompt/seed must reproduce the prior sealed mathematical forecast's initial noise, conditioning, terminal latent and PNG exactly. That check certifies extraction fidelity to that forecast. A new prompt is an actual new own-state generation; its agreement with native CUDA execution is unmeasured unless separately compared.

## Saved-only prompt-composition diagnostics

`prompt_closure_study.py` stages three source-bound native prompt histories and a separately scored fourth corner. Its prediction child seals204 complete candidate-array hashes before target-exclusive file reads. Six fixed composition/control rules are compared at34 independently supplied boundaries; these are not a connected state rollout. `prompt_span_study.py` separately fits the best two-direction plane using the target, explicitly as a capacity diagnostic with no serving claim.

The source is an existing authenticated campaign; these workers execute no model, original dense weights, encoder or VAE. They still use historical native states and rendered images. Stage and submit once through the workspace's configured mrun environment:

```sh
../common/bin/common exec python generator/prompt_closure_study.py stage artifacts/my-closure \
  --source ../artifacts/mdb-flux-prompt-factorial-20261105/run-01
../common/bin/common exec python generator/prompt_closure_study.py submit artifacts/my-closure
# Inspect the returned job; collect only that job after success.
../common/bin/common exec python generator/prompt_closure_study.py collect artifacts/my-closure
../common/bin/common exec python generator/prompt_span_study.py stage artifacts/my-span --parent artifacts/my-closure
../common/bin/common exec python generator/prompt_span_study.py submit artifacts/my-span
../common/bin/common exec python generator/prompt_span_study.py collect artifacts/my-span
```

These launchers require the retained Beast source paths and the qualified worker environment. The checked-in [results](../docs/results/prompt-composition/provenance.json) retain the original requests, raw reports, predictions, controls and custody subset without copying large native state arrays into this repository. Runtime figures are analysis costs, not generator latency.

## Coefficient-only MLP write

The [compiled MLP experiment](../docs/compiled-mlp-write.md) removes original dense projections from serving one supplied32-row input line. Five offline-derived arrays feed a fixed supplied residual update. It does not interpret a prompt or render an image.

## Compiled state-only receiver

The [next-reader experiment](../docs/compiled-state-reader.md) recovers coordinates and exact normalization scale from the producer's emitted BF16 state, with no dose field or native state refresh. It serves the next joint layer's pre-head-normalization, pre-RoPE Q/K/V affines through three learned reduced banks. It calls no neural framework and evaluates no original dense projection online; learned-map evaluation remains part of its arithmetic.

`state_receiver_study.py stage|submit|collect` runs the source-bound compiler, fresh state-only serving and six post-seal native CPU references in one finite mrun lease. `state_receiver_native.py` is deliberately a separate entry point: serving helpers prohibit importing neural frameworks. `state_reader_attribution.py stage|submit|collect` subsequently analyzes only saved arrays, including a same-final-quantizer full-projection counterfactual.

The checked-in reader needs NumPy/Pillow and one [1,32,3072] array of FP32-widened BF16 state bytes:

```bash
python generator/render_state_receiver.py \
  --bundle docs/results/compiled-state-reader \
  --state docs/results/compiled-state-reader/state-22185a19edaa2661.npy \
  --output artifacts/my-reader-output
```

The supplied midpoint packet reproduces this bounded input. Accuracy for arbitrary states, contexts or prompts is unmeasured. Output is Q/K/V and residual diagnostics, not an image.

## Compiled global head

The [global-head experiment](../docs/compiled-global-head.md) advances those state-fed Q/K/V through learned head normalization, captured RoPE and complete1536-query head0 attention. Its fixed-source cache excludes all32 changed rows; changed queries read every updated key. The same-arithmetic full calculation verifies the partition identity, while prediction errors and region/row counter-trends remain separate.

```bash
python generator/render_global_head.py \
  --bundle docs/results/compiled-global-head \
  --state docs/results/compiled-global-head/state-22185a19edaa2661.npy \
  --output artifacts/my-global-head
```

The portable utility requires NumPy/Pillow and returns a[1536,128] head field plus computational controls. It accepts this supplied state geometry, not a raw prompt. `global_attention_study.py stage|submit|collect` handles the finite source-bound experiment; `global_value_address.py` runs the saved-only matched value-delta address diagnostic. `publish_global_attention.py` preserves collected metadata/banks and labels derived score projections and omitted arrays.

No neural framework runs in these experiments. Offline fixed-context preparation still evaluates six selected original-weight projections; online state evaluation uses three reduced learned maps and two learned head scales. There is no native Flash/output-projection/decoder claim. The [next input-frontier plan](../docs/plans/raw-prompt-encoder-atoms.md) targets the real Qwen3 token-local prefix, with complete-vocabulary design and explicit smoke coverage.

```sh
../common/bin/common exec python generator/render_compiled_mlp.py \
  --alpha .5 --output artifacts/my-compiled-write
```

The published3.75MiB coefficient payload and residual/gate context need only NumPy/Pillow. Native outputs and original parameters are absent from primary serving inputs. Native comparisons exist at0,.5,1 only. The result record retains per-row counter-trends, BF16 differences and compilation/serving costs.

## Provenance and checks

`kernel-manifest.json` pins the original accepted forecast sources, parameter recipes and model-file hashes. `NOTICE.md` and `UPSTREAM-LICENSE` retain attribution and license. `prepare.py` can reproduce those files from the identified immutable campaign run; it is not needed for ordinary generation. Model weights remain external and are not included in Git or the website.

```sh
../common/bin/common exec python -m unittest discover -s generator -p 'test_*.py' -v
npm test
```

Mechanics tests use tiny synthetic banks and stubbed orchestration; they do not execute a pretrained model. The real benchmark report is separate evidence of full generator execution.
