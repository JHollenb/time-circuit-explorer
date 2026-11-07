# Time Circuit Explorer

An interactive field guide to model computation: native execution, recovered time circuits, and the state that connects one computation to the next.

Two interchangeable model packs explore the frozen R3 Mamba ISA model and FLUX.2 Klein 4B. Both connect an expandable architecture, recorded native evidence, mathematical reconstructions, and the state that makes the next input. Small browser calculations expose the relevant algebra without downloading a model.

The FLUX pack follows a four-pass, 112-transition history from initial noise: joint and single blocks, before/after boundaries, full and matched forecasts, spatial latent previews, BF16 Euler calculations, synthetic attention/SiLU sandboxes, and original terminal images with pixel-error inspection. Open it directly with `?model=flux2-klein-4b`.

The [prediction-only generator](generator/README.md) turns those reconstructed equations into fresh images, without loading the native transformer. Its website section shows original measured outputs and per-stage compute. Start `../common/bin/common exec python generator/server.py` to enable live local generation through mrun; the hosted Site displays saved benchmark examples. Open `?model=flux2-klein-4b#generator`.

The [first measured benchmark](docs/prediction-only-benchmark.md) records 365.55–418.85 seconds of denoiser wall time per image on four configured CPU threads, with native encoding and decoding measured separately.

The [complete masked-encoder result](docs/complete-masked-encoder-image.md) now follows two fresh left/right prompts through their own complete conditioning and original image consumers. Candidates match all retained computed fields and original PNG bytes while avoiding 146.15 billion nominal encoder linear MACs across the pair (5.24%). The 109.59-second assay still pays original learned attention, MLP, denoiser and VAE work; it establishes no runtime speedup or arbitrary-prompt zero-neural backend.

The [compiled clock result](docs/compiled-flux-clock.md) excludes six learned clock tensors from fresh image serving: 341.31 MB of parameters become 442.37 KB of deployed constants. Both images still match their references byte for byte. This removes only 0.0036% of nominal transformer linear MACs; the live denoiser/VAE remain paid. The 54.53-second assay establishes a bounded compiler reduction, with arbitrary-prompt zero-neural generation still open.

The [raw-prompt rank-one reader](docs/raw-rank-one-image.md) now supplies all32 first-layer heads and follows its own state through complete conditioning to two images. Both preserve the left/right swap, with10.72% signed image-response error despite125.43% conditioning-response error. Original encoder/FLUX/VAE work remains paid; the subsequent zero-reader control below measures its contribution beyond lexical residual input.

The [zero-reader control](docs/zero-reader-image.md) now confirms useful contribution on that pair: deleting the read raises signed PNG response error to104.17% and degrades the objects, while rank-one remains10.72%. Aggregate conditioning-response error ranks them in the opposite order. The subsequent uniform-value reduction below tests a much smaller live state; most learned computation remains paid.

The [uniform causal summary](docs/uniform-reader-image.md) now runs with8,208bytes of numerical state and5.5ms reader arithmetic per prompt. It matches its own law exactly but merges the objects, with122.68%image-response error. Endpoint and row improvements over deletion are retained. The selected-only follow-up below tests the tail's contribution.

The [selected-only reader](docs/selected-reader-image.md) deletes tail moments and envelopes while preserving the selected computation exactly. Both images retain the objects and spatial swap, at 19.37% signed image-response error versus 10.72% with the tail. Comparable allocated numerical state falls from 18.98 MB to 2.10 MB. Full K/V scans and original learned consumers remain paid.

The [direct-score reader](docs/direct-reader.md) now removes anchors, routing and score caches too, reducing comparable numerical state to1.05MB. All demanded BF16 heads, H0, masks and positions match the selected-only parent, so its images are inherited through exact input identity. The CPU assay takes9.93s with no new dense consumers or images; this is known-input result reuse. Runtime centers are8,192B, with full K/V/score scans and learned lexical storage still paid. Arbitrary-prompt zero-neural generation remains open.

The [uncentered reader](docs/uncentered-reader.md) removes the remaining source geometry/centers dependency too. All complete consumed inputs match the actual original consumer on the known pair, qualifying inherited images. Its8.42-second CPU assay performs no new neural consumers or image generation; full K/V, score scans and learned lexical storage remain. The subsequent fresh prompt square below tests new compositional coverage.

The [fresh prompt square](docs/fresh-prompt-square.md) now generates four own candidate images before four new full-attention references. Body colors and spatial order survive across the fixed pitcher/watering-can square, with3.61–9.51%endpoint PNG error and31.99–45.43%signed-edge error. Native/candidate spout defects and large padding-conditioning discrepancies remain visible. The194.34-second assay pays212native encoder blocks,32denoisers and8VAE decodes; arbitrary-prompt zero-neural generation remains open.

The [full-input padding MLP tangent](docs/padding-mlp-tangent.md) tests a source-only local replacement with every receiver coordinate retained. Its four new images show worse color-response fidelity than a constant anchor:91.97%versus61.16%error against full attention. A separate post-seal source diagnostic attributes the direct-write discrepancy to a large nonlinear remainder and signed cancellation, far above measured rounding terms. The183.73-second primary and20.96-second diagnostic retain all original downstream learned work; this is an informative limit of the chosen local law, with the arbitrary-prompt zero-neural goal still open.

The [retained-query encoder](docs/retained-padding-queries.md) now executes 53 original query trajectories and reconstructs the complete conditioning field. Two new images retain coarse vessel colors/order, with 7.23%/11.54% endpoint error and 48.93% color-response error against historical full attention. The 56.47-second assay avoids 89.65% of nominal encoder affine MACs, while original weights, eight denoisers and two VAE decodes remain. Numerical execution and padding reconstruction are separately retained; this is reduced learned work, with arbitrary-prompt zero-neural generation still unresolved.

The [source-native global reader](docs/source-native-global-reader.md) tests one first-joint image attention read plus a discovery-only shared velocity writer at 1.092% of full final-call affine MACs, with attention/fitting/VAE costs separate. Its two images preserve rough structure but distort color/surfaces; hold is better on all feedback/evaluation/main velocity endpoints. The 65.57-second assay uses46reads, one closed-form fit and two VAE decodes, with228passing mechanics tests and full-byte custody. This remains learned computation; arbitrary-prompt zero-neural generation is open.

The [complete-state rotating-write result](docs/rotating-residual-writes.md) preserves all fine rows and full current K/V while updating fewer residual rows. It uses 57.81% of full final-call affine MACs, but both images become grid-like texture: PNG RMS error is 95.44/94.35 versus 9.46/13.06 for hold. The 49.27-second run pays two routed denoisers/two VAE decodes plus original prefixes. Exact carry, own-state handoffs, physical projection slices and224mechanics tests verify the executed program. This fixed approximation is inadequate; arbitrary-prompt zero-neural generation remains open.

The [coarse global innovation result](docs/coarse-global-innovation.md) executes a source-fixed globally conditioned update at 50.12% of full final-call affine MACs. It is worse than holding the old velocity on both cases; about 65% of endpoint error energy is in the coarse producer, with 35% in excluded fine innovation. The corrected run takes 41.24 seconds and pays two coarse denoisers/two VAE decodes plus original prefixes. Numerical failures, actual consumed key mass and complete error terms are retained. This is reduced learned computation; arbitrary-prompt zero-neural generation remains open.

The [two-clock omission/recovery result](docs/two-clock-velocity-factorial.md) shows that the final native read improves signed PNG-response error from36.31% to25.53% after an earlier omission. Keeping the earlier read instead gives17.66% at the same three-call budget. A separate saved-only test finds4.04×/3.74× amplification of incoming latent differences, with80–84% of changed write energy orthogonal to them. The28.83-second primary pays2new denoisers/4VAE; the2.23-second diagnostic uses no new neural evaluations. The simple radial-cancellation law does not explain these cases; arbitrary-prompt zero-neural generation remains open.

The [last-velocity experiment](docs/last-velocity-forecast.md) replaces the fourth denoiser output on its actual own prefix. Hold preserves the parent image much better than the fixed secant: 17.66% versus 54.52% signed PNG response error. Four new VAE-decoded images cost 38.35 scheduler seconds, with zero new encoder/denoiser calls and six historical prefix denoisers still charged. A serving version could remove 25% of denoiser affine work; this replay establishes neither fresh serving speedup nor arbitrary-prompt zero-neural generation.

The [zero-network image response experiment](docs/zero-network-image-response.md) goes further within a supplied scene: a compiled RGB chart renders new doses with no encoder, denoiser or VAE evaluation. It measured 1.075 ms of arithmetic per primary output. The public chart and CLI are runnable with NumPy/Pillow; broad changes interpolate well, while fine contextual effects remain weak. Arbitrary new prompts and seeds remain outside this experiment.

Open `?model=flux2-klein-4b#zero-network` for live browser rendering. Switch among four supplied contexts, move the dose slider, compare tested doses with historical native targets, and inspect current-device timing alongside the original CPU assay. Browser float32 arithmetic reproduces all 16 recorded primary outputs pixel for pixel. Calibration RGB packs load only when the section comes into view; no model runtime is downloaded.

The [arbitrary-prompt research record](docs/zero-neural-research.md) tracks the remaining gap. Two saved-only diagnostics inspect prompt composition at34 boundaries, retaining the prediction seals, native comparisons and target-assisted span analysis. They use no new neural evaluations; each boundary still uses native calibration states, so this is research toward a generator rather than arbitrary-prompt serving.

The [compiled MLP write](docs/compiled-mlp-write.md) eliminates original dense projections online for a supplied32-row input line. Five offline-derived output arrays feed a residual update at about .38ms arithmetic per evaluation. Errors and row heterogeneity remain measured; arbitrary-prompt generation remains open.

The [raw-prompt encoder prefix](docs/raw-prompt-encoder-atoms.md) compiles all 151,936 token-local embedding/QKV operands. Eight frozen raw strings match the isolated native prefix exactly across 56.6 million BF16 scalars, including positions. Serving uses a 2.645 GB learned-output table and no original projections; warm calls take 12.8–13.9 ms, with substantial first-use storage latency. Contextual encoding and images remain open.

The [streaming context-state assay](docs/compiled-context-statistics.md) now tests a 66.6 KB state on raw-text-derived first-layer attention operands. Its finite-feature kernel executes correctly, but whole-head error remains 183% and sampled key probabilities collapse. Retrospective centering helps without fixing the reader. Complete errors, counters and saved-only diagnostics are public; this is research evidence, not a new image generator.

The [source-frozen context reader](docs/source-conditioned-context.md) improves coordinates using offline token/position moments. It accepts eight fresh strings without prompt-fitted parameters and improves broad head error, but several prompt-change errors grow sharply. Padding and active-query countertrends, state costs and original raw evidence remain explicit. Arbitrary-prompt zero-neural image generation is still open.

The [common-bank rank diagnostic](docs/common-bank-context.md) finds a better shared state for three of those four changes. Rank128 errors are20–37%, with a retained Japanese-padding failure. All eight prompts fit the signed factors; this is finite-bank capacity, with no feature rule for unseen queries or image-generation claim.

The [live query-transport reader](docs/query-transport-context.md) adds actual prefix value–key covariance. It improves all four new paired head changes and their measured downstream MLP/state effects, at 8.52 MB of core state per head. The consumer still supplies the other 31 heads and evaluates original learned equations; complete conditioning and zero-neural images remain open.

The [complete attention-bank experiment](docs/complete-attention-bank.md) predicts all 32 heads without reference replacement and consumes all 512 rows through the original first-block equations. Endpoint states improve, but the paired state error remains 116%. A saved-only attribution separates the missing partition and value-weighted exponential terms and their cancellation. The 273 MB logical moment state is about 130 times the BF16 K/V payload; efficiency and arbitrary-prompt zero-neural images remain open.

The [sparse joint-tilt experiment](docs/sparse-joint-tilt.md) replaces that covariance tensor with retained keys and a positive tail. Four-key core state falls to8.55MB; active-row state response improves, but padding remains inaccurate. A separate full-query-score oracle performs much better. The next test freezes support and changes only the tail approximation; this is retrospective component research, with no arbitrary-prompt zero-neural image claim.

The [matched tail-centroid experiment](docs/tail-centroid-tilt.md) runs that comparison at identical state and selected keys. Eligible paired state error improves 126.3%→96.0%; the frozen oracle slightly worsens 41.8%→43.7%. This isolates a useful tail-law change on the saved pair while retaining the remaining support gap and original learned consumer costs.

The [source-witness selector](docs/source-witness-support.md) tests four frozen query witnesses per anchor under that same read. Active state response improves slightly, while padding and overall response worsen. The evidence preserves source compilation, additional storage/work and all countertrends; the next diagnostic separates query coverage from the key-ranking objective. Arbitrary-prompt zero-neural images remain open.

The [current-query selector diagnostic](docs/full-prefix-selector-objective.md) now reduces paired state error95.96%→41.76%, with active error32.85%→8.33%, at the same four-key budget and centroid law. Full causal-prefix access remains paid; signed capture, endpoint/per-head countertrends and original consumer costs are explicit. The subsequent retrieval-cost result is measured below. This does not serve arbitrary-prompt zero-neural images.

The [certified retrieval-cost assay](docs/certified-current-query-retrieval.md) preserves those ordered supports and outputs exactly while skipping about10.6%of score products. Bound/sort overhead makes this implementation much slower; its full FP64 index is also larger than BF16 K/V. This is measured component research, not a zero-neural image backend.

The [fixed-support tail factorial](docs/current-support-tail-factorial.md) reduces paired state error 41.76%→33.46% with exact tail mass, or 33.77% with exact tail value. Value repair improves all 32 paired heads; mass improves only 12 active heads. Signed cancellation and original learned-consumer costs remain explicit.

The [joint positive-quadratic law](docs/positive-tail-moment-closure.md) predicts both tail factors together and reduces paired state error to24.15%, with6.08% active error. All32 pooled heads improve, while five active heads and one padded head worsen. Full-prefix operands and original receiving computations remain paid; the subsequent rank-one state test below measures how much of this gain survives. Arbitrary-prompt zero-neural images remain open.

The [rank-one streaming state](docs/rank-one-tail-state.md) now matches its direct positive-law calculation at every BF16 output, with no fallbacks. It retains less of the full-direction gain: paired state error is39.02%, versus41.76% centroid and24.15% full quadratic. Core moments, safety envelopes, full retrieval index and original receiving computations remain paid. This separates representation limits from streaming arithmetic; arbitrary-prompt zero-neural images remain open.

The [omitted-score diagnostic](docs/rank-one-omission-diagnostic.md) factors that gap into signed mass and value transport without model or consumer evaluation. A mixed value term points toward lower head error in all32 heads, while finite-dose countertrends remain. Full-direction inputs are diagnostic resources; no corrected serving state or image is claimed.

The [conditional reuse and composition result](docs/sign-composition-boundary.md) verifies exact sign reuse on support-closed rows and a missed response on mixed rows. A new saved-state assay finds that exact one-block reuse still leaves the output outside the sign and row-scaling families: 98.34% of its discrepancy energy survives the best per-row scale. Complete metadata, row accounts and computation costs are public. This constrains rule chaining; arbitrary-prompt zero-neural images remain open.

The [transformed-QKV result](docs/cached-qkv-reuse.md) predicts mixed-support updates exactly while omitting three input projections. The [first-prompt mask-aware assay](docs/masked-kv-own-state.md) then starts from fresh raw strings and advances its own H1→H2. It avoids 2.538 billion nominal K/V MACs on the short prompt, with 0.385% endpoint state error from changed projection geometry. Every query and MLP row remains. These are compiler reductions with substantial learned work still present.

The [fixed accumulation-policy follow-up](docs/kv-accumulation-profile.md) preserves that saving while lowering H2 error to 0.183%. Full formation remains exact; compact formation improves on 500/512 rows, with twelve padding reversals retained. The next result follows complete conditioning and its actual first FLUX projection; arbitrary-prompt zero-neural image generation remains open.

The [complete conditioning consequence](docs/conditioning-demand-consequence.md) carries three actual H2 states through H9/H18/H27 and the native FLUX context projection. C error improves 15.76%, with an H18 pooled reversal and 165 C rows worse. All remaining learned work is charged.

The [actual image consequence](docs/conditioning-image-consequence.md) now lowers PNG RMS 8.73% while increasing MAE 8.11%; most pixels worsen by squared error to full. The site includes an original-image divider and fixed-scale error view. Three same-noise native images cost 41.29 scheduler seconds, 3.42 GB sampled RSS and 9.17 GB VRAM; twelve denoiser calls and three VAE decodes remain paid. This supplies no arbitrary-prompt zero-neural backend.

The [source-only landmark reader](docs/source-landmark-context.md) supplies an explicit rule for eight new strings, with64live positive prefix statistics. Its initial mixture is inaccurate. Saved-only routing/capacity diagnostics distinguish excessive smoothing from the added restriction of fixed convex response mixing; target-assisted fitted weights are not serving assets. Contextual/image completion remains open.

The [compiled state reader](docs/compiled-state-reader.md) now consumes that writer's actual BF16 state without a dose input, recovering coordinates and the next layer's Q/K/V affines in about .93ms for32rows. It uses learned reduced banks, with native and per-row errors retained. A saved-array follow-up separates basis omission, rounding, numerical policy and inherited writer error. This is one supplied-context edge, before attention.

The [global-head continuation](docs/compiled-global-head.md) carries those actual-state operands through head0's complete1536-query attention read in about1.8ms. Incremental and full mathematical attention have identical BF16 outputs; paired prediction error remains35.1% after rounding. Matched value-delta controls expose address sensitivity. Frozen context, learned maps and independent-arithmetic limits remain explicit; the next proposed cut starts at [raw-prompt encoder operands](docs/plans/raw-prompt-encoder-atoms.md).

## Run locally

Requires Node.js 20 or newer. There are no package dependencies to install.

```sh
npm run dev     # http://127.0.0.1:4177; rebuilds on source changes
npm test        # numerical evidence and model-host checks
npm run build  # creates the ignored root dist/ directory
```

Refresh the browser after editing source. The generated `dist/` directory can be served by any static host; the existing Sites configuration uses that directory.

## Repository layout

```text
time-circuit-explorer/
├── viz/
│   ├── src/          # shared components, templates, model host and family adapters
│   ├── public/data/  # recorded numerical evidence and research notes
│   ├── scripts/      # static build, local server and offline evidence exporter
│   └── tests/        # dependency-free Node tests
├── docs/plans/       # plans for additional visualized models
├── generator/        # prediction-only worker, mrun launcher and local live API
├── .openai/          # existing private Site deployment identity
└── package.json      # root workspace commands
```

See [visualization documentation](viz/README.md) for numerical scope, evidence provenance, and model-pack interfaces. The [FLUX pipeline plan](docs/plans/flux-pipeline.md) records the implemented design and acceptance checks.

The [hosted explorer](https://time-circuit-explorer.jholl-sub.chatgpt.site) currently requires its owner's ChatGPT sign-in. This Git repository is public; Site access is managed separately.
