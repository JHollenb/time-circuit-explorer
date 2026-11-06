# Time Circuit Explorer

An interactive field guide to model computation: native execution, recovered time circuits, and the state that connects one computation to the next.

Two interchangeable model packs explore the frozen R3 Mamba ISA model and FLUX.2 Klein 4B. Both connect an expandable architecture, recorded native evidence, mathematical reconstructions, and the state that makes the next input. Small browser calculations expose the relevant algebra without downloading a model.

The FLUX pack follows a four-pass, 112-transition history from initial noise: joint and single blocks, before/after boundaries, full and matched forecasts, spatial latent previews, BF16 Euler calculations, synthetic attention/SiLU sandboxes, and original terminal images with pixel-error inspection. Open it directly with `?model=flux2-klein-4b`.

The [prediction-only generator](generator/README.md) turns those reconstructed equations into fresh images, without loading the native transformer. Its website section shows original measured outputs and per-stage compute. Start `../common/bin/common exec python generator/server.py` to enable live local generation through mrun; the hosted Site displays saved benchmark examples. Open `?model=flux2-klein-4b#generator`.

The [first measured benchmark](docs/prediction-only-benchmark.md) records 365.55–418.85 seconds of denoiser wall time per image on four configured CPU threads, with native encoding and decoding measured separately.

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
