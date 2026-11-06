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
