# Time Circuit Explorer

An interactive field guide to model computation: native execution, recovered time circuits, and the state that connects one computation to the next.

Two interchangeable model packs explore the frozen R3 Mamba ISA model and FLUX.2 Klein 4B. Both connect an expandable architecture, recorded native evidence, mathematical reconstructions, and the state that makes the next input. Small browser calculations expose the relevant algebra without downloading a model.

The FLUX pack follows a four-pass, 112-transition history from initial noise: joint and single blocks, before/after boundaries, full and matched forecasts, spatial latent previews, BF16 Euler calculations, synthetic attention/SiLU sandboxes, and original terminal images with pixel-error inspection. Open it directly with `?model=flux2-klein-4b`.

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
├── .openai/          # existing private Site deployment identity
└── package.json      # root workspace commands
```

See [visualization documentation](viz/README.md) for numerical scope, evidence provenance, and model-pack interfaces. The [FLUX pipeline plan](docs/plans/flux-pipeline.md) records the implemented design and acceptance checks.

The [hosted explorer](https://time-circuit-explorer.jholl-sub.chatgpt.site) currently requires its owner's ChatGPT sign-in. This Git repository is public; Site access is managed separately.
