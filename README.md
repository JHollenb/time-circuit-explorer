# Time Circuit Explorer

An interactive field guide to model computation: native execution, recovered time circuits, and the state that connects one computation to the next.

The first model pack explores the frozen R3 Mamba ISA model with recorded native evidence and small browser calculations. It includes an expandable architecture, scan and MLP mathematics, independent own-state playback, and retained counterexamples. The model host supports additional packs and family-specific adapters.

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
│   ├── src/          # HTML, CSS, browser modules, model host and ISA adapter
│   ├── public/data/  # recorded numerical evidence and research notes
│   ├── scripts/      # static build, local server and offline evidence exporter
│   └── tests/        # dependency-free Node tests
├── docs/plans/       # plans for additional visualized models
├── .openai/          # existing private Site deployment identity
└── package.json      # root workspace commands
```

See [visualization documentation](viz/README.md) for numerical scope, evidence provenance, and model-pack interfaces.

The [hosted explorer](https://time-circuit-explorer.jholl-sub.chatgpt.site) currently requires its owner's ChatGPT sign-in. This Git repository is public; Site access is managed separately.
