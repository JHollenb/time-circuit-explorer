# Time Circuit Explorer

A buildless, browser-local teaching site for the frozen R3 Mamba direct-effect ISA model. The interface links the physical architecture, recorded native values, recovered local mathematics, a compact response law, and own-state architectural feedback.

## Run and verify

```sh
npm run dev     # http://127.0.0.1:4177
npm test        # Node built-in test runner; no install needed
```

Deployable content is `dist/`. No framework, inference engine, CDN, GPU, account system, or backend is required. Sites supplies private hosting. The tracked `.openai/hosting.json` binds the project identity.

## Evidence and numerical scope

Checkpoint: `9399e921776d7a5437c5bbd51bfc1928f09d2ddea444fe2aa1c9cf282baf7f1e`.

The initial numerical microscope is `carry-depth-0` from the compiled-scan experiment. All 24,576 H coordinates, 64 final-mixer outputs, and 64 post-MLP rows are checked against saved observations. The native lane is always a recording. The browser evaluates the source equation in double precision, with a small erf approximation for GELU. Moving a local control recomputes only the declared conditional calculation; it never supplies a fictional native counterfactual.

The amber H curve is recomputed sequentially from zero using the captured factor sequence. A clock edit at the selected row propagates through the remaining conditional recurrence. Expanding the equation shows the equivalent product-of-retentions expansion of all earlier writes. The consumer strip reconstructs the selected mixer row, residual MLP, and register readout including its explicit input-bit copy term. It does not commit custom local edits to the separate compact-program state loop.

The state machine also offers the two recorded four-instruction symbolic-loop paths. It computes its own register/flag/offset trajectory from initial inputs, separately from the saved native states and sealed predictions. The scan/MLP microscope remains the fixed carry-depth-0 recording when that outer-loop context changes; this distinction is shown in the UI. Phase names map conceptual dependencies, not measured execution time.

The compact Boolean law is a finite response hypothesis with 93/96 held words exact. All three exceptions are retained. The frontier factorial panel uses source-qualified streamed equations with native baseline calibration. Flag rules are not uniquely identified, and the site makes no general ISA semantic or reduced-state sufficiency claim.

## Re-export the evidence

From this checkout:

```sh
/Users/jakeholl/domains/common/bin/common exec python scripts/export_evidence.py
```

This reads existing artifacts and a weights-only tensor archive. It makes zero model forwards and does not modify research sources. `dist/data/provenance.json` records SHA-256 hashes of every source. Architectural words are serialized without JavaScript precision loss. Reading notes load on demand and preserve original Markdown, mathematical notation, and Obsidian references.

## Add a model

`dist/models.js` defines ModelPack v1 and validates topology. A pack owns its identity, graph geometry and component metadata, clock dimensions, state vocabulary, lesson descriptions, evidence URLs, and kernel binding. Graph nodes have `id`, `kind`, `x/y/w/h`, `label`, `sub`, `math`, `text`, `dims`, `equation`, and `phase`; edges refer to node IDs. Groups have a rectangle and label. The graph renderer traverses the supplied topology.

Add the new pack to `registry`, with an `adapter` module URL. Each adapter exports `mount(pack, environment)` and returns `{dispose()}` for cleanup. `environment.signal` is aborted when switching packs. `ModelHost` disposes the old adapter, restores the presentation shell, and mounts the new one through the model menu. The host is independent of ISA state shapes, node counts, and clocks.

For another ISA checkpoint with matching evidence contracts, reuse `isa-adapter.js` and provide the new recorded JSON and weights. A different family needs its own numerical/playback adapter: attention state, diffusion scheduler state, and ISA architectural words are different interfaces. Reuse the presentation shell and styling; register family-specific computations instead of forcing the ISA recurrence onto the new model. Small pure ISA kernels live in `kernels.js`.

The model menu intentionally lists only the shipped specimen. The host test mounts and swaps an internal two-node, single-clock fixture, verifies its different clock dimensions, and checks adapter cleanup.

## Teaching references

- [Bartosz Ciechanowski — Gears](https://ciechanow.ski/gears/): direct manipulation that explains an equation.
- [Red Blob Games — A*](https://www.redblobgames.com/pathfinding/a-star/introduction.html): progressive examples and linked algorithm state.
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/): expandable architecture with coordinated numerical views.

No external scripts or model weights are downloaded at runtime.
