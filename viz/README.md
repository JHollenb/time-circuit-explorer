# Time Circuit Explorer visualization

A browser-local teaching site for the frozen R3 Mamba direct-effect ISA model and FLUX.2 Klein 4B distilled. The interface links the physical architecture, recorded native values, recovered mathematics, and each model's state feedback.

## Run and verify

```sh
npm run dev     # from the repository root or viz/; http://127.0.0.1:4177
npm test        # Node built-in test runner; no install needed
npm run build  # copies viz/src and viz/public to root dist/
```

Source code lives in `viz/src/`, and recorded evidence lives in `viz/public/data/`. The build copies these into the ignored root `dist/` directory. No framework, inference engine, CDN, GPU, account system, or backend is required. Sites supplies private hosting. The tracked root `.openai/hosting.json` binds the existing project identity.

## Evidence and numerical scope

Checkpoint: `9399e921776d7a5437c5bbd51bfc1928f09d2ddea444fe2aa1c9cf282baf7f1e`.

The initial numerical microscope is `carry-depth-0` from the compiled-scan experiment. All 24,576 H coordinates, 64 final-mixer outputs, and 64 post-MLP rows are checked against saved observations. The native lane is always a recording. The browser evaluates the source equation in double precision, with a small erf approximation for GELU. Moving a local control recomputes only the declared conditional calculation; it never supplies a fictional native counterfactual.

The amber H curve is recomputed sequentially from zero using the captured factor sequence. A clock edit at the selected row propagates through the remaining conditional recurrence. Expanding the equation shows the equivalent product-of-retentions expansion of all earlier writes. The consumer strip reconstructs the selected mixer row, residual MLP, and register readout including its explicit input-bit copy term. It does not commit custom local edits to the separate compact-program state loop.

The state machine also offers the two recorded four-instruction symbolic-loop paths. It computes its own register/flag/offset trajectory from initial inputs, separately from the saved native states and sealed predictions. The scan/MLP microscope remains the fixed carry-depth-0 recording when that outer-loop context changes; this distinction is shown in the UI. Phase names map conceptual dependencies, not measured execution time.

The compact Boolean law is a finite response hypothesis with 93/96 held words exact. All three exceptions are retained. The frontier factorial panel uses source-qualified streamed equations with native baseline calibration. Flag rules are not uniquely identified, and the site makes no general ISA semantic or reduced-state sufficiency claim.

## Re-export the evidence

From this checkout:

```sh
../common/bin/common exec python viz/scripts/export_evidence.py --workspace ..
```

Run that command from the repository root inside the research workspace. Outside that workspace, use its `common exec` launcher and an explicit `--workspace` path. Re-exporting requires the existing research artifacts and a Torch environment; running, building, and testing the website do not. The exporter makes zero model forwards and does not modify research sources. `viz/public/data/provenance.json` records SHA-256 hashes of every source. Architectural words are serialized without JavaScript precision loss. Reading notes load on demand and preserve original Markdown, mathematical notation, and Obsidian references.

## Add a model

`viz/src/models.js` defines ModelPack v1 and validates topology. A pack owns its identity, graph geometry and component metadata, clock dimensions, state vocabulary, lesson descriptions, evidence URLs, and kernel binding. Graph nodes have `id`, `kind`, `x/y/w/h`, `label`, `sub`, `math`, `text`, `dims`, `equation`, and `phase`; edges refer to node IDs. Groups have a rectangle and label. The graph renderer traverses the supplied topology.

Add the new pack to `registry`, with an `adapter` module URL. Each adapter exports `mount(pack, environment)` and returns `{dispose()}` for cleanup. `environment.signal` is aborted when switching packs. `ModelHost` disposes the old adapter, restores the presentation shell, and mounts the new one through the model menu. The host is independent of ISA state shapes, node counts, and clocks.

For another ISA checkpoint with matching evidence contracts, reuse `isa-adapter.js` and provide the new recorded JSON and weights. A different family needs its own numerical/playback adapter: attention state, diffusion scheduler state, and ISA architectural words are different interfaces. Reuse the presentation shell and styling; register family-specific computations instead of forcing the ISA recurrence onto the new model. Small pure ISA kernels live in `kernels.js`.

The model menu lists both shipped specimens. Shared architecture, walkthrough, inspector, dialog, and playback components live in `components.js`. Each adapter mounts its own HTML template. Adapters may return `getState()`; the host retains that model-specific state across switches, aborts pending work, and serializes mounts to prevent an old adapter writing into a new view.

## FLUX evidence and scope

`flux-model.js` supplies the graph and six chapters; `flux-adapter.js` owns the two-stream state view and nested clocks; `flux-kernels.js` contains the scalar BF16 Euler and synthetic attention/SiLU computations. The 112 transitions are four scheduler passes with 28 operations each, not 112 denoising steps. All 5 joint and 20 single block addresses remain selectable.

The full forecast starts at actual Z₀. The matched forecast begins at observed native Z₂ after 56 transitions; its earlier clocks are unavailable. Before/after selects operation inputs/outputs. Image, text, modulation, and denoiser fields clear at commit; the latent and bound conditioning, IDs, and schedules carry forward. State selection, coordinate selection, and latent-channel selection are separate controls.

`public/data/flux-klein/` retains coordinate samples and full-field error reductions with explicit separate scopes. Tile files contain four complete packed latent channels, arranged using the recorded spatial ID axes 1 and 2, on a shared fixed color scale. These are not RGB reconstructions. Terminal images are byte copies of the four original native-VAE PNGs. Pixel error is computed from those images in the browser; complete channel histograms and worst pixels remain inspectable.

The full endpoint has 3.3254% latent relative RMS error, RGB MAE 0.116628 on a 0–255 scale, and maximum channel error 42. Matched has 1.2509%, 0.066078, and 10 respectively. Zero mismatches across 740,327,424 scalar values describe independent reconstruction of the sealed program, a different comparison from native CUDA. The shared native conditioning/VAE and one-seed scope remain explicit. No speedup or generalization claim is made.

Re-export the existing artifacts without running a model:

```sh
../common/bin/common exec python viz/scripts/export_flux_evidence.py --workspace ..
```

The exporter uses NumPy and the public corrected `tensor_archive.load_tensor_archive` from the sibling `saturn-pub` checkout. It pins the accepted audit SHA-256, verifies native BF16/int64 framing and bytes, streams forecast-bundle digests, and verifies selected NPY leaves and content hashes. It leaves multi-gigabyte arrays and all pretrained weights outside the website. `provenance.json` binds 236 source identities, 199 selected forecast leaves, and every generated artifact. Full-field error numbers come from the accepted audit, not sample reductions.

All 192 retained Euler coordinates reproduce their source results exactly. Tests also cover ties-to-even, signed zero, subnormals, hostile interval-promotion cases, all clock joins and field lifetimes, own-state pass joins, toy algebra, original artifact hashes, session restoration, cancelled mounts, and transfer budgets. Initial compressed assets are about 815 KB for ISA and 253 KB for FLUX, including per-sample source paths and hashes; tiles, Euler fixtures, endpoint images, and provenance load on demand. Both families were checked in the browser at desktop and 390-pixel mobile widths.

## Teaching references

- [Bartosz Ciechanowski — Gears](https://ciechanow.ski/gears/): direct manipulation that explains an equation.
- [Red Blob Games — A*](https://www.redblobgames.com/pathfinding/a-star/introduction.html): progressive examples and linked algorithm state.
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/): expandable architecture with coordinated numerical views.

No external scripts or model weights are downloaded at runtime.
