# Zero-network image response: frozen exploratory protocol

Question: can a compiled response program produce images along a supplied diffusion-state path without evaluating any neural network at serving time?

The current generator evaluates the complete dense denoiser equations; bypassing a framework's `forward` method does not eliminate that computation. This experiment instead compiles a restricted output function from historical native observations. It cannot establish arbitrary text-to-image generation or discovery of a sufficient general model state.

Astra advised using the existing late-latent neighborhood because dependency analysis already isolates a real reuse boundary. After the last latent read, unchanged working image/text/context imply unchanged velocity. The qualified BF16 Euler law supplies terminal latents. The new assay approximates the remaining restricted map `alpha → native-VAE RGB`; it does not execute unused Euler arithmetic or claim the pixel program consumes predicted latent arrays.

Source: `artifacts/mdb-latent-neighborhood-20261105/run-02`, job `job-93f45da60821`, FLUX.2 Klein4B distilled, CUDA BF16, known seeds60666/60667, two I0/I1 branches per seed, 512×512. Its24 historical images cost one model load,24 VAE decodes and4 native scheduler checks, with zero denoiser/readout calls. Earlier source acquisition is additional cost. See MDB `FLUX-SUFFIX-CONTEXT.md` and `DEPENDENCY-REUSE.md`.

Freeze before loading image pixels:

- Four context-bound curves:60666.I0,60666.I1,60667.I0,60667.I1. These are dependent curves from two known seeds.
- Compiler reads only alpha0,0.25,1: twelve calibration PNGs.
- Primary is piecewise affine interpolation between neighboring anchors; a global three-anchor quadratic is a fixed comparison.
- Controls: copy-zero, nearest calibration image, endpoint-only linear interpolation, and the other seed's response anchored at the recipient's zero image.
- Held evaluation doses0.05,0.1,0.5: twelve historical PNGs, opened by the scorer only after all predictions and hashes seal.
- Novel serving dose0.075, predeclared here. No native target exists in this source; its accuracy remains unmeasured.
- Preserve full-image and fixed source eye-window MAE, error relative to native change from alpha0, clipping, background error, and paired I1−I0 effect error/cosine. No arbitrary accuracy threshold may stop discovery or erase a control that wins.

For anchors `Y0,Yq,Y1` and q=0.25, the primary is `(1−alpha/q)Y0+(alpha/q)Yq` below q and `((1−alpha)/(1−q))Yq+((alpha−q)/(1−q))Y1` above q. Quadratic uses the unique Lagrange polynomial through those three images. Clip to0–255 and round to nearest integer once when producing the actual PNG; score those saved bytes. The coefficients are calibrated pixel response bases, not pretrained network weights.

Compile and serve in separate processes. Serving receives only the compiled bundle and bound context/dose specification. Deny neural-framework imports and evaluation-directory reads; record imports, opened files, timings and memory. These are trusted-code audit guards, not an adversarial OS sandbox. Save predictions before invoking the separate scorer. Seal the request, compiled bundle, serving receipt and individual PNG hashes in a dedicated manifest before scoring; recheck the seal and prediction bytes. Execute the finite saved-only assay through mrun, CPU, no model acquisition, no GPU, no retries, strict preflight and guarded custody.

Pre-execution instrumentation amendment: Astra found that copying MDB `ForecastTransaction` alone omits its package dependencies. The assay instead uses the explicit hash manifest above, avoiding extra runtime packages. This changes the measurement harness, not the frozen curves, doses, methods or metrics. It is not an MDB transaction certification.

This is retrospective held-file isolation: outcomes already exist and prior findings informed this design. Neither held-file separation nor a transaction seal makes it a prospective discovery. A pleasing image does not establish correct causal response, novel prompt/seed generalization, or freedom from historical native computation.
