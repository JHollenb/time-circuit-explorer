# Prediction-only generator: first measured run

The extracted generator produced two original 512 × 512 images from prompts and seeds, with **zero native denoiser calls and no native transformer loaded**. Each image consumed its own predicted state through all 112 transitions. The native encoder and VAE remain part of the pipeline; all original denoiser parameter banks remain required.

The qualified green-cone/yellow-cylinder scene (seed 113) exactly reproduced the earlier forecast's initial noise, conditioning and final FP32 latent, and its PNG was byte-identical. A new blue-glass-sphere/orange-cube prompt (seed 2718) also completed. Its native CUDA accuracy comparison is unmeasured; the observed image visibly contains the requested objects.

| Measurement | Qualified scene | New prompt |
|---|---:|---:|
| CPU denoiser wall time | 418.85 s | 365.55 s |
| Denoiser process CPU time | 1,309.34 CPU-s | 1,184.62 CPU-s |
| Native encoding, including reload/transfers | 59.66 s | 39.44 s |
| Native VAE, including transfers | 35.60 s | 14.80 s |
| Image wall time after initial setup | 515.32 s | 421.33 s |
| Peak sampled denoiser RSS | 3.03 GiB | 3.56 GiB |
| Logical parameter reads | 31.00 GB | 31.00 GB |
| Parameter-bank loads | 676 | 676 |

The worker used four configured CPU threads on the shared Ryzen 9 7950X / RTX 4080 host. Cold setup took **425.54 seconds**, mostly outside the explicitly measured checkpoint/loading stages, during cold package imports on the HDD-backed runtime cache. The first image including setup took **940.86 seconds**. These are single observations under concurrent fleet load, not a speedup estimate or a latency distribution.

The two-image worker took **1,363.34 seconds**. The scheduler measured **1,467.92 seconds** for execution including interpreter/process overhead. Queue wait, staging and collection are excluded. The worker's process high-water RSS was **7.69 GiB**; peak PyTorch CUDA allocation was **7.83 GiB**, set by text encoding. During CPU prediction, CUDA allocator residency was about 32 MiB; fleet telemetry also includes driver/context residency. The encoder is released after each encoding and reloaded for the next image, with that cost charged to encoding.

The original benchmark retained its 900-second planning estimate and 1,500-second hard timeout. Neither is rewritten to fit the outcome. Future launcher estimates are 1,000/1,500 seconds for one/two images, with 1,500/1,800-second hard timeouts, following independent review of the cold-start measurement. Numerical worker and kernel bytes are unchanged.

The job succeeded with return code zero. Original request, raw worker report, original PNGs, hashes and collection provenance are published in [the website evidence directory](../viz/public/data/generator/). The scheduler's declared and sealed payload hashes agree. This initial ordinary submission did **not** provide an execution-time payload seal; that field remains null. Authenticated post-run hashes of every staged worker input matched before and after collection, as did remote-before, remote-after and local result inventories. This supports the stated source-audited completed-generation observation; it is not a retroactive execution attestation. Future submissions require mrun's guarded payload path.

Job: `job-2dee3c5989a6` · Request: `4a01c75c475fa9af79a67339ee2d13b4dddfa5ebaf72ac11074fec0d588275f9`.

See [generator setup and measurement definitions](../generator/README.md), [the FLUX visualization plan](plans/flux-pipeline.md), and [the original reconstructed execution evidence](../viz/public/data/flux-klein/provenance.json). This generator evaluates the full dense mathematical reconstruction with FP64 reductions and explicit BF16 boundaries. It has no measured native speedup, new-prompt parity certificate, FLOP count or energy measurement.
