# Six state normalization assay

Implementation contract for cycle109, refining the immutable [selected108 question](coupled-baseline-response-state.md), SHA256 c22e3dfb640a472ef8d6267a5dd13dcb258d86d635bddda82e8f84b3d14fae8f. Status at authoring: unimported, untested, unstaged and unrun. Execution freezes these source and contract bytes; later results belong in separate records. The goal remains arbitrary raw prompt to faithful image with zero neural serving evaluations.

## Specimen and source boundary

Reuse the six actual authenticated full BF16 H1 states a,b,p,q,Xno,Xon from the [107 crossover](../finite-consumer-crossover.md), discovery00/10, unchanged active row9. Four edges are native a→b, own p→q, no a→Xno and on p→Xon. Five contrasts are E=own−native, Uno=own−no, Vno=no−native, Uon=own−on and Von=on−native. Every contrast component uses this same incidence map. Use actual admitted crossed endpoints, with no replacement by nominal unrounded changes.

Reproduce the exact original CUDA statements from the authenticated contrast_writer_worker.Consumer normalization closure: h.float(); work*rsqrt(work.pow(2).mean(last,keepdim=True)+epsilon); original BF16 gamma*work.to(h.dtype). Preserve epsilon1e−6, Torch2.13.0+cu130, the parent numerical profile, full contiguous[1,512,2560]shape, stride[1310720,2560,1] and BF16 output. These are new source-policy reconstructions; old107 normalized fields were not retained, so direct old-output parity is unavailable.

Authenticate the entire selected first safetensors shard, its header and gamma payload. Materialize only model.layers.1.input_layernorm.weight,2560BF16 values/5120bytes. Do not call the old four-parameter loader or Consumer constructor. Q/K/V matrices are never materialized, and their affine maps are never evaluated. Hashing4,967,215,360shard bytes remains paid I/O. Existing107 Q/K/V scalar observations are authenticated context, with no fresh projection or unique attribution inferred from pre-affine norms.

## Evaluated explanatory law

On row9 only, use contiguous FP64 products with np.sum(dtype=float64): squaredRadius=sum(h*h)/2560+epsilon, radius=sqrt(squaredRadius), inverse=1/radius, ideal=gamma*(h*inverse). Define rho=actual finite Z−evaluated ideal. It includes native finite effects and rounding of this declared FP64 explanation; it is not exclusively BF16 error.

For each actual edge h→k retain delta=k−h, D=gamma*(delta/radius(k)), B=gamma*(h*(inverse(k)−inverse(h))), R=rho(k)−rho(h), Z=actualZ(k)−actualZ(h), g=Z−orderedSum(D,B,R). The evaluated edge explanation is orderedSum(D,B,R,g). Its separately evaluated closure is measured as a scalar squared norm.

For each contrast retain D,B,R edge differences, G=left.edge.g−right.edge.g, g=targetResponse−orderedSum(D,B,R,G), and closure=targetResponse−orderedSum(D,B,R,G,g). The computed residual need not make a floating sum close exactly. Retain both g and final closure. Do not simplify them symbolically to zero.

Retain all signed Gram terms and cosines, with math.fsum for scalar sums. Terms are not unsigned additive causal shares. Norms, dot products and actual normalization outputs remain source-qualified producer measurements; the publisher rebuilds scalar summaries and verifies bytes/typed joins, without independently replaying vectors. A large pre-affine term cannot establish its importance after Q/K/V projection.

## Exact geometry and controls

Six original normalization Acts consume the immutable CUDA gamma parent and full H1 tensors. One CPU FP64 explanation Act binds all six full H1 tensors, six full Z outputs and gamma. Each Act records actual tensor payload identities, canonical Frame ancestry, immutable inputs, operation schema and read/write domains. Gamma identity is checked before/after all six source Acts.

Retain86 numerical files on Beast:6fullBF16 Z,1BF16 gamma,79FP64 row vectors. The row fields are12state ideal/rho,24edge delta/D/B/R/Z/g,11historical response fields,30contrast D/B/R/G/g/closure, and2endpoint errors. Logical numeric payload is17,351,680bytes;79NPY headers of128bytes make expected file bytes17,361,792. Actual sizes remain authoritative.

Raw controls compare Z(Xno) against Z(a), and Z(Xon) against Z(p), on all512rows each:1024comparisons,2intervened and1022unmodified. Retain6pooled groups:each branch×all/intervened/unmodified. Preserve all36pooled metrics and2unmodified-row change counts. Unexpected unmodified changes require instrument/profile diagnosis before clean row9 interpretation; they do not erase completed evidence or establish a scientific null.

Analysis publication rebuilds402derived scalars from declared primitive energies/dots:47response,120four-edge,205five-contrast,18radii and12endpoint metrics; plus36pooled metrics. Scalar-close tolerances check consistent publication arithmetic, not scientific quality. Zero native reference retains absolute RMS and null relative values. Closure size, cosine, component energy and evaluator agreement are observations, without promotion thresholds.

## Mechanics and execution

Before first import, obtain exact source/math/resource/custody review. Seven supplied tiny cases cover independently calculated D/B with nonunit radii/gamma, supplied finite discrepancies, negative signed cross terms and contrast signs, zero reference, known endpoint geometry, scalar/domain tampering, and invalid ABI/nonfinite inputs plus a nonzero final closure witness. No real state/model array runs locally. Freeze executed source/test bytes after this one mechanics run.

One guarded mrun lease on Beast is justified by host-local retained inputs and shard. CUDA is required; intent-first sizing, strict preflight, no guessed memory reservation, CPU numerical threads1,6logical branches/7operations/1gamma load/0replays. Expected under5minutes, hard cap900seconds; no retries, child jobs or automatic expansion. Inspect hosts, queue note, why and log. This is below the15minute review threshold; Astra separately reviews the whole source before execution.

Work is6normalizations/3072rows;7,864,320coordinates at each FP32square/rescale and BF16gamma stage;3072row reductions/epsilon/rsqrt;0learned affineMACs. Record full shard authentication, gamma/header/load, state authentication/read, H2D transfer, Act timing with hashing, CUDA event span including Act/hash work, output transfer/retention, FP64 analysis and controls separately. CUDA event timings are not isolated kernel latency. Report RSS and CUDA allocated peak separately from scheduler sampled peaks/reservations. Six source normalizations remain paid neural arithmetic.

Collect compact evidence directly from the single successful job. Publisher verifies request/source closure, current successful guarded custody, complete collection, parent107 publication, exact6state/gamma/profile bindings,86retained ABIs,7typed transactions,402+36rebuilt scalars and1024raw rows. Fresh remote full-file hashes cover6inputs+1shard+86newfields=93files; NPY payload/header validation is byte-only. Scientific arrays stay near compute. Run and verify the standard Saturn evidence sweep separately, retaining unsupported-instrument outcomes if the schema remains unsupported.

## Decision scope

The result should guide what an input-owned state writer must jointly preserve: directional transition, baseline-aligned denominator feedback and finite behavior. It does not justify another fixed-nine-atom routing fit, radius-only endpoint repair or a claim about unique projection geometry. The independent106padding-capacity gap, raw-history contextual initializer, owned diffusion transitions and pixel decoder remain open. No writer fit, new prompt, Q/K/V, attention, MLP, full encoder, denoiser, VAE or image is authorized by this assay. Next full research/storage review remains111 or earlier material evidence/pressure, with108 the current full audit.
