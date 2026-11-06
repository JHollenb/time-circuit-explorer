# Raw strings now assemble the first encoder operands

We compiled the complete **151,936-token** vocabulary of the pinned FLUX.2 Klein Qwen3 text encoder through its first token-local prefix. A fresh serving process accepts raw text, executes the exact chat template and tokenizer policy, gathers the resulting token atoms, and applies positions. It evaluates no original dense projection and imports no neural runtime.

All eight frozen prompt specimens matched the post-seal native prefix **bit for bit across 56,623,104 BF16 scalars**: residual embeddings, pre-position Q/K/V and positioned Q/K. Token IDs, masks, positions and rendered template bytes also matched. This moves the input frontier from supplied hidden states to raw text. It does **not** complete contextual text encoding or produce an image.

The table memoizes learned-function outputs. Its 2.645 GB atom payload is larger than the 809 MB embedding and prefix parameters it replaces. This is an exact finite lexical compilation under the measured contract, not compression, a discovered semantic dictionary or a representation-independent elimination of learned computation.

## The finite function that can be compiled

Before the first attention read, each token's operands depend only on its token ID and fixed parameters. With B denoting a BF16 boundary, the source performs

\[
e_t=E[t],\quad n_t=N_{\gamma_{in}}(e_t),\quad
q_t=N_{\gamma_q}(B(W_qn_t)),\quad
k_t=N_{\gamma_k}(B(W_kn_t)),\quad v_t=B(W_vn_t).
\]

The query and key normalizations operate separately over each128-wide head. There are32 query heads and8 key/value heads: widths4096/1024/1024, despite the residual width being2560. The atom is

\[
A[t]=(e_t,q_t,k_t,v_t)\in\mathrm{BF16}^{8704}.
\]

This is a single-token function with a finite complete domain. We compile every embedding ID, including IDs the pinned tokenizer may never emit. The compiler does not select or fit atoms to the held strings. A preliminary813-ID mechanics run remains separately recorded; it was not substituted for coverage.

Qwen's numeric order matters. Its RMS operation squares, averages and applies reciprocal-square-root inFP32, casts the normalized value back toBF16, then multiplies by BF16 gamma with another BF16 result. The compiler calls the installed isolated Qwen RMS and Torch Linear primitives with the original BF16 parameters. It does not construct a full encoder. This qualifies the actual eager CUDA prefix, not a smooth real-number formula or universal parity across backends and matrix layouts.

## Raw text, positions and the exact interface

Serving evaluates the pinned `chat_template.jinja` through an immutable Jinja sandbox with the same whitespace settings. The input is a single user message; generation prompt is enabled and thinking disabled. Low-level Rust `tokenizers` reads the exact tokenizer JSON. It adds special tokens, pads on the right with token151643 and truncates on the right to512 positions. EOS is `<|im_end|>`.

The token IDs gather A[t]. Position IDs are **0 through511**, including padding positions; they are not a cumulative sum of the attention mask. The emitted attention mask records padding, but no contextual attention runs in this experiment.

We compile the nonlearned positional cosines/sines for this fixed512-position policy from the source RoPE operation. Serving then applies the Qwen half-rotation J with eager BF16 multiply/add boundaries:

\[
q_i^{pos}=B\left(B(q_{t_i}\odot c_i)+B(Jq_{t_i}\odot s_i)\right),
\]

and likewise for k. This is the split-half Qwen rotation; the earlier FLUX head used a different interleaved positional convention.

The learned table is sharded in512-row chunks. A manifest binds every shard, token index, row-checksum array, tokenizer, template and positional table. Serving maps shards lazily and authenticates each consumed row against the bound checksum array. Repeated tokens reuse verified lexical rows. Missing IDs raise an explicit coverage error; there is no neural fallback or online table fill.

The serving child blocks neural imports, original checkpoint reads and network connections. It receives raw strings and the compiled store, not a contextual hidden-state history or complete-prompt cache. Complete primary and interface-control operands seal before the separate native-reference process starts.

## What the comparisons establish

The eight strings were frozen in the staged request before compilation and comparison. They cover an empty prompt, a new object description, reordered wording, repetitions, whitespace and punctuation, Japanese/accented/emoji text, literal special-token/tool-response strings, and a long prompt that truncates. Their active lengths are12,25,26,27,27,35,26 and512 tokens. These are interface specimens, not independent semantic image-quality trials.

For all eight, all512 rows of every field matched the native isolated prefix. The [4,096 per-position observations](results/raw-prompt-encoder-atoms/observations.jsonl) include padded and active positions separately. The full vocabulary was compiled, but native parity was measured on these eight sequences under B1/T512 geometry; it was not independently tested for every possible sequence, layout or backend.

Compilation and reference share a source-qualified helper that assembles the original embedding/Linear modules and installed Qwen RMS primitives; the fresh reference recomputes them on each complete prompt. This tests vocabulary-table assembly and the independent NumPy positional program. It is not a hooked comparison against a whole Qwen model, and it does not independently validate the helper's boundary selection beyond the bound source inspection.

Three controls identify different interface errors:

| Control | Observed effect |
|---|---|
| Substitute one token with another covered ID | Exactly one token-ID mismatch; changed lexical operands and positioned fields |
| Shift the position table cyclically by one | Lexical operands remain exact; all512 position IDs differ; positioned Q/K change |
| Omit the generation suffix | Seven token/mask differences for seven shorter prompts; no difference for the long truncated prompt |

The last null is expected and retained: right truncation already removes that suffix in the long specimen. It would be incorrect to demand that every control perturb every input. These are interface diagnostics, not semantic-address controls or tests of image fidelity.

## Compute and storage

| Stage or quantity | Measured result or dimension-derived work |
|---|---|
| Useful compiled domain | 151,936 IDs;0 uncovered |
| Atom payload | **2,644,901,888 bytes**; E plus pre-position Q/K/V |
| Bound store members | 2,662,098,950 bytes including shard headers, tokenizer/template, index/checksums and positions; manifest/receipt extra |
| Original prefix parameters | 809,375,232 bytes, including the full embedding |
| Useful offline projection work | 2,389,746,647,040 MACs |
| Physical offline projection work | 2,391,759,912,960 MACs;297 batches of512, with128 repeated padding rows in the final batch |
| Offline primitive evaluations | 297 embedding,891 Linear,891 RMSNorm,1 positional-table call;0 full encoders/denoisers/VAEs |
| Compiler | 51.163s wall including setup;35.761s atom loop;9.833s process CPU |
| Compiler memory | 1,588.8 MiB process high-water RSS;844,377,088 CUDA-allocated bytes peak;866,123,776 reserved |
| Serving initialization | .430s after Python imports |
| First primary call in the serving sequence | **14.7ms–2.141s**, including tokenizer, newly needed row reads/checks and positions |
| Warm primary call | **12.80–13.93ms**, median of7 repeats for each string; cached lexical rows |
| Serving high-water RSS | 230.5 MiB |
| Actual unique row payload fetched | 1,201,152 bytes across69 IDs for the complete serving/control sequence |
| Logical atom payload per evaluation | 8,912,896 bytes for512 gathered rows; repeated rows may come from cache |
| Complete serving sequence | 8.834s after imports;9.397s fresh child wall including imports;1.817s process CPU |
| Post-seal native references | 8 isolated prefixes:8 embedding,24 Linear,24 RMSNorm,8 positional-table and8 rotation calls;19.806s including setup |
| Entire mrun lease | 127.00s; sampled tree peak1,488.7 MB RSS and1,116 MB GPU |

The compiler and reference each hash the4,967,215,360-byte source shard. Compiler/source loading and reference validation are offline costs, not zero-cost prerequisites. The CUDA arithmetic and data movement are included in synchronized CPU-returning prefix operations; no energy measurement or hardware FLOP counter is claimed. MACs are calculated from dimensions.

The first-call range exposes storage latency. These are calls in one sequential fresh process, with rows reused across prompts and controls; they are **not** independently cold disk-cache trials. Warm latency must not replace those first-call costs. There is no controlled end-to-end speedup claim, and these numbers cover only the first token-local prefix.

The full run reused the smoke's admitted4,096 MB RAM/VRAM profile, with one CPU thread and4 GB disk. It streamed fixed512-row chunks. Both jobs had finite600s timeouts, no automatic retry, strict preflight, CUDA admission and matching guarded declared/sealed/executed payload identities.

## What remains between these operands and an image

The next contextual operation reads the complete causal, padding-masked bank. For query head h, with grouped key/value head g(h)=floor(h/4),

\[
p_{ij}^{h}=\operatorname{softmax}_{j\in\mathcal A_i}
\left(\frac{(q_i^{h,pos})^\top k_j^{g(h),pos}}{\sqrt{128}}\right),\qquad
a_i^h=\sum_{j\in\mathcal A_i}p_{ij}^h v_j^{g(h)}.
\]

Unlike A[t], this depends on the other tokens and their positions. Computing this explicit attention equation would consume the new raw-text-derived operands, but it would still evaluate model attention. It cannot be renamed zero-neural image prediction. Output projection, residual writes, gated MLPs and subsequent layers remain beyond this result.

Source inspection additionally identifies an output-demand opportunity: the pipeline stacks `hidden_states[9,18,27]`. The installed output-capturing code records the initial embedding and each decoder output, replacing only the final entry with the normalized last state. These indices therefore refer to outputs after decoder layers8,17 and26. Omitting the unused final nine layers, final normalization and language-model head appears structurally valid for this consumer, but execution parity for that omission is **not established here**. Do not drop padded query rows: FLUX still consumes a512-row conditioning field.

After contextual conditioning, arbitrary own-state denoising and a general non-neural pixel decoder remain open. The [compiled MLP write](compiled-mlp-write.md), [state-only reader](compiled-state-reader.md) and [global attention head](compiled-global-head.md) solve bounded supplied-context cuts on the other side of that gap. The [RGB response chart](zero-network-image-response.md) supplies a bounded no-neural endpoint. None currently connects these pieces into arbitrary-prompt zero-neural images.

Astra's cycle5 review recommends testing a [smaller streaming context state](plans/compiled-context-statistics.md) next, with exact attention restricted to offline reference work. It also reviewed the newest [next-consumer study](/Users/jakeholl/domains/obsidian/blog/2026-11-06-001355-the-next-consumer-changes-which-model-write-matters.md): local attention/MLP rankings reverse after the following block, reinforcing that current read error alone is insufficient. That link names an unpublished working post.

## Reproduce and inspect

The [full report](results/raw-prompt-encoder-atoms/report.json), [coverage/shard manifest](results/raw-prompt-encoder-atoms/manifest.json), [prediction seal](results/raw-prompt-encoder-atoms/prediction-seal.json), [byte verification](results/raw-prompt-encoder-atoms/retained-store-verification.json) and [guarded custody](results/raw-prompt-encoder-atoms/provenance.json) retain the evidence. Full tables and complete forecast/reference arrays stay on Beast; public Git contains compact immutable measurements and manifests.

From the domains research workspace with its configured scheduler and pinned offline environment:

```sh
../common/bin/common exec python generator/encoder_atoms_study.py stage artifacts/my-atoms-smoke --coverage smoke
../common/bin/common exec python generator/encoder_atoms_study.py submit artifacts/my-atoms-smoke
# Inspect that exact job; collect only after success.
../common/bin/common exec python generator/encoder_atoms_study.py collect artifacts/my-atoms-smoke
../common/bin/common exec python generator/encoder_atoms_study.py stage artifacts/my-atoms-full --coverage full --calibration artifacts/my-atoms-smoke
../common/bin/common exec python generator/encoder_atoms_study.py submit artifacts/my-atoms-full
# Inspect, then collect the same successful full job.
../common/bin/common exec python generator/encoder_atoms_study.py collect artifacts/my-atoms-full
../common/bin/common exec python generator/verify_encoder_store.py artifacts/my-atoms-full
```

An existing compiled store can serve new raw strings with `encoder_atoms.py SPEC OUTPUT`: SPEC supplies `store`, its `manifestSHA256`, `prompts` containing opaque IDs/raw strings, and `repeats`. Output is the prefix operands and seal, **not an image**. It needs NumPy, tokenizers and Jinja2, with no Torch or Transformers dependency at serving.
