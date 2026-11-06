# Move the next cut to raw prompt operands

Status: implemented and executed in cycle5; see [complete-vocabulary results](../raw-prompt-encoder-atoms.md). All151936IDs were compiled, and eight frozen raw strings exactly matched the isolated native prefix. The source-grounded proposal below is retained as the experiment design. Contextual encoding and arbitrary-prompt-to-image remain open.

The global head now consumes a supplied producer state. Repeating known-context head/block reductions alone would leave the input frontier unchanged. The next experiment should accept raw text and remove an actual computation at the **beginning** of the existing conditioning pipeline, without supplying a contextual hidden-state history.

## The actual encoder and its boundary

Read-only inspection of the pinned Beast model identifies `Qwen3ForCausalLM`, with vocabulary151936, width2560,36 layers,32 query heads,8 key/value heads and head width128. RMS epsilon is1e-6 and RoPE theta is1e6. The text-encoder config hash is`214b4c29a0d975e9fddf9994a5673f22cb2c4c5750352f9227c2c3251ebeab40`.

The installed FLUX.2 Klein pipeline wraps a raw string as a user message, applies the actual chat template with generation prompt and `enable_thinking=False`, pads/truncates to512 tokens, and calls the encoder without cache. It stacks `hidden_states[9,18,27]` into a7680-wide conditioning field. A layer0 prefix result is therefore **not** complete prompt conditioning.

Inspected source hashes:

- `diffusers/pipelines/flux2/pipeline_flux2_klein.py`: `8a57a8a7f1c22fc0c5bc5d3fd1bd9bc76c17c1f4d6644cc73370651dd955358e`.
- `transformers/models/qwen3/modeling_qwen3.py`: `fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab`.
- `tokenizer/tokenizer_config.json`: `443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0`.

Before staging, bind the exact encoder weight files/ranges, every tokenizer asset, the actual separate chat template, added/special-token configuration, padding/truncation sides and native numeric policy. The tokenizer config itself has no inline chat template; do not invent a replacement string format from that absence.

## Compile finite token-local operations

At the first decoder layer, before contextual attention, the source program is

\[
e_t=E[t],\quad n_t=\mathrm{RMSNorm}_{\rm in}(e_t),\quad
q_t=\mathrm{RMSNorm}_q(W_qn_t),\quad
k_t=\mathrm{RMSNorm}_k(W_kn_t),\quad v_t=W_vn_t.
\]

These operations depend on token ID t and fixed parameters, before RoPE and masking. Verify that statement against the installed source and controlled execution. Compile the residual embedding and pre-position Q/K/V atoms offline. Serving tokenizes a new prompt, gathers its atoms, and applies the declared position transform; it receives no prompt encoding or native contextual state.

Preserve the **Qwen** normalization order. Its source casts BF16 input toFP32, computes variance/rsqrt and scaling inFP32, casts back toBF16, then multiplies by learned BF16 gamma. It is not the same numerical program as one FP64 normalization/gamma product rounded only at the end. The per-head query/key normalizations occur before RoPE. Reference arithmetic and backend-specific discrepancies must remain explicit.

The source fixes a finite vocabulary, so design the compiler and storage format for the **complete151936-row domain**. A small, frozen smoke may establish byte-range mechanics and instrument geometry first, but it must report every uncovered ID and cannot become a substitute task or a claimed arbitrary-prompt capability. No neural fallback or dynamic table fill is allowed during serving. Compile exactly defined tokens independently of the held prompt combinations.

The atom table is learned-function memoization, not a semantic dictionary, prompt/image lookup, universal non-neural model or compression proof. Its scope is a finite token-local prefix; later contextual computation remains unsolved.

## Costs and resource shape

For BF16 residual E plus pre-RoPE Q/K/V atoms:

\[
\begin{aligned}
|E|&=151936\cdot2560\cdot2=777912320\ \mathrm{bytes},\\
|QKV|&=151936\cdot(4096+1024+1024)\cdot2=1866989568\ \mathrm{bytes},\\
|\mathrm{total}|&=2644901888\ \mathrm{bytes}.
\end{aligned}
\]

The three original projection matrices total31457280 BF16 bytes. Full-vocabulary projection work is2389746647040 MACs before normalization and I/O. The table is larger than the original embedding plus these projection matrices; the proposed gain is removal of repeated serving projections, not compression. These are estimates from inspected dimensions, not measured runtime or memory.

Keep tables near compute and collect compact manifests, selected rows and receipts, not gigabytes onto the Mac or into the public Git repo. Use one finite mrun lease, CUDA admission for any CUDA work, actual measured chunk/shape policy, strict preflight, no retries and explicit compiler/serving/reference costs. Start with the smallest informative smoke and compare viable qualified configurations before expanding. Use the required long-run advisor review if a measured expansion is expected to exceed15minutes.

## Held prompts and controls

Freeze a development/smoke token set before inspecting held combinations, then expand to complete reachable-token coverage under the pinned tokenizer. Held prompt strings should vary order, repetition, punctuation, spacing, Unicode, length, padding, truncation and familiar/unfamiliar word combinations. Novel combinations test assembly; novel vocabulary tests coverage. Neither alone certifies semantic image quality.

Use the exact user-message rendering and tokenizer policy in fresh serving with neural imports and original matrix reads prohibited. Bind token IDs, attention mask, position IDs, selected atom hashes, shaped residual/Q/K/V arrays and final positioned operands. Record uncovered IDs explicitly rather than silently using zero rows or neural evaluation.

Seal complete prefix forecasts before matched references. Compare with the original isolated first-layer prefix in the qualified backend and fixed512-row geometry, with repeated token IDs at varied positions and ordering. Batch/row-layout-dependent numerical effects are instrument diagnostics; do not assume a small native matmul has the same reduction as the declared sequence geometry. A wrong token-row index and wrong position transform should be distinguishable from missing coverage. Keep raw and BF16 discrepancies, per-token outcomes and covariance rather than using accuracy to prune discovery.

An extended consumer can let the gathered positioned Q/K/V drive the first causal attention read, using the actual mask and grouped key/value heads. Keep its independent-equation reference distinct from native SDPA/Flash, and count any native continuation. The first-layer output projection, MLP, later contextual encoder layers, FLUX state transitions and image decoder remain separate unsolved computations. The native encoder has36layers, but conditioning only gathers indices9/18/27: inspect exact hidden-state indexing and output-demand dependencies before charging or executing an unused suffix.

## Progress that would count

Raw prompts would form a real, source-qualified encoder operand boundary without supplied hidden states or online original projections. New word orders would assemble from fixed token-local atoms, with full-vocabulary coverage or explicitly incomplete smoke coverage. That advances the input side of the original objective while preserving the much larger remaining requirements; it would not establish arbitrary-prompt image generation or a representation-independent ban on learned maps.
