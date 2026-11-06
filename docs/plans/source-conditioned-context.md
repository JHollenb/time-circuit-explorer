# Freeze context geometry from source atoms, then test new strings

Status: **executed and measured in cycle7**. The [source-frozen result](../source-conditioned-context.md) preserves this protocol: on eight fresh strings, rank64 whole-head error improves to92.8% from162.7% unconditioned, but all-query prompt-change errors include780.5% above/below and1,743.0% inside/outside. Active/padding countertrends remain separate. This is a reusable first-head approximate read with source-frozen coordinates, not faithful contextual encoding or image generation.

The original goal remains arbitrary-prompt image generation without neural evaluations. This experiment addresses one first-layer contextual read. It does not supply the later encoder, denoiser, decoder or image-quality evidence.

## Freeze the source distribution

Sample2,048 token IDs without replacement from the complete151,936-ID domain using PCG64(seed1707), sort and retain the exact int64 ID-list bytes and SHA-256. Seed alone is not the identity. Use every position0–511 uniformly, with the real eager BF16 positional transform. No prompts or contextual target arrays enter the fit.

Key moments use this uniform token×position pool. Query moments use a50/50 mixture of that pool and the checkpoint's padding-token ID at every position. This is a declared source-distribution heuristic, not a natural-language/deployment distribution or an inherent model property. Uniform vocabulary sampling includes unusual/special IDs; test active, padding and long unpadded results separately. Source/test token overlap is expected lexical compilation, not access to contextual targets.

Compute stable FP64 block means and centered M2 with Chan merging, in64-token chunks. Do not materialize the complete token×position pool. For pool U and pad P, the query mixture is

\[
a=\tfrac12\mu_U+\tfrac12\mu_P,\qquad
A=\tfrac12\operatorname{Var}_U+\tfrac12\operatorname{Var}_P
+\tfrac14(\mu_U-\mu_P)^2.
\]

The between-component term is required. Key center b and variance B come from the uniform key pool. Freeze

\[
D_\ell=\left[\frac{\max(B_\ell,10^{-12}\overline B)}
{\max(A_\ell,10^{-12}\overline A)}\right]^{1/4}.
\]

Record raw moments, floors, affected coordinates, geometry bytes and compilation costs. If an entire moment side is zero, unit scale is a labelled degenerate rule. This is numerical construction, not an accuracy gate.

## Serve without prompt-fitted geometry

Use the same frozen64×128 Gaussian array as cycle6 and nested ranks16/32/64. A fresh guarded process receives only raw strings, the lexical atom store, frozen a/b/D and feature constants. It applies

\[
u=D(x-a),\quad w=D^{-1}(y-b),\quad \beta(y)=a^\top y
\]

and adds β to key-feature logs before the stable prefix update. No per-prompt refit, contextual target cache, original projection or exact-attention fallback is allowed. Maintain and consume the actual prior z/M/maxima/count/clock state. Save complete heads and selected state checkpoints before reference access. A uniform control and unchanged rank64 feature reader accompany the three conditioned ranks.

The exact gauge changes the partition by a query constant. Retain both the transformed partition and its original-gauge conversion,

\[
\log Z_{original}=\log Z_{gauge}+x^\top b-a^\top b.
\]

Applying this factor to an approximate partition does not make it exact. Charge query/key coordinate transforms, bias/correction contractions, features, normalization, exponentials, state arithmetic, preformed log arrays and outputs separately. Core state alone is not total working memory or retained learned information.

## Frozen new strings and reference order

Eight new strings/four pairs are fixed in [context_source_study.py](../../generator/context_source_study.py): kite above/below boat, key inside/outside box, Japanese umbrella/bicycle color assignment, and long repeated marble/carpet/willow/fountain order/reversal. The request binds exact strings, all source files, ID bytes and inherited atom/feature identities before execution. Three warm repeats per method/string measure component cost, not controlled native speedup.

After complete forecasts seal, canonical full-prefix assembly verifies selected ports and exact FP64 causal attention supplies references. No native encoder/attention/VAE runs. A saved authenticated original W_o head slice transports errors offline without new checkpoint reads. It remains an ideal linear consumer diagnostic, not full native fused projection/residual/image parity. Direct finite-feature calculations verify the conditioned streaming law on one short and one long string after sealing.

Retain all/BF16/active/padding head errors, four prompt-change errors, per-position metrics, feature concentration, range bounds and exact-reference diagnostics. Changed token lengths also change positions/masks; split paired regions accordingly. No accuracy gate suppresses negative arms or controls.

Use one guarded finite mrun CPU lease near the host-local atom store, strict preflight and no automatic retries. The largest compilation chunks are bounded; package/runtime presence is separate from actual imports/evaluations. This is model-derived approximate computation with paid source compilation and a2.645GB learned table, not the strict zero-neural image result.
