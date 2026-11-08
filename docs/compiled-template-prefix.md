# A contextual template prefix compiled into constants

The partial constructor now runs from raw input: a fresh non-neural tokenizer/copy child emits actual contextual template states at H0–H27. The 286,720-byte bank matches the corresponding historical native B words at all 28 states. Eight fixed raw-input cases retain their own complete input packets, copy these same two rows and expose the remaining 510 rows as unresolved. The arbitrary-prompt zero-neural image goal remains open.

The first two tokenizer outputs of the bound single-user frontend are `[151644, 872]`: the opening role marker and `user`. These are fixed before the raw content. The next newline can merge with leading content whitespace for direct tokenizer callers, so the compiler fixes two rows. It preserves the full 151,936-token vocabulary and 512-position input interface.

Let P={0,1}. With causal attention and row-local residual, normalization and MLP operations,

$$H^{\ell+1}_i=F_{\ell,i}(H^\ell_0,\ldots,H^\ell_i),\qquad i\in P.$$

Since H0 at these positions and their positions/masks are fixed, induction gives constants C0,...,C27 satisfying

$$C_{\ell+1}=F_{\ell,P}(C_\ell).$$

This is partial evaluation of an input-independent subgraph. The compiler extracts previously computed source-qualified native A words; it does not discover new numerical values without historical neural work. The resulting H0–H27 bank has 28×2×2,560×2 = 286,720 payload bytes. A clean consumer initializes its own full token packet from raw text, copies these prefix states and labels positions 2–511 unresolved. A wrong prefix makes state unavailable. No zeros, native-state refresh or neural fallback fill the remainder.

The useful distinction from the [lexical atom table](raw-prompt-encoder-atoms.md) is that these are actual contextual states after residual/attention/MLP transitions, albeit for fixed template rows. The [masked-KV](complete-masked-encoder-image.md) and [query-demand](conditioning-demand-consequence.md) cuts still execute original blocks on their live rows. This compiler avoids those operations only for this fixed subgraph. It does not implement a dynamic encoder supplied with compiled prefix K/V.

The mathematical constant law is about the qualified causal source arithmetic. Copying one retained finite result does not prove universal CUDA bit parity. Padded and fully attended inputs can choose different native mask/SDPA dispatches. The post-serving historical A/B comparison measures its own recorded profile; B has been public and is not a fresh held-out example.

The executed [report](results/template-prefix-cofactor/report.json) and [child receipt](results/template-prefix-cofactor/receipt.json) retain every case and every layer. Both original complete token/mask packets match the raw tokenizer. The leading-newline case changes the third token from 198 to 1406, confirming why only two rows were compiled. Novel text, embedded role markers, Unicode, a truncation case with 512 active rows and an empty tokenizer-boundary control all keep the same first two tokens. The last two cases describe the tokenizer interface, not the image server's 1–1,000-character input contract. A wrong first token returns `unsupported-prefix` and makes previous states unavailable.

The zero change of compiled states across content is expected: this subgraph cannot see that content. The historical native A/B records change one lexical row at position 8. At every H1–H27, all 504 rows from 8 through 511 differ, while earlier rows remain byte-identical. That observation prevents treating a fixed template spelling, a fixed padding token or a clean constant prefix as a replacement for the dynamic contextual computation. In particular, the assistant suffix follows the user content and is not compiled here.

One guarded CPU job, `job-521d4678edb9`, completes in **4.29 s**, with sampled peak process RSS 276.8 MB and zero VRAM. Compilation/authentication/copy takes 0.9654 s; the fresh child takes 1.0655 s wall time. Its tokenizer initialization/warmup before the read guard takes 0.9737 s and compiled-state loading afterward 0.4096 ms. Per input, the state update and 28 layer copies take 0.0971–0.2169 ms. Tokenization is 0.0880–0.1624 ms for the seven short cases and 7.4833 ms for the long case; serialization/hash costs 0.3406–0.5103 ms per case. These are separate scopes from queue/setup and do not establish end-to-end speedup. The child's 192,184,320-byte RSS is a high-water measurement with possible inherited pre-exec history, not isolated steady-state memory.

The bank alone is 286,720 B; all compiled assets including the 11.42 MB tokenizer total **11,714,597 B**. Eight served bank copies add 2,293,760 B near compute. The compiler authenticates 73,403,904 B of A files; post-seal diagnostics read 146,807,808 B of A/B files. No full native arrays hydrate locally. This new job does byte extraction/comparison and non-neural tokenization, with zero new original-model evaluations, scientific floating arithmetic or images. Historical A alone paid 1,395,193,282,560 nominal encoder linear MACs for its complete 512-row run; this is inherited preparation, not a measured incremental cost or a saving for the two-row compiler. Original parameter acquisition, attention, nonlinearities, earlier model work, decoding and source storage remain additional costs.

The [source review](results/template-prefix-cofactor-evidence/advisor-source.json) precedes the five supplied-byte mechanics checks and the single run. The [actual-result review](results/template-prefix-cofactor-evidence/advisor-actual.json) confirms the scoped construction and corrected metadata publisher; the [publication](results/template-prefix-cofactor/publication.json) authenticates all nine compact members and freshly hashes the bank plus eight served outputs near compute. The guarded payload has matching declared/sealed/executed SHA256 `ef62e20fe28286551245ca91ada74bb87ab5403f748a0eb1758c1014150fcddc` and 64,347 bytes. The guard runs after explicitly measured non-neural tokenizer initialization; it denies original-state/source-tokenizer paths and numerical-runtime import canaries. It is a trusted-code audit boundary. All executed program/study/test/plan sources remain frozen.

The remaining function is substantial. Writing the source encoder as

$$\Phi(u)=(C,\Psi(u;C)),$$

this program supplies C and owns u, while **Psi remains unimplemented**. Prefix K/V are not yet compiled; no native dynamic consumer or image suffix runs. The next construction must produce content-dependent state from its own input, with its actual operations and assets exposed. This result does not authorize automatic cache growth, another original-MLP approximation campaign or an image-generation claim. Full research/storage review remains due 102.

The [frozen plan](plans/template-prefix-cofactor.md) declares eight raw-input cases, source authentication, byte comparisons and refusal controls. The [program](../generator/template_cofactor_program.py), [study](../generator/template_cofactor_study.py) and [mechanics checks](../generator/test_template_cofactor.py) keep all dynamic rows, contextual QKV, denoising and pixels explicitly open. The preceding [selected-weight compiler](compiled-selected-mass.md) remains a separate bounded coefficient producer.
