# The assumption that turns examples into an advancing program

The [exact compiler](exact-behavioral-compiler.md) recovered a small machine and emitted a copy/constant program. Its global proof used a supplied complete graph. We have now removed that graph from a separate analysis and reconstructed the same rational candidate from the frozen word/output transcript alone. It reproduces all 37 distinct recorded words, while a different finite-state function can reproduce those same observations and disagree on an unqueried word.

There is a constructive qualification: an **independently justified global rank bound** would make the separating observations sufficient. The bound is extra knowledge about the whole behavior, rather than a conclusion obtained by observing a small rank in a finite panel. This resolves the inference contract for this learning method. It does not establish that FLUX has such a bound, or produce an image.

## Factor the function, with the assumption visible

Let the complete scalar behavioral Hankel matrix be \(H[u,v]=f(uv)\). Suppose its global rank is at most r, and observations identify an invertible r×r minor \(B=H[P,E]\). Its rank is then exactly r. The selected full rows span every full row; the selected columns separate their coefficients.

For \(z(u)=H[u,E]\), define \((B_a)_{ij}=f(p_i a e_j)\) and \((F_P)_i=f(p_i)\). The coefficients of the full row after u are \(z(u)B^{-1}\). Shifting that row by a yields

\[
z(ua)=z(u)B^{-1}B_a,
\qquad f(u)=z(u)B^{-1}F_P.
\]

Thus the observed initial row, shifted panels and readout determine

\[
z_0=H[\epsilon,E],\qquad M_a=B^{-1}B_a,\qquad C=B^{-1}F_P.
\]

Induction gives the complete realization. Every equality uses the global row-space premise. The inverse of a finite minor supplies a lower bound; it cannot supply the missing upper bound. This is the familiar field-linear viewpoint developed in [Kiefer's weighted-automata notes](https://arxiv.org/html/2009.01217v1), applied here to the retained transcript.

The new [analysis](../generator/behavioral_authority.py) reads eleven pinned transcript/proposal/native-interface files, plus its own source for identity. It does not read the supplied source graph or complete-graph certificate. The recovered B, initial state, two operators and readout match the prior rational artifact exactly. All 37 recorded words agree. The report explicitly marks the rank≤3 premise as **not established by the transcript**.

## A different source can hide beyond the transcript

The records contain all 31 binary words through length four, two selected length-five words and four length-32 words. The shallow query ledger adds no distinct words. Define

\[
g(w)=f_{\rm recovered}(w)+\mathbf 1[w=a^{33}].
\]

It agrees on every recorded word. At a³³ the recovered machine returns zero and g returns one. This is an analytically constructed alternative, without a new oracle call or a second queried organism. It does not challenge the previous supplied-source certificate: it is a different source that the weaker interface cannot distinguish.

The singleton difference \(\delta=g-f\) has exact Hankel rank 34. Its rows aⁱ and columns aʲ for 0≤i,j≤33 form an invertible anti-diagonal identity. A 34-state weighted chain realizes the singleton, giving the matching upper bound. Rank subadditivity then yields

\[
31\leq\operatorname{rank}(g)\leq 37.
\]

The bounds need not be tight. They suffice to show why this alternative cannot live in the assumed rank≤3 class. The global assumption excludes it; the recorded observations alone do not.

For comparison, two field-linear representations of dimensions r₁ and r₂ have a difference representation of dimension at most r₁+r₂. Its reachable row span stabilizes by depth r₁+r₂−1, so a nonzero difference has a witness no longer than that bound. For dimensions three and three, exhaustive agreement through length five would suffice. Our records omit 30 of the 32 length-five words. This does not defeat the separating-basis argument above: exhaustive testing is a sufficient alternative, without being necessary. See [Kiefer et al., Proposition 3.1 and the difference construction](https://www.cs.ox.ac.uk/james.worrell/fossacs12-journal.pdf).

## What this changes for native models

| Possible authority | Current native-image position |
|---|---|
| Economical global rank or hypothesis-class bound | Unestablished |
| Qualified complete equivalence oracle | Unestablished |
| Exhaustive finite-domain coverage | Unavailable at useful scale |
| Source-derived reduced commuting proof | Unestablished |
| Empirical symbolic candidate with future tests | Still open to exploration |

The [live wrapper](../generator/worker.py) accepts nonempty text with incoming length at most 1,000 Python characters, strips it, and accepts a uint32 seed. Its native encoder call requests 512 tokens. The separately qualified [lexical prefix](../generator/encoder_atoms.py) uses a chat template, tokenizer, right padding/truncation, masks, positions and RoPE over a 151,936-ID compiled store. It remains token-local; it supplies no complete contextual encoder. The [native continuation](../generator/fresh_square_encoder.py) still evaluates learned contextual layers.

A bounded finite-vocabulary interface has finitely many effective token sequences under fixed framing and controls. Token truncation alone does not bound raw strings; the wrapper's character cap is a separate restriction. Neither supplies economical exhaustive coverage. Raw text concatenation can also retokenize a boundary, so an a/b word theorem cannot silently become a raw-prompt theorem. Seed, noise, schedule, resolution and numerical semantics belong to the function definition too.

Global proof and useful exploration remain separate. Failure to establish a rank bound limits the universal claim, without disproving a candidate or stopping reversible evidence collection. A compact symbolic register program can also have high behavioral rank; small linear rank is a sufficient hypothesis for this particular learning procedure, without being a requirement on every possible solution.

Finally, a recovered dense recurrence can still execute learned arithmetic. [Copy/constant lowering](exact-behavioral-compiler.md) is a separate requirement, and neither inference method supplies a source-derived general pixel program. Contextual production, autonomous transitions and final pixels remain the substantive image-generation gaps.

## Actual cost and next direction

The local standard-library analysis takes **10.763 ms before report writing**, with 91.593 ms command process wall. It pays **2,598 exact rational linear MACs** to reconstruct and check the toy candidate and evaluate the proposed witness; elimination, hashes, JSON, source reads and interpreter work are additional. Three targeted tests pass in 0.001 s, with 104.057 ms test-process wall. Historical test counts remain separate. It reads no source graph or complete-graph certificate, evaluates no native model, and reads or produces no scientific tensor or image. No new source-oracle calls occur.

The [report](results/behavioral-authority/report.json) preserves all 37 comparisons, the supplied premise, rank bounds and native-interface mapping. The [evidence index](results/behavioral-authority-evidence/README.md) pins sources, exact copies, checks and primary references. The [frozen plan](plans/behavioral-certificate-authority.md) remains unchanged as historical design evidence; its analysis is executed here.

This closes one inference-contract question. Subsequent research should seek a source-derived register/update/readout family with an explicit contextual producer, rather than expand toy families or launch a native-image Hankel survey. ISA's finite/register programs are a concrete research family to examine, within their measured applicability; they do not establish the corresponding FLUX registers or pixels. Learned QKV lookup plus attention arithmetic and generic dense weighted automata retain learned computation.

The [living log](/Users/jakeholl/domains/obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md), [state/context account](/Users/jakeholl/domains/obsidian/blog/2026-11-05-153154-state-context-and-the-mathematics-of-time-circuits.md), [simplifying-assumption post](/Users/jakeholl/domains/obsidian/blog/2026-11-05-161109-the-one-assumption-that-would-make-model-computation-easier-to-understand.md) and [own-state transition work](/Users/jakeholl/domains/obsidian/blog/2026-11-06-184700-compiling-cheaper-transitions-on-the-states-they-create.md) retain the broader mathematical goal. Full research/storage review 93 is completed for this cycle, with exact scopes in their receipts and next scheduled reviews 96. No complete zero-neural arbitrary-prompt image generator is established or queued.
