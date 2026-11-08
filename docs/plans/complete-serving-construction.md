# Specify a complete serving construction from raw text to pixels

The next question is whether we can specify one executable raw-prompt → program → pixels construction whose serving dependencies omit original learned transitions, supplied contextual states, prompt-keyed final images and a supplied procedural renderer. This is a written feasibility cut, **unexecuted**, with no implementation or submitted job. Source inspection and symbolic derivation come before another GPU experiment.

The [matched template experiment](../frozen-template-transport.md) improves response discrepancy while worsening actual own-input update error. It does not resolve any missing dependency in arbitrary-prompt zero-neural generation. Astra recommends ending that single-block matrix-compression sequence as the primary route. More rank, depth, seeds, precision or response-norm tuning is not the default next experiment.

## Account for every function

Write a candidate as

\[
c=c(p;A),\qquad s_0=I(c,\xi;A),\qquad
s_{k+1}=T_k(s_k,c;A),\qquad Y=R(s_K;A),
\]

where p is the raw string, ξ is any declared seed/noise input and A contains all compiled artifacts. A candidate must give equations or an implementable algorithm, input/output types, numerical boundaries and an explicit artifact list for **c, I, every Tₖ and R**. An unspecified semantic parser, sufficient state, decoder or compiled learned map leaves a function unresolved. Track those gaps directly rather than presenting them as a complete architecture.

For each artifact in A, record what produced it offline, its domain, parameter/data provenance, storage size and whether compilation used native evaluations. Offline native work is allowed research cost. Serving must consume candidate-produced state at every step; an oracle refresh changes the claim. Distinguish storage access, scalar/nonlinear arithmetic and learned map evaluation rather than counting function names.

The first deliverable is a dependency table with an explicit unresolved cell whenever no construction is available. The second is either one complete candidate with a symbolic compute/storage estimate or a precise explanation of the missing algorithm. This is architectural feasibility work, not a new scientific result or a quality gate.

## Start from the two actual endpoints

The [complete-vocabulary atom compiler](../raw-prompt-encoder-atoms.md) already gives a concrete partial c. With token IDs tᵢ from the pinned template/tokenizer, gather

\[
A[t]=(e_t,q_t,k_t,v_t),\quad
q_i^{pos}=B(B(q_{t_i}\odot c_i)+B(Jq_{t_i}\odot s_i)),
\]

and likewise for k. All 151,936 embedding IDs are compiled; the 512-position policy and BF16 order are explicit. The 2.645 GB atom payload memoizes learned-function outputs. This covers the first token-local prefix, not contextual text encoding. No supported I/T/R composition currently follows from these atoms.

The [closed RGB response chart](../zero-network-image-response.md) gives an explicit R only on four supplied context curves:

\[
\hat Y_r(a)=
\begin{cases}
Y_{r,0}+4a(Y_{r,1/4}-Y_{r,0}),&a\leq1/4,\\
Y_{r,1/4}+\frac{a-1/4}{3/4}(Y_{r,1}-Y_{r,1/4}),&a>1/4.
\end{cases}
\]

The key r and three whole-image anchors per curve are supplied. There is no demonstrated map from arbitrary lexical atoms to a new r or new scene basis. The missing composition cannot be filled by silently choosing the nearest known image. The [fresh-prompt square](../fresh-prompt-square.md) shows own-state contextual/image continuation on new strings while retaining original learned consumers; it supplies a reference boundary, not the missing zero-neural construction.

## Separate finite compilation from general composition

The lexical table is feasible because its input has one finite token index. A complete contextual table has an input sequence. For a vocabulary V and n unconstrained positions its domain is \(|V|^n\); even 32 positions at V=151,936 produce more than \(2^{550}\) sequences before seed variation. This is a combinatorial illustration, not an exact tokenizer-reachable-domain census. Restricting length makes the domain finite without making complete storage economical.

Whole-prompt image lookup would also violate the requested construction. Rewriting each original learned matrix into scalar operations or a Boolean circuit preserves its learned computation under another notation. Neither is evidence that the serving dependency has been removed. A useful alternative representation needs an explicit composition rule that generates new contextual operands and pixels from reusable artifacts.

Possible assumptions must therefore be stated as assumptions. For example, a bounded finite scene grammar may make c and a program state tractable, but narrows arbitrary-prompt coverage; supplying a renderer would provide R while leaving the requested learned-to-executable visual construction unresolved. Approximate state closure may reduce T's size only after identifying an executable state and testing future consumption on its own outputs. No such assumption is adopted as an established result here.

## Review the construction before scheduling work

Use source-bound inspection of the four linked endpoint/reference documents, the [strict original goal](arbitrary-prompt-zero-neural.md), relevant compiler/runtime sources and current evidence. State what each source actually provides. Keep compilation cost, complete storage and serving work separate. Give bounds or symbolic counts where a measurement does not exist, without presenting estimates as observed runtime.

If a complete candidate emerges, ask Astra to review its exact equations and dependency table. Only then write the smallest informative implementation/experiment plan, with native work through mrun and an explicit own-state/pixel consumer. If no complete candidate can be specified, preserve the unresolved c/I/T/R cells and the mathematical obstruction. Do not choose another learned-block assay just because it is easy to execute.

Full newest-post and storage reviews are complete through cycle 90; next scheduled reviews are cycle 93 or earlier material evidence/pressure. This plan has no implementation or submitted job. The active research objective remains arbitrary raw prompts to images with zero neural evaluations.
