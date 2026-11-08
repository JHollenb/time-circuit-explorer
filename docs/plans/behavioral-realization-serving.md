# Separate behavioral realization from permissible serving

This written, unexecuted cut asks whether a predictive-state compiler can produce an explicit program without quietly retaining a learned recurrent model or a whole-image cache. No implementation, native-image campaign or GPU job is queued. It follows the [executed construction analysis](../serving-construction.md), whose raw-word collisions demonstrate why future distinctions and color assignment must remain available.

## A conditional construction for every serving function

For a declared alphabet Σ, let f(w) be the complete P-scalar image under a fixed model, input-framing policy, seed/noise, schedule, resolution and numerical contract. Use scalar future-test columns:

\[
H[u,(v,j)]=f_j(uv).
\]

Suppose the complete matrix has exact rank r. Choose r prefixes pᵢ and r suffix/pixel tests (vⱼ,ℓⱼ) such that

\[
B_{ij}=f_{\ell_j}(p_i v_j)
\]

is invertible. Define b(u)ⱼ=fℓⱼ(uvⱼ), (Bₐ)ᵢⱼ=fℓⱼ(pᵢavⱼ), and Fₚ as the matrix whose rows are complete images f(pᵢ). Then

\[
z_0=b(\epsilon),\quad M_a=B^{-1}B_a,\quad C=B^{-1}F_P,
\]
\[
z_{k+1}=z_kM_{a_k},\qquad
f(a_1\cdots a_n)=z_0M_{a_1}\cdots M_{a_n}C.
\]

Every full Hankel row is a combination of the selected rows, and the separating columns recover its coefficients. Appending a symbol shifts its future tests, so the same relation determines the next predictive state. This uses the **global** row-space hypothesis. An invertible sampled panel alone does not establish that hypothesis or shift closure.

This is a conditional all-function realization: c supplies the exact ordered input symbols, I supplies z₀, T applies the symbol operators in order, and R applies C. Operators can fail to commute, retaining order distinctions that a word bag discards. Noncommutativity alone does not establish the required semantic binding.

The input interface must be resolved before any native experiment. Raw-byte concatenation, raw-word concatenation and tokenizer-ID concatenation define different functions; BPE can retokenize across a raw-text boundary. Declare framing, masks, truncation/overflow, invalid strings and EOS exactly once. A construction for one fixed seed gives no shared realization across seeds, resolutions or schedules. The native alphabet/frontend contract remains unresolved here; the first mechanics organism would use a fully supplied finite machine.

## Finite output changes the state-count argument

For exact uint8 RGB, each entry of b(u) is in {0,…,255}. Because its r separating entries determine the full future row,

\[
r\leq N_{\rm behavioral}\leq256^r.
\]

Thus finite rank implies a finite future-equivalence index for this exact finite-valued output contract. The upper bound can be exponentially expensive. A real-valued example with rank one and infinitely many states is not an RGB counterexample. A lookup transition δ(state,symbol) exists in principle for a finite quotient; enumerating it and storing a complete RGB image at each terminal state can still be the forbidden image-cache construction.

## Charge the learned work and offline queries

Generic dense Mₐ and C evaluate learned coefficients and form a linear recurrent model. Naming it a weighted automaton does not establish zero-neural serving. Its nominal coefficient and serving counts are

\[
|\Sigma|r^2+rP+r,\qquad nr^2+rP\text{ MACs}.
\]

As an illustration only, r=128, |Σ|=151,936, n=512 and P=786,432 require 2,489,319,424 token-operator coefficients and 100,663,296 readout coefficients. FP64 storage totals 20,719,862,784 bytes before metadata; serving uses 109,051,904 learned-coefficient MACs. Exact rational arithmetic can require larger representations and different arithmetic. These are symbolic counts, not observed model rank, storage or runtime.

A naive known-basis compiler can issue up to (|Σ|+1)r²+2r complete-image oracle calls before deduplication, basis discovery and validation. Observing one pixel test still requires the underlying image oracle unless a qualified partial consumer exists. All native compilation/reference production remains paid. No native-image Hankel campaign should begin from these illustrative dimensions.

## The next informative deliverable

Write one representation and compilation contract separating three questions: what a finite panel observes, what a complete closure certificate would establish, and what a permissible economical serving program actually executes. If useful, validate an exact rational basis compiler on one supplied tiny finite machine, including noncommuting order, distinguishing suffixes and a representation collision. Such a toy validates mechanics only. A finite oracle-query panel cannot certify an unrestricted image function globally without an additional structural assumption or equivalence mechanism.

Potentially compatible serving needs explicit symbolic/discrete operation lowering and a source-derived compositional pixel program. Neither may be inserted as a placeholder. Dense learned recurrences and complete-image lookup remain separately charged alternatives, not a completed strict-goal candidate.

Saturn's existing [closed-state compiler](../../../saturn/docs/CLOSED-STATE-COMPILER.md) and [implementation](../../../saturn/src/saturn/closed_state.py) provide affine finite-horizon Hankel diagnostics and compact kernels. The runtime still uses linear learned arithmetic; it is not an existing raw-token black-box behavioral compiler or zero-neural image generator. The existing `behavioral-hankel-predictive-state-compiler` request has been [upvoted](../results/serving-construction/runtime-request.json), without a new implementation claim. Full research/storage90 remains current,next93. Arbitrary-prompt zero-neural image generation remains active and unsolved.
