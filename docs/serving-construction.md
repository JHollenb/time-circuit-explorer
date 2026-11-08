# An executable state must preserve future distinctions

A zero-neural image program needs a representation that retains what later computations will use, together with executable rules for advancing it and producing pixels. The current components do not yet close that loop. Ordered lexical atoms accept raw text; the RGB chart produces pixels in four supplied contexts. Contextual operand formation, general state transitions and general pixel production remain unresolved between them.

The [complete-serving feasibility cut](plans/complete-serving-construction.md) now has an executed, source-bound [collision analysis](results/serving-construction/collision-witnesses.json). It establishes one concrete restriction: an unordered multiset of raw words cannot reproduce the recorded color-binding image pairs. Any deterministic program receiving only that multiset and common controls has a worst-endpoint error floor of about **30.20 and 31.11 pixel levels RMS** on the two pairs. This excludes one representation under native pixel parity; it does not exclude all compact representations or an independently prompt-faithful generator.

## The four functions and their actual dependencies

Write the proposed program as

\[
c=c(p;A),\quad z_0=I(c,\xi;A),\quad
z_{k+1}=T_k(z_k,c;A),\quad Y=R(z_K;A),
\]

where p is raw text, ξ supplies the declared randomness and A contains every compiled artifact. The algorithm and cost of each function matter. Calling a function a semantic parser, state compiler or decoder does not specify its computation.

| Function | Current source-backed component | Remaining dependency |
| --- | --- | --- |
| Lexical part of c | [Prefix](../generator/encoder_atoms.py) returns ordered IDs, masks, positions and token-local E/Q/K/V/RoPE from the complete vocabulary store | First contextual attention and the remaining encoder computation |
| Complete c | [Own-state encoder](../generator/fresh_square_encoder.py) advances real contextual states | It loads and evaluates original learned weights |
| I at the latent boundary | [Noise draw and packing](../generator/fresh_square_study.py), conditioning and finite schedule state | This is conditional on c being available; learned image projections are inside T |
| T | [Own-state denoiser and scheduler](../generator/fresh_square_images.py) consume the actual preceding latent | General advancement still evaluates learned maps |
| R | Native VAE, or the [RGB chart](../generator/response_chart.py) on supplied contexts | No general pixel producer for novel candidate states |

The [chart CLI](../generator/render_response.py) explicitly rejects unknown context keys. Its twelve supplied whole-image anchors are not a compositional visual vocabulary. Conversely, the 151,936-entry [lexical atom store](raw-prompt-encoder-atoms.md) is a real finite compilation, but its outputs are initial operands rather than complete prompt interpretation. These interfaces do not currently share an executable intermediate representation.

## Exact closure is a commuting computation

Let sₖ be the full native state at a declared boundary. Include every future-relevant tensor, conditioning field, clock, position, RNG state and persistent history. Controls held outside the state must be explicitly fixed or supplied. Let Fₖ(s,u) be the next transition under admitted control u, and O(sK) the terminal observable.

A proposed smaller state is zₖ=qₖ(sₖ). A reduced transition Tₖ can be well-defined on the declared reachable domain only if two states that share a code always receive the same successor code under the same control:

\[
q_k(s)=q_k(s')\implies
q_{k+1}(F_k(s,u))=q_{k+1}(F_k(s',u)).
\]

At the final boundary we also need

\[
q_K(s)=q_K(s')\implies O(s)=O(s').
\]

These conditions are sufficient for well-defined functions on represented states: choose any representative of each code and set

\[
T_k(q_k(s),u)=q_{k+1}(F_k(s,u)),\qquad
R(q_K(s))=O(s).
\]

The implications ensure that changing the representative changes neither answer. The initial producer has its own obligation. If Iₙ is native initialization, then for common ξ and controls,

\[
c(p)=c(p')\implies
q_0(I_N(p,\xi))=q_0(I_N(p',\xi)).
\]

Together these statements establish mathematical factorability on the declared domain. They do not provide economical algorithms for computing the codes, transitions or readout. The identity q(s)=s retains everything while leaving the original computation intact. Defining q(s) as its final image can also be sufficient while requiring the entire model to compute that code. Neither supplies the missing serving program.

The corresponding future equivalence is

\[
s\sim_k s'\iff
O_k(s,\mathbf u)=O_k(s',\mathbf u)
\quad\text{for every admitted continuation }\mathbf u.
\]

The consumer, horizon, controls and finite numerical policy define this relation. Agreement at the current readout is weaker. In the analyzer's supplied four-state toy, the visible register v is the current output and h is hidden. Under `hold`, (v,h) remains unchanged; under `reveal`, it becomes (h,h). States (0,0) and (0,1) both currently output zero, but `reveal` distinguishes them. Retaining (v,h) closes all eight state/control transitions of this toy. This is known-wiring mechanics, without an inferred binary register or quotient in FLUX.

## A collision creates an unavoidable error floor

Consider the exact representation

\[
c_{\rm bag}(p)=\operatorname{sorted}(\operatorname{Counter}(p.\operatorname{split}()).\operatorname{items}()).
\]

Case and punctuation remain unchanged. This is a raw whitespace-word multiset, not a BPE token bag or a sum of learned token vectors. The [frozen fresh-square strings](results/fresh-prompt-square/request.json) contain these opposing assignments:

| Cases | First assignment | Second assignment |
| --- | --- | --- |
| 00 / 10 | Orange ceramic pitcher left of teal metal watering can | Teal ceramic pitcher left of orange metal watering can |
| 01 / 11 | Orange ceramic pitcher right of teal metal watering can | Teal ceramic pitcher right of orange metal watering can |

Each pair has exactly equal canonical word counts and different ordered word sequences. A deterministic I/T/R pipeline receiving only that code, common artifacts and common randomness produces one shared image z for both prompts. If the native reference images are Y₀ and Y₁, then in the same RMS norm,

\[
\max(\|z-Y_0\|,\|z-Y_1\|)\geq\frac{\|Y_0-Y_1\|}{2}.
\]

The stronger mean-square account follows by expanding around the midpoint:

\[
\frac{\|z-Y_0\|^2+\|z-Y_1\|^2}{2}
=\left\|z-\frac{Y_0+Y_1}{2}\right\|^2
+\frac{\|Y_0-Y_1\|^2}{4}.
\]

The [published report](results/fresh-prompt-square/report.json) gives the full-reference PNG response norm over the same 786,432 pixel scalars. The analyzer uses `referenceRms`, rather than candidate error or rounded article values:

| Pair | Native image distance RMS | Worst-endpoint lower bound RMS | Mean endpoint squared-RMS lower bound |
| --- | ---: | ---: | ---: |
| 00 / 10 | 60.40180436 | 30.20090218 | 912.09449259 |
| 01 / 11 | 62.21173060 | 31.10586530 | 967.57485612 |

Units are pixel levels on 0–255, squared in the last column. Values inherit the historical measurement's numerical precision. The midpoint attains the real-valued pair bound, but that midpoint is not a proposed image generator; integer output restrictions can make the attainable error larger.

The shared controls are stronger than matching seed numbers. The [noise seal](results/fresh-prompt-square/noise-seal.json) records one common draw, the image source clones its actual raw noise for every case, and all four [reference image records](results/fresh-prompt-square/images-full.json) share identical schedule payloads. Local metadata joins bind those facts to the reference seal and original request. Historical raw arrays are not freshly rehashed or numerically replayed here.

This excludes exact native-image factoring through that bag on these pairs, including a truly order-independent additive function of these same raw words. It does not exclude position-aware sums, ordered atoms, role-sensitive codes or every compact quotient. Ordered words have no collision among the four inspected strings; that is not proof of their general sufficiency. The references also have [reported object-role defects](fresh-prompt-square.md), so native pixel parity and independent prompt quality remain distinct contracts.

## Approximation needs a radius and an executable recurrence

If all states in one code's fiber share a generated output, the best worst-case error is the radius of their future-output set around an admissible output. Its diameter divided by two is a lower bound. Diameter at most 2ε is necessary for error at most ε, but generally insufficient: several points can require a larger enclosing radius. Even a small radius does not tell us how to find its center under the serving constraints.

An executable approximation needs a separate propagation account. If reduced states require a reconstruction Dₖ into a common state space, define eₖ=‖Dₖ(zₖ)−sₖ‖ and require a bound δₖ on ‖Dₖ₊₁(Tₖ(z))−Fₖ(Dₖ(z))‖. The reconstruction algorithm and its cost are additional obligations. Suppose Fₖ has sensitivity bound Lₖ on a domain containing the native and reconstructed own states. Then

\[
e_{k+1}\leq L_k e_k+\delta_k,
\]
\[
\|Y-\hat Y\|\leq\epsilon_R+L_R\left[
e_0\prod_{j=0}^{K-1}L_j+
\sum_{i=0}^{K-1}\delta_i\prod_{j=i+1}^{K-1}L_j\right].
\]

Here εR bounds ‖R(zK)−O(DK(zK))‖ and LR bounds O's sensitivity. These constants are assumptions until established on the relevant domain; local response ratios are not uniform bounds. BF16 rounding also prevents silently borrowing smooth real-arithmetic guarantees. The [template experiment](frozen-template-transport.md) gives a related warning: smaller response discrepancy can still worsen the update through its alignment with existing error.

## What this changes

The next contextual representation must preserve color-to-object assignment and expose a rule for how that information is consumed and updated. Adding a hand-authored color parser would supply the answer for a narrow grammar while leaving general model-derived interpretation unresolved. Ordered atoms already preserve order; the missing algorithm starts when those operands must become contextual computations and eventually pixels.

The [analyzer](../generator/serving_collision_witness.py) completes in **0.01225s before receipt writing** using public scalar JSON, raw text and tiny finite-state mechanics. Five independent [mechanics checks](results/serving-construction/tests.json) pass. It reads no scientific arrays or images, performs no new model evaluation or learned matrix product, and generates no image. Historical learned reference production remains paid. The [evidence index](results/serving-construction/README.md) pins seventeen source/evidence files and separates counterexamples, no counterexample in inspected records, incomparable controls and missing evidence.

No complete permissible c/I/T/R candidate is specified by this result, and no GPU assay follows automatically. The feasibility analysis has narrowed a representation assumption and made the missing producer explicit. Full research/storage review90 remains current, next93; this cycle adds only compact source, scalar receipts and prose, with no archive or deletion. Arbitrary-prompt zero-neural image generation remains active and unsolved.
