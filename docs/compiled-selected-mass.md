# Producing selected attention weights from a compiled score function

The selected-weight producer now runs from clean token IDs, masks and positions, using its own address selector. On four covered histories, all 1,908 padding-query supports, selected score words and weight words match a separately coded reference exactly under the declared source profile. No online Q/K products or historical weight packets enter serving. This closes one previously supplied coefficient boundary. Scalar attention normalization still runs, offline learned compilation remains paid, and arbitrary-prompt zero-neural image generation remains unsolved.

The [full review 99](results/research-review-cycle99/README.md) selected this single construction after the [allocation/value crossover](first-read-route-payload.md) and [source-cost feasibility study](relative-mass-producer-feasibility.md). Its [original plan](plans/compiled-selected-mass.md) and [implementation contract](plans/compiled-selected-mass-implementation.md) remain frozen. There is no vocabulary/head expansion, new native model, value mixing, contextual suffix or image in this result.

## The computation

Let the admitted history be $H_i=\{(u_j,j):j<i\}$, with 30 covered key IDs, one fixed padding-query token/head, and 512 absolute positions. The qualified source score is

$$S(i,u,j)=\operatorname{fsum}_r\left[\operatorname{FP64}(q_{i,r}k_{u,j,r})\right].$$

The operands come from the existing source-qualified BF16 rotary words, widened and scaled as in cycle 95. Compilation stores these scalars in little-endian FP64 at element address

$$\operatorname{offset}(i,a,j)=30\,i(i-1)/2+a i+j,$$

where $a$ is the token's alphabet index. Every complete position block matches its previously retained score hash. The asset contains 3,924,480 entries and 31,395,840 bytes; it is a reusable primitive table, without a whole-history answer cache.

The unchanged [integer selector](../generator/source_order_program.py) maintains its own history and chooses up to four addresses $J(i)$. The fresh [stdlib consumer](../generator/compiled_mass_program.py) retrieves their scores and executes, in selected order,

$$m=\max_{j\in J(i)}S(i,u_j,j),\qquad e_j=\exp(S(i,u_j,j)-m),\qquad Z=\operatorname{fsum}_j e_j,\qquad w_j=e_j/Z.$$

That displayed equation describes the real-arithmetic law; the implementation fixes the actual max/subtract/`math.exp`/`math.fsum`/division order. Python3.14.6, Linux/glibc2.39 and the actual mapped libm hash are recorded. Empty support emits empty weights and four padded zeros; short support normalizes only valid keys; underflow zeros remain. Unknown admitted tokens invalidate subsequent queries even when a particular edited address would otherwise go unselected. Tiny supplied fixtures cover these cases.

## What the measured histories show

Historical A/B and fixed synthetic C/D each have 35 active keys followed by 477 padding queries. The [complete 2,560-row record](results/compiled-selected-mass/observations.jsonl.gz) includes all five packets, active-query refusals and the unknown-token control. The [panels](results/compiled-selected-mass/panels.json) preserve 1,908 supported rows and 652 unsupported rows. The separately coded Python coordinate-product reference uses the same qualified operands and numerical policy; it is not a new native-reader reference. All supported addresses, selected scores and weights agree bit-for-bit in that profile.

The historical pair changes 426 ordered supports and changes weights on two further unchanged-support rows; 49 rows have zero weight change. Comparison with the old NumPy weights has maximum absolute discrepancy $4.9960\times10^{-16}$, with no support difference. The [local scalar audit](results/compiled-selected-mass/scalar-audit.json) recomputes the selected normalization with this host's math library: six words differ, by at most $5.5511\times10^{-17}$. Both discrepancies remain observations; neither is converted into a claim of universal native finite parity.

The [new pair](results/compiled-selected-mass/pairs.json) has **zero response on all 477 rows**. Its fixed position-8 substitution never enters either selected support, as the [additive coverage audit](results/compiled-selected-mass/edited-address-coverage.json) confirms. C/D prediction packets are byte-identical. This exercises a new covered history but does not demonstrate responsiveness to a new selected edit or prompt semantics. The cases were retained unchanged. Replacing position 8 with an uncovered ID causes all 477 later padding queries to be refused, rather than silently ignoring that token.

## Work, time and storage

One guarded CPU mrun lease, `job-833def2c0ba2`, succeeds in 14.64 scheduler seconds with sampled peak RSS141.1MB and zero VRAM. Admission chooses4096MB/four scheduler threads; actual numerical-library threads are one. No retries or second scientific job run.

| Scope | Measured work or time |
|---|---:|
| Offline compilation |502,333,440 coordinate products;3,924,480 score reductions;11.7258s including serialization/hash/metadata|
| Shared serving asset authentication/load/decode |68.039ms|
| Per known512-position packet, routing plus weights |6.259–9.609ms|
| Incremental selected-score normalization and packet formation |0.840–1.065ms per known packet|
| Serialization and hash |8.456–8.581ms per known packet|
| All five packets, fresh child wall |170.928ms including startup|
| Separate post-seal scalar reference |8,547,840 coordinate products;66,780 score reductions;0.4376s diagnostics including output writes|

Serving pays 7,632 score lookups, subtractions, exponentials and divisions, plus1,908 normalization sums and66,780 integer rank lookups. Max comparisons, history, authentication, runtime setup and I/O are additional. Parent CPU12.5775s excludes child CPU0.1374s. Worker wall13.3953s starts at execute entry and ends before the final report/provenance copies. These scopes are distinct; no matched speedup is measured.

The new score asset31,395,840B plus existing raw ranks7,848,960B totals39,244,800B before runtime, indices, inputs and outputs. The history's560logical integer bytes do not describe total memory. Child high-water RSS104,189,952B starts from an already103,628,800B high-water baseline and can include inherited pre-exec usage; it is not isolated steady-state serving RAM. [Storage practice](results/compiled-selected-mass-evidence/storage-practice.json) keeps the full score file near compute. Only30original compact output files/1,364,961bytes plus review/QA records are collected. No archive/deletion occurs; full storage99 remains current,next102,last actual archive78.

## What is now missing

Cycle 97 needed supplied weights; this port forms them from its allowed input and source assets. It still computes attention normalization and retains compiled learned coefficients. It supplies neither learned values nor their mixed write, the first contextual residual/MLP update, the next layer's Q/K/V, denoising or pixels. Literal table scaling from cycle 98 remains a warning, not an information-theoretic lower bound.

The next question is the first contextual-state producer boundary where token-local identity ceases to determine the operand. The [source-bound construction contract](plans/contextual-state-construction.md) fixes own H1 and its next-layer Q/K/V consumer, with a raw-input initializer and explicit paid inputs. It is not an implemented non-neural successor or a queued assay. Completing this port does not authorize another cache, head, codec or dense-factor campaign. The strict arbitrary-prompt zero-neural goal remains active.

The [publication](results/compiled-selected-mass/publication.json), [execution receipt](results/compiled-selected-mass/execution.json), [compiler ledger](results/compiled-selected-mass/compile-compiler.json), [serving receipt](results/compiled-selected-mass/prediction-receipt.json), [six-test receipt](results/compiled-selected-mass/mechanics.json), [source review](results/compiled-selected-mass-evidence/advisor-source.json), [driver](../generator/compiled_mass_study.py) and [independent byte/scalar publisher](../generator/publish_compiled_mass.py) preserve the evidence. Public-scalar QA does not independently recompute full source QK arithmetic; all complete score bytes and512segments are freshly verified remotely without local array hydration.
