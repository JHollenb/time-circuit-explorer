# Recovering a state machine, then compiling away its matrices

An exact compiler now recovers a small machine's future-relevant state from output queries and emits a two-register program using only copies and constants. A fresh serving process matches all 35 valid test words and refuses the invalid symbol. A separate complete finite-graph certificate proves the transition and output identities for every finite word over its declared alphabet.

The organism, alphabet and scalar output are fully supplied. This is a compiler mechanics result, with no model-derived language semantics or pixels. **Arbitrary raw prompt → image without neural evaluations remains unsolved.** The site's separate prediction-only generator accepts new text but still evaluates the original learned arithmetic, native encoder and VAE; the zero-neural response chart accepts known contexts and doses. Those interfaces remain distinct.

## Remember what a future can reveal

The machine has two bits, a currently visible value \(v\) and a hidden value \(h\), initially zero. Its two input symbols act as

\[
(v,h)\xrightarrow{a}(h,0),\qquad
(v,h)\xrightarrow{b}(h,1),\qquad O(v,h)=v.
\]

The input is an ordered list of `a` and `b`, with one terminal output. The empty word leaves the initial state unchanged. There is no EOS symbol or raw-text tokenizer. An unknown symbol is outside the contract and is refused.

States \((0,0)\) and \((0,1)\) have the same current output. Appending `a` reveals their difference. A representation that records only the current output therefore cannot advance correctly. This is the small version of the [serving-construction problem](serving-construction.md): compression must preserve distinctions that later consumers can expose.

Let \(f(w)\) be the machine's terminal output after word \(w\). A state described by future observations is

\[
q(s)=\big(O(s),O(T_a s),O(T_{ba}s)\big)=(v,h,1).
\]

The constant third coordinate is useful to linear realization, although it needs no persistent runtime register after lowering. Order matters: \(f(ab)=0\), while \(f(ba)=1\).

## Factor the observations into advancing state

The [frozen construction specification](plans/behavioral-realization-serving.md) defines scalar future-test columns of a behavioral Hankel matrix:

\[
H[u,(x,j)]=f_j(ux).
\]

Here the output has one scalar. The exact rational compiler queries a panel whose prefix and suffix inventories contain all words of length at most two. It selects separating prefixes \(P=(\epsilon,b,ba)\) and future tests \(E=(\epsilon,a,ba)\), giving

\[
B=H[P,E]=
\begin{bmatrix}0&0&1\\0&1&1\\1&0&1\end{bmatrix}.
\]

For a symbol \(x\), put \((B_x)_{ij}=f(p_i x e_j)\), and let \(F_P\) hold the current outputs at the basis prefixes. In row-vector convention,

\[
M_x=B^{-1}B_x,\qquad C=B^{-1}F_P,\qquad z_0=(0,0,1).
\]

The recovered operators are

\[
M_a=\begin{bmatrix}0&0&0\\1&0&0\\0&0&1\end{bmatrix},\qquad
M_b=\begin{bmatrix}0&0&0\\1&0&0\\0&1&1\end{bmatrix},\qquad
C=\begin{bmatrix}1\\0\\0\end{bmatrix}.
\]

Thus \(z_{k+1}=z_kM_{x_k}\) and \(f(w)=z_0M_wC\). The compiler obtains these coefficients through its word-output query interface. A separate verifier has privileged access to the supplied finite graph. It checks the initial state, all eight reachable state/symbol edges and all four outputs. Those equalities support induction over every finite input word. The nonsingular minor supplies a global rank lower bound of three; the complete three-coordinate realization supplies the upper bound. The exact global behavioral rank is therefore three **for this supplied machine**.

An invertible sampled panel by itself would establish neither the global rank nor transition closure. For an unrestricted native image function, the finite graph used by this certificate is unavailable. That missing authority is a substantive research gap.

## Lower the computation, including its readout

Running generic learned \(M_x\) and \(C\) would still evaluate a learned recurrent map. The next compiler stage therefore inspects the actual expressions and permits only copying an old register or writing a byte constant. It proves the third coordinate invariant and eliminates it.

| Instruction | New register 0 | New register 1 |
|---|---|---|
| Initialize | constant 0 | constant 0 |
| Read `a` | old register 1 | constant 0 |
| Read `b` | old register 1 | constant 1 |
| Terminal output | copy register 0 | — |

Every transition reads the complete old state before committing either new register. The emitted [program](results/behavioral-realization/program.json) has two persistent byte registers. Python objects, dispatch and the temporary next-state list require additional memory. Only bit values are reachable, but this implementation does not claim a two-bit total allocation.

The fresh `python -I` child opens the program, request and interpreter source. Twelve denied-file canaries cover the source graph, compiler/study sources, rational realizations, certificates and query ledgers; two denied-import canaries cover NumPy and PyTorch. The child contains no coefficient-matrix evaluator or source-oracle call. Its prediction hashes seal after exit and before source comparison. These are audited trusted-code checks, without an adversarial sandbox or hardware operation counter claim.

## Keep the informative failures

Using only the empty suffix produces a rank-one sampled panel and an all-zero trajectory from the initial state. It reproduces the current readout map on encoded source states, yet merges states with different futures. The shortest suffix distinguishing `00` from `01` is `a`; the first whole-word failure from the initial state is `ba`. The [shallow certificate](results/behavioral-realization/shallow-certificate.json) preserves all four failed edges and both merged-state witnesses.

An exact coordinate change exposes a different limit. Set

\[
w=(v+h,h,1).
\]

This representation still closes exactly. Under `b`, its first new coordinate is \(h+1\); the readout is \(w_0-w_1\). The copy/constant lowerer refuses those expressions. Closure has survived, while this coordinate system requires operations outside the declared instruction set. The [gauge control](results/behavioral-realization/gauge-lowering.json) is a fixed declared transformation, with no coordinate search. Its refusal does not prove that another program or coordinate choice is impossible.

The useful mathematical target is therefore a representation that is both future sufficient and economical under a declared execution language. Low rank is one description of behavior. It does not determine the cost of executing that behavior.

## What ran and what it cost

One tiny local standard-library mechanics run produced 17 raw JSON files totaling 51,859 bytes. No model, neural runtime, scientific tensor array or image was loaded. The supplied graph is an oracle for mechanics; its output calls remain charged as supplied-source work.

| Scope | Observed work |
|---|---:|
| Main compiler | 83 query requests; 33 physical queries; 50 cache hits |
| Main compiler source transitions | 108; maximum physical query length 5 |
| Shallow compiler | 13 requests; 9 physical queries; 4 cache hits |
| Shallow source transitions | 16; maximum physical query length 3 |
| Whole main study before report | 170 source output calls; 636 source symbol transitions |
| Main-process row/matrix arithmetic | 940 exact rational linear MACs |
| Fresh child | 226 transitions, 226 register copies, 226 constant writes, 35 readout copies |
| Fresh-child matrix evaluations / source calls | 0 / 0 |
| Fresh-child valid results / explicit invalid refusal | 35 / 1 |
| Child serving before receipt write | 0.538 ms |
| Study elapsed before report write | 36.521 ms |
| Complete command process wall | 156.101 ms |

The 31 short words cover lengths zero through four; four additional words have length 32. The compiler's inventories stop at length two, but concatenated shifted queries reach length five. The complete graph certificate supplies the all-word result; the 35 serving examples alone do not.

Exact Gaussian elimination, graph traversal, JSON, file/import guards, interpreter dispatch and process overhead are additional to the rational linear-MAC count. A nominal dense rank-three evaluator would use nine coefficient MACs per symbol and three per scalar readout; that is an arithmetic illustration, without a matched runtime benchmark. No energy or speedup is established by these single-run times.

Six targeted mechanics tests pass at the executed source snapshots. They include independent fraction/inverse checks, the penultimate-bit output identity, shallow counterexamples, constant elimination, the coordinate limitation, simultaneous register updates, invalid operators/symbols and rank-zero handling. Historical broad-suite counts are separate. [Evidence](results/behavioral-realization-evidence/README.md) retains exact copies, hashes, query/answer ledgers, tests and advisor scopes.

## The image-generation gap remains concrete

For this toy, the source graph supplies the behavior and the future tests discover how to represent it. No component discovers the meaning of a natural-language prompt, creates a contextual image state or derives a general pixel program. Supplying a new renderer at the last boundary would assume the computation we are trying to recover. Storing full output images for prompts would retain the excluded cache construction.

The original question therefore has two answers: arbitrary new text is accepted by the [reconstructed learned generator](prediction-only-benchmark.md), with its learned arithmetic paid; arbitrary text is not supported by the [zero-neural response chart](zero-network-image-response.md). This compiler result advances the mechanics required to investigate the stricter goal, without changing either interface's capability.

Full research and storage review 90 remains current; both are due again in cycle 93. Astra's actual-result review recommends a [certificate-authority analysis](plans/behavioral-certificate-authority.md): identify which extra structural assumption or source-derived proof can turn a finite query transcript into a complete realization. A finite-state alternative can agree on every recorded word yet differ at an unqueried longer word. That leaves this complete supplied-graph certificate intact while exposing what a native black-box interface lacks. The next full review may supersede this written, unexecuted cut; no native-image Hankel campaign follows from toy success. The [living log](/Users/jakeholl/domains/obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md) keeps that decision and the active goal together. Related posts develop [state and context](/Users/jakeholl/domains/obsidian/blog/2026-11-05-153154-state-context-and-the-mathematics-of-time-circuits.md), [the simplifying assumption](/Users/jakeholl/domains/obsidian/blog/2026-11-05-161109-the-one-assumption-that-would-make-model-computation-easier-to-understand.md), [consumer metrics across time](/Users/jakeholl/domains/obsidian/blog/2026-11-06-173328-the-consumer-metric-needs-a-clock.md), and [advancing the states a program creates](/Users/jakeholl/domains/obsidian/blog/2026-11-06-184700-compiling-cheaper-transitions-on-the-states-they-create.md). These are local workspace backlinks.

To reproduce the supplied mechanics in a new output directory:

```sh
rtk proxy ../common/bin/common exec python generator/behavioral_realization_study.py artifacts/my-behavioral-realization
```

The [compiler](../generator/behavioral_realization.py), [interpreter](../generator/behavioral_program.py), [study](../generator/behavioral_realization_study.py) and [tests](../generator/test_behavioral_realization.py) are source pinned. The original written specification remains unchanged as historical design evidence; its supplied-machine mechanics portion is now executed here. No complete image generator or GPU assay is queued.
