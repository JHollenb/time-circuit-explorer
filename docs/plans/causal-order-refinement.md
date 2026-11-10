# Preserve the parent while refining a contextual address

The [finite Cauchy update](../cauchy-writer.md) improves the current first contextual-state approximation, but produces no new responding row. The next construction should change one distinction that the program can express. It should not spend another iteration improving the same constants while the vessel histories remain indistinguishable. This is a mathematical construction decision, not a frozen assay or execution admission.

## The state that the current writer actually consumes

At row i, the writer reads a lexical state h⁰ᵢ and two resolved addresses ℓ₀(i), ℓ₁(i). Its finite computation is

\[
\widehat h_i=Q\bigl(Q(h_i^0+a_{0,\ell_0(i)})+a_{1,\ell_1(i)}\bigr),
\]

where Q is the declared FP32 then BF16 admission. Equal lexical rows and equal addresses produce equal outputs for every choice of shared atoms. Applying a deterministic next normalization cannot recover a distinction that the writer never produced.

The vessel histories already have different causal first-position registers for token18575 and token94607. Their padding rows nevertheless reach the same two atoms. This is a collision in the selected address, not a proof that the155 available features contain no order information. Define the consumed address as aᵢ=(h⁰ᵢ,ℓ₀(i),ℓ₁(i)). A deterministic writer is constant on each equivalence class of aᵢ. Refining that partition is necessary to produce a difference inside one such class; it does not by itself make the difference useful to the next consumer.

For two colliding rows with distinct native targets y₊,y₋ and positive weights w₊,w₋, any shared prediction p obeys the exact factorization

\[
w_+\|p-y_+\|^2+w_-\|p-y_-\|^2
=(w_++w_-)\|p-\bar y\|^2
+\frac{w_+w_-}{w_++w_-}\|y_+-y_-\|^2,
\]

with the weighted target mean ȳ. The last term is an unavoidable endpoint error inside that consumed-address class. A paired-response target is sharper: its predicted difference is zero for every atom update, so any nonzero target difference has100% relative response error. This restriction belongs to the current writer/address, not every possible non-neural program.

## Read a relation rather than an absolute clock threshold

From the existing causal registers f_A(i), f_B(i), define

\[
o_i=\begin{cases}
+1&0\le f_A(i)<f_B(i),\\
-1&0\le f_B(i)<f_A(i),\\
0&\text{one is missing, equal, or otherwise unknown.}
\end{cases}
\]

Only already observed active tokens update these registers. Padding retains the last available relation; future tokens cannot change a previous row's feature. Translating both first positions preserves o. The entire inherited tree is not translation invariant: it still contains its old position and lexical predicates.

Append this one derived feature,155→156. At old slot1 leaf8, use two ordinary binary threshold nodes: `o <= -0.5` selects a negative clone, otherwise `o <= 0.5` keeps the original neutral atom, otherwise select a positive clone. Do not label a ternary operation as one old binary predicate. The actual maximum tree depth changes4→6, slot1 atoms9→11 and total atoms23→25. The existing16-leaf bound per slot still fits; its depth bound must explicitly change. Two additional2560-coordinate FP32 atoms cost20,480bytes, plus tree/schema metadata. At most two presence tests and one integer order comparison form the relation; traversing this leaf adds one or two threshold tests per affected row. No new vector multiplication is required during serving.

Both clones start as exact byte copies of the accepted113 atom8. All old atoms, slot0, the prefix, and all other paths stay fixed. This gives an exact semantic parent null for all valid raw inputs, including inputs outside the measured prompts. It does not give the refined artifact the old program hash. Retaining the parent after a rejected update should restore the original parent directory and identity. Neutral rows and paths outside old leaf8 retain their parent computation at every dose.

The selected literal pair and eligible old leaf are evidence-driven restrictions. They are not an arbitrary-prompt semantic resolver. A successful local result must not become a claim that a two-word order rule solves contextual encoding.

## Optimize only the newly available contrast

Let C₊ and C₋ be discovery rows that reach old leaf8 and have positive or negative order. Neutral rows are outside the update. Let wᵢ be the original positive endpoint weight, m₊=Σ_{C₊}wᵢ, m₋=Σ_{C₋}wᵢ and M=m₊+m₋. If either mass is zero, retain the parent; an unobserved branch must not acquire a fitted value by invention.

Choose the fixed contrast design

\[
b_i=\begin{cases}m_-/M&i\in C_+,\\-m_+/M&i\in C_-,\\0&\text{otherwise},\end{cases}
\qquad D=\sum_iw_ib_i^2=\frac{m_+m_-}{M}.
\]

For g in the existing objective convention, with its endpoint and pair weights already included, form

\[
S=\sum_i b_i g_i,\quad r=-S/D,\quad q_i=b_i r.
\]

Then, in ideal arithmetic,

\[
\sum_iw_iq_i=0,\qquad g^Tq=-\|S\|^2/D\le0.
\]

Do not apply w a second time to g. This construction has one learned2560-dimensional contrast, despite storing two child atoms. It removes a shared baseline degree of freedom, so another general improvement in old constants cannot conceal whether the new contextual distinction helps. Mean preservation concerns the ideal weighted perturbation, not individual rows, pair responses, finite admitted writes, or behavior on a new prompt distribution.

Recompute g at the accepted113 H and its actual normalized Z. Reusing the old110 gradient would answer a different question. Use the complete ideal RMS/gamma forward Jacobian J at those own parent rows. With the original endpoint/pair metrics M_H and M_Z,

\[
\beta=q^TM_Hq+(Jq)^TM_Z(Jq),\quad
m(\alpha)=L_0+\tfrac\alpha2g^Tq+\tfrac{\alpha^2}{4}\beta.
\]

For a nonzero ray choose one analytic dose α_model=−gᵀq/β, capped by the same explicitly exploratory η=1/16 row-relative radius. This is a ray minimum of a declared Gauss–Newton model. It is not a minimizer or derivative guarantee for the discrete finite program.

Update only the two clones as `FP32(oldAtom8 + alpha*b±*r)`. Re-execute the original two finite additions. FP32 atom rounding and BF16 state admission can break exact centering and alter the ideal direction; measure both. Zero dose must use byte copies, including signed-zero bits.

## One finite future and a consequential decision

The future assay must bind the accepted113 program/H/Z, original targets/weights/gamma/profile and fresh owned raw input. First build a refined null, then one finite candidate. Evaluate original normalization only as explicitly paid offline feedback. Keep a strict ceiling of six fresh full512-row original normalization/gamma evaluations: four discovery histories, then two known-transfer histories after sealing the discovery choice. Parent and refined-null states alias the already captured parent if their bytes agree. No new native affine, image, route search, extra dose, new prompt set or data expansion is implied.

Seal selection from the original discovery objective; the exact parent wins a tie. Keep all four loss terms, active/padding endpoint and paired responses, every regression and the rejected candidate. Known transfer is not an independent untouched test. Fresh stdlib serving should demonstrate the original parent, refined null, candidate and selected program on the existing seven prompts, including the novel-input demonstration. Runtime and storage work must include that extra null arm. These are dependent demonstrations, not28 independent cases or a per-prompt benchmark.

The informative outcome is whether a genuinely new input-owned address produces a useful finite contextual response, and what remains wrong after the collision is removed. It does not require every row to improve. If the new response is poor or the candidate loses, preserve its sign, magnitude and consumer effects. Do not turn that into a proof that context is impossible, and do not start an automatic leaf/depth/dose campaign. The next architectural decision must use those effects to choose a richer recurrent state or a different writer law.

## The simplifying assumption and its limit

For a general time circuit, the powerful assumption would be a compact input-owned state s with an inexpensive update T and a sufficient consumer map R:

\[
s_{t+1}=T(s_t,x_{t+1}),\qquad y_t=R(s_t),
\]

where future behavior depends on the prefix only through s. Exact future equivalence is a demanding global condition; local internal-coordinate equality is neither necessary nor sufficient for useful approximate image behavior. The order relation refines one known collision in the current state partition. It is one test of how to construct such a state, not evidence that a small universal sufficient state has been found.

The new [algorithm review](../../../obsidian/blog/2026-11-10-113100-math-models-and-faster-algorithms-where-the-research-stands.md) makes simple baselines and full work accounting consequential. Native fewer-step images and native draft verification can be useful speedups while retaining neural evaluations. They do not satisfy this goal. The [symbol-table review](../../../obsidian/blog/2026-11-10-111430-where-the-symbol-table-work-stands-against-the-time-circuit-architecture.md) identifies the unresolved resolver/operand formation, and the [own-state MLP work](../../../obsidian/blog/2026-11-06-111533-improving-an-mlp-on-its-own-predicted-states.md) requires the state actually produced to survive the next consumer. Complete conditioning, autonomous denoising, general pixel formation and independent prompt faithfulness remain open.
