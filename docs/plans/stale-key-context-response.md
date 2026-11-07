# Test whether retained keys can predict the changing context response

Status: **written, unexecuted**. This is a conditional computation-elimination test after an existing baseline, not a zero-neural image backend.

The [propagated-context result](../propagated-context-query.md) finds a 2.050% next-query endpoint change when the whole complement changes, while full K/V formation consumes 98.398% of new matrix MACs. The next question concerns one expensive source dependency: can the changed values be consumed under the retained key allocation and still predict the actual context response?

## Freeze the primary before measuring the missing corners

Keep the exact case, support S=[4,11,12,13], source cuts, selected P residual/Q/K/V, global modulation, absolute positions, numerical profile and operator geometry from job-52b170ce67d8. Authenticate its request, full-byte receipt, phase seals and original parameter recipes. Do not select another support or candidate after examining the results.

Write F_ab for the complete selected attention→own residual→own sequential MLP→own joint2-query trajectory, with keys from a and values from b outside S. Every selected row is restored to the identical shared P operands. Existing immutable observations are F_PP (P-control) and F_QQ (changed complementary context).

Compute exactly two new corners:

- **Primary F_PQ:** retain P complementary keys, use Q complementary values. This is the fixed candidate for the already measured F_QQ continuation.
- **Diagnostic F_QP:** use Q complementary keys, retain P complementary values. It identifies the alternative source dependency and the interaction; it is not an outcome-selected replacement candidate.

Both run identical rectangular attention kernels, original output/MLP maps and their own next-query maps. No Q-attention splice or target refresh occurs. Primary seals and exits before the diagnostic starts. All archived diagonal attention/MLP/T/query outputs remain denied to both actors and open only after both new outputs seal and exit. The shared selected operands and required source K/V banks are supplied, authenticated and paid.

## Score the response instead of hiding behind its large endpoint

For every captured field, compare the candidate response

\[
\widehat D=F_{PQ}-F_{PP}
\]

with the actual changed-context response

\[
D=F_{QQ}-F_{PP}.
\]

Primary error is RMS(Ď−D)/RMS(D), with absolute numerator/denominator retained. Include the zero-response/hold baseline, endpoints, all four selected rows and all 96 query-head addresses. Preserve undefined ratios, signed direction, magnitude, heterogeneity and near misses. Do not substitute the roughly 2% endpoint scale for the much smaller response denominator.

Retain the finite four-corner interaction

\[
I=F_{QQ}-F_{QP}-F_{PQ}+F_{PP}.
\]

At fixed query, the ideal attention interaction is (Π_Q−Π_P)(V_Q−V_P), where Π depends on the changed complementary keys. Native finite reductions and later nonlinear receiving computations remain in the observed interaction. Do not attribute a later interaction entirely to the first attention primitive, or fit a correction from the target.

## Charge the proposed saving and the actual assay separately

The two new actors cost **905,969,664 nominal matrix MACs**, eight affines and two rectangular attention calls. No full bank is projected again in this assay. Existing banks, selected operands, original source prefixes, two diagonal trajectories and original maps remain paid. Strict preflight and one finite CUDA mrun lease apply; no retries, images, full suffix, fitting or precision arm.

If the prespecified stale-key response is useful, its dependency graph could omit one **Q-side full key formation**, 1,536×3,072² = **14,495,514,624 matrix MACs**, plus key-specific head normalization/RoPE. Fresh Q value formation and its input normalization would still be required. This is a conditional opportunity after an available P baseline; it is not a measured serving speedup, first-prompt cost or elimination of learned computation. A useful outcome would motivate producing only fresh values from the actual source state in a later own-serving test.

The [new third-family result](../../../obsidian/blog/2026-11-07-005623-a-third-family-separates-attention-routing-from-values.md) motivates separating allocation from payload. It transports coherent Q/K routes at step0/single4 through full native futures; its route ranking cannot establish which factor dominates this fixed-query joint1 cut. Our next test remains source-bound and quantitative. The full arbitrary-prompt zero-neural goal remains unchanged.
