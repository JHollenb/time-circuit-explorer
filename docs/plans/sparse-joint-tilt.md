# Sparse joint tilt with an append-only memory tail

Status: **proposed after cycle11 and Astra review; not staged or measured**. The [complete-bank attribution](../complete-attention-bank.md) shows substantial partition and value-weighted exponential terms with signed cancellation. A partition-only oracle still leaves 88.4% paired head error. The 273 MB moment core is about 130 times the complete BF16 K/V payload. The next cut tests a joint numerator/partition response while removing C_vy entirely.

Use the already-scored complete-bank pair as a retrospective representation test, not fresh validation. Preserve the same 64 source landmarks per query head and nearest-source route. Compare fixed support budgets m=1,2,4. No target head, output error or fitted coefficient selects the serving support.

For each anchor, retain its m highest-scoring actual prefix keys under the **anchor** score ℓᵀy, with deterministic address/clock tie order. Keep key/value addresses and a stable weighted tail state. At a query, form the full anchor key mean μ from retained keys and tail, then freeze the centered gauge

\[
t_j=(x-\ell)^T(y_j-\mu).
\]

Partial tilt is gauge-dependent: applying an arbitrary constant only to selected keys changes their mass relative to the tail. Use this declared live mean gauge in every arm; do not tune it from target errors.

For support S, tilt its retained keys exactly and leave unretained keys at their anchor measure:

\[
\widehat h_S(x)=
\frac{H+\sum_{j\in S}p_j(e^{t_j}-1)v_j}
{1+\sum_{j\in S}p_j(e^{t_j}-1)}.
\]

This updates numerator and partition together and is a positive mixture of actual values. Empty support gives the anchor read; complete support gives exact attention. Those identities validate mechanics, not small-support accuracy. No value-range repair or exact fallback is admitted.

Implement the equivalent positive tail-plus-retained aggregation rather than subtracting nearly equal sums. Fixed anchor scores make the top-m threshold monotone as new keys arrive. If an incoming key enters the retained set, append the evicted minimum to the tail; otherwise append the incoming key directly. A rejected or evicted key never needs to re-enter top-m, so the tail only advances by additions and running-maximum rescaling. Masked keys do not enter; repeated token occurrences remain distinct addresses. Preserve checkpoint/resume identities and actual prefix clocks.

Keep a separate **full-query-score oracle** support ranked by |p_j expm1(t_j)| using every allowed key in the current causal prefix. Form it only after the serving-eligible complete banks seal. It scans keys absent from compact serving state, without reading future keys. It tests whether this mass-ranked support can repair the response, not whether the anchor-score selector can find it. It is not selected by target head error, but still uses complete current-query key access and must never be reported as compact serving or a deployment candidate. A poor result would not exclude every sparse support: this is not an output-optimal subset search.

Freeze all methods before scoring. Seal complete candidate banks, then report paired head/output/full-block state responses with active/padding/before8/at8/after8 splits. Original full-block equations, if evaluated, remain a separately paid offline consumer. Retain all per-head countertrends, numerator/partition effects, support overlap, normalized remainder mass captured, actual value-weighted correction and signed residuals. Positive range alone does not establish fidelity. The parent already qualifies the one-address exact control; do not relabel that baseline assistance as prediction.

Charge source assets, all anchor score/routing work, tail statistics, support score/address records, retained K/V payload, workspaces and serialization. Distinguish a simple duplicated per-anchor payload from exact address deduplication within a KV group; shared K/V does not share distinct anchor statistics. Use the 2,097,152-byte full BF16 K/V payload as a reference, not a claimed saving. Do not add higher-order tensors or increase the landmark count in this test.

Mechanical checks cover direct restricted-tilt equality, empty/full support, positive aggregation under extreme scores, stable selection/eviction/ties, masked/repeated occurrences, GQA mapping and resumed own-state execution. Real execution remains one finite guarded CPU mrun lease near retained data, strict preflight, no automatic retry, no duplicate suite. If the representation is promising, a subsequent fresh raw-string test must advance this state from its own history before claiming reusable serving. Full contextual encoding, original MLP replacement, diffusion, general decoding and arbitrary-prompt zero-neural images remain open.
