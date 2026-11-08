# An exact state update can still be the wrong representation

We can advance the compiled attention order exactly using integer state updates. The proposed representation is nevertheless larger and slower than the previous table. Its compressed bytecode occupies8.06MB versus7.14MB, its packed serving index occupies15.64MB versus7.85MB of original ranks, and recorded A/B serving takes31.85–35.73ms versus the prior5.62–5.70ms. Those timings are unmatched historical observations, not a controlled speed comparison.

The result closes one fixed encoding branch. It does not show that attention lacks a compact discrete program. In particular, this table orders alternative tokens at the same position that cannot coexist in a real history. A cheaper comparator could exploit that weaker legal-history requirement. No grammar tuning, vocabulary/head expansion or image continuation follows automatically.

## The function being preserved

The [previous experiment](source-ordering-program.md) compiled one padding query token, head0, a30-token alphabet and512absolute positions. It replaced fresh Q/K scoring with integer ordering. For each clock i, a total order π_i sorts every candidate pair(token,j) with j<i. A legal live history chooses one actual token per admitted position; restricting π_i to that subset recovers its top four.

This is our [approximate first-reader support port](uncentered-reader.md), not native semantic dispatch. All-history correctness concerns the new pinned bank-independent score contract. The954old-reader matches are separate finite observations. We retain all those distinctions: soft weights, values, the contextual write and later model computation are not supplied by the address program.

The [scheduled Astra review96](results/research-review-cycle96/README.md) inspected current source-bound evidence across the image, chat, ISA and contextual-runtime work. Forty-eight of49prior content pins are unchanged; the living log advanced. A newly reviewed history-reader-role finding sharpens the address/payload distinction: a fixed initial allocation can carry changed values, while later queries change after the native MLP. No inspected newer result supplies a general cheap contextual initializer or pixel producer. The review selected one fixed recurrence feasibility cut rather than another dense-response campaign.

## From copies to a sparse state law

The compiler scans π_(i+1) in target order. A new candidate(token,i) emits INSERT(token). An existing candidate begins a maximal contiguous COPY from π_i. The grammar is fixed before execution:5bytes for COPY(start,length),3bytes for INSERT(token), and a2-byte operation count per clock. There is no alternative decomposition or format search.

The first draft would have expanded every global order for every serving packet. Before any execution, a mathematical factorization removed that unnecessary work. If a source interval[a,a+ℓ) is copied to target interval[b,b+ℓ), each candidate in it obeys

\[
r_{i+1}=r_i+(b-a),\qquad a\le r_i<a+\ell.
\]

Each clock’s COPY source intervals form a disjoint complete partition of the old ordinal domain. Sort their starts once, store signed deltas, and binary-search the interval containing each live key’s rank. INSERT gives the new key’s initial rank. With at most15,330candidates, starts fit unsigned16bits and deltas fit signed16bits.

The live state contains only admitted positions, token indices and their current ranks. A key admitted at the current active position remains pending until the next clock includes it. The clock+1 control owns a separate lookahead state and can initialize the current key from its already available INSERT. Unknown active IDs remain in history as unsupported coverage, never as silently dropped keys.

Thus serving advances its own compact history through a piecewise integer translation. It reads no Q/K vectors, scores, original complete rank-table arrays or native outputs. The [program](../generator/order_clock_program.py), [driver](../generator/order_clock_study.py) and [frozen plan](plans/order-clock-recurrence.md) define the complete computation.

## Exact execution, poor economy

One guarded CPU assay validates every3,924,480sparse transported/inserted ordinal against a separate full-order decoder. All512decoded-clock hashes then match the old rank-table ledger after serving exits and seals. All52prediction gzip files are byte-identical to the preceding experiment, covering26,624method rows. Supported coverage remains3,788queries, including954historical and523short-prefix rows; all four transfer histories remain unsupported. There is no new prompt or image capability.

The COPY distribution explains the poor representation:

| COPY length | Operations |
|---|---:|
| 1 | 3,894,152 |
| 2 | 7,400 |
| 3 | 57 |
| 4 | 4 |
| 5 | 1 |
| 6 | 1 |

There are3,901,615COPY operations and15,330INSERTs. Fully99.8087%of copies contain just one identity; mean copied length is1.00193. A changing total order need not preserve long adjacent runs, even when individual scores vary continuously.

The fixed raw byte formula is

\[
B=5C+3(15{,}330)+2(511)=5C+47{,}012.
\]

Beating the original7,848,960raw bytes would require C≤1,560,389. The actual program is19,555,087raw bytes,2.4914times the table. Gzip9 compresses it to8,062,001bytes,12.92%larger than the previous7,139,604bytes. This is an observed cost of the chosen grammar; maximal contiguous runs do not imply globally optimal compression.

Serving also needs the source-sorted translation index. Packed starts/deltas cost4Cbytes, clock offsets2,052bytes, end sentinels1,024bytes and INSERT ranks30,720bytes. Total resident numerical index is15,640,256bytes,1.9927times the old raw ranks. The compressed program remains a separate stored artifact. Smaller serialized code, had it occurred, would not by itself imply smaller resident memory.

## The measured work

| Work | Measurement |
|---|---|
| Integer compilation | 2.414s;3,924,480source rank reads;511permutation checks/sorts and511target-order sorts |
| Fixed gzip serialization and metadata | 18.340s |
| Load, decompression and hashing | 50.292ms |
| Build serving index | 1.797s;511source-interval sorts |
| Separate full/sparse ordinal proof | 8.513s;3,924,480checks;51,754,593interval comparisons |
| A / B primary serving, excluding startup/outputIO | 35.730 / 31.853ms |
| A / B rank transport | 17,255binary searches and17,290rank writes each |
| A / B interval comparisons | 218,770 / 218,786 |
| A / B top-four comparisons | 57,838 / 57,398 |
| Complete fresh child, including index/proof/all52methods/IO | 12.519s |
| Complete CPU job | 33.87s;162.7MBreported peakRSS;zeroVRAM |
| New learned score products, original model/encoder/denoiser/VAE evaluations, images | Zero |

A/B still admit35keys and support477queries each. Their840logical history bytes describe three int64 fields per key, not Python objects, outputs, unknown lists, index storage or peak process memory. Full order/inverse/raw bytecode references are released after the separate proof before packet serving. Verification temporarily uses coexisting order payloads up to61,260bytes, a30,720-byte inverse array,15,360validation bytes and small COPY slices. Child high-waterRSS99,036KiBcan include inherited pre-exec process history; it is not isolated steady serving memory.

Unsupported transfer packets still pay known-key transport; their timings are not successful fast predictions. The synthetic511-active-key packet pays130,305binary searches for its one supported final query. All cases and controls remain in the [serving receipt](results/order-clock-recurrence/prediction-receipt.json). These costs show why only counting top-four reads would miss the state update.

The original lexical assets and502,333,440offline score products remain paid. This study adds integer compilation and proof work, rather than making that preparation disappear. No parameter vector or learned matrix is evaluated online, but the finite program remains source-derived learned information.

## What this changes

The working inference is narrow: adjacent-clock contiguous copying does not expose economical structure in this stronger global order. The sparse evaluator proves a useful state law and avoids expanding irrelevant candidates per packet, but its enormous translation index and extra searches erase that advantage here. We retain the exact program as evidence and keep the simpler table as the better measured representation for this domain.

The next goal-facing dependency is the payload and its receiving update. Ordered addresses alone do not determine attention magnitudes: scores with the same ordering can give different normalized weights. A value table alone likewise does not establish the contextual read or the state consumed by the next query. The full96review’s history evidence and [ISA register work](../../isa/docs/executable-symbol-scan-lowering.md) favor specifying that actual producer/read/update contract, rather than interpreting a compressed address interface as a full model program.

Astra recommends one [saved-only allocation/value question](plans/first-read-route-payload.md) at the same477padding queries. Crossing captured allocations with the two token-local value banks would separate their read effects. Because only address8changes value, ideal payload and interaction both lie along that one value difference; source arithmetic and BF16 can add a finite remainder. Existing value-event/covariance work must first be checked for the same specimen, and retained fields must authenticate. At most488,448value products form the two cross reads, with endpoint replay separately charged. This is a proposed paid diagnostic of the missing function, not its new producer. The plan is written and unexecuted; no further assay is queued.

## Evidence and review

Eight targeted supplied-integer checks pass once in0.050s, including81tiny legal histories through all four methods and fresh file/import exclusions. The [independent publisher](../generator/publish_order_clock.py) separately decodes every ordinal, validates maximal COPY intervals and byte/index counts, and checks every method row, source/test/request/payload/seal join. All64original output files,8,558,900bytes, are [published byte-exact](results/order-clock-recurrence/publication.json). Complete neural operands remain near compute.

The [evidence index](results/order-clock-recurrence-evidence/README.md) separates full96survey, exact source review before the first test/stage/run, actual-result review and later narrative review. Reporting-only timing/status/counter corrections occurred before any execution. Sources, tests, plan and raw outputs remain frozen afterward.

The [storage96audit](results/storage-audit-cycle96/README.md) checks93recovery pointers, with nine fresh full hashes totaling3.806GBand84older metadata-only remote checks. There are zero large transport candidates, no archive/deletion, and no bulk neural-array hydration. Next full research/storage review is99or earlier material evidence/pressure. Arbitrary-prompt zero-neural image generation remains active and unsolved.
