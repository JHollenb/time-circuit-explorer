# Early storage review and bounded archive

The [audit](result.json) is a read-only snapshot before cleanup. Its broader recursive scan covers 193 artifact directories and finds 74 archive/log size-and-name candidates totaling 28,696,662,553 logical bytes. The field `eligibleTransports` means eligible for inspection by that filter, not verified safe to remove. No volume-wide deletion follows from the list. Logical bytes can share APFS storage and do not predict physical recovery.

The subsequent [cleanup receipt](../storage-cleanup-cycle10/result.json) selects only two transport copies from one terminal run, `job-947395d68069`. Original custody checks, complete remote SHA256 verification, restoration pointers and fresh local identity checks precede removal. The exact archived bytes total 2,335,517,150; extracted scientific evidence and original custody records remain local. The public receipt and plan are byte-identical copies of their originals.

Shared-volume available bytes change from 6,863,237,120 to 10,741,833,728 during cleanup. Concurrent activity and APFS effects mean that entire observed increase cannot be attributed to these two files. An existing separate workspace dedup receipt is attributed separately; this cycle does not claim that work.

The last actual archive is now cycle10, following cycle5's six transport copies. Next review is due by cycle13, earlier under pressure. Full new scientific arrays remain near compute; the new compact scientific publications total 1,962,798 bytes.
