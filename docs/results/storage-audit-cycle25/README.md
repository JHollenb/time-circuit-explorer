# Cycle 25 storage audit

The [pre-cleanup audit](before.json) found 64 name/size candidates totaling 20,181,806,531 logical bytes. The [post-cleanup audit](result.json) finds **62 candidates / 18,645,388,207 logical bytes** after the [verified two-file archive](../storage-cleanup-cycle25/README.md). Candidates are review leads, not deletion certificates or unique physical bytes.

All **18** restoration pointers match archive receipts and original custody; fresh remote regular-file/size checks cover all 18 cold objects. The earlier 16 objects were not full-byte rehashed in this audit. Only the new two received fresh full-SHA verification during their archive transaction. Post-audit shared-volume free space was observed at 15,617,187,840 bytes. Next periodic review is due by cycle 28 or earlier under pressure.
