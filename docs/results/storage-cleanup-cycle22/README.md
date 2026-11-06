# Cycle 22 transport cleanup

The [result](result.json) records two redundant transport copies from terminal job `job-b80f449f3667`, totaling **1,537,861,671 bytes**. Its archive and compressed worker log moved to content-addressed cold storage on Beast. Complete remote SHA-256 checks, durable restoration pointers, and fresh local file/custody identity checks preceded removal. Extracted scientific evidence and original receipts remain local.

The [plan](plan.json) binds each source path, size, hash and original custody record. Shared-volume free space was observed at 13,237,059,584 bytes before and 14,754,340,864 after. These APFS/shared-volume figures are not an isolated physical-space attribution. The [post-cleanup audit](../storage-audit-cycle22/README.md) separately checks restoration records and lists remaining review candidates. The next periodic review is due by cycle 25, earlier under pressure.
