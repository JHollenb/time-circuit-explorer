# Cycle 13 transport archive

The scheduled cleanup moves two transport copies from terminal job `job-68295609457d`: archive 810,756,640 bytes and compressed worker log 817,991,919 bytes, totaling **1,628,748,559 bytes**. All extracted evidence and original custody remain local.

The [plan](plan.json) binds file identity and checksums; the [result](result.json) records full remote SHA256 verification, restoration commands, neighboring `.cold.json` pointers and completed local transport removal. Source: [archive_transports.py](../../../generator/archive_transports.py).

Shared-volume free space is observed 9,524,436,992→11,117,727,744 bytes during the operation. Concurrent/APFS effects prevent assigning that entire increase to this archive. The separate [post-cleanup review](../storage-audit-cycle13/result.json) finds 70 transport name/size candidates totaling 24.732 GB logical; this does not certify them all safe to remove. Ten pointers agree with receipts. Fresh remote type/size checks cover all ten; only the current two objects receive a fresh full-byte rehash this cycle.

Next storage review is due by cycle 16 or earlier pressure. New scientific arrays, lexical atoms and model stores remain near compute.
