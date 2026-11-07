# Cycle75 transport archive

Two complete-custody terminal-failed transport snapshots were archived losslessly: **677,763,634 bytes** from job `job-a84ab88fb0eb`. The job remains failed. Extracted scientific evidence and original custody records remain unchanged.

Both new cold copies received complete SHA-256 verification before durable restore pointers were fsynced and local transport copies removed. [Plan](plan.json), [result](result.json), and [events](events.jsonl) retain exact paths, hashes and restore commands. The archive driver is [storage_terminal_snapshots.py](../../../generator/storage_terminal_snapshots.py); its terminal-state extension preserves failed outcomes.

The [post-audit](../storage-audit-cycle75/result.json) checks 84 existing pointers against custody and remote regular-file/size metadata. Only the two new cold copies received fresh full hashes in this transaction. Nine remaining candidates total 3,806,240,603 logical bytes; their names and sizes do not authorize deletion. Shared-volume free-space changes are observations, not exclusive physical attribution.
