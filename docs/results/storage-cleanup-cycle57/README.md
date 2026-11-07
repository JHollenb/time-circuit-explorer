# Cycle57 scheduled transport cleanup

Four terminal-run archive/log transport copies totaling **626,249,557 bytes** were copied to content-addressed Beast cold storage, fully SHA-256 verified and given durable restoration pointers before local unlink. All extracted scientific evidence and original custody remain local.

The [frozen cleanup plan](plan.json), [event stream](events.jsonl) and [result](result.json) record the original files, terminal jobs, local identities, complete checksums, cold paths and exact rsync recovery commands. This transaction used [storage_transports.py](../../../generator/storage_transports.py). It did not remove model parameters, raw extracted results or failed scientific outcomes.

Shared-volume free space was observed at **10,739,851,264→11,367,358,464 bytes** during the transaction. Concurrent workspace/APFS changes prevent attributing the entire difference physically to these transports. The [post-audit](../storage-audit-cycle57/README.md) checks 60 restoration-pointer/custody joins and fresh remote regular-file sizes. The new four objects have full-byte checks in this transaction; the previous 56 are not freshly byte-rehashed by the post-audit. Next scheduled review is cycle60 or earlier pressure.
