# Cycle60 transport cleanup

Archived **four redundant transport files /1,508,098,618 bytes** from two succeeded older runs. Fresh original custody, local SHA256, complete cold-copy SHA256 and durable restoration pointers precede each local unlink. All extracted scientific evidence and original custody records remain local.

[Plan](plan.json), [actual result and exact restore commands](result.json), [fsynced transaction events](events.jsonl) and [publication/source hashes](publication.json) preserve the transaction. The source is [storage_transports.py](../../../generator/storage_transports.py).

Observed shared-volume free space moves from **7,762,280,448** to **9,279,070,208 bytes**. The difference is not an exclusive physical-space attribution. The [post-cleanup audit](../storage-audit-cycle60/README.md) joins all 64 restore pointers and reviews remaining transport candidates. Next review is due cycle63 or earlier pressure.
