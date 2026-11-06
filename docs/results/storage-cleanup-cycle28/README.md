# Cycle28 bounded transport cleanup

Four redundant archive/log transport copies from two completed jobs were moved to Beast cold storage, totaling **968,631,521 bytes**. The jobs are `job-2f0cd41dfc45` (eye program) and `job-13cddbd2f9ab` (edit onset). Extracted scientific artifacts and original custody receipts remain local.

The [plan](plan.json), [event stream](events.jsonl), and [result](result.json) retain each file's identity, custody binding, destination, SHA256 and restoration command. Each destination was fully hashed and fsynced before a durable `.cold.json` restoration pointer and fresh local identity check allowed unlinking its transport copy. This is archival, not deletion of scientific evidence.

Observed local free space was 13,242,306,560 bytes before and 14,201,638,912 afterward. Concurrent activity and filesystem accounting prevent interpreting that whole difference as uniquely reclaimed physical space. The [post-audit](../storage-audit-cycle28/README.md) records the subsequent state. Review again at cycle31 or earlier under storage pressure.
