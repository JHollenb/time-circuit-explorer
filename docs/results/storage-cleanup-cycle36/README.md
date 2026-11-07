# Early transport cleanup while the next experiment waits

Four completed-run transport copies totaling1,631,201,526bytes have moved to verified Beast cold storage. They are the archive and compressed log for `job-85125cf2ae94` and `job-5500509e5fdd`. All extracted scientific evidence and original custody receipts remain local.

The [plan](plan.json) binds terminal job identities, complete original custody, full local SHA256 and file identities before transfer. Each removal follows a fresh complete remote SHA256/size check, durable restoration pointer and another local identity/custody check. The [result](result.json) and [events](events.jsonl) preserve every transaction and restoration command. No model arrays or extracted results were deleted.

Shared-volume available space is observed7,622,496,256→9,250,029,568bytes during the archive operation. This observed change is not an exact physical-byte attribution; concurrent workspace activity and filesystem allocation can differ from logical transport sizes. The review was brought forward after the prior9.9GBspace observation fell to about7.6GB.

The [post-cleanup audit](../storage-audit-cycle36/result.json) checks32restoration-pointer/custody joins and fresh cold-object regular-file sizes. The newly archived4objects received full-byte checks; the older28are not freshly byte-rehashed by this audit. It inventories48remaining archive/log name-and-size candidates totaling13,611,295,315logical bytes. They are not deletion certificates. The older failed snapshot lacking qualified custody remains local. Next storage review is due bycycle39or earlier pressure.
