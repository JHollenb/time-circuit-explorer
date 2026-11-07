# Storage maintenance72

Six redundant archive/log copies from three completed runs are archived to content-addressed Beast cold storage: **991,780,817 bytes**. Each original custody receipt, fresh scheduler state and local hash is checked. Full remote SHA verification and a durable local restoration pointer precede local transport unlink. Extracted science and original custody remain local.

The [plan](plan.json), [result with exact restore commands](result.json), [durable event log](events.jsonl) and [publication receipt](publication.json) retain the transaction. The [post-audit](../storage-audit-cycle72/README.md) checks the resulting pointer inventory.

Shared-volume free space changes from 13,539,590,144 to 14,515,818,496 observed bytes during this transaction. That change is not an exclusive physical-byte attribution. Failed, unbound and unpaired candidates are retained.
