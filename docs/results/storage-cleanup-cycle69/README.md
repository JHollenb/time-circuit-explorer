# Cycle69: four redundant transports moved to verified cold storage

Four archive/log copies from two completed runs total **1,150,830,575 bytes**. Full local custody/hash checks and full remote SHA-256 checks precede durable restore pointers and local transport unlink. All extracted scientific evidence and original custody receipts remain local.

The selected runs are `mdb-suffix-context-20261105/run-01` (`job-dcae09a1f5c9`) and `mdb-text-program-20261104/run-01` (`job-da6d4cedfd1d`). Each has a retained extracted tree, complete archive/log custody and an authoritative succeeded job. Failed or unbound snapshots and standalone logs lacking the required paired custody were left untouched.

[The exact plan](plan.json), [transaction result](result.json), [fsynced events](events.jsonl) and [publication hashes](publication.json) preserve source/cold identities and restore commands. Restore pointers remain beside each original transport path. Cold objects live under `/mnt/hdd8tb/cold-storage/mbp1/time-circuit-explorer/transports/sha256` on Beast; use the exact `restoreCommand` from each pointer or result row.

Shared local free space is observed at **12,522,688,512 → 13,671,899,136 bytes** during the transaction. That change is not exclusive physical attribution because other workspace activity and filesystem allocation can contribute. The [post-audit](../storage-audit-cycle69/README.md) checks 76 pointer/custody joins and remote regular-file/size metadata. Only the new four receive fresh full-byte hashes this cycle; the older 72 do not.

The unchanged [bounded archiver](../../../generator/storage_transports.py) performs the move; the [publisher](../../../generator/publish_storage_cycle69.py) copies exact receipts. Remaining candidates are a name/size inventory, not deletion certificates. Next scheduled maintenance is cycle72 or earlier pressure.
