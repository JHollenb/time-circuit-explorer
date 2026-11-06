# Artifact storage and recovery

The active research workflow audits storage alongside Astra's review every few cycles, with the next review due by cycle 8. Keep the narrative, requests, manifests, custody and compact measurements local. Full compiled tables and complete large operand arrays stay near compute. Move completed transport copies to verified cold storage when they consume useful development space.

Cycle 5 archived six archive/compressed-log files from three completed runs: **7,701,454,599 bytes**. The extracted research artifacts and original custody records remain local. Available local space increased from 10,885,238,784 to 18,569,580,544 bytes during the operation, an observed 7,684,341,760-byte increase. Other system activity can affect that measurement; archived logical bytes are recorded separately.

The [exact plan](results/storage-cleanup-cycle5/plan.json), [completion receipt](results/storage-cleanup-cycle5/result.json), [per-file events](results/storage-cleanup-cycle5/events.jsonl) and [archiver source identity](results/storage-cleanup-cycle5/provenance.json) preserve the operation. Cold objects live on Beast under `/mnt/hdd8tb/cold-storage/mbp1/time-circuit-explorer/transports/sha256/`.

The [bounded archiver](../generator/archive_transports.py) accepts explicitly selected completed-run directories. Staging verifies each transport against its original custody digest, checks the scheduler's terminal state and requires extracted evidence to remain local. Archiving transfers to a temporary remote path, verifies the complete bytes, publishes the cold object, writes a restoration pointer, then rechecks the local identity before removing that local transport. The command does byte I/O only; it executes no model.

Each moved `archive.tar.gz` or `worker.log.gz` has a neighboring `.cold.json` pointer. Its `restoreCommand` is an argument array for restoring the exact original path with rsync; `sha256`, `bytes`, `remoteHost` and `remote` identify the object. After restoration, verify the local SHA256 against the pointer before consuming it. The original capture receipts remain unchanged.

The complete encoder-atom table is active input to the next experiment, so it remains on Beast. Its independent [342-file byte verification](results/raw-prompt-encoder-atoms/retained-store-verification.json) binds 3,228,908,150 bytes of store and operand evidence. Public Git contains 3.743 MB of compact prefix evidence, rather than the complete multigigabyte store.
