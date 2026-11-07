# Cycle66 transport cleanup

Four redundant transports from two authoritatively successful jobs move to verified cold storage: **1,296,894,473 bytes**. Every cold file receives a complete SHA256 check before a durable restore pointer is written and the local transport is unlinked. Extracted scientific evidence and original custody remain local.

The [plan](plan.json), [actual transaction](result.json), [fsynced events](events.jsonl) and [exact publication](publication.json) preserve the operation. Observed shared-volume free space changes 11,581,157,376→12,883,542,016 bytes; the complete difference is not attributed physically to this archive.

An initial selection also included a failed binding-search job. Qualification stopped before creating a cleanup plan or unlinking any file; those transports remain local. The executed plan instead uses the successful binding-writer and carrier-restoration jobs. [The72-pointer post-audit](../storage-audit-cycle66/README.md) checks recovery metadata. Next scheduled storage review69 or earlier pressure.
