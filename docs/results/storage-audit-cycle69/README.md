# Cycle69 storage audit: 76 restore pointers, 17 remaining candidates

[The actual read-only audit](result.json) verifies 76 pointer/original-custody joins and fresh remote regular-file/size identities. The four new archived objects were fully SHA-256 checked in the [archive transaction](../storage-cleanup-cycle69/result.json). The previous 72 cold objects are not freshly full-byte rehashed here.

Seventeen remaining transport-name/size candidates total **5,475,785,054 logical bytes**. This does not establish that they are safe to delete or that those bytes are unique physical allocation. Several failed/unbound snapshots or unpaired logs remain in place because this archiver requires completed paired custody and retained extracted evidence.

[Publication hashes](publication.json) bind the copied audit. [The source](../../../generator/storage_transports.py) remains unchanged; [the cleanup summary](../storage-cleanup-cycle69/README.md) explains restore procedures and space observations. New scientific arrays stay near compute, with compact metadata and original PNGs collected locally. Next scheduled maintenance is cycle72 or earlier pressure.
