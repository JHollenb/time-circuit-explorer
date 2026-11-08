# Scheduled storage review: recovery retained, no new archive needed

The two artifact roots contain no remaining archive.tar.gz or worker.log.gz transport copies of at least 100 MB. Cycle 84 archives and deletes no files. The last actual archive remains cycle 78; the next review is due at cycle 87, earlier if storage pressure or large artifacts warrant it.

The [receipt](publication.json) preserves two custody classes. The [84 older pointers](terminal-pointers.json) have fresh local original-custody joins and remote regular-file/size checks, without a fresh full-byte rehash. The [nine current-byte snapshots](current-snapshots.json) receive fresh complete remote SHA checks over 3,806,240,603 bytes and local neighboring-metadata checks. Those checks do not repair historical scientific custody limitations.

All extracted evidence, outcomes and recovery pointers remain. Full scientific arrays stay near compute. These read-only operations neither evaluate a model nor certify the scientific content of a transport.
