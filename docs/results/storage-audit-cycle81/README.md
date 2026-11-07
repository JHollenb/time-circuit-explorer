# Scheduled storage review: 93 recovery pointers, no new archive needed

The two local artifact roots contain no remaining `archive.tar.gz` or `worker.log.gz` files of at least 100 MB. No transport is deleted or newly archived in cycle 81. Last actual archiving remains cycle 78; the next scheduled review is cycle 84, earlier if storage pressure or large new artifacts warrant it.

The [publication receipt](publication.json) keeps two custody classes separate. The [84 older pointers](terminal-pointers.json) have fresh local original-custody joins and remote regular-file/size checks, without a fresh full-byte rehash. The [nine current-byte snapshots](current-snapshots.json), previously archived in cycle 78, receive fresh complete remote SHA checks over 3,806,240,603 bytes and local neighbor-metadata checks. Their original scientific custody limitations remain unchanged.

All extracted evidence, historical outcomes and recovery pointers remain in place. Full scientific arrays stay near compute; compact metadata is published here. These read-only storage operations neither evaluate a model nor certify the scientific contents of a transport.
