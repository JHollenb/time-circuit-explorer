# Cycle 25 transport cleanup

Two redundant archive/log transports from terminal job `job-c829133410d8`, totaling **1,536,418,324 bytes**, moved to content-addressed cold storage on Beast. The [completion receipt](result.json), [plan](plan.json) and [events](events.jsonl) preserve full remote SHA checks, durable restoration pointers and fresh local custody/identity checks before removal. Extracted scientific evidence and original custody remain local.

Shared-volume free space was observed at 14,104,309,760 bytes before and 15,636,279,296 after; that change is not an isolated physical-space attribution. The [post-cleanup audit](../storage-audit-cycle25/README.md) records remaining candidates and recovery checks. Next review is due by cycle 28 or earlier under pressure.

An earlier [staging refusal](stage-refusal.json) rejected a different failed snapshot because its original archive-custody record was absent. No transport or extracted evidence from that snapshot was removed.
