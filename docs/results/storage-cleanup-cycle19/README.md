# Cycle19 transport cleanup

The [result](result.json) records two redundant transport copies from succeeded job `job-37be2daf859f`, totaling **1,393,328,482 bytes**. The archive and compressed worker log were copied to content-addressed cold storage on Beast and fully SHA-256 verified. Fresh local identity checks and durable restoration pointers preceded removal. Extracted scientific artifacts and their original receipts remain local.

The [plan](plan.json) binds the selected paths and original custody. Free space was observed at12,154,871,808 bytes before and13,527,924,736 after; these shared/APFS measurements are not an isolated physical-space attribution. The [post-cleanup audit](../storage-audit-cycle19/README.md) distinguishes restoration checks from remaining review candidates.
