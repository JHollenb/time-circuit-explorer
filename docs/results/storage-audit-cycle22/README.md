# Cycle 22 storage audit

The [read-only audit](result.json) finds **64 transport-name/size candidates totaling 20,181,806,531 logical bytes** after the separately receipted [cleanup](../storage-cleanup-cycle22/README.md). These are candidates for review, not deletion certificates or unique physical bytes.

All 16 cold pointers match their archive receipts and unchanged original custody. Fresh remote regular-file/size checks pass. The previous 14 objects were not fully byte-rehashed this cycle; the archiver separately verified complete SHA-256 for the two new copies before removing local transports. Extracted scientific evidence remains local. Shared-volume free space was observed at 14,752,497,664 bytes. The next periodic review is due by cycle 25, earlier under pressure.

The reusable [auditor](../../../generator/audit_transports.py) performs read-only checks. Selection and removal remain explicit operations in `archive_transports.py`; the inventory alone authorizes neither.
