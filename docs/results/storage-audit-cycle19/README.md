# Cycle19 storage audit

The [bounded audit](result.json) finds66 transport-name/size candidates totaling21,719,668,202 logical bytes after the separately receipted [cleanup](../storage-cleanup-cycle19/README.md). These are review candidates, not safe-deletion certificates or unique physical bytes.

All14 cold pointers matched their archive receipts, and fresh remote regular-file/size checks succeeded. The previous12 cold objects were not fully byte-rehashed this cycle; the archiver separately verified complete SHA-256 for the two new copies. Extracted scientific evidence remains local. Observed free space was13,528,760,320 bytes. Next review is due by cycle22, sooner under storage pressure.
