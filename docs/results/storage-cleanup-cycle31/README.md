# Cycle31 transport cleanup

Four completed-job archive/log transport copies totaling **1,852,490,354 bytes** were moved to Beast cold storage. Every remote copy passed a full SHA check before a durable restoration pointer was written and the unchanged local transport removed. All extracted scientific evidence and original custody receipts remain local. The [plan](plan.json) and [result](result.json) bind the selected files, checksums and restore commands.

Shared-volume free space increased from13,170,712,576 to15,016,808,448 bytes during the operation. This observation includes concurrent workspace activity and is not an exact physical reclamation measurement.

The [storage review](../storage-audit-cycle31/result.json) subsequently checks26 restoration-pointer/custody joins and remote regular-file/size identities. The older22 objects are not freshly full-byte rehashed; their original archive receipts retain that evidence. One larger failed snapshot lacked its original archive custody receipt and was left local, as recorded in the [decisions](../storage-audit-cycle31/decisions.json). The50 remaining name/size candidates total15,332,838,602 logical bytes; they are not certified deletions or unique physical bytes. Next review is due by cycle34, or earlier under pressure.
