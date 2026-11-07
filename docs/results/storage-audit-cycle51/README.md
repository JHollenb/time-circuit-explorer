# Cycle 51 transport and restoration audit

The [post-cleanup audit](result.json) checks **52 restoration-pointer/custody joins** and fresh remote regular-file sizes. The four new cold copies received full checksum verification in the [cleanup transaction](../storage-cleanup-cycle51/result.json); the previous 48 were not freshly byte-rehashed here.

The [pre-audit](pre-audit.json) found 42 name/size candidates. After cleanup, 38 candidates total 11,044,501,235 logical bytes. These are possible transport copies to inspect, not deletion certificates or unique physical storage. All extracted research evidence remains local.
