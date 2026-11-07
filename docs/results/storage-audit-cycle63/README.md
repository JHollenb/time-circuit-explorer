# Cycle 63 bounded storage audit

[Actual audit](result.json) joins **68 restore pointers** to original custody, absent local transports and fresh remote ordinary-file/size checks. The four files in the [new cleanup](../storage-cleanup-cycle63/README.md) have complete cold-copy SHA256 checks. The other 64 historical cold files were not freshly byte-rehashed here.

The name/size scan finds **25 remaining candidates / 7,923,510,102 logical bytes**. Those are not deletion certificates or unique physical bytes. Extracted evidence is retained. [Publication/source hashes](publication.json) bind the exact copied receipt. Next review is cycle 66 or earlier pressure.
