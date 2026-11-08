# Exact supplied-machine compiler result

The 17 JSON files in this directory are byte-exact copies of `artifacts/behavioral-realization-v1`, totaling 51,859 bytes. This README is publication prose, additional to that raw inventory. [files.json](files.json) binds the other 16 raw files; [collection.json](../behavioral-realization-evidence/collection.json) binds all 17 copies and the four source snapshots.

[report.json](report.json) records exact rank three, the complete finite-graph certificate and all 36 serving comparisons: 35 valid words exact and one invalid symbol explicitly refused. [program.json](program.json) emits two persistent byte registers with only copies/constants/readout. The separate shallow and fixed-coordinate controls preserve their counterexamples/refusal. [discovery-queries.json](discovery-queries.json) and [shallow-queries.json](shallow-queries.json) retain every physical word/output query.

All behavior, alphabet and scalar readout belong to a supplied finite machine. There are no learned-model tensors, images or arbitrary-prompt semantics. The [article](../../exact-behavioral-compiler.md) explains the math, costs and limits; the [evidence README](../behavioral-realization-evidence/README.md) distinguishes actual execution, source reviews and publication checks.
