# First coarse-global attempt: attention-bias contract failure

Job job-976d565ca3f5 is authoritatively failed after38.21seconds,8200.5MB sampled RSS and7690MB VRAM. The first CUDA attention dispatch rejects the source-fixed FP32 bias for BF16 queries. No image, complete velocity, candidate seal or target-scoring phase exists. Original weight loading and partial learned projections were performed and remain charged.

The [scheduler result](scheduler-job.json), [request](request.json), [frozen plan](frozen-plan.md) and [partial-array inventory](retained-partial-array-inventory.json) retain the original failure. Seven complete partial arrays/3,497,856bytes remain near compute, with authenticated byte inventory; they are not scientific outcome measurements. This is a broken numerical contract, not a null result for the coarse hypothesis.

A fresh v2 explicitly records the requested FP32 and consumed BF16 key weights using a shared conversion helper, tested first with synthetic attention. The original failed payload is unchanged.
