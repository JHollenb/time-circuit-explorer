# Failed first terminal-state capture

Guarded job `job-c266fd593526` fails after **69.80 seconds**, sampled peak RSS1,931.1MB/device1,192MB, at its first single19 StateCut save. The pinned deployed Saturn wheel lacks the optional `saturn_pub.shared_store` module imported by v1's wrapper. No scientific arrays, cuts or result files were persisted.

The traceback position implies joint1–4 and single0–18 executed first; live work counters were not durable, so exact realized MACs are unavailable. This is a package/persistence failure, not evidence against output liveness or a completed capture. Its runtime remains paid separately.

[Failure and custody](failure.json), [full scheduler log](job.log), [original request](request.json), [frozen plan](frozen-plan.md), [resource basis](resource-basis.json) and [exact publication](publication.json) remain unchanged. The [separate v2 plan](../../plans/terminal-native-capture-v2.md) qualifies the actual installed package before model work and saves durable stage progress. The [successful capture](../../terminal-native-capture.md) does not erase this attempt.
