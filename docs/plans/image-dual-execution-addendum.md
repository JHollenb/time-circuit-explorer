# Image-dual execution and reference accounting

Status: written before staging. This addendum corrects an axis-order premise in the committed [scientific plan](image-dual-score-producer.md), specifies the finite policy and charges the reference work needed to interpret it. The smooth numerator/stale-denominator candidate remains the same.

Astra's pre-freeze code review identified that native IDs use **(T, H, W, L)** rather than the plan's assumed (T, L, H, W). An authenticated raw-byte inventory of the actual captured cosine/sine factors confirms image group counts **[1,32,32,1]**, text counts **[1,1,1,512]**, and exact row membership for height at axis 1 and width at axis 2. Constant image axes are therefore 0 and 3; text L occupies channels 96:128. The implementation uses those actual positions and rechecks all groups before compilation. The original committed plan remains intact. No model-state run was staged or submitted under the incorrect order, and no scientific outcome selected the correction. The 65 groups and MAC counts are unchanged.

Compiler and actor deny the original full P/Q key-array paths. A separate byte-only intake authenticates the native Q key file/header/payload and copies its contiguous 512-row text prefix. It does not numerically decode image keys. The actor does not hydrate the image-key matrix. Its actual native BF16 ModLN input is captured and supplies the later Q reference projection.

All new grouped contractions, softmax and value contractions use CPU FP64 with four threads. Native BF16 ModLN and original receiving affines use CUDA under the preceding fixed profile. The head admitted to the consumer follows CPU FP64→FP32→BF16 RNE→CUDA. No precision arm or numerical outcome selection is added.

After actor seal and process exit, one smooth Q-image projection and two image QK contractions verify the numerator and expose combined projection/RMS/RoPE finite-policy differences. Two FP64 full AV references reuse the actor's exact text logits and restored values. They produce attention-head diagnostics only, with no additional receiving MLP or next query.

| New reference work beyond the original plan's Q projection | MACs |
|---|---:|
| Direct smooth image QK |12,582,912|
| Native-key image QK |12,582,912|
| Smooth and native-key full AV |37,748,736|
| **Reference addendum** |**62,914,560**|

The revised complete nominal matrix count is **21,359,493,120 MACs**: compiler 10,286,530,560; actor 1,346,371,584; post-seal references 9,726,590,976. These are prospective counts until the actual operation ledger runs. The coefficient payload remains 153,354,240 bytes, excluding its NPY header and all other inputs, parameters and captures.

Use the signed vector telescope

\[
B(H_c)-H_{SDPA}=[B(H_c)-H_c]+[H_c-H_{smoothQ}]+[H_{smoothQ}-H_{nativeKeys64}]+[H_{nativeKeys64}-H_{SDPA}].
\]

The terms respectively measure admission rounding, retained-denominator plus observed dual-contraction arithmetic, combined key finite-policy differences, and attention numerical-policy differences. RMS magnitudes are not additive; retain signed arrays, their actual-response scales and per-row/head effects. Use log partitions including the softmax peak when comparing different score systems. This is not a new consumer arm or a causal attribution to an individual rounding boundary.

Prior sources, native text keys/full values, complete baseline trajectories, original learned maps and all compilation/reference work remain paid. No first-prompt saving, memory saving, runtime speedup, new image or arbitrary-prompt zero-neural capability is claimed from this assay.
