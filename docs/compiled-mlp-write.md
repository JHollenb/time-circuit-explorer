# A dense MLP write, compiled into five output arrays

We removed the original dense projections from serving one supplied MLP input line, then let its predicted write feed a residual update. The primary process reads only five output-coefficient arrays and its fixed parent/gate context. It receives no original matrices, expanded hidden channels, native output endpoints or midpoint answer. Arbitrary prompts and autonomous image generation remain open.

The specimen is FLUX.2 Klein4B's image MLP at denoising pass 2, joint block 1 (both indexed from zero), with 32 selected rows and 3072 output channels per row. Native CPU BF16 references already exist in `job-f0cd6a040823`; parent context comes from `job-5500509e5fdd`. The new job executes no native neural calls. Its offline compiler still evaluates original learned-weight projections and sigmoid arithmetic; that work is counted.

## The computation that disappears online

Ignoring native rounding boundaries for this algebra,

\[
m(x)=W_{out}\big[(W_gx)\odot\sigma(W_gx)\odot(W_vx)\big].
\]

For the supplied x(α)=x0+αΔx, both projected operands are affine in α. Approximate sigmoid locally to degree2 around Wg·x0. Multiplication by its gate operand gives degree3; multiplication by the value operand gives degree4. Projecting the five coefficients offline leaves

\[
\widehat m(\alpha)=c_0+\alpha\big[c_1+\alpha(c_2+\alpha(c_3+\alpha c_4))\big].
\]

The arrays are calculated from original weights and endpoint inputs; no native output is fitted, including c0. Serving uses four array multiplications and four additions for Horner evaluation, followed by

\[
\widehat I'=B\big(U_0+B(g_M\odot B(\widehat m))\big).
\]

The predicted write is consumed. U0 and gM remain supplied and fixed. The reference uses the recorded isolated native MLP return in that same update; it is a reconstructed consumer comparison, not an observed complete-block midpoint history. The [design](plans/compiled-mlp-write.md) records those boundaries.

## Measured cost

| Scope | Measurement |
|---|---:|
| Primary arithmetic, write plus residual, α=.5 | .378ms median |
| Primary medians over α=0,.5,1 | .378–.387ms |
| Fresh primary process, all doses, repetitions and I/O | .246s |
| Primary self-reported peak RSS | 51.36MiB |
| Five FP64 coefficient arrays | 3,932,160 payload bytes /3.75MiB |
| Fixed residual/gate context | 405,504 payload bytes |
| Compiler after imports | 4.136s wall /4.090s CPU |
| Compiler self-reported high-water RSS | 990.36MiB |
| Original BF16 parameters used offline | 169,869,312 bytes /84,934,656 parameters |
| Complete checkpoint hashed offline | 7,751,109,744 bytes |
| Entire mrun lease | 6.29s,0VRAM |

One configured CPU thread performs the arithmetic; mrun reserved four CPU slots under its default probe profile. Each timing includes the complete 32-row write and residual, with 31 repetitions after one charged warmup per dose. The fresh process includes imports, loading, 96 evaluations, saving and hashing three outputs; it is not single-query latency. Compiler high-water RSS exceeds mrun's sampled 753.4MB; both remain visible. Logical bytes are not physical disk traffic or energy. No matched native timing baseline exists, so no speedup is claimed.

This representation is specific to these 32 rows and this line. Five FP64 coefficient arrays for all 1024 independently specialized row paths would occupy 125,829,120 bytes before consumer context, compared with 169,869,312 bytes of shared BF16 MLP weights. New independently compiled contexts need new coefficients. The smaller measured payload is not general model compression.

## Aggregate improvement hides a strong counter-trend

At α=.5, rounded-write endpoint error is .001824RMS, or .350% of native output RMS. Receiving-state error is .002634RMS, or .182% of native state RMS. To expose small changes within those large states, also measure

\[
e_{\Delta}(\alpha)=[P(\alpha)-P(0)]-[T(\alpha)-T(0)].
\]

Its denominator is native change RMS. Direct endpoint error normalized by that change remains separate; it is not paired-response error.

| Midpoint method | Raw-write response error | Rounded-write response error | Receiving-state response error |
|---|---:|---:|---:|
| Derived polynomial | 8.48% | 9.43% | 12.59% |
| Native endpoint interpolation | 11.17% | 11.96% | 15.42% |
| Copy native baseline | 100% | 100% | 100% |

These are pooled RMS ratios, not mean row percentages. The polynomial beats interpolation on only 6/32 raw-write rows, 7/32 rounded-write rows and 9/32 state rows. Median per-row response errors are 58.9%, 65.1% and 52.6%; the largest state ratio is 187.3%. Four source image rows (600, 533, 436 and 531) contain 90.43% of native midpoint write-effect energy and 89.78% of state-effect energy. Those larger-change rows drive the pooled advantage. All 288 case/row records remain in the [raw report](results/compiled-mlp-write/report.json), without an accuracy filter. They are dependent views of one source pair, not 288 independent experiments.

Native midpoint curvature RMS is .002714. The polynomial predicts .002069RMS curvature with .001791 residual,66.0% of the native curvature norm. Endpoint interpolation predicts zero curvature. This retains a useful trend and considerable unexplained structure.

## Rounding remains unresolved

The analytic line uses unrounded x0+αΔx; the archived midpoint uses B((x0+x1)/2). Rounding moves 24,805 of 98,304 midpoint input scalars, with .0003878 RMS and .00390625 maximum displacement. The compiler also removes intermediate projection, SiLU and product BF16 boundaries. Error therefore combines approximation and numerical-policy differences; it cannot be assigned solely to sigmoid truncation.

At α=0, rounded-write relative RMS error is .325%. This independently computed baseline discrepancy is not a failed no-change intervention. Paired response and curvature remove some baseline effects without creating native BF16 parity.

## Run the bounded program

The published bundle includes exact collected coefficients and consumer context. It needs NumPy and Pillow, without a neural runtime or model download:

```sh
../common/bin/common exec python generator/render_compiled_mlp.py \
  --alpha .5 --output artifacts/my-compiled-write
```

The output contains raw and rounded writes, updated state and a hash seal. Doses other than 0,.5,1 have no native comparison here. This produces state arrays, not an image. The [implementation](../generator/mlp_polynomial.py), [launcher](../generator/mlp_polynomial_study.py) and [publication tool](../generator/publish_mlp_polynomial.py) retain the calculation and custody path.

The successful guarded job is `job-80041334757d`, request `cf49511d1c0f94f9832d7ae927b92df41f3e5c82aa26e018ef1d378a6471531d`. Source archives were verified and rebound to fresh scheduler records before staging. The primary sealed before the worker opened native result bundles. Endpoint controls ran separately afterward, then the midpoint was decoded. This remains retrospective: original outcomes informed the design and staging integrity checks read historical archives. Python file/import audits are cooperative instrumentation, not an OS sandbox.

Earlier `job-63f71d0f7684` stopped when the compiler's own manifest was disallowed during receipt generation. It produced no serving seal or scored result. Its [failure record](results/compiled-mlp-write/failed-run.json), original request and log remain separate. The fix permits that declared input; a complete synthetic compiler regression exercises the path. All 22 Python mechanics tests pass.

The Saturn sweep and verification completed on 288 rows with `target-streams-only` status, no address joins and no terminal-claim receipts. This checks the instrument, not broad generation. The larger [zero-neural objective](zero-neural-research.md) remains active.

The next experiment should connect this emitted state to the actual next joint block's row-local reader: normalization/modulation and QKV projection. That reader must consume the program's BF16 state, without receiving α or a native replacement. Compile its projection onto a small producer-derived basis offline, expose the out-of-basis residual, and separate receiver approximation from inherited writer error. See the [state-only receiver plan](plans/compiled-state-receiver.md).
