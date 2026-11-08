# A change vector needs the right state underneath it

A nearly correct contextual change can fail at the next computation because it starts from the wrong baseline. In this finite crossover, our predicted change on the native baseline gives 5.21% Q-response error. The native change on our own baseline gives 118.90%. K and V show the same pattern. This is evidence about one unchanged token and its immediate consumer; arbitrary-prompt images with zero neural evaluations remain unsolved.

The [contrast-aware writer](contrast-aware-contextual-writer.md) gave vessel row9 a first-layer contextual response with only 4.74% relative error, yet its next Q/K responses had over 100% error. The [nine-write capacity study](finite-writer-capacity.md) separately found that better routing cannot recover most vessel padding response in that frozen output family. This assay asks a different question: why does the promising active-row change fail when consumed?

## Exchange the change and the baseline

Let a,b be the native BF16 H1 states for discovery histories00/10, and p,q the accepted writer's own states. Row9 is active and has the same token in both histories. Define the differences after promotion to FP64:

\[
\delta_n=b_9-a_9,\qquad\delta_o=q_9-p_9.
\]

Keep the four stored diagonal consumer results. Construct two additional full states, changing only row9:

\[
X_{no,9}=A(a_9+\delta_o),\qquad
X_{on,9}=A(p_9+\delta_n),
\]

where A is FP32 rounding followed by BF16 rounding, both to nearest even. Xno otherwise copies a; Xon otherwise copies p. The same original normalization/Q/K/V consumer F runs on both complete [1,512,2560] inputs. Four historical consumer units are reused through exact input-payload, parameter and numerical-profile bindings; two new units are evaluated.

At each port, including identity at H1, measure

\[
N=F(b)-F(a),\quad O=F(q)-F(p),\quad
C_{no}=F(X_{no})-F(a),\quad C_{on}=F(X_{on})-F(p).
\]

The response error is the distance to N, divided by its norm. The magnitude of C alone is a different quantity.

| Port | Own response O error | Own change on native baseline, Cno error | Native change on own baseline, Con error |
|---|---:|---:|---:|
| H1 | 4.7432% | 4.7785% | 0.3557% |
| Q | 113.7085% | 5.2146% | 118.9018% |
| K | 111.3169% | 6.2896% | 117.0901% |
| V | 69.4903% | 4.9661% | 74.3448% |

Our change remains useful at Q/K/V when supplied the native baseline. Giving our baseline the native change does not recover native behavior. Thus a change-vector-only account misses something necessary in this specimen. Baseline direction, normalization gain, finite arithmetic and receiver geometry remain mixed; this result does not identify RMS scale alone as the cause.

## Factor the discrepancy without discarding signs

Let E=O−N. Two finite paths split the observed discrepancy:

\[
U_{no}=O-C_{no},\quad V_{no}=C_{no}-N,\qquad
U_{on}=O-C_{on},\quad V_{on}=C_{on}-N.
\]

Each evaluated path also retains the regrouping vector g=E−(U+V). Its full signed identity is

\[
\|E\|^2=\|U\|^2+\|V\|^2+2\langle U,V\rangle
+2\langle U+V,g\rangle+\|g\|^2.
\]

Both g vectors and the evaluated Gram residual are zero at all four ports in this run. Both paths have negative U/V cross terms. For Q, one path gives

\[
0.6386022266=0.6577216545+0.0013430516-0.0204624796.
\]

The terms cancel; their squared norms are not independent causal percentages. The second path gives the same total with a different split. Both include their respective admission effects, rather than responses to unrounded states outside the BF16 interface.

The two diagonal re-admissions reproduce b9/q9 exactly. Crossed admission changes 1,591 coordinates on the native baseline and 801 on the own baseline; their remainder norms relative to the native H1 change are 0.8168% and 0.3557%. All 3,066 unmodified Q/K/V row controls have zero error against their stored baselines. Four fresh byte-segment checks separately establish that the crossed states are unchanged outside row9.

## Why the baseline can matter

Ignoring finite arithmetic for explanation, write one normalization/projection as

\[
F_W(h)=W\operatorname{diag}(\gamma)\frac{h}{s(h)},\qquad
s(h)=\sqrt{\|h\|^2/d+\epsilon}.
\]

Its directional derivative is

\[
J_h[v]=\frac{W\operatorname{diag}(\gamma)}{s(h)}
\left(v-h\frac{h^\top v}{d\,s(h)^2}\right).
\]

The same v encounters a different gain and radial subtraction at another h. A first-order expansion suggests

\[
O-N\approx (J_p-J_a)\delta_n+J_p(\delta_o-\delta_n).
\]

This formula explains how small change-vector error can coexist with large consumer-response error. It is an ideal real-arithmetic interpretation, not an attribution computed by this experiment. The actual consumer has FP32 normalization and BF16 admissions; the measured finite corners above have stronger authority than this local approximation.

An affine consumer on a common reachable region would simplify matters: its response would depend only on the increment, provided admission were exact or accounted for separately. That assumption needs evidence. For the present finite states, transplanting the same native change between baselines clearly does not preserve the next response.

This connects directly to [MLP continuation](../../cmachine/research/transition_laws/FINDINGS.md), the [state quotient counterexample](../../cmachine/research/state_quotient/FINDINGS.md), and [time circuits](../../obsidian/blog/2026-11-05-153154-state-context-and-the-mathematics-of-time-circuits.md). A useful state code must preserve the distinctions its future computation consumes. Matching one readout, a state norm or a current change is insufficient. We need an economical state representation and an update law that remains useful on that representation's own states.

## Measured work and evidence

One guarded CUDA mrun job, job-ef24db06d909, completes in 9.4 seconds on Beast. Its worker timer after imports/request verification is 7.003 seconds, with 4.373 process CPU seconds. Selected-parameter loading and shard authentication take 3.047 seconds. The two consumer timers, including hash capture, are 0.593 and 0.0158 seconds; the first includes startup effects, so they do not establish a speed comparison. Numerical CPU threads are one; intent-first planning reserves four scheduler threads, 4,096 MB RAM and 4,096 MB VRAM. Sampled peaks are 1,569.9 MB RSS and 412 MB VRAM; PyTorch's separately reported allocation peak is 93,328,384 bytes. The process high-water mark is 1,531,888 KiB.

The assay pays six original learned linears and 16,106,127,360 nominal matrix MACs, plus normalization, finite construction, analysis, hashing and file work. Four parameter tensors contain 31,462,400 bytes; their 4,967,215,360-byte shard is freshly hashed before loading and again by publication. Historical state construction and neural preparation remain paid. No attention, MLP, complete encoder, denoiser, VAE, new prompt or image runs.

Eight tiny supplied mechanics checks pass once. Initial staging stops before creating a directory or job because the parent fresh-file inventory has no weight-shard row. The preserved [stop record](results/finite-consumer-crossover-evidence/staging-v1-stop.json) and [versioned correction](plans/finite-consumer-crossover-staging-v2.md) distinguish authenticated historical parameter metadata from current byte checks. The numerical worker remains unchanged. Strict preflight passes command/root/reservation checks; undeclared verifier and geometry hooks are skipped. Declared, sealed and executed payloads match SHA8287403f8c36252ad3a1cc9f581f3be58930d0c2b6bccf9d85caaade0a64c821.

The [publication](results/finite-consumer-crossover/publication.json) authenticates 3,072 raw comparisons, 108 pooled scalars, 188 signed summaries, seven typed Acts and sixty retained fields. It freshly checks 77 remote files totaling 5,021,631,744 bytes. Scalar reconstruction and byte custody do not independently replay coordinate norms or historical model outputs. The [generic Saturn sweep](results/finite-consumer-crossover-evidence/saturn-instrument.json) recognizes neither the gzip row stream nor these gate-free typed records and reports an empty unsupported target; it supplies no generic scientific coverage.

All sixty new numerical fields, 18,763,008 bytes, remain near compute. The primary compact publication is 2,226,744 bytes, plus evidence and narratives. No tensor hydrates locally and no archive/delete occurs. Full [research](results/research-review-cycle105/README.md) and [storage](results/storage-audit-cycle105/README.md) reviews105 remain current; review108 is next.

The proposed [next diagnostic](plans/finite-normalization-operand-audit.md) reconstructs the six exact pre-affine normalization operands and separates gain, radial and finite effects before another writer fit or QKV evaluation. It is unimplemented and unrun, and comes after the full108 review. It cannot supply missing padding response, full conditioning or pixels by itself.

Sources: [complete report](results/finite-consumer-crossover/report.json), [signed analyses](results/finite-consumer-crossover/analysis.json), [source review](results/finite-consumer-crossover-evidence/advisor-source.json), [final preparation review](results/finite-consumer-crossover-evidence/advisor-source-v2-final.json), [actual result review](results/finite-consumer-crossover-evidence/advisor-actual.json), [living research log](../../obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md).
