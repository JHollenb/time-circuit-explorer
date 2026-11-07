# The row-dependent part of an MLP write

The mean-activation component predicts a useful but limited part of the missing write on one actual FLUX state. It removes **7.46% of full discrepancy energy**, including **21.35% text** and **4.97% image**. Full remaining RMS is 96.20% pooled, 88.68% text and 97.48% image. Most of the discrepancy still depends on the interaction between changing activation and changing Up values. This is a measured component of one reconstructed MLP write; no new image or economical generator is established.

![Full and centered correction errors and signed centered component Gram.](results/mean-gate-figure/mean-gate.png)

The [previous diagnostic](own-state-gated-covariance.md) found that the best per-stream constant leaves 83.21% of missing-write RMS. Its covariance mean nearly reaches that constant-family optimum. A better constant cannot explain most of the remaining row variation. We now test a linear component inside that variation, following the [frozen plan](plans/mean-gate-variable-gate.md).

## Factor the product before projecting it

Fix clock 3 / single 0 from the [mean-Up generator's own trajectory](mean-up-trajectory.md). Let A be its admitted BF16 SiLU activation, U* the previously paid FP64 Up reference, S the actual selected quarter of channels, D the saved FP64 original Down map, and g the residual gate. A and g are different quantities: A varies across rows and MLP channels; this block's g is shared across rows in the output coordinates.

For text and image separately, write

\[
\delta U_i=U_i^*-\bar U_s^*,\quad
R_i=1_{\bar S}\odot A_i\odot\delta U_i.
\]

The exact real-arithmetic expansion is

\[
R_i=\underbrace{1_{\bar S}\odot\bar A_s\odot\delta U_i}_{L_i}
+\underbrace{1_{\bar S}\odot(A_i-\bar A_s)\odot\delta U_i}_{B_i}.
\]

We derive the two activation means from the current A, then compute one new projection, Y_L=LDᵀ. The producer receives no old error field or oracle correction. It seals six arrays and exits. Only afterward does another process open the previous Y_R, Y_Q and Y_E fields and define Y_B^Δ=Y_R−Y_L. This changing term is subtraction-defined; we did not independently evaluate BDᵀ. Product-space closure is measured separately, with maximum absolute residual 8.88×10⁻¹⁶.

Define gated quantities X=g⊙Y_E, ℓ=g⊙Y_L, b=g⊙Y_B^Δ, q=g⊙Y_Q. The last term retains the previous combined policy/reassociation boundary. The measured closure X−(ℓ+b+q) has relative RMS 1.07×10⁻¹⁶. This reconstructed FP64 account excludes the actual native concatenated-output and residual rounding.

## Full error and centered error answer different questions

The primary proposed correction leaves r=X−ℓ. Its accuracy is ||r||/||X||. We also inspect C_s(r), where C_s subtracts each stream's own mean. Pooled centering still uses those two means, rather than one mean across all rows.

| Stratum | Full residual / full discrepancy RMS | Centered residual / centered discrepancy RMS | Full energy removed | Centered energy removed |
|---|---:|---:|---:|---:|
| All | 96.20% | 94.46% | 7.46% | 10.78% |
| Text | 88.68% | 88.38% | 21.35% | 21.89% |
| Image | 97.48% | 96.05% | 4.97% | 7.74% |

Centering removes the residual mean as a paid oracle diagnostic. It does not add a mean correction to the producer. The complementary account remains explicit:

\[
\|r\|^2=\sum_s n_s\|\bar r_s\|^2+\sum_s\|C_s(r)\|^2.
\]

Pooled residual energy is 12,723,318.997: 4,228,349.468 in its mean and 8,494,969.530 in centered variation, with a −1.86×10⁻⁹ measured closure discrepancy. The centered residual divided by the **full** discrepancy RMS is 78.60%, a third explicitly distinct denominator. The paid constant-only comparator remains 83.21% pooled / 98.77% text / 80.10% image RMS.

In real arithmetic, L has zero stream mean because δU does. Its measured mean is at FP64 rounding scale; we retain it rather than forcing it to zero. The mean-reduction comparison has a large relative ratio only because its reference is about 10⁻¹⁶ RMS. That ratio is not evidence of a meaningful mean effect. The covariance mean remains unproduced by this linear term.

## Keep the signed interaction

For the pooled centered quantities, ||ℓ_c||²=2,435,454.485 and ||b_c||²=8,494,893.444, while 2⟨ℓ_c,b_c⟩=−1,409,311.137. Their cosine is −0.1549. The policy/reassociation component has centered energy 97.580; its cross terms remain in the [full report](results/mean-gate/report.json).

These energies are not additive importance scores. The negative linear/changing cross term cancels part of their individual energy. Mean/centered orthogonality follows the centering construction; it does not discover independent model circuits.

The proposed full correction improves **1,289 dependent rows and worsens 247**: 482/30 text, 807/217 image. All 95 negative-alignment rows are image rows. Another 152 rows worsen despite positive alignment, including 30 text and 122 image rows. The row identity explains why:

\[
\Delta_i=\|X_i\|^2-\|X_i-\ell_i\|^2
=2\langle X_i,\ell_i\rangle-\|\ell_i\|^2.
\]

Centered scoring improves 1,354 rows and worsens 182, using a different target. These are coupled rows from one state, not independent prompt trials. Complete signed rows are retained losslessly.

## What computation was paid

One strict-preflight CUDA mrun lease, `job-d5b3a1b3bad4`, succeeds in **27.94 seconds** with sampled peaks of **2,043.6 MB RAM / 1,104 MB device memory**. It pays exactly **43,486,543,872 matrix MACs** for a 1,536×9,216 by 9,216×3,072 Down projection. It newly loads the **226,492,416-byte saved original FP64 Down map**, with zero original checkpoint tensors loaded. Historical generation, captured activation/gate/support, full Up formation and map promotion remain paid.

The synchronized instrumented operation takes 1.643 seconds including internal projection fingerprints. Actor wall time after imports is 12.647 seconds; post-exit analysis takes 8.612 seconds. These intervals have different scopes. No matched serving-speed improvement is measured.

All **341 mechanics tests** pass, including an independent tiny scalar mean/product/Down reference and signed residual identities. The audit binds one Act, six producer inputs, six producer outputs, fifteen post-exit arrays and the physical projection operands. Complete byte checks cover **48 files / 1,202,839,396 bytes**, including **31 NPY payloads**. Reduction and contraction arithmetic is source-audited and tiny-tested; the byte verifier does not independently replay real-array numerical calculations. The standard Saturn run/verify both return zero for one 1,536-row target stream, with no fires, formal claims or qualified addresses and one missing expected fire ledger.

All **21 new arrays / 604,572,936 file bytes** remain near compute. The public core is 16 files / 1,352,747 bytes before evidence and figure. No new archive or deletion occurs this cycle; the last actual cleanup is [cycle75](results/storage-cleanup-cycle75/README.md), with the next review due 78 or earlier under pressure. [Evidence and provenance](results/mean-gate-evidence/README.md) retain the scopes and receipts.

The working inference is that common activation coefficients describe a modest part of this receiver's missing write, while changing activation and Up must be understood together. The proposed two-stream linear-map compilation still costs 130.460 billion MACs before 14.496 billion application MACs, with receiver-dependent coefficients and no demonstrated reuse. This decomposition supplies no arbitrary-prompt zero-neural image backend. The last generated image remains cycle75.

Astra's [next unexecuted plan](plans/odd-gate-quadratic.md) uses SiLU(G)=G/2+H(G), with H(G)=G tanh(G/2)/2 even. It tests the fixed quadratic product ½δG*⊙δU. Unlike L, that product can carry a covariance mean. G was not saved, so the plan pays a fresh original-weight FP64 gate projection and one Down: 86.973 billion prospective MACs. Actual A and old error fields remain unavailable until producer exit. Smooth-even and finite-activation remainders are separated only in post-release product bookkeeping. A dense quadratic tensor would exceed 231 GB in FP64, so the identity alone supplies no cheap implementation. The next cycle also owes a full newest-post and storage review before staging.
