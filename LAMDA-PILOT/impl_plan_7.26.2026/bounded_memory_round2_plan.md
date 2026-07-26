# Bounded-Memory Task-Agnostic Evaluation — Round 2 (standalone plan)

Purpose: repair the bounded_memory harness whose first-round results were
artifact-dominated (head-loss suppression shared by all methods), remove the two
documented unfairnesses against SketchLoRA (un-swept lr; ambiguous eviction
rule), and run the pre-registered budget grid properly. Fairness rule for the
whole round: every fix is either (a) uniform across methods or (b) the removal
of a documented asymmetry; no fix is admissible whose only justification is
changing the ranking. All configured runs are reported; none are dropped after
results are seen.

Round-1 status: all four bounded_memory runs (100MB/50T, 50MB/15T, 150MB/30T,
200MB/30T) are DIAGNOSTIC ONLY — no number from them appears in the paper. The
stream_run frozen-vs-pre-freeze comparison (58.59 → 64.07) IS retained as the
freeze-validation result.

## 1. Harness fixes (uniform; the round-1 artifact)

1.1 **Masked cross-entropy.** Loss computed only over classes present in the
    current cycle's data (cycle membership is data-derived — boundary-blind).
    Implementation: per-batch, mask logits to the union of classes in the cycle;
    optionally ACE-style (mask only for classes new to the stream) as a logged
    alternative — pick ONE for the round, uniformly, and record the choice.
    Rationale: unmasked 1000-way CE made every absent (including every
    previously-learned) class a permanent negative — the canonical logit-
    suppression failure (cf. ACE, Caccia et al.; MVP's logit masking) — and is
    the primary suspect for the round-1 collapse (first-checkpoint deficit
    ~30 pts; 50–60 pt drop vs stream_run at matched 15-task horizon;
    all-method convergence).
1.2 **Head weight decay = 0** (adapter LoRA wd already 0). Absent-class rows
    must not decay toward zero across a 122-cycle run.
1.3 Eval unchanged: volume checkpoints, logits masked to seen-so-far
    (data-derived), top-1 and top-5, per-latent-task breakdown write-only.
1.4 **Sanity anchor (gate):** re-run the 15-task/50MB configuration with fixes
    1.1–1.2 for SeqLoRA and InfLoRA. Acceptance: first-checkpoint accuracy
    within ~10 pts of the stream_run early-curve level (high 80s–low 90s) and
    final accuracy within ~15 pts of the stream_run finals. If the anchor still
    sits 30+ pts low, a second artifact exists — STOP and diagnose before any
    grid run.

## 2. Gating (this path has never been gated)

2.1 Golden-oracle test on the bounded_memory path: SketchLoRA with ε=0 and
    r̂ ≥ (cycles)·r reproduces the compression-disabled run end-to-end.
2.2 Eval-routing identity asserts extended to volume checkpoints (log + assert
    the parameter-state hash evaluated at each checkpoint).
2.3 Boundary-leak audit of bounded_memory_mixin.py: no task-indexed control
    flow anywhere in the training path (grep + review; commit the audit note).
2.4 Eviction-rule disambiguation: implement BOTH readings of the bounded-
    eviction count formula behind a switch; unit-test each on synthetic
    spectra; the spec-conformant rule is: below cap, evict
    t = min(r_residual, k_ε) trailing directions of the composite; at cap,
    evict exactly (composite_rank − r̂_max). Select the conformant reading,
    document the other's existence and test result. The non-conformant reading
    is never run in production.
2.5 Fold-noise reduction (uniform to SketchLoRA's own folds; touches no other
    method): either exact SVD at d=768 or RandSVD with oversampling scaled
    p = max(10, r̂/2) + 1 power iteration. At round-1 mean ranks (~54–61),
    fixed p=10 sits in the known error-inflation regime. Choose per the Plan A
    A5.3 criterion if that decision is still open; otherwise exact SVD.

## 3. Method configuration parity

3.1 **SketchLoRA lr sweep (removes the documented asymmetry):** identical
    protocol to the others — 3-point sweep {1e-4, 3e-4, 1e-3} on CIFAR-100 at
    the 20-epoch oracle convention, validation-split selection, winner
    propagated per the project rule. No in-setting tuning for any method.
3.2 Frozen SketchLoRA otherwise: rank 10, cap 128, bounded eviction
    (conformant reading), ε=0.01, LoRA wd 0.
3.3 **Lazy-merge arm** (boundary-blind self-clock; Online-LoRA precedent):
    accumulate the residual across cycles, fold when residual energy/rank
    saturates against internal thresholds. Runs as a LABELED VARIANT
    (SketchLoRA-LM) alongside default SketchLoRA, both reported. Supervisor
    sign-off required before it appears in any table.
3.4 O-LoRA, InfLoRA, SeqLoRA: unchanged (λ₁ 0.5; lamb 0.95/lame 1.0 with
    T := ceil(stream/cycle); swept lrs as round 1). If any baseline author-
    default includes head-bias handling we disabled, note it; none do here.

## 4. Experimental grid (pre-specified; replaces the round-1 ad-hoc grid)

Fixed task count within each comparison — budgets are the ONLY axis.
4.1 **Dose-response (primary figure):** OmniBenchmark-1K, 30 latent tasks,
    budgets as fractions of mean task size (~1,690 imgs ≈ 245MB):
    **{0.2×, 0.5×, 1×, 2×}** (≈ 49/122/245/490 MB). Methods: SeqLoRA, O-LoRA,
    InfLoRA, SketchLoRA, SketchLoRA-LM. Seeds: 2 (1993, 1996); +1 seed on any
    cell within 1 pt of a claimed crossover. 5×4×2 = 40 runs, each ≈ 30-task
    oracle cost (compute invariant to budget).
4.2 **Long-horizon (headline Pareto):** Omni-1K, FULL 100 tasks, budget 0.5×,
    all five configurations, 1–2 seeds. This is where O-LoRA's ~200-slot state
    and step-cost growth, InfLoRA's ~2,000-direction claim vs d=768, and
    SketchLoRA's flat resource profile are measured, not asserted.
4.3 ImageNet-R repeat of 4.1 at {0.2×, 1×} only, 2 seeds (cross-dataset check,
    20 runs, cheap).
4.4 Metrics per run: top-1/top-5 at volume checkpoints; A_AUC; final;
    degradation-from-own-oracle (Plan B main-table runs); per-method seed
    spread vs budget (first-class metric); M_inf/M_train and step-time vs
    cycle; SketchLoRA rank/energy/admitted-orthogonal-energy;
    accuracy-at-arrival per latent task (analysis-side).
Cost: ≈ 40 + ~10 + 20 runs; at post-fix throughput roughly 100–160 GPU-h.
Schedule: fixes + gates + anchor on A100s (~2–3 days); 4.1/4.3 backfill H200
gaps; 4.2 queues with the other Omni work.

## 5. Pre-registered predictions and decision tree (commit before 4.1 launches)

P1. With the head fixed, absolute numbers recover to the stream_run range and
    the methods separate.
P2. Comfortable budgets (1×, 2×): O-LoRA ≥ InfLoRA > SketchLoRA ≥ SeqLoRA.
P3. Thin budget (0.2×): InfLoRA degrades fastest; SketchLoRA's mean and seed
    variance are flattest; crossover of SketchLoRA above InfLoRA possible,
    above O-LoRA uncertain.
P4. 100-task: InfLoRA late-stream accuracy-at-arrival degradation; O-LoRA
    linear state/step-cost growth (measured); SketchLoRA within a few points
    of the accuracy leader at O(1) memory and flat step time.
Decision tree for the paper's task-agnostic section:
- If P3's crossover appears → headline = thin-budget robustness + Pareto.
- If not → headline = Pareto + variance only; the accuracy ordering is
  reported as-is, including SketchLoRA's position.
- Under NO branch are round-1 numbers cited, a budget cell dropped, or a new
  setting variant introduced post hoc to change a ranking. A further setting
  redesign is admissible only to fix a demonstrated artifact (anchor-gate
  failure in §1.4), never in response to an unwelcome ordering.

## 6. Supervisor items (gate 4.2, not the fixes)

- [ ] Masked-CE choice (cycle-mask vs ACE) acknowledged as the corrected spec.
- [ ] Lazy-merge admissibility (variant vs default).
- [ ] §5 predictions + decision tree reviewed and frozen.
- [ ] Agreement that round-1 bounded_memory numbers are diagnostic-only.
