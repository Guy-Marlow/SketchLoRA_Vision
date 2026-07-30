# SketchLoRA v2 Plan — CA Repair, Admission Floor, InfLoRA Positioning
# Standalone; follows the bolt-on/CE plan. Local A100 cluster = variant ranking
# only (declared methodology validation); H200 same-seed runs = admissible.

## 0. Decisions locked by the ablation data (record in repo)
- FD shrinkage: OFF the v2 path (cost 2-4 top1 at 15T). Retained as the
  compression dial (lazy@100MB: 65.82 top1 at r_hat 22 vs 66.42 at 93 — report
  as the rank-accuracy tradeoff figure). Long-horizon FD claim remains untested
  at 15T — do not describe FD as refuted, only as off-path.
- Annealing-epsilon proposal: REJECTED (needs a clock/horizon — reintroduces
  the stream-length oracle we criticize in InfLoRA; fights symptom not cause).
- Tree-merge proposal: DEFERRED to v3/future work. Note for the paper: pairwise
  balanced merging = mergeable-summaries / distributed-FD structure (Agarwal et
  al. 2012; Ghashami et al. 2016 merge theorem); error composes additively over
  a log-depth root path; retained state becomes O(log K) spine (not O(1));
  loss is scheduled fairly, not eliminated. Pilot only if reserve remains.

## 1. Admission rule v2: floor + cap-turnover (fixes the cap-collapse bug)
Rule (single codepath replacing guaranteed_admission and force_increase):
- Below cap: evict t = min(r_residual, k_eps) trailing directions, EXCEPT the
  top-k directions of the residual's component orthogonal to the pre-merge
  sketch, which are protected this merge (k = admission_floor, sweep {1, 5}).
- At cap: evict (composite_rank - cap) directions FROM THE NON-PROTECTED SET
  ONLY (incumbent turnover). The floor must survive the cap branch — unit-test
  exactly the failure found in force_increase (evict computed irrespective of
  floor once rank hits 128).
- Uncapped variant (rank_cap=None) remains a labeled config for the adaptive
  flagship if the memory story tolerates r_hat ~150 (~25MB — still << 92/345MB).
- Protection is per-merge only (protected directions become ordinary incumbents
  next merge) — document; a multi-merge protection window is v3 territory.
Tests: synthetic spectra (floor survives at cap); golden (k=0, cap=128 ==
current bounded_eviction bit-exact).

## 2. CA repair sweep (local, 100MB 15T Omni slice, 1 seed, ~1 afternoon)
Target: keep the +4.6 top1, recover the -3.7/-5.3 top5. Variants:
  a. ca_steps {50, 100, 300} with early stop on a held-out pseudo-feature set;
  b. covariance: per-class diagonal (current) vs shared full (768^2, one
     matrix, ~2.25MB) vs per-class low-rank(8)+diag;
  c. real-feature mixing: alignment batches = 50% stored-stat samples + 50%
     real current-cycle features;
  d. logit-adjustment-only arm: no head retraining; additive per-class prior
     correction from stored counts/means (cannot damage within-class ranking);
  e. (a-winner) x (b-winner) combined;
  f. current CA as control.
Selection metric: top1 + top5 jointly (reject any variant with top5 < baseline
- 1.0). Winner becomes ca_v2.
Fairness (binding): the money cell runs InfLoRA+CA_v2 alongside SketchLoRA+
CA_v2 — CA is method-agnostic and InfLoRA+CA exists in its own paper; we run
the comparison before reviewers ask. O-LoRA/SeqLoRA+CA: 1 local run each to
report the delta; include in appendix.

## 3. v2 assembly and the money rerun (H200)
- v2 = frozen v1 + floor(k from sweep) + ca_v2. Lazy/FD stay available as
  labeled compression variants. Version-tag; goldens green.
- Pre-register before launch: expected v2 100-task final in the mid-30s;
  decision rule — if v2 within 6-8 top1 of InfLoRA+CA_v2 at 100 tasks, the
  paper leads accuracy-competitive + frontier; if not, the paper leads the
  frontier + oracle-concession + compression-dial story. Either branch is
  written; no third round of variant mining on the money cell.
- Runs (all same seed 1993, 100MB, 100-task Omni): SketchLoRA-v2,
  SketchLoRA-v2-lazy (compression point), InfLoRA+CA_v2, and the missing
  same-seed cells from the current grid (SeqLoRA/SketchLoRA were 1996 —
  rerun on 1993 or rerun O-LoRA/InfLoRA on 1996; pick ONE seed set, fix the
  cross-seed confound). TreeLoRA joins now that the hook bug is fixed.
  ~6-8 runs; second seed on the headline pair if ladder allows.

## 4. InfLoRA positioning package (analysis, ~0 GPU-h beyond above)
Report, per method, at t = {1, 50, 100}:
- M_inf as-implemented AND best-case-folded (InfLoRA: 92MB -> fold frozen_delta
  into W; state both; the fold does NOT apply to M_train);
- M_train (InfLoRA: DualGPM feature_mat d^2/layer — required to keep learning);
- inference latency + GFLOPs (already measured: SketchLoRA 1.078ms/36.5G vs
  InfLoRA 1.235ms/40.9G — SketchLoRA leads; make this a stated result);
- CE with the hook audit resolved (when exactly do covariance hooks run; the
  ~16%-of-forward estimate stands or falls on this — audit before any sentence
  about InfLoRA's CE is written);
- the ORACLE CONCESSION table: per method, what boundary-free deployment
  requires — InfLoRA: total horizon T a priori (documented harness injection);
  TreeLoRA/O-LoRA: none but linear state; SketchLoRA: none. This is a headline
  qualitative result of the setting, not a footnote.
- TreeLoRA leaf-growth curve added to the frontier figure (IN-R grid shows it
  accuracy-strong at 60/63 — the frontier story must include it honestly).

## 5. Sequencing
Day 1: §1 patch + tests; §2 sweep launched (parallel). Day 2: v2 tag, §3
pre-registration committed, H200 queue. Days 3-4: §4 analysis while runs
complete. Everything else in flight (oracle grid, wave 3) is untouched.
