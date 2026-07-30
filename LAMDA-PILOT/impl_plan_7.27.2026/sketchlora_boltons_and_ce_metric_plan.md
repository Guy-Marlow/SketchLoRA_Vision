# SketchLoRA Bolt-Ons (FD / Lazy Merge / CA) + Computational-Efficiency Metric
# Standalone addition to the running round-2 campaign. No existing plan is modified.

Scope: (Part 1) three individually-flaggable SketchLoRA extensions as SEPARATE
codepaths, composable in any combination; (Part 2) the Computational Efficiency
(CE) metric of Diaz-Rodriguez, Lomonaco, Filliat & Maltoni, "Don't forget, there
is more than forgetting: new metrics for Continual Learning" (2018), implemented
for every method in the campaign. Fairness rules from prior plans bind: bolt-ons
are labeled variants until sign-off; CE is counted identically for everyone;
all-flags-off must be bit-identical to the current frozen SketchLoRA.

================================================================================
PART 1 — SKETCHLORA BOLT-ONS
================================================================================

Config surface (all default false/off; any subset may be enabled):
  fd_shrinkage: bool
  lazy_merge: {off, period, plateau}   + lazy_merge_period: int,
                                         lazy_merge_delta: float,
                                         lazy_merge_max_holdoff: int
  classifier_alignment: bool           + ca_steps, ca_lr, ca_batch, ca_store
Each flag guards ONE function in ONE module (fd.py, lazy.py, ca.py); the merge
core calls optional hooks. No flag may alter any code executed when it is off.

## 1.1 FD shrinkage (fd_shrinkage)

Placement: inside the merge, AFTER the eviction count t is chosen (bounded-
eviction rule unchanged) and the composite SVD is truncated to rank l:
  sigma_shrunk = sqrt(clamp(Sigma[:l]**2 - Sigma[l]**2, min=0))
where Sigma[l] is the FIRST DISCARDED singular value. Refactor into the stored
factorization exactly as now (B = U*diag(sqrt(s)), A = diag(sqrt(s))*V^T).
Notes:
- If nothing is discarded (composite rank <= l), no shrinkage occurs (Sigma[l]
  is defined as 0). At the cap, shrinkage is the FD "pay rent" step that stops
  incumbent-mass runaway — the intended effect.
- Log per merge: pre/post-shrink total energy, Sigma[l]^2 (the rent), cumulative
  rent per module. Prediction to verify: ||sketch||_F plateaus instead of
  growing monotonically; k_eps stops saturating the eviction bound.
- Optional later variant (documented, NOT implemented now): robust-FD-style
  partial shrinkage sigma' = sqrt(Sigma^2 - alpha*Sigma[l]^2), alpha in (0,1].
- Theory hook (for the supervisor, not the coder): shrinkage upgrades the
  per-merge Eckart-Young bound to a stream-global FD/COD-type bound; routes:
  (a) corollary via Co-occurring Directions (Mroueh et al., AISTATS 2017) for
  the product-of-concatenations form of the adapter sum, or (b) direct one-page
  adaptation of the standard FD proof (Liberty 2013; Ghashami, Liberty,
  Phillips, Woodruff, SICOMP 2016). Cite SketchOGD (arXiv 2305.16424) and
  Sketchy (arXiv 2302.03764) as deep-learning/CL precedent for FD.

## 1.2 Lazy merge (lazy_merge)

Semantics: the residual slot keeps training across cycles; the fold fires on an
internal, boundary-blind trigger instead of every cycle.
- mode=period: fold every lazy_merge_period cycles (note: this generalizes the
  existing `period` hyperparameter; alias them, do not duplicate).
- mode=plateau (the new content; Online-LoRA precedent): at each cycle end
  compute drift d_c = ||R_c - R_{c-1}||_F / (||R_{c-1}||_F + eps) over the
  residual factors' product. Fold when d_c < lazy_merge_delta for 2 consecutive
  cycles, OR when cycles-since-last-fold == lazy_merge_max_holdoff (staleness
  cap; default 10). Log the trigger reason per fold.
- Constraints: optimizer state for the residual persists across non-folding
  cycles (do NOT reset per cycle when lazy is on — document this deviation);
  eval always routes through sketch + current residual composite so checkpoints
  never see a half-consolidated model penalty (assert via routing test).
- Rationale on record: at 100MB/Omni the eager rule produced 243 folds; each
  newcomer arrives with ~1/243 of accumulated energy. Plateau-lazy is expected
  to cut fold count 3-6x, raising newcomer relative weight accordingly.

## 1.3 Classifier alignment (classifier_alignment)

SLCA-style, exemplar-free:
- During each cycle, accumulate per-class feature mean and diagonal variance
  (768-d each) for classes present in the cycle, computed online from the
  penultimate features under the CURRENT model. ca_store selects {mean+diagvar}
  (default; ~6.1 MB for 1000 classes fp32 — add to M_train ledger).
- After each fold (or each cycle if no fold), run head-only alignment:
  ca_steps (default 300) steps, batch ca_batch (default 128) of pseudo-features
  sampled N(mu_c, diag(var_c)) uniformly over SEEN classes, CE loss masked to
  seen classes, lr ca_lr (default 1e-3), head parameters only, adapters frozen.
- Statistics are NEVER recomputed retroactively (no stored data); note the
  known drift caveat in logs. SDC-style drift compensation (transporting old
  means through sketch_old->sketch_new at fold time) is a documented follow-up,
  not in this round.
- Fairness: reported as SketchLoRA+CA labeled arm; if promoted beyond a variant,
  the same bolt-on must be offered to every method (it is method-agnostic).
- CA compute is charged to the method in the CE metric (Part 2).

## 1.4 Testing and run matrix

- Golden: all flags off == current frozen tag, bit-identical accuracy trace.
- Unit: FD on synthetic spectra (known shrinkage outcome); plateau trigger on
  synthetic drift sequences (fires exactly when specified); CA improves a
  synthetic drifted-head construction; routing assert with lazy on.
- A100 factorial (30-task Omni configs, 0.2x and 0.5x budgets, 1 seed):
  {off, FD, LM-plateau, CA, FD+LM, FD+LM+CA} = 6 arms x 2 budgets = 12 runs.
  Judge on late-task accuracy-at-arrival, admitted-orthogonal-energy,
  ||sketch||_F trajectory, final A; then ONE configuration goes to supervisor
  sign-off, gets tagged v2, and reruns the 100-task H200 cell (same seed as the
  subspace methods). One method, one configuration, every table; the v1->v2
  delta is itself reported.

================================================================================
PART 2 — COMPUTATIONAL EFFICIENCY (CE) METRIC
================================================================================

## 2.1 Definition and campaign instantiation

CE = min(1, (1/N) * sum_i [ Ops_fb(Tr_i) * eps / Ops(Tr_i) ])
- Ops_fb(Tr_i): mul-add operations for ONE forward + ONE backward pass over
  training set Tr_i.
- Ops(Tr_i): TOTAL mul-adds actually spent learning Tr_i — all epochs, all
  auxiliary machinery (penalties, hooks, consolidation, alignment, extra
  stages/passes).
- eps: we set eps = E = 20 (the shared epoch budget). Justification per the
  source paper: eps > 1 rescales CE when Ops_fb << Ops, moving the benchmark-
  dependent lower bound for interpretability. With eps = 20, a method with ZERO
  overhead beyond the matched budget scores exactly 1.0 (the min-cap), and CE
  reads directly as 1/(1 + overhead fraction). Record eps in every report.
- Unit Tr_i: ORACLE runs -> real task i, N = task count. BOUNDED-MEMORY runs ->
  cycle i, N = cycle count (no tasks exist; cycles are the learning exposures;
  document this instantiation in the paper). Because compute is stream-uniform
  by construction, also report the equivalent global ratio as a sanity check.
- CLscore/CLstability: if computed at all, criteria are reported individually
  first; weights are UNIFORM and pre-registered (free weight choice is a
  p-hacking vector — never tune weights after seeing criteria values).
  CLstability uses across-seed std of each criterion per Eq. 9.

## 2.2 Counting conventions (identical for every method)

- Count MACs (mul-adds). All FLOPs quoted elsewhere are 2x MACs — never mix.
- Ops_fb measured, not assumed: torch profiler (with_flops) on one training
  step, batch 48, per method, split into fwd and bwd; cross-check fwd against
  fvcore within 5%. Backward is whatever the profiler reports for that
  method's actual trainable set (frozen-backbone activation-grads included).
- STATE-DEPENDENT costs are recomputed, not amortized as constants: any per-
  step cost that grows with slots/rank/history is re-measured or re-derived at
  every eval checkpoint and integrated piecewise over cycles.
- Everything is counted. An item may be EXCLUDED from tables only with a logged
  analytic estimate proving it < 0.1% of Ops(Tr_i); the estimate ships in the
  artifact ledger regardless.
- Per-run artifact: ops_ledger.json — per cycle: step MACs (fwd, bwd, penalty),
  steps, boundary-op MACs itemized by category, auxiliary-pass MACs. CE is
  computed OFFLINE from the ledger (consistent with persist-everything design).

## 2.3 Per-method ledgers (campaign implementations, as built)

Reference magnitudes (ViT-B/16, 224^2, 197 tokens): forward ~17.6 GMACs/image.

- SeqLoRA: fwd + bwd only; optimizer resets negligible (log estimate). The
  eps=20 anchor: CE = 1.0 by construction. Serves as the counting sanity check.
- O-LoRA: (a) fwd uses the summed/merged delta -> constant fwd cost; (b) per-
  step orthogonality penalty: for each of 24 modules, current-A x each frozen
  prev-A^T: r^2*d MACs per slot-pair (~7.7e4), TIMES slot count s(c) at cycle c,
  plus its backward (~2x). Linear in s(c): integrate piecewise; at s=243 this
  is ~0.05% of step MACs — likely reportable-negligible in MACs even though
  wall-clock grows (Python/loop overhead): SAY SO EXPLICITLY, this is why CE
  and wall-clock are complementary, not redundant. (c) per-boundary slot
  allocation + merge construction: count, log, likely <0.1%.
- InfLoRA: (a) B-only weight-grads (slightly cheaper bwd — measured, not
  assumed); (b) FEATURE-COVARIANCE ACCUMULATION HOOKS: per image per hooked
  module ~tokens*d^2 ~ 1.16e8 MACs; over 24 modules ~2.8 GMACs/image ~ 16% of
  a forward — NOT negligible, and it recurs for whatever fraction of the
  stream the hooks are active (audit the implementation for exactly when hooks
  run: every batch? final epoch per cycle? count what the code actually does);
  (c) per-boundary d x d SVDs + DualGPM projections per module: ~768^3-order
  per module per boundary; 243 x 24 total -> log, expect ~0.01-0.1%; (d) the
  analytic A-initialization products. InfLoRA will report the LOWEST CE of the
  subspace methods primarily due to (b) — verify the audit before publishing
  that sentence.
- SketchLoRA: (a) sketch-inclusive training forward: slot-0 adds 2*d*r_hat(c)
  MACs/token/module; at r_hat=95 ~ 4% of forward and GROWING over the stream —
  recompute r_hat(c) from the diagnostics log and integrate piecewise (this is
  the dominant SketchLoRA overhead; do not average it); (b) per-fold merge:
  composite materialization + SVD (~d^3 exact, or randsvd cost) per module per
  fold — with lazy merge the fold count drops and CE IMPROVES: the ledger must
  make that visible; (c) CA if enabled: sampling + head-only steps
  (ca_steps * ca_batch * d * n_classes MACs + bwd) per alignment event —
  roughly one image-forward-equivalent per event; count and attribute.
- TreeLoRA (post bug-fix runs): (a) per-step sparse-update regularizer and
  gradient-similarity estimates (r*d-order — log estimate); (b) per-boundary
  bandit/LCB tree search and node updates (negligible-tier, log); (c) leaf
  adapter allocation. Expect CE ~ SeqLoRA within noise; the ledger proves it.
- Wave-3 methods (oracle campaign; brief but binding): HiDeLoRA — count ALL
  stages (main + correction epochs + statistics computation + task-inference
  training) inside Ops(Tr_i); its multi-stage design is exactly what CE exists
  to expose. CL-LoRA — the early-exit KD pass adds a partial forward per step
  (count the l-block fraction) + distillation loss ops. TUNA — per-task adapter
  training + fusion ops + the class-statistics extraction pass (extra forwards
  over task data: count). EASE — per-task adapter training + prototype
  extraction forwards + (task 2+) the prototype-complement computation.
  RainbowPrompt — prompt-evolution transforms per step (attention-based
  transformation + alignment: measured via profiler, non-trivial); its
  100-epoch author arm gets the same eps=20, i.e., its CE will be ~5x lower —
  that asymmetry is the metric working as intended; report it.

## 2.4 Validation and reporting

- Validation: (1) SeqLoRA CE == 1.0 exactly; (2) analytic vs profiler step MACs
  within 5% per method; (3) ledger totals x images == profiler-measured run
  totals within 10% on one A100 run per method; (4) recompute one method's CE
  by hand from its ledger in the analysis notebook and match the pipeline.
- Reporting: CE column beside accuracy/memory in every efficiency table, both
  regimes (oracle, bounded-memory); per-cycle CE trajectory figure for the
  state-dependent methods (O-LoRA penalty growth, SketchLoRA r_hat growth,
  InfLoRA hook regime); criteria table before any CLscore; weights uniform,
  pre-registered, stated.

## 2.5 Cost and sequencing

Ledger instrumentation is logging, not compute: ~0 GPU-h beyond the profiler
micro-runs (one step per method per checkpoint config, A100, < 2 h total).
Part 1 factorial: 12 A100 runs (~15-25 GPU-h post-fix). One H200 rerun of the
100-task cell for the tagged v2 (+CA arm if signed off). Gates: golden test
green before any flagged run; supervisor sign-offs (FD theory note, lazy-merge
admissibility, CA arm status) before the H200 rerun; CE ledgers active on that
rerun and on all subsequent campaign runs.
