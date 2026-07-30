"""Per-method analytic auxiliary-MAC formulas for the CE ledger
(impl_plan_7.27.2026 sec 2.3). Two provenance tiers, kept distinct in the
docstrings below:

  CONFIRMED FROM CODE this session: O-LoRA's per-step orthogonality-penalty
  call frequency and slot-count-vs-cycle relationship (models/olora.py); the
  fact that InfLoRA's feature-covariance hooks fire during two DEDICATED
  extra full passes over each chunk's data (_init_lora_A + _update_dualgpm in
  models/inflora.py), not every regular training step.

  TAKEN FROM THE PLAN'S OWN STATED MAGNITUDES, not independently re-derived
  against a profiler line-by-line in this session: the exact GMACs/module
  constants (tokens*d^2 for InfLoRA's hooks, r^2*d for O-LoRA's penalty
  pair-cost, d^3-order DualGPM/exact-SVD boundary costs, ca_steps*ca_batch*
  d*n_classes for CA). These match the plan's sec 2.3 text; treat them as
  documented estimates until profiled and cross-checked per sec 2.4.

Every function returns MACs (mul-adds), never FLOPs.
"""

DIM = 768          # ViT-B/16 embed dim
TOKENS = 197        # 224^2 patch16 + cls token
N_MODULES = 24      # 12 blocks x {q, v}


def seqlora_aux_macs_per_step():
    """fwd+bwd only -- the eps=20 anchor, CE=1.0 by construction (sec 2.3)."""
    return 0.0


def olora_aux_macs_per_step(slot_count, rank=10, dim=DIM, n_modules=N_MODULES):
    """CONFIRMED (models/olora.py::_orth_and_l2, called every step via
    _stream_extra_loss): current-A x each frozen prev-A^T, r^2*d MACs per
    slot-pair (~7.7e4 at r=10,d=768 -- matches plan sec 2.3's stated
    magnitude exactly), TIMES slot_count (= self._stream_chunk, one new slot
    per chunk boundary), PLUS backward (~2x forward, per the plan). Linear in
    slot_count -- integrate piecewise over cycles using each cycle's own
    slot_count, do not average."""
    fwd = (rank ** 2) * dim * n_modules * slot_count
    return fwd * 3.0   # forward + ~2x backward, per plan sec 2.3(b)


def inflora_boundary_macs(chunk_images, dim=DIM, tokens=TOKENS, n_modules=N_MODULES):
    """CONFIRMED (models/inflora.py::_init_lora_A + _update_dualgpm, each doing
    ONE FULL EXTRA PASS over the chunk's train_loader to accumulate
    cur_matrix -- not every regular training step): 2 passes x chunk_images x
    tokens*d^2 MACs/module x n_modules. tokens*d^2 ~ 1.16e8 MACs/module/image
    (plan's stated magnitude, ~16% of one forward) -- NOT independently
    re-derived here, taken as given."""
    per_image_per_module = tokens * (dim ** 2)
    return 2 * chunk_images * per_image_per_module * n_modules


def inflora_dualgpm_svd_macs(n_modules=N_MODULES, dim=DIM):
    """TAKEN FROM PLAN (sec 2.3c): d^3-order per module per boundary SVD +
    projection cost for DualGPM's update_DualGPM. Plan's own expectation:
    ~0.01-0.1% of Ops(Tr_i) -- logged for completeness, not because it's
    expected to matter."""
    return n_modules * (dim ** 3)


def sketchlora_step_macs_sketch_inclusion(r_hat, dim=DIM, tokens=TOKENS, n_modules=N_MODULES):
    """TAKEN FROM PLAN (sec 2.3, SketchLoRA(a)): slot-0 (the frozen sketch)
    adds 2*d*r_hat MACs/token/module to every training forward, growing over
    the stream as r_hat grows -- the DOMINANT SketchLoRA overhead per the
    plan ('do not average it'). r_hat should come from this run's own
    sketchlora_diag_*.json r_hat_mean at the cycle in question, not a
    campaign-wide constant."""
    return 2 * dim * r_hat * tokens * n_modules


def sketchlora_fold_macs(r_hat, dim=DIM, oversampling=10, merge_op="randsvd"):
    """TAKEN FROM PLAN (sec 2.3, SketchLoRA(b)): composite materialization +
    SVD cost per module per FOLD (not per step) -- exact-SVD order d^3, or
    randsvd's cheaper (d, r_hat+oversampling)-order cost. With lazy merge the
    fold COUNT drops (fewer calls to this function per run), which is exactly
    how CE is expected to improve under lazy merge -- the ledger must show
    this via fewer auxiliary_pass_macs entries, not a smaller per-fold cost."""
    if merge_op == "exactsvd":
        return dim ** 3
    # randsvd: dominant cost is forming the sketch (dim x (r_hat+oversampling))
    # and its SVD, ~ dim * (r_hat + oversampling)^2 order (standard randomized-
    # SVD complexity), taken as the plan's implied cost model for this ablation.
    k = r_hat + oversampling
    return dim * (k ** 2)


def sketchlora_ca_macs(ca_steps, ca_batch, n_classes, dim=DIM):
    """TAKEN FROM PLAN (sec 2.3c / sec 1.3): ca_steps * ca_batch * d * n_classes
    MACs (forward) + backward (~2x) per alignment event -- 'roughly one
    image-forward-equivalent per event' per the plan's own framing."""
    fwd = ca_steps * ca_batch * dim * n_classes
    return fwd * 3.0


def treelora_aux_macs_per_step(rank=10, dim=DIM, n_modules=N_MODULES):
    """TAKEN FROM PLAN (sec 2.3, TreeLoRA(a)): per-step sparse-update
    regularizer + gradient-similarity estimate, r*d-order per module. Plan's
    expectation: CE ~ SeqLoRA within noise -- this formula is what the ledger
    uses to verify that expectation, not assume it."""
    return rank * dim * n_modules * 3.0   # fwd-order estimate + ~2x backward
