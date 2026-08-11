"""InfLoRA (Interference-Free Low-Rank Adaptation) for LAMDA-PILOT.

Port of svd_sketching_vision/InfLoRA (methods/inflora.py + models/vit_inflora.py)
onto the shared LoRA scaffold.  Per task t:

  1. Collect the input second-moment matrix ``cur_matrix`` (E[x x^T]) at every
     attention block by forwarding the task's data with the *accumulated* model
     (LoRAs 0..t-1 applied).
  2. Set the new LoRA down-projection A analytically from the SVD of
     ``cur_matrix`` projected to *remove* the subspace already spanned by past
     tasks (DualGPM).  A is then frozen; only the up-projection B (and the head)
     is trained.  This makes task t's update live in directions that barely
     affect the inputs of previous tasks -> "interference free".
  3. After training, recompute ``cur_matrix`` and grow the DualGPM feature
     memory ``feature_list`` / projection matrices ``feature_mat``.

LoRA is on the query and value projections (our shared convention; the original
uses key/value).  Inference reuses the merged-sum (CIL) / task-routed (TIL)
evaluation of the baseline ``LoRAVitNet`` + ``TILLearner``.
"""

import logging
import math
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.lora import Learner as LoRALearner
# *** UNTESTED as of 2026-08-03 *** -- measured-CE region tagging
# (docs/ce_profiling_implementation_plan.md sec 4.3). No-op unless a profiling
# session is active (utils/ce_profiler.py).
from utils.ce_profiler import ce_region, run_boundary
from utils.ce2_profiler import ce2_boundary

# 2026-08-10: 8->4, see models/lora.py's identical change for rationale.
num_workers = 8


class Learner(LoRALearner):
    def __init__(self, args):
        super().__init__(args)
        # InfLoRA's frozen slots are never modified once trained (A is set once
        # analytically, B is trained once, neither touched again) -- safe to fold
        # into a dense delta for O(1) merged forward (plan doc §6 item 2).
        self._network.enable_frozen_folding()
        self.rank = args.get("lora_rank", 10)
        self.lamb = args.get("lamb", 0.95)        # DualGPM threshold lower bound
        self.lame = args.get("lame", 1.0)         # DualGPM threshold upper bound
        self.total_sessions = args["nb_tasks"]
        self.train_merge = True                   # accumulated forward (sum 0..t)
        self.feature_list = []                    # per-layer orthonormal bases (np)
        self.project_type = []                    # 'remove' | 'retain' per layer
        self.feature_mat = []                     # per-layer projection matrices

    # FLAGGED CHANGE (2026-07-21): exact persistent_state() override, added
    # alongside the free_folded_slots() fix above. Previously InfLoRA had no
    # override and fell back to utils.metrics_logger.default_persistent_state(),
    # which was measuring the wrong thing entirely: it sums named_parameters(),
    # which (a) INCLUDED the (now-freed) dead per-task lora_A/lora_B bank, and
    # (b) EXCLUDED frozen_delta_q/frozen_delta_v entirely, since those are
    # register_buffer, not nn.Parameter, so named_parameters() never saw them.
    # This override counts what InfLoRA's algorithm actually needs to persist:
    # the folded delta (bounded, O(d^2) per block), the DualGPM bases that seed
    # future tasks' analytic A, and the current (not-yet-folded) task's own live
    # slot -- matching exactly what free_folded_slots() leaves allocated.
    def persistent_state(self):
        net = self._network.module if hasattr(self._network, "module") else self._network
        frozen_delta_bytes = 0
        for attn in net.attn_modules():
            frozen_delta_bytes += attn.frozen_delta_q.numel() * attn.frozen_delta_q.element_size()
            frozen_delta_bytes += attn.frozen_delta_v.numel() * attn.frozen_delta_v.element_size()
        cur_slot_bytes = 0
        for attn in net.attn_modules():
            for mlist in (attn.lora_A_q, attn.lora_B_q, attn.lora_A_v, attn.lora_B_v):
                for p in mlist[self._cur_task].parameters():
                    cur_slot_bytes += p.numel() * p.element_size()
        # feature_mat is a cached derivative of feature_list (feature_mat[i] =
        # feature_list[i] @ feature_list[i].T) -- recomputable, not fundamentally
        # needed to persist, but counted here anyway since it's actually held in
        # memory right now (this project's "raw allocation" accounting convention,
        # same as sketchlora's persistent_state -- see its docstring).
        dualgpm_bytes = 0
        for f in self.feature_list:
            dualgpm_bytes += f.numel() * f.element_size()
        for m in self.feature_mat:
            dualgpm_bytes += m.numel() * m.element_size()
        fc_bytes = sum(p.numel() * p.element_size() for p in net.fc.parameters()) if net.fc is not None else 0
        total_bytes = frozen_delta_bytes + cur_slot_bytes + dualgpm_bytes + fc_bytes
        return {"params": int(total_bytes // 4), "bytes": int(total_bytes),
                "breakdown": {"frozen_delta": frozen_delta_bytes, "current_slot": cur_slot_bytes,
                             "dualgpm_bases": dualgpm_bytes, "fc": fc_bytes}}

    # -- TIL eval must use InfLoRA's *merged* adapter -------------------
    # Like O-LoRA, InfLoRA trains and infers with the merged sum (merge=True).
    # The inherited TIL routing (slot t alone, merge=False) mismatches training
    # and depresses TIL; override to evaluate the merged adapter (masking to the
    # task's class slice still happens in _eval_til).
    def _forward_task(self, inputs, task):
        net = self._network.module if hasattr(self._network, "module") else self._network
        return net(inputs, task=self._cur_task, merge=True)

    # -- sample-boundary streaming hooks (reuse _init_lora_A / _update_dualgpm) --
    # One DualGPM slot per CHUNK; at each chunk start A is set analytically from the
    # input covariance of the CURRENTLY-available data (the in-progress class-task),
    # projected to remove past chunks' subspace. Because chunk edges straddle real
    # class-groups, that covariance mixes class-groups -> the subspace the method
    # carves is "messy" relative to the true tasks. With lamb=lame the DualGPM
    # threshold is constant (no per-task annealing).
    def _stream_init(self):
        self._stream_chunk = -1
        self._stream_train_a = False             # InfLoRA: A analytic+frozen, train only B

    def _stream_slot(self):
        return self._stream_chunk

    # Plan C §C1 concession: total_sessions (T, used by update_DualGPM's
    # threshold ramp below) is normally the real task count; under bounded-
    # memory streaming there is no real task count available a priori in the
    # same sense, so the harness computes T := ceil(stream_images/cycle_images)
    # and sets it once via this hook before training starts (models/
    # bounded_memory_mixin.py::bounded_memory_run). No-op default lives on
    # BoundedMemoryMixin; only InfLoRA needs this concession.
    def _bounded_set_total_sessions(self, total_sessions):
        self.total_sessions = total_sessions

    def _stream_begin_chunk(self, loader):
        self._stream_chunk += 1
        if self._stream_chunk > 0:
            # slot count is not generically bounded by nb_tasks under this clock --
            # see BOUNDARY_AGNOSTIC_IMPLEMENTATION_LOG.md's "BLOCKING ARCHITECTURAL GAP"
            self._network.add_task_slot()
        self._cur_task = self._stream_chunk      # _init_lora_A / DualGPM read _cur_task as slot
        self._network.freeze_to_task(self._stream_chunk, train_a=False)
        # FLAGGED CHANGE (2026-07-21): freeze_to_task just folded chunk-1's slot into
        # frozen_delta -- free its now-fully-redundant weight memory (see
        # Attention_LoRA.free_folded_slot's docstring; InfLoRA-specific, safe here).
        # *** UNTESTED as of 2026-08-03 *** -- plan sec 4.3: tagged "for
        # completeness" per the plan -- expected negligible (nn.Identity()
        # replacement, no tensor compute), included so the ledger shows it was
        # considered rather than silently absent.
        with ce_region("inflora/free_folded_slots"):
            self._network.free_folded_slots(self._stream_chunk - 1)
        for p in self._network.fc.parameters():
            p.requires_grad = True
        self._network.to(self._device)
        self._init_lora_A(loader)                # analytic, DualGPM-projected A for this chunk
        self._stream_new_optimizer()

    def _stream_end_chunk(self, loader):
        self._update_dualgpm(loader)             # grow DualGPM feature memory

    def _ce_boundary_macs_this_cycle(self, chunk_images, macs_per_image_fwd=0.0):
        # impl_plan_7.27.2026 sec 2.3: CONFIRMED two dedicated extra full passes
        # over the chunk's own data per cycle (_init_lora_A at begin, _update_
        # dualgpm at end, both above) -- not a per-step cost.
        #
        # *** UNTESTED as of 2026-08-03 *** -- local GPUs were unavailable
        # (thermal/damage risk) at the time this was written, so this has NOT
        # been exercised on a live run. Verified only by static tracing: the
        # macs_per_image_fwd plumbing (stream_mixin.py's base signature ->
        # bounded_memory_mixin.py's call site -> here) has exactly one call
        # site in the whole codebase (grepped), and the arithmetic/units were
        # checked by hand, but no profiler run has confirmed this executes
        # without error or produces the expected magnitude. Confirm on the
        # first real run (e.g. imagenetr_slurm_grid) before trusting its CE
        # numbers -- check the written ops_ledger_*.json for a populated
        # "covariance_hooks_base_forward" key with a plausible (nonzero,
        # roughly 7x the "covariance_hooks_bookkeeping" key) magnitude.
        #
        # FIXED 2026-08-03 (undercounting bug found during a persistent-memory/
        # CE audit): this used to charge ONLY the incremental cost of the
        # covariance accumulation itself (inflora_boundary_macs -- the extra
        # outer-product bookkeeping, ~16% of one forward per the plan's own
        # figure), never the BASE forward-pass cost of running those two
        # passes at all. But _init_lora_A/_update_dualgpm each call
        # net(inputs, ..., merge=True) -- a genuine full ViT forward, not just
        # a covariance-accumulation step -- so the covariance bookkeeping is a
        # small increment riding on top of an otherwise-full-price forward
        # pass, and only the small increment was ever being charged. Audited
        # the actual ce_ledger.record_unit(...) call site
        # (bounded_memory_mixin.py) and confirmed auxiliary_pass_macs (the
        # field meant to hold "a full extra pass costs this much") was never
        # populated for InfLoRA anywhere -- nothing was covering this gap.
        # Fixed by adding the base cost explicitly: 2 passes x chunk_images x
        # macs_per_image_fwd (the same profiler-measured per-image forward
        # cost already used for Ops_fb, now threaded through this hook -- see
        # models/stream_mixin.py's base signature and bounded_memory_mixin.py's
        # call site). Back-of-envelope: this missing term was found to be
        # roughly 7x larger than the covariance-bookkeeping term alone, i.e.
        # the previously-reported CE (~0.993-0.994) understated InfLoRA's true
        # overhead by roughly 4x (true CE closer to ~0.96). NOT applied
        # retroactively to already-completed runs (round2_slurm_grid) -- this
        # fix takes effect on new runs only (e.g. the imagenetr_slurm_grid
        # submission), per explicit user instruction not to estimate/backfill
        # old CE numbers.
        from utils.ce_formulas import inflora_boundary_macs, inflora_dualgpm_svd_macs
        return {
            "covariance_hooks_base_forward": 2 * chunk_images * macs_per_image_fwd,
            "covariance_hooks_bookkeeping": inflora_boundary_macs(chunk_images),
            "dualgpm_svd": inflora_dualgpm_svd_macs(),
        }

    # -- task loop ------------------------------------------------------
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.data_manager = data_manager
        self._network.update_fc(self._total_classes)
        self._register_task_range()
        self._network.default_task = self._cur_task
        logging.info("[InfLoRA] Learning on {}-{}".format(self._known_classes, self._total_classes))

        # Grow one adapter slot if this task needs an index that isn't allocated
        # yet (construction only preallocates slot 0 -- see utils/inc_net.py).
        if self._cur_task >= self._network.backbone.n_tasks:
            self._network.add_task_slot()

        # only B (+ head) trainable; A is set analytically below
        self._network.freeze_to_task(self._cur_task, train_a=False)
        # FLAGGED CHANGE (2026-07-21): freeze_to_task just folded task-1's slot into
        # frozen_delta -- free its now-fully-redundant weight memory (see
        # Attention_LoRA.free_folded_slot's docstring; InfLoRA-specific, safe here;
        # NOT applied to O-LoRA/TreeLoRA/HideLoRA, which also opt into folding but
        # may still need individual past slots -- O-LoRA's orthogonality penalty
        # genuinely reads every past lora_A forever).
        # *** UNTESTED as of 2026-08-03 *** -- same tag as _stream_begin_chunk's
        # call, for the oracle (non-bounded-memory) per-task path; CE profiling
        # is currently only wired into bounded_memory_mixin.py, so this is a
        # no-op today, kept for consistency if that ever changes.
        with ce_region("inflora/free_folded_slots"):
            self._network.free_folded_slots(self._cur_task - 1)
        for p in self._network.fc.parameters():
            p.requires_grad = True

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                       shuffle=True, num_workers=num_workers, persistent_workers=True)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                      shuffle=False, num_workers=num_workers, persistent_workers=True)

        self._network.to(self._device)
        # oracle-mode boundary bookkeeping (docs/ce_step_boundary_isolation_plan.md
        # sec 7): InfLoRA has no per-step aux (train_merge=True's fold-branch
        # overhead is embedded in the ordinary forward and only measurable via R2,
        # see trainer.py) -- its ENTIRE step-vs-SeqLoRA overhead is these two
        # boundary calls, each a genuine extra full forward pass over the task's
        # data. Two DISTINCT kind names (not one shared "boundary") because
        # CEProfileController.commit() overwrites its held value per kind -- using
        # two kinds and merging via all_current() at ledger-write time is the same
        # pattern models/bounded_memory_mixin.py already uses for
        # boundary_begin/boundary_end. `_ce_boundary_ctrl` is None (and
        # run_boundary falls back to a direct call) whenever final_metrics/CE-
        # logging is off, under bounded_memory streaming (that path's own
        # _stream_begin_chunk/_stream_end_chunk are already wrapped end-to-end by
        # the driver, see there), and for any other trainer.py track that never
        # sets this attribute.
        _ctrl = getattr(self, "_ce_boundary_ctrl", None)
        with ce2_boundary(self):
            run_boundary(_ctrl, "inflora_init_a", lambda: self._init_lora_A(self.train_loader))
        self._log_trainable()
        self._train(self.train_loader)           # train B + head (inherited)
        with ce2_boundary(self):
            run_boundary(_ctrl, "inflora_dualgpm", lambda: self._update_dualgpm(self.train_loader))

    # -- (1)+(2) analytic init of the down-projection A -----------------
    @torch.no_grad()
    def _init_lora_A(self, train_loader):
        net = self._network
        net.eval()
        net.reset_cur_matrix()
        net.set_collect(True)
        # *** UNTESTED as of 2026-08-03 *** -- plan sec 4.3 "init_lora_A_forward":
        # a genuine FULL extra forward pass over the whole chunk (this is the
        # base-forward cost the 2026-08-03 CE-undercounting fix added to the
        # analytic formula -- this tag is its measured counterpart). Includes
        # backbone/vit_lora.py::_accumulate_cov's bmm(x^T,x) via
        # set_collect(True) above (tagged separately inside that method as
        # "inflora/covariance_accumulate").
        with ce_region("inflora/init_lora_A_forward"):
            for _, inputs, _t in train_loader:
                net(inputs.to(self._device), task=self._cur_task, merge=True)
        net.set_collect(False)

        # *** UNTESTED as of 2026-08-03 *** -- plan sec 4.3 "init_lora_A_svd":
        # per-module SVD in float64 (NOT float32 -- a real cost multiplier the
        # old flat formula never modeled) + the feature_mat projection.
        # Previously UNCOUNTED as its own line item (folded into
        # inflora_dualgpm_svd_macs's single flat n_modules*dim^3 constant,
        # which also covers update_DualGPM below -- conflating two genuinely
        # different operations into one number).
        with ce_region("inflora/init_lora_A_svd"):
            for kk, attn in enumerate(net.attn_modules()):
                cur = attn.cur_matrix.clone().double()           # [dim, dim] on cpu
                if self._cur_task == 0:
                    U, S, _ = torch.linalg.svd(cur)
                    basis = U[:, :self.rank]
                else:
                    fmat = self.feature_mat[kk].double()
                    if self.project_type[kk] == "remove":
                        cur = cur - fmat @ cur
                    else:
                        assert self.project_type[kk] == "retain"
                        cur = fmat @ cur
                    U, S, _ = torch.linalg.svd(cur, full_matrices=False)
                    basis = U[:, :self.rank]
                A = (basis.t() / math.sqrt(3)).float().to(self._device)   # [rank, dim]
                attn.lora_A_q[self._cur_task].weight.data.copy_(A)
                attn.lora_A_v[self._cur_task].weight.data.copy_(A)
                attn.reset_cur_matrix()

    # -- (3) grow the DualGPM feature memory ----------------------------
    @torch.no_grad()
    def _update_dualgpm(self, train_loader):
        net = self._network
        net.eval()
        net.reset_cur_matrix()
        net.set_collect(True)
        # *** UNTESTED as of 2026-08-03 *** -- plan sec 4.3 "update_dualgpm_forward":
        # the SECOND full extra forward pass this cycle (measured counterpart
        # of the 2026-08-03 base-forward-cost fix).
        with ce_region("inflora/update_dualgpm_forward"):
            for _, inputs, _t in train_loader:
                net(inputs.to(self._device), task=self._cur_task, merge=True)
        net.set_collect(False)

        mat_list = [attn.cur_matrix.clone() for attn in net.attn_modules()]
        for attn in net.attn_modules():
            attn.reset_cur_matrix()
        # update_DualGPM tags its own internals (inflora/update_dualgpm_linalg,
        # inflora/dualgpm_python_loop) -- see that method, plan sec 4.3.
        self.update_DualGPM(mat_list)

        # *** UNTESTED as of 2026-08-03 *** -- plan sec 4.3 "feature_mat_rebuild":
        # O(r_i * dim^2), grows with r_i (the CURRENT DualGPM basis size for
        # layer p) -- previously UNCOUNTED as its own line item. This is
        # DISTINCT from update_dualgpm_linalg (computed above, inside
        # update_DualGPM) even though both read feature_list -- this rebuild
        # happens AFTER update_DualGPM has already finished growing
        # feature_list for this cycle, producing the CACHED projector
        # _init_lora_A reads next cycle.
        with ce_region("inflora/feature_mat_rebuild"):
            self.feature_mat = []
            for p in range(len(self.feature_list)):
                Uf = self.feature_list[p] @ self.feature_list[p].t()
                self.feature_mat.append(Uf)

    # -- DualGPM (torch/GPU port of InfLoRA/methods/inflora.py::update_DualGPM --
    # verbatim algorithm, same SVD-throughout structure as the reference (no eigh
    # substitution -- the reference never uses eigh even for the symmetric
    # covariance case, so torch.linalg.svd is the faithful match). Only the
    # tensor backend changes: np.linalg.svd/CPU-float64 -> torch.linalg.svd/GPU,
    # mirroring cur_matrix's own device (see vit_lora.py's Attention_LoRA, which
    # used to force cur_matrix onto CPU on every accumulation step -- a device
    # placement bug independent of this numpy->torch port, fixed alongside it).
    def update_DualGPM(self, mat_list):
        threshold = (self.lame - self.lamb) * self._cur_task / self.total_sessions + self.lamb
        logging.info("[InfLoRA] DualGPM threshold: {:.4f}".format(threshold))
        # *** UNTESTED as of 2026-08-03 *** -- plan sec 4.3, THE region expected
        # to reproduce the observed 36.2->38.8 s/cycle wall-clock climb (docs/
        # ce_profiling_implementation_plan.md sec 0 defect #2): the OLD analytic
        # formula (inflora_dualgpm_svd_macs) is a flat n_modules*dim^3 constant
        # that ignores feature_list[i]'s growing column count r_i -- but
        # `feature_list[i] @ (feature_list[i].t() @ activation)` below is
        # genuinely O(r_i * dim^2), and r_i grows every cycle this branch runs.
        # Two DISTINCT regions inside this function, not one: this tag covers
        # the tensor/SVD ops (which DO grow with r_i and are real MACs); the
        # rank-selection for-loops below are tagged separately as
        # "inflora/dualgpm_python_loop" (R5 -- Python-loop + implicit
        # tensor->bool sync cost via `if accumulated_sval < threshold:`,
        # invisible to a MAC-only view regardless of r_i).
        if len(self.feature_list) == 0:
            with ce_region("inflora/update_dualgpm_linalg"):
                for activation in mat_list:
                    U, S, Vh = torch.linalg.svd(activation, full_matrices=False)
                    sval_total = (S ** 2).sum()
                    sval_ratio = (S ** 2) / sval_total
                    r = int(torch.sum(torch.cumsum(sval_ratio, 0) < threshold).item())
                    self.feature_list.append(U[:, 0:max(r, 1)])
                    self.project_type.append('remove' if r < (activation.shape[0] / 2) else 'retain')
        else:
            for i in range(len(mat_list)):
                activation = mat_list[i]
                if self.project_type[i] == 'remove':
                    with ce_region("inflora/update_dualgpm_linalg"):
                        U1, S1, Vh1 = torch.linalg.svd(activation, full_matrices=False)
                        sval_total = (S1 ** 2).sum()
                        act_hat = activation - self.feature_list[i] @ (self.feature_list[i].t() @ activation)
                        U, S, Vh = torch.linalg.svd(act_hat, full_matrices=False)
                        sval_hat = (S ** 2).sum()
                        sval_ratio = (S ** 2) / sval_total
                        accumulated_sval = (sval_total - sval_hat) / sval_total
                    r = 0
                    with ce_region("inflora/dualgpm_python_loop"):
                        for ii in range(sval_ratio.shape[0]):
                            if accumulated_sval < threshold:
                                accumulated_sval += sval_ratio[ii]
                                r += 1
                            else:
                                break
                    if r == 0:
                        continue
                    with ce_region("inflora/update_dualgpm_linalg"):
                        Ui = torch.cat([self.feature_list[i], U[:, 0:r]], dim=1)
                        self.feature_list[i] = Ui[:, 0:Ui.shape[0]] if Ui.shape[1] > Ui.shape[0] else Ui
                else:
                    assert self.project_type[i] == 'retain'
                    with ce_region("inflora/update_dualgpm_linalg"):
                        U1, S1, Vh1 = torch.linalg.svd(activation, full_matrices=False)
                        sval_total = (S1 ** 2).sum()
                        act_hat = self.feature_list[i] @ (self.feature_list[i].t() @ activation)
                        U, S, Vh = torch.linalg.svd(act_hat, full_matrices=False)
                        sval_hat = (S ** 2).sum()
                        sval_ratio = (S ** 2) / sval_total
                        accumulated_sval = sval_hat / sval_total
                    r = 0
                    with ce_region("inflora/dualgpm_python_loop"):
                        for ii in range(sval_ratio.shape[0]):
                            if accumulated_sval >= (1 - threshold):
                                accumulated_sval -= sval_ratio[ii]
                                r += 1
                            else:
                                break
                    if r == 0:
                        continue
                    with ce_region("inflora/update_dualgpm_linalg"):
                        act_feature = self.feature_list[i] - U[:, 0:r] @ (U[:, 0:r].t() @ self.feature_list[i])
                        Ui, Si, Vi = torch.linalg.svd(act_feature, full_matrices=True)
                        self.feature_list[i] = Ui[:, :self.feature_list[i].shape[1] - r]

        # convert any over-grown 'remove' layer to its 'retain' complement
        # *** UNTESTED as of 2026-08-03 *** -- same "update_dualgpm_linalg" tag:
        # an SVD-dominated pass, no per-iteration tensor->bool sync (the shape
        # comparisons here are plain Python ints, .shape[1] is not a tensor).
        with ce_region("inflora/update_dualgpm_linalg"):
            for i in range(len(self.feature_list)):
                if self.project_type[i] == 'remove' and \
                        (self.feature_list[i].shape[1] > (self.feature_list[i].shape[0] / 2)):
                    feature = self.feature_list[i]
                    U, S, V = torch.linalg.svd(feature, full_matrices=True)
                    self.feature_list[i] = U[:, feature.shape[1]:]
                    self.project_type[i] = 'retain'
                elif self.project_type[i] == 'retain':
                    assert self.feature_list[i].shape[1] <= (self.feature_list[i].shape[0] / 2)
                logging.info("[InfLoRA] Layer {}: {}/{} type {}".format(
                    i + 1, self.feature_list[i].shape[1], self.feature_list[i].shape[0], self.project_type[i]))
