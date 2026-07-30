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
        self._network.free_folded_slots(self._stream_chunk - 1)
        for p in self._network.fc.parameters():
            p.requires_grad = True
        self._network.to(self._device)
        self._init_lora_A(loader)                # analytic, DualGPM-projected A for this chunk
        self._stream_new_optimizer()

    def _stream_end_chunk(self, loader):
        self._update_dualgpm(loader)             # grow DualGPM feature memory

    def _ce_boundary_macs_this_cycle(self, chunk_images):
        # impl_plan_7.27.2026 sec 2.3: CONFIRMED two dedicated extra full passes
        # over the chunk's own data per cycle (_init_lora_A at begin, _update_
        # dualgpm at end, both above) -- not a per-step cost. Expected to
        # dominate InfLoRA's non-fwd/bwd overhead (~16% of one forward per
        # image, recurring every image of every cycle).
        from utils.ce_formulas import inflora_boundary_macs, inflora_dualgpm_svd_macs
        return {
            "covariance_hooks": inflora_boundary_macs(chunk_images),
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
        self._network.free_folded_slots(self._cur_task - 1)
        for p in self._network.fc.parameters():
            p.requires_grad = True

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                       shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                      shuffle=False, num_workers=num_workers)

        self._network.to(self._device)
        self._init_lora_A(self.train_loader)     # DualGPM-projected init of A
        self._log_trainable()
        self._train(self.train_loader)           # train B + head (inherited)
        self._update_dualgpm(self.train_loader)  # grow feature memory

    # -- (1)+(2) analytic init of the down-projection A -----------------
    @torch.no_grad()
    def _init_lora_A(self, train_loader):
        net = self._network
        net.eval()
        net.reset_cur_matrix()
        net.set_collect(True)
        for _, inputs, _t in train_loader:
            net(inputs.to(self._device), task=self._cur_task, merge=True)
        net.set_collect(False)

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
        for _, inputs, _t in train_loader:
            net(inputs.to(self._device), task=self._cur_task, merge=True)
        net.set_collect(False)

        mat_list = [attn.cur_matrix.clone() for attn in net.attn_modules()]
        for attn in net.attn_modules():
            attn.reset_cur_matrix()
        self.update_DualGPM(mat_list)

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
        if len(self.feature_list) == 0:
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
                    U1, S1, Vh1 = torch.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1 ** 2).sum()
                    act_hat = activation - self.feature_list[i] @ (self.feature_list[i].t() @ activation)
                    U, S, Vh = torch.linalg.svd(act_hat, full_matrices=False)
                    sval_hat = (S ** 2).sum()
                    sval_ratio = (S ** 2) / sval_total
                    accumulated_sval = (sval_total - sval_hat) / sval_total
                    r = 0
                    for ii in range(sval_ratio.shape[0]):
                        if accumulated_sval < threshold:
                            accumulated_sval += sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        continue
                    Ui = torch.cat([self.feature_list[i], U[:, 0:r]], dim=1)
                    self.feature_list[i] = Ui[:, 0:Ui.shape[0]] if Ui.shape[1] > Ui.shape[0] else Ui
                else:
                    assert self.project_type[i] == 'retain'
                    U1, S1, Vh1 = torch.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1 ** 2).sum()
                    act_hat = self.feature_list[i] @ (self.feature_list[i].t() @ activation)
                    U, S, Vh = torch.linalg.svd(act_hat, full_matrices=False)
                    sval_hat = (S ** 2).sum()
                    sval_ratio = (S ** 2) / sval_total
                    accumulated_sval = sval_hat / sval_total
                    r = 0
                    for ii in range(sval_ratio.shape[0]):
                        if accumulated_sval >= (1 - threshold):
                            accumulated_sval -= sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        continue
                    act_feature = self.feature_list[i] - U[:, 0:r] @ (U[:, 0:r].t() @ self.feature_list[i])
                    Ui, Si, Vi = torch.linalg.svd(act_feature, full_matrices=True)
                    self.feature_list[i] = Ui[:, :self.feature_list[i].shape[1] - r]

        # convert any over-grown 'remove' layer to its 'retain' complement
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
