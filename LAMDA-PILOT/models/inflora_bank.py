"""InfLoRA-Bank: InfLoRA with an EXPLICIT, GROWING per-task adapter bank --
built purely as a comparison baseline so InfLoRA-Sketch's memory-reduction
claim has something concrete (not hypothetical) to be measured against.

THREE prior InfLoRA designs existed in this codebase, in order:
  (1) original port: dense frozen_delta buffer, += task's B@A every task
      (O(dim^2) per block regardless of lora_rank).
  (2) (same as (1), superseded before this file existed).
  (3) current models/inflora.py (2026-11-03): folds each task's (A,B)
      directly into q_proj/v_proj's own weight, in place, then FREES that
      slot's memory -- costs ZERO additional persistent memory beyond the
      base model, but means there is no "bank" left to compress by the time
      InfLoRA-Sketch's own fold mechanism would otherwise apply.

This file adds a FOURTH design, matching InfLoRA-Ref's own actual mechanism
(InfLoRA-Ref/models/vit_inflora.py::Attention_LoRA.forward, InfLoRA-Ref/
methods/inflora.py): keep EVERY task's trained (A,B) pair allocated and
frozen FOREVER (``lora_A_q``/``lora_B_q``/.../``lora_B_v`` ModuleLists grow by
one slot per task and are never folded or freed), and reconstruct the
deployed function's dense delta EPHEMERALLY at every forward by summing all
allocated slots (``sum_{t=0}^{task} B_t(A_t(x))``) -- exactly the reference's
own ``torch.stack([... for t in range(task+1)]).sum(dim=0)``. Reference does
this even though InfLoRA's own algorithm has no architectural NEED for a
per-task bank (DualGPM's orthogonality guarantee only needs A frozen and the
running feature_list/feature_mat, not individual past B's kept separately);
it is simply how the reference happens to be written. This file reproduces
that choice deliberately so SketchLoRA's compression claim ("keeps an
otherwise-growing bank bounded") has a literal growing bank to point at for
InfLoRA, the same way it already does for LoRA-Subtraction
(models/lorasub.py's growing bank vs. models/lorasub_sketch.py).

Mathematically IDENTICAL to plain InfLoRA (models/inflora.py) in every other
respect -- same analytic A from DualGPM-projected input covariance, same B
training, same DualGPM growth, same loss. The ONLY difference is WHERE a
trained task's B lives afterward: forever in its own slot here, folded into
q_proj/v_proj (and then freed) there. This is NOT a new method for the
headline accuracy comparison -- it should score identically (up to
nondeterminism) to models/inflora.py on accuracy; it exists only to give
InfLoRA-Sketch a fair, literal "what would have grown" baseline.

model_name="inflora_bank". Backbone/config keys identical to plain InfLoRA
(backbone_type="vit_base_patch16_224_lora", lora_rank, lamb, lame, etc.) --
no new backbone module needed, since the shared backbone (backbone/
vit_lora.py) already supports exactly this via its generic merge=True
forward (``_lora_delta``'s non-folded branch, taken whenever
``_fold_enabled`` stays False -- never toggled on here, same as plain
InfLoRA never toggles it on either).

NOT adapted for streaming/boundary-mode use (see _stream_init below) --
InfLoRALearner's own streaming hooks assume the fold-into-base design this
variant deliberately does not use, so this variant is restricted to the
plain oracle per-task loop (incremental_train), which is the only context
the comparison this file exists for actually needs.
"""

import logging
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.inflora import Learner as InfLoRALearner
from utils.ce_profiler import ce_region, run_boundary
from utils.ce2_profiler import ce2_boundary
from utils.rank_diagnostics import log_bank_effective_rank

num_workers = 8


class Learner(InfLoRALearner):
    def __init__(self, args):
        super().__init__(args)
        # Undo InfLoRALearner's fold-into-base routing (merge=False there,
        # since q_proj/v_proj already carry every past task's contribution).
        # Here nothing is ever folded into the base weight, so forward must
        # sum over every allocated slot instead -- exactly the reference's
        # own mechanism, and exactly what backbone/vit_lora.py::_lora_delta's
        # merge=True branch already does generically once _fold_enabled
        # stays False (never toggled on by this class, same as InfLoRALearner
        # itself never toggles it on -- see that file's __init__ comment).
        self.train_merge = True
        self._network.merge = True
        # 2026-11-05 user request: summed-bank effective-rank diagnostic
        # (utils/rank_diagnostics.py) -- opt-in, see _log_bank_effective_rank.
        self.log_bank_rank = bool(args.get("log_bank_rank", False))
        self.bank_rank_energy_target = args.get("bank_rank_energy_target", 0.99)

    # -- streaming/boundary-mode hooks: deliberately NOT supported. ------
    # InfLoRALearner's own _stream_begin_chunk calls _fold_into_base +
    # free_folded_slots and its _init_lora_A/_update_dualgpm hardcode
    # merge=False -- all three assumptions this variant violates by design.
    # Failing fast here (rather than silently inheriting wrong behavior) --
    # this variant was built only for the oracle per-task comparison
    # (incremental_train below), which is the only thing the InfLoRA-vs-
    # InfLoRA-Sketch memory comparison actually needs.
    def _stream_init(self):
        raise NotImplementedError(
            "InfLoRA-Bank (model_name='inflora_bank') does not support streaming/"
            "boundary-mode training -- it was built only for the plain oracle "
            "per-task loop. InfLoRALearner's streaming hooks assume the fold-"
            "into-base design (merge=False, _fold_into_base, free_folded_slots) "
            "this variant deliberately replaces with an always-growing, never-"
            "folded bank, so inheriting them unmodified would silently train/eval "
            "against the wrong function rather than merely failing to compile.")

    # -- task loop: IDENTICAL to InfLoRA's own incremental_train except it
    # never calls _fold_into_base / free_folded_slots -- every past task's
    # (A,B) slot stays allocated, frozen (requires_grad=False), and intact
    # forever. task=self._cur_task / train_a=False routing is unchanged. --
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.data_manager = data_manager
        self._network.update_fc(self._total_classes)
        self._register_task_range()
        self._network.default_task = self._cur_task
        logging.info("[InfLoRA-Bank] Learning on {}-{}".format(self._known_classes, self._total_classes))

        if self._cur_task >= self._network.backbone.n_tasks:
            self._network.add_task_slot()

        # train_a=False: A stays analytic/frozen, same as plain InfLoRA.
        # freeze_to_task's own folding (fold_frozen_slots) is a permanent
        # no-op here too (_fold_enabled is never toggled on by this class or
        # its parent) -- this call is kept purely for its freeze/unfreeze
        # side effect, same reasoning as plain InfLoRA's own comment.
        self._network.freeze_to_task(self._cur_task, train_a=False)
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
        _ctrl = getattr(self, "_ce_boundary_ctrl", None)
        with ce2_boundary(self):
            run_boundary(_ctrl, "inflora_bank_init_a", lambda: self._init_lora_A_bank(self.train_loader))
        self._log_trainable()
        self._train(self.train_loader)           # train B + head (inherited)
        if self._bank_admit(0):
            with ce2_boundary(self):
                run_boundary(_ctrl, "inflora_bank_dualgpm",
                            lambda: self._update_dualgpm_bank(self.train_loader))
        self._log_memory_accounting()
        self._log_bank_effective_rank()

    # -- 2026-11-05 user request: plain InfLoRA, InfLoRA-Bank, and InfLoRA-
    # Sketch had NO per-task INFO-level memory log at all before this (unlike
    # models/lorasub.py's own _log_memory_accounting) -- persistent_state()
    # was only ever read by trainer.py's own metrics-JSON writer at eval
    # checkpoints, invisible in the live .out log. persistent_state() itself
    # was already correct (bank_bytes sums whatever's currently in each
    # ModuleList, which starts at 1 slot via lora_n_slots default=1 and grows
    # by exactly one slot per task via add_task_slot() -- never a large
    # fixed allocation; the backbone's OWN always-present buffers,
    # frozen_delta_q/v + cur_matrix, ~81MB total across 12 blocks, are
    # register_buffers, never nn.Parameters, and were never touched by this
    # method's byte-counting loop) -- this just makes that growth visible.
    def _log_memory_accounting(self):
        net = self._network.module if hasattr(self._network, "module") else self._network
        state = self.persistent_state()
        logging.info(
            "[InfLoRA-Bank] task {} persistent bank: {} slot(s) allocated (Q+V, A+B, every "
            "block) = {:,} bytes (adapter_bank={:,}, dualgpm_bases={:,}, fc={:,})".format(
                self._cur_task, net.backbone.n_tasks, state["bytes"],
                state["breakdown"]["adapter_bank"], state["breakdown"]["dualgpm_bases"],
                state["breakdown"]["fc"]))

    @torch.no_grad()
    def _log_bank_effective_rank(self):
        """2026-11-05 user request: the summed bank's TRUE combined rank
        (not each slot's own fixed lora_rank). Forms the dense delta =
        sum_{s=0}^{cur_task} B_s @ A_s per (block, q/v) -- the exact
        uncompressed update merge=True's forward applies -- then SVDs it for
        its effective rank. No-op unless log_bank_rank is set."""
        if not self.log_bank_rank:
            return
        net = self._network.module if hasattr(self._network, "module") else self._network
        t = self._cur_task
        deltas = {}
        for li, attn in enumerate(net.attn_modules()):
            for proj, (A_list, B_list) in (("q", (attn.lora_A_q, attn.lora_B_q)),
                                            ("v", (attn.lora_A_v, attn.lora_B_v))):
                B_cat = torch.cat([B_list[s].weight for s in range(t + 1)], dim=1).float()
                A_cat = torch.cat([A_list[s].weight for s in range(t + 1)], dim=0).float()
                deltas[(li, proj)] = B_cat @ A_cat
        log_bank_effective_rank("InfLoRA-Bank", t, deltas, self.bank_rank_energy_target)

    # -- analytic A: IDENTICAL algorithm to InfLoRA's own _init_lora_A, with
    # ONE change -- the covariance-collection forward uses merge=True, not
    # merge=False. Plain InfLoRA can safely use merge=False there because
    # q_proj/v_proj already carry every past task's folded contribution; this
    # variant never folds anything into the base weight, so merge=False
    # would see only the pretrained base and completely miss every past
    # task's effect on the input distribution feeding this covariance. The
    # current task's own (fresh kaiming-A/zero-B) slot still contributes
    # exactly zero regardless, same as plain InfLoRA. ---------------------
    @torch.no_grad()
    def _init_lora_A_bank(self, train_loader):
        net = self._network
        net.eval()
        net.reset_cur_matrix()
        net.set_collect(True)
        with ce_region("inflora_bank/init_lora_A_forward"):
            for _, inputs, _t in train_loader:
                net(inputs.to(self._device), task=self._cur_task, merge=True)
        net.set_collect(False)

        with ce_region("inflora_bank/init_lora_A_svd"):
            for kk, attn in enumerate(net.attn_modules()):
                cur = attn.cur_matrix.clone().double()
                if self._cur_task == 0 or kk >= len(self.feature_mat):
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
                A = (basis.t() / math.sqrt(3)).float().to(self._device)
                attn.lora_A_q[self._cur_task].weight.data.copy_(A)
                attn.lora_A_v[self._cur_task].weight.data.copy_(A)
                attn.reset_cur_matrix()

    # -- DualGPM growth: IDENTICAL to InfLoRA's own _update_dualgpm, same
    # merge=True change and same reasoning as _init_lora_A_bank's above. --
    @torch.no_grad()
    def _update_dualgpm_bank(self, train_loader):
        net = self._network
        net.eval()
        net.reset_cur_matrix()
        net.set_collect(True)
        with ce_region("inflora_bank/update_dualgpm_forward"):
            for _, inputs, _t in train_loader:
                net(inputs.to(self._device), task=self._cur_task, merge=True)
        net.set_collect(False)

        mat_list = [attn.cur_matrix.clone() for attn in net.attn_modules()]
        for attn in net.attn_modules():
            attn.reset_cur_matrix()
        self.update_DualGPM(mat_list)     # inherited, unchanged

        with ce_region("inflora_bank/feature_mat_rebuild"):
            self.feature_mat = []
            for p in range(len(self.feature_list)):
                Uf = self.feature_list[p] @ self.feature_list[p].t()
                self.feature_mat.append(Uf)

    # -- persistent memory: the WHOLE growing bank (every allocated task's
    # A+B, q+v, every block) -- NOT InfLoRALearner's fold-aware override
    # (which correctly counts only the current live slot THERE, since
    # everything else lives for free inside q_proj/v_proj). That override's
    # whole premise -- past tasks cost zero additional memory because
    # they're folded into pre-existing base weights -- does not hold for
    # this variant, so this restores the generic "raw allocation, never
    # freed" convention models/lora.py's own base persistent_state already
    # uses for the ordinary growing-bank case. --------------------------
    def persistent_state(self):
        net = self._network.module if hasattr(self._network, "module") else self._network
        bank_bytes = 0
        for attn in net.attn_modules():
            for mlist in (attn.lora_A_q, attn.lora_B_q, attn.lora_A_v, attn.lora_B_v):
                for m in mlist:
                    for p in m.parameters():
                        bank_bytes += p.numel() * p.element_size()
        dualgpm_bytes = 0
        for f in self.feature_list:
            dualgpm_bytes += f.numel() * f.element_size()
        for m in self.feature_mat:
            dualgpm_bytes += m.numel() * m.element_size()
        fc_bytes = sum(p.numel() * p.element_size() for p in net.fc.parameters()) if net.fc is not None else 0
        total_bytes = bank_bytes + dualgpm_bytes + fc_bytes
        return {"params": int(total_bytes // 4), "bytes": int(total_bytes),
                "breakdown": {"adapter_bank": bank_bytes, "dualgpm_bases": dualgpm_bytes, "fc": fc_bytes}}

    def _bank_bytes(self):
        """BankCapMixin hook: every FROZEN past task's slot (0..cur_task-1 --
        excludes the live/currently-training slot, same "never count the
        live slot" convention every other method in this project uses) +
        DualGPM bases. This is the structure a bank_cap_mb would actually
        need to cap for this variant (unlike InfLoRALearner's own
        _bank_bytes, which only has DualGPM to cap since the adapter bank
        itself costs nothing extra there)."""
        net = self._network.module if hasattr(self._network, "module") else self._network
        bank_bytes = 0
        for attn in net.attn_modules():
            for mlist in (attn.lora_A_q, attn.lora_B_q, attn.lora_A_v, attn.lora_B_v):
                for slot_idx in range(self._cur_task):
                    for p in mlist[slot_idx].parameters():
                        bank_bytes += p.numel() * p.element_size()
        dualgpm_bytes = 0
        for f in self.feature_list:
            dualgpm_bytes += f.numel() * f.element_size()
        for m in self.feature_mat:
            dualgpm_bytes += m.numel() * m.element_size()
        return bank_bytes + dualgpm_bytes
