"""TreeLoRA for LAMDA-PILOT: per-task LoRA (O-LoRA-style scaffold -- train_merge=True,
one slot per task, merged CIL/TIL forward) with O-LoRA's orthogonality penalty
REPLACED by a hierarchical gradient-similarity-tree regularizer. Port of
TreeLoRA/model/Regular/Tree_LoRA.py (train_one_task) +
TreeLoRA/utils/kd_lora_tree.py (see utils/kd_tree.py, ported near-verbatim) onto our
shared q/v scaffold. `reg=0.5` confirmed from the reference's actual launch script
(TreeLoRA/scripts/lora_based_methods/Tree_LoRA.sh), not a config default -- unlike
HiDeLoRA, TreeLoRA's train_one_task does NOT warm-start a new task's adapter from the
previous one (verified: no A/B copy anywhere in the reference's per-task loop).

Depth axis ("lora_depth") = our 24 wrapped LoRA-A projections (12 blocks x {q,v}).
Each training step collects the CURRENT TASK's trainable A parameter VALUES (not
literal .grad -- faithful to what the reference's insert_grad actually consumes,
despite calling it "grad") as one row per module, running-averaged across the
task's steps into a single per-task snapshot used for the tree/bandit machinery.
"""

import logging

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm

from models.lora import Learner as LoRALearner
from utils.kd_tree import KD_LoRA_Tree
from utils.toolkit import tensor2numpy


class Learner(LoRALearner):
    def __init__(self, args):
        super().__init__(args)
        # TreeLoRA's frozen slots are never modified once trained (only ever read,
        # for the tree-gradient-similarity regularizer) -- safe to fold into a
        # dense delta for O(1) merged forward (plan doc §6 item 2).
        self._network.enable_frozen_folding()
        self.reg = args.get("reg", 0.5)
        self.train_merge = True   # accumulated forward (sum 0..t), same as O-LoRA
        self.tree = KD_LoRA_Tree(num_tasks=args["nb_tasks"], reg=self.reg)

    # -- TIL eval must use the merged adapter (same fairness fix as O-LoRA/InfLoRA:
    # training forward is merge=True, so TIL should evaluate that same state) ----
    def _forward_task(self, inputs, task):
        net = self._network.module if hasattr(self._network, "module") else self._network
        return net(inputs, task=self._cur_task, merge=True)

    def _stacked_A(self):
        """Current task's trainable A (down-proj), one row per wrapped module,
        flattened -- matches the reference's `loranew_A` parameter collection."""
        net = self._network.module if hasattr(self._network, "module") else self._network
        t = self._cur_task
        rows = []
        for attn in net.attn_modules():
            rows.append(attn.lora_A_q[t].weight.reshape(-1))
            rows.append(attn.lora_A_v[t].weight.reshape(-1))
        return torch.stack(rows)   # [lora_depth, dim*rank]

    def _train(self, train_loader):
        self._network.to(self._device)
        params = [p for p in self._network.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.min_lr) if self.lr_anneal else None

        t = self._cur_task
        lo, hi = self._ce_slice()
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            self.tree.new_epoch_init(len(train_loader))
            losses, reg_run, correct, total = 0.0, 0.0, 0, 0
            for _, inputs, targets in train_loader:
                if self.reg > 0:
                    self.tree.step()
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs, task=t, merge=True)["logits"]
                local_logits = logits[:, lo:hi]
                local_targets = targets - lo
                loss = F.cross_entropy(local_logits, local_targets)

                if self.reg > 0:
                    grad_current = self._stacked_A()
                    self.tree.insert_grad(grad_current)
                    if t > 0:
                        prev_id_matrix = self.tree.tree_search(t, self._device)
                        reg_loss = self.tree.get_loss(grad_current, loss, prev_id_matrix)
                        loss = loss - reg_loss
                        reg_run += float(reg_loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                preds = local_logits.argmax(dim=1)
                correct += preds.eq(local_targets).cpu().sum()
                total += len(targets)
            if scheduler is not None:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            prog_bar.set_description(
                "[TreeLoRA] Task {}, Epoch {}/{} => Loss {:.3f} (reg {:.3f}), Acc {:.2f}".format(
                    t, epoch + 1, self.epochs, losses / len(train_loader),
                    reg_run / len(train_loader), train_acc))
        logging.info("[TreeLoRA] Task {} done. Acc {:.2f}".format(t, train_acc))
        if self.reg > 0:
            self.tree.end_task(t)

    def persistent_state(self):
        base = super().persistent_state()
        grad_bytes = sum(g.numel() * 4 for g in self.tree.all_accumulate_grads if g is not None)
        return {"params": base["params"], "bytes": base["bytes"] + grad_bytes,
                "breakdown": {**base["breakdown"], "tree_grad_store": grad_bytes}}

    # ==================================================================
    # Boundary-agnostic streaming hooks (models/stream_mixin.py). Adapter slot
    # = CHUNK index (self._stream_chunk), same convention as O-LoRA/InfLoRA --
    # one slot per adapter-boundary event, decoupled from real task boundaries.
    # CORRECTED (was wrong): the fold count is driven by a memory-constraint
    # sample threshold, not real task count, so it is NOT generically bounded by
    # nb_tasks -- confirmed by direct crashes on other methods sharing this same
    # backbone (see BOUNDARY_AGNOSTIC_IMPLEMENTATION_LOG.md's "BLOCKING
    # ARCHITECTURAL GAP" section). Both the shared LoRA scaffold's adapter slots
    # (backbone/vit_lora.py::add_task_slot) and KD_LoRA_Tree's own
    # `all_accumulate_grads`/`num_of_selected` (utils/kd_tree.py) now grow on
    # demand instead of assuming the nb_tasks preallocation is an upper bound.
    # ==================================================================
    def _stream_init(self):
        self._stream_chunk = -1

    def _stream_slot(self):
        return self._stream_chunk

    def _stream_begin_chunk(self, loader):
        self._stream_chunk += 1
        if self._stream_chunk > 0:
            self._network.add_task_slot()
        self._cur_task = self._stream_chunk   # _stacked_A/_forward_task read _cur_task as slot
        self._network.freeze_to_task(self._stream_chunk, train_a=True)
        for p in self._network.fc.parameters():
            p.requires_grad = True
        self._stream_new_optimizer()

    def _stream_end_chunk(self, loader):
        if self.reg > 0:
            self.tree.end_task(self._stream_chunk)

    def _stream_train_epoch(self, loader, lo, hi):
        """Full override (not just _stream_extra_loss) -- TreeLoRA's regularizer
        needs the raw CE loss VALUE to scale itself (kd_tree.py::get_loss does
        `reg_loss * loss.detach()`), which the generic _stream_extra_loss(lo, hi)
        hook has no way to supply. Mirrors _train's per-batch loop exactly,
        against self._stream_optim/_stream_sched instead of a locally-built one."""
        self._network.train()
        t = self._stream_chunk
        if self.reg > 0:
            self.tree.new_epoch_init(len(loader))
        for _, inputs, targets in loader:
            if self.reg > 0:
                self.tree.step()
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            logits = self._network(inputs, task=t, merge=True)["logits"]
            local_logits = logits[:, lo:hi]
            local_targets = targets - lo
            loss = F.cross_entropy(local_logits, local_targets)
            if self.reg > 0:
                grad_current = self._stacked_A()
                self.tree.insert_grad(grad_current)
                if t > 0:
                    prev_id_matrix = self.tree.tree_search(t, self._device)
                    reg_loss = self.tree.get_loss(grad_current, loss, prev_id_matrix)
                    loss = loss - reg_loss
            self._stream_optim.zero_grad()
            loss.backward()
            self._stream_optim.step()
        if self._stream_sched is not None:
            self._stream_sched.step()
