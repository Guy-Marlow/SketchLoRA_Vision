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
# *** UNTESTED as of 2026-08-03 *** -- measured-CE region tagging
# (docs/ce_profiling_implementation_plan.md sec 4.4). No-op unless a profiling
# session is active (utils/ce_profiler.py).
from utils.ce_profiler import ce_region


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
        # *** UNTESTED as of 2026-08-03 *** -- plan sec 4.4 "stacked_A_build":
        # torch.stack of 24 reshaped [dim*rank] rows, called EVERY training
        # step -- previously UNCOUNTED as its own line item (folded into
        # treelora_aux_macs_per_step's flat constant).
        with ce_region("treelora/stacked_A_build"):
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

    # *** UNTESTED as of 2026-08-03 *** -- local GPUs were unavailable
    # (thermal/damage risk) at fix time, so this has NOT been exercised on a
    # live run. Verified only by static tracing: confirmed attn.frozen_delta_q/
    # frozen_delta_v are unconditionally registered buffers (backbone/
    # vit_lora.py, always exist regardless of enable_frozen_folding()), and
    # confirmed self._cur_task correctly points to the live, not-yet-folded
    # slot at read time by tracing freeze_to_task()'s fold_frozen_slots(task-1)
    # call (folds slot t-1 when STARTING slot t, so slot t itself stays
    # unfolded for its entire lifetime as "current") -- the same invariant
    # InfLoRA's own persistent_state() already relies on in production. No
    # live run has confirmed this executes without error or matches the
    # ~399MB estimate computed by hand below. Confirm on the first real run
    # before trusting its memory numbers.
    #
    # FIXED 2026-08-03 (found during a persistent-memory audit prompted by an
    # anomalously large reported figure -- 1026MB at 50MB budget, LARGER than
    # O-LoRA's own unbounded bank). The OLD override just added tree_grad_store
    # on top of super().persistent_state() (models/lora.py's generic "every
    # slot still in the ModuleList" accounting), which has the exact same two
    # problems InfLoRA's own persistent_state() was written to fix back on
    # 2026-07-21 (see models/inflora.py's docstring): (a) it counts every
    # historical per-task lora_A/lora_B slot as still live, even after
    # enable_frozen_folding() has folded them into frozen_delta_q/v, and (b) it
    # never reads frozen_delta_q/v at all, since those are register_buffer, not
    # nn.Parameter. Verified via utils/kd_tree.py that TreeLoRA's regularizer
    # (tree_search/get_loss/_update_similarity) ONLY ever reads
    # self.tree.all_accumulate_grads -- its own separate per-task snapshot
    # store -- and NEVER reads back a folded task's live adapter weights, so
    # (unlike O-LoRA, whose orthogonality penalty genuinely needs every past
    # lora_A forever) there is no structural reason old TreeLoRA slots need to
    # stay counted as persistent once folded. This override now matches
    # InfLoRA's exact convention: frozen_delta (fixed, O(d^2) per block) +
    # the current (not-yet-folded) task's own live slot + the tree's own
    # gradient-snapshot store (tree_grad_store, TreeLoRA's genuine O(cycles)
    # cost) + head. Measured effect on one real checkpoint (50MB, final):
    # reported figure drops from 1026MB to ~399MB -- the removed ~682MB was
    # entirely redundant, already-folded dead weight; NOTE this is an
    # ACCOUNTING fix only, not a memory-reclamation fix -- old slots are still
    # allocated on the GPU (free_folded_slots() is not called here, mirroring
    # what old runs' actual behavior was), so this override describes what
    # SHOULD be counted as "persistent," not a change to what's allocated.
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
        grad_bytes = sum(g.numel() * 4 for g in self.tree.all_accumulate_grads if g is not None)
        fc_bytes = sum(p.numel() * p.element_size() for p in net.fc.parameters()) if net.fc is not None else 0
        total_bytes = frozen_delta_bytes + cur_slot_bytes + grad_bytes + fc_bytes
        return {"params": int(total_bytes // 4), "bytes": int(total_bytes),
                "breakdown": {"frozen_delta": frozen_delta_bytes, "current_slot": cur_slot_bytes,
                             "tree_grad_store": grad_bytes, "fc": fc_bytes}}

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

    def _ce_aux_macs_per_step(self):
        # impl_plan_7.27.2026 sec 2.3: per-step sparse-update regularizer +
        # gradient-similarity estimate, r*d-order per module (only charged
        # when reg>0, matching the actual code path in _bounded_train_epoch).
        if self.reg <= 0:
            return 0.0
        from utils.ce_formulas import treelora_aux_macs_per_step
        net = self._network.module if hasattr(self._network, "module") else self._network
        rank = net.attn_modules()[0].rank
        return treelora_aux_macs_per_step(rank=rank)

    def _bounded_train_epoch(self, loader, optimizer, scheduler, cycle_class_mask):
        """bounded_memory_mixin.py's own driver never calls _stream_train_epoch
        below (that hook only exists on the stream_run path, models/stream_mixin.py) --
        it inlines a generic loop (BoundedMemoryMixin._bounded_train_epoch) that only
        offers the narrow _stream_extra_loss(lo, hi) hook, which cannot carry the tree
        regularizer (needs new_epoch_init/step/insert_grad/tree_search/get_loss and the
        raw pre-penalty loss value -- see _stream_train_epoch's own docstring below for
        why _stream_extra_loss's signature is insufficient). Without this override,
        self.tree.current_grad is never populated by insert_grad, so end_task's
        `self.current_grad.shape[0]` crashes with AttributeError on the very first
        cycle. Full override, mirroring the base class's masked-CE convention (additive
        -inf mask over the full head, not lo:hi slicing) instead of stream_run's slice
        convention -- the only other difference from _stream_train_epoch below."""
        self._network.train()
        slot, merge = self._stream_slot(), self._stream_train_merge()
        t = self._stream_chunk
        if self.reg > 0:
            self.tree.new_epoch_init(len(loader))
        for _, inputs, targets in loader:
            if self.reg > 0:
                self.tree.step()
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            logits = self._network(inputs, task=slot, merge=merge)["logits"]
            masked_logits = logits + cycle_class_mask
            loss = F.cross_entropy(masked_logits, targets)
            if self.reg > 0:
                grad_current = self._stacked_A()
                self.tree.insert_grad(grad_current)
                if t > 0:
                    prev_id_matrix = self.tree.tree_search(t, self._device)
                    reg_loss = self.tree.get_loss(grad_current, loss, prev_id_matrix)
                    loss = loss - reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

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
