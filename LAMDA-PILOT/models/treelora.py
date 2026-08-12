"""TreeLoRA for LAMDA-PILOT: a SINGLE, continuously-fine-tuned rank-r LoRA
adapter (pinned to slot 0, never frozen, never replaced) regularized by a
hierarchical gradient-similarity tree + UCB bandit that nudges each step's
update toward whichever past task's adapter snapshot looks most related.
Port of TreeLoRA/model/Regular/Tree_LoRA.py (train_one_task) +
TreeLoRA/utils/kd_lora_tree.py (see utils/kd_tree.py, ported near-verbatim).
`reg=0.5` confirmed from the reference's actual launch script
(TreeLoRA/scripts/lora_based_methods/Tree_LoRA.sh), not a config default --
unlike HiDeLoRA, TreeLoRA's train_one_task does NOT warm-start a new task's
adapter from the previous one (verified: no A/B copy anywhere in the
reference's per-task loop).

Depth axis ("lora_depth") = our 24 wrapped LoRA-A projections (12 blocks x {q,v}).
Each training step collects the live adapter's A parameter VALUES (not literal
.grad -- faithful to what the reference's insert_grad actually consumes,
despite calling it "grad") as one row per module, running-averaged across the
task's steps into a single per-task snapshot used for the tree/bandit machinery.

2026-08-11: CORRECTED to match the reference's actual architecture, after a
2026-08-10 investigation (see the ce_profiling_methodology memory) found this
port had inherited a per-task adapter BANK from O-LoRA's scaffold (one new
LoRA slot per task, all summed via merge=True at forward time) that has NO
counterpart in the published algorithm. Verified directly against
TreeLoRA-Ref/utils/my_peft/tuners/lora.py: the reference's LoRA layer has two
adapter pairs, `loranew_A/B` (the live, trainable rank-r adapter) and
`lora_A/B` (a "previous-tasks" bank sized `r_sum`) -- and
TreeLoRA-Ref/model/Regular/Tree_LoRA.py FORCES `r_sum=0` at every save
("This is the key point to be compatible with O_LoRA!!!"), making `lora_A/B`
permanently empty. No `.merge()`/`merge_adapter()` call appears anywhere in
the reference's training loop either. Net effect: `loranew_A/B` is one
fixed-size adapter, continuously fine-tuned from task 0 through the last task
with NO growth in parameters, forward/backward FLOPs, or memory as tasks
accumulate -- exactly SeqLoRA's scaffold (pin slot 0, never freeze it, never
add a slot), NOT O-LoRA's. Fixed by adopting SeqLoRA's `_train_adapter`/
`_eval_adapter` pattern (see models/seqlora.py) and `train_merge=False`; the
tree/UCB regularizer itself is UNCHANGED, still the sole continual-learning
mechanism, still genuinely reads/writes real per-task state (its own
`all_accumulate_grads` snapshot store and rebuilt KD-tree, which DO grow with
task count -- that part was always faithful and stays exactly as before).
TIL evaluation is no longer performed by this project at all, so no
TIL-specific routing is needed here anymore (the old `_forward_task` override
that forced merge=True for TIL fairness is removed, since there is no merge
concept left to be unfair about).

This is a real, first-order behavioral change, not a cleanup: the old port
had model capacity the published method was never designed to have (a
permanent per-task memory bank on top of the regularizer), so its
accuracy numbers do not reflect the actual algorithm and should not be
compared against pre-fix results.

Does NOT use frozen-slot dense folding (see backbone/vit_lora.py's
``enable_frozen_folding``) -- moot now (a single never-frozen slot has
nothing to fold), but left un-called for the same reason SeqLoRA never calls
it either.
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
from utils.ce_profiler import ce_region, run_boundary, run_step_narrow
from utils.ce2_profiler import ce2_boundary


class Learner(LoRALearner):
    def __init__(self, args):
        super().__init__(args)
        # 2026-08-10: NOT calling enable_frozen_folding() -- see module
        # docstring. TreeLoRA-Ref never folds (it has no per-task bank at
        # all), and this port's fold was >99.99% of its measured CE overhead
        # with no algorithmic counterpart. _lora_delta (backbone/vit_lora.py)
        # falls through to its factored/non-fold loop automatically whenever
        # _fold_enabled stays False (the class default) -- no other code
        # change needed for this to take effect.
        self.reg = args.get("reg", 0.5)
        # single continuously-trained adapter -> nothing to merge/accumulate,
        # same as SeqLoRA (see module docstring for why this, not O-LoRA's
        # accumulated-forward convention, is what matches the reference).
        self.train_merge = False
        self._network.merge = False
        self.tree = KD_LoRA_Tree(num_tasks=args["nb_tasks"], reg=self.reg)

    # -- single adapter, pinned to slot 0 for both training and (former TIL)
    # evaluation -- identical pattern to models/seqlora.py. `_cur_task` still
    # advances normally and is used BELOW as the tree regularizer's own task
    # identity (which snapshot row to write/read) -- that is bookkeeping for
    # the regularizer, not adapter routing, and the two must not be conflated.
    def _train_adapter(self):
        return 0

    def _eval_adapter(self, task):
        return 0

    def _stacked_A(self):
        """Live adapter's trainable A (down-proj), one row per wrapped module,
        flattened -- matches the reference's `loranew_A` parameter collection."""
        # *** UNTESTED as of 2026-08-03 *** -- plan sec 4.4 "stacked_A_build":
        # torch.stack of 24 reshaped [dim*rank] rows, called EVERY training
        # step -- previously UNCOUNTED as its own line item (folded into
        # treelora_aux_macs_per_step's flat constant).
        with ce_region("treelora/stacked_A_build"):
            net = self._network.module if hasattr(self._network, "module") else self._network
            t = self._train_adapter()   # always 0 -- the single live slot, NOT self._cur_task
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

        t = self._cur_task   # tree regularizer's task identity ONLY -- adapter routing is always slot 0
        lo, hi = self._ce_slice()
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            self.tree.new_epoch_init(len(train_loader))
            # Step-type measurement (docs/ce_step_boundary_isolation_plan.md sec
            # 1a/2/7): only wrap the isolated tree-regularizer block below (not the
            # surrounding forward/backward/optimizer.step()), only on epoch 0. The
            # SAME epoch-0 harvest also captures tree_search's per-epoch
            # `all_grad` rebuild (tagged "treelora/per_epoch/...", see
            # utils/kd_tree.py) alongside the genuinely per-step tags -- both are
            # accumulated together here and split apart downstream by
            # utils.ce_profiler.split_by_recurrence, not here.
            step_acc = getattr(self, "_ce_step_acc", None) if epoch == 0 else None
            losses, reg_run, correct, total = 0.0, 0.0, 0, 0
            for _, inputs, targets in train_loader:
                if self.reg > 0:
                    self.tree.step()
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs, task=self._train_adapter(), merge=self.train_merge)["logits"]
                local_logits = logits[:, lo:hi]
                local_targets = targets - lo
                loss = F.cross_entropy(local_logits, local_targets)

                if self.reg > 0:
                    def _tree_regularizer(_loss=loss):
                        grad_current = self._stacked_A()
                        self.tree.insert_grad(grad_current)
                        if t > 0:
                            prev_id_matrix = self.tree.tree_search(t, self._device)
                            return self.tree.get_loss(grad_current, _loss, prev_id_matrix)
                        return None
                    reg_loss = run_step_narrow(step_acc, "treelora_step", _tree_regularizer)
                    if reg_loss is not None:
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
            # oracle-mode boundary bookkeeping (docs/ce_step_boundary_isolation_
            # plan.md sec 7): end_task's tree-build genuinely grows with task
            # count (see utils/kd_tree.py) -- wrap it so that growth is visible
            # per-task, not averaged away. Under bounded_memory streaming this
            # call site isn't used (_stream_end_chunk, below, is already wrapped
            # end-to-end by the driver's own boundary_end session).
            with ce2_boundary(self):
                run_boundary(getattr(self, "_ce_boundary_ctrl", None), "boundary",
                             lambda: self.tree.end_task(t))

    # 2026-08-11: base off models/lora.py's generic persistent_state() -- with
    # only ever ONE slot allocated now (see module docstring), that already
    # correctly reports just slot 0's A+B+head, no override needed for that
    # part. TreeLoRA's own genuine extra state is the tree's gradient-snapshot
    # store (tree_grad_store, real O(tasks-seen) cost, has no analog in the
    # generic base class) -- that's the only thing added here.
    def persistent_state(self):
        base = super().persistent_state()
        grad_bytes = sum(g.numel() * 4 for g in self.tree.all_accumulate_grads if g is not None)
        total_bytes = base["bytes"] + grad_bytes
        breakdown = dict(base["breakdown"])
        breakdown["tree_grad_store"] = grad_bytes
        return {"params": int(total_bytes // 4), "bytes": int(total_bytes), "breakdown": breakdown}

    # ==================================================================
    # Boundary-agnostic streaming hooks (models/stream_mixin.py). Adapter slot
    # is pinned to 0 for the whole stream (StreamMixin's own default
    # _stream_slot already returns 0 -- no override needed here, same as
    # SeqLoRA). `self._stream_chunk` still advances every chunk and still
    # feeds the tree regularizer's own task identity (which snapshot row to
    # write/read) and KD_LoRA_Tree's `all_accumulate_grads`/`num_of_selected`
    # (utils/kd_tree.py), which genuinely grow on demand with chunk count --
    # that part of the design is unchanged and unaffected by this fix.
    # ==================================================================
    def _stream_init(self):
        self._stream_chunk = -1

    def _stream_begin_chunk(self, loader):
        self._stream_chunk += 1
        self._cur_task = self._stream_chunk   # tree task identity ONLY -- adapter routing is always slot 0
        self._network.freeze_to_task(0, train_a=True)   # single continuously-trained adapter -- never add a slot
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

    def _bounded_train_epoch(self, loader, optimizer, scheduler, cycle_class_mask, step_acc=None):
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
                # Step-type measurement (docs/ce_step_boundary_isolation_plan.md
                # sec 1a/2/7/8): same narrow-wrap-the-isolated-block pattern as
                # the oracle _train() override above -- step_acc is non-None
                # only during epoch 0 of a profiled cycle (see
                # bounded_memory_mixin.py's driver).
                def _tree_regularizer(_loss=loss):
                    grad_current = self._stacked_A()
                    self.tree.insert_grad(grad_current)
                    if t > 0:
                        prev_id_matrix = self.tree.tree_search(t, self._device)
                        return self.tree.get_loss(grad_current, _loss, prev_id_matrix)
                    return None
                reg_loss = run_step_narrow(step_acc, "treelora_step", _tree_regularizer)
                if reg_loss is not None:
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
        t = self._stream_chunk   # tree task identity ONLY -- adapter routing is always slot 0
        if self.reg > 0:
            self.tree.new_epoch_init(len(loader))
        for _, inputs, targets in loader:
            if self.reg > 0:
                self.tree.step()
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            logits = self._network(inputs, task=self._stream_slot(), merge=self._stream_train_merge())["logits"]
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
