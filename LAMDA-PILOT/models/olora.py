"""O-LoRA (Orthogonal subspace LoRA) for LAMDA-PILOT.

Port of the regularizer from the original O-LoRA repo
(svd_sketching_vision/O-LoRA/src/uie_trainer_lora.py): in addition to the
task-local cross-entropy, the current task's LoRA-A directions are pushed to be
orthogonal to every previous task's (frozen) LoRA-A, and the current LoRA is
L2-regularized:

    loss = CE  +  lambda_1 * sum_layers sum_{s<t} | A_t @ A_s^T |.sum()
               +  lambda_2 * || current-task LoRA ||_2

Here A is applied to the query and value projections (our shared convention).
Inference is the merged sum of LoRAs 0..t (CIL) or task-routed (TIL), reusing
the baseline ``LoRAVitNet`` and the TIL eval in ``models/til_base.py``.
"""

import logging
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm

from models.lora import Learner as LoRALearner
from utils.toolkit import tensor2numpy
# *** UNTESTED as of 2026-08-03 *** -- measured-CE region tagging
# (docs/ce_profiling_implementation_plan.md sec 4.2). No-op unless a profiling
# session is active (utils/ce_profiler.py).
from utils.ce_profiler import ce_region


class Learner(LoRALearner):
    def __init__(self, args):
        super().__init__(args)
        # O-LoRA's frozen slots are never modified once trained -- safe to fold
        # into a dense delta for O(1) merged forward (plan doc §6 item 2).
        self._network.enable_frozen_folding()
        self.lamda_1 = args.get("lamda_1", 0.5)   # orthogonality weight
        self.lamda_2 = args.get("lamda_2", 0.0)   # L2 weight on current LoRA
        # O-LoRA accumulates: the forward sums previous (frozen) + current LoRA.
        self.train_merge = True

    # -- TIL eval must use O-LoRA's *merged* adapter ---------------------
    # O-LoRA's inference model is the merged sum (W + Σ_{k≤cur} B_k A_k) -- the
    # same adapter its CIL uses and the one each task's head was trained against
    # (training forward is merge=True).  The inherited lora.Learner._forward_task
    # routes TIL to slot t *alone* (merge=False), which mismatches training and
    # unfairly depresses TIL.  Override to evaluate the merged adapter; _eval_til
    # still masks logits to task t's class slice.
    def _forward_task(self, inputs, task):
        net = self._network.module if hasattr(self._network, "module") else self._network
        return net(inputs, task=self._cur_task, merge=True)

    # -- sample-boundary streaming hooks --------------------------------
    # One adapter slot per CHUNK (not per class-task); the orthogonality penalty
    # snapshots/orthogonalises against the previous CHUNK adapters (slots < chunk).
    # Because chunks straddle real class-groups, those snapshots are taken over
    # class-mixed data -- the "messy subspace" regime under test.
    def _stream_init(self):
        self._stream_chunk = -1

    def _stream_slot(self):
        return self._stream_chunk

    def _stream_begin_chunk(self, loader):
        self._stream_chunk += 1
        if self._stream_chunk > 0:
            # slot count is not generically bounded by nb_tasks under this clock --
            # see BOUNDARY_AGNOSTIC_IMPLEMENTATION_LOG.md's "BLOCKING ARCHITECTURAL GAP"
            self._network.add_task_slot()
        self._cur_task = self._stream_chunk        # _orth_and_l2 reads _cur_task as current slot
        self._network.freeze_to_task(self._stream_chunk, train_a=True)
        for p in self._network.fc.parameters():
            p.requires_grad = True
        self._stream_new_optimizer()

    def _stream_extra_loss(self, lo, hi):
        orth, l2 = self._orth_and_l2()
        return self.lamda_1 * orth + self.lamda_2 * l2

    def _ce_aux_macs_per_step(self):
        # impl_plan_7.27.2026 sec 2.3: current-A x each frozen prev-A^T, r^2*d
        # MACs per slot-pair, TIMES slot count at this cycle, plus backward (~2x).
        #
        # *** UNTESTED as of 2026-08-03 *** -- FIXED off-by-one (docs/
        # ce_profiling_implementation_plan.md sec 4.2/6.2, found during the
        # measured-CE audit). The PREVIOUS formula used
        # `slot_count = self._stream_chunk + 1` ("current slot count including
        # the current task"), but _orth_and_l2's actual loop below is
        # `for s in range(t)` with `t = self._cur_task == self._stream_chunk`
        # (set equal in _stream_begin_chunk) -- range(t) iterates over exactly
        # `t` PREVIOUS slots, excluding the current one, so the real slot count
        # the penalty pays for is `self._stream_chunk`, not `+ 1`. The bug
        # overcounted by exactly one slot-pair's cost at EVERY cycle -- most
        # visible at cycle 0 (t=0: the `if t > 0:` guard below means orth is
        # EXACTLY zero, since there is no previous slot to compare against, yet
        # the old formula charged slot_count=1 anyway).
        from utils.ce_formulas import olora_aux_macs_per_step
        slot_count = self._stream_chunk
        net = self._network.module if hasattr(self._network, "module") else self._network
        rank = net.attn_modules()[0].rank
        return olora_aux_macs_per_step(slot_count, rank=rank)

    def _orth_and_l2(self):
        """Orthogonality penalty (current vs all previous tasks) + L2 on the
        current task's LoRA, summed over all attention blocks.

        Vectorised (mirrors svd_sketching_language/tokmem/atomic/multislot_lora.py::
        orthogonality_penalty): the frozen A_{<t} for a given (block, projection) are
        stacked into one [t*r, dim] matrix once per task -- cached on the attention
        module, invalidated when _cur_task advances -- so the cross-term is a SINGLE
        matmul+abs+sum instead of a per-s loop. Numerically identical: for the
        concatenation Y = [Y_0; Y_1; ...; Y_{t-1}] (stacked along dim 0), X @ Y^T
        stacks the individual X @ Y_s^T blocks along dim 1 (columns), and abs() is
        elementwise, so |X @ Y^T|.sum() == sum_s |X @ Y_s^T|.sum() exactly. Dominant
        win in budget mode (up to 475 tiny matmuls/step collapses to 1)."""
        t = self._cur_task
        orth = 0.0
        l2 = 0.0
        for attn in self._network.attn_modules():
            for proj, (A_list, B_list) in (("q", (attn.lora_A_q, attn.lora_B_q)),
                                           ("v", (attn.lora_A_v, attn.lora_B_v))):
                A_t = A_list[t].weight                       # [r, dim]
                # *** UNTESTED as of 2026-08-03 *** -- plan sec 4.2: previously
                # UNCOUNTED (small on its own, but real -- two torch.norm calls
                # per block-projection, every step).
                with ce_region("olora/orth_l2_norms"):
                    l2 = l2 + torch.norm(A_t, p=2) + torch.norm(B_list[t].weight, p=2)
                if t > 0:
                    cache_attr, task_attr = "_orth_prev_" + proj, "_orth_prev_task_" + proj
                    if getattr(attn, task_attr, None) != t:
                        # *** UNTESTED as of 2026-08-03 *** -- plan sec 4.2
                        # "orth_prev_cache_rebuild": previously UNCOUNTED entirely.
                        # At t=484 this concatenates a [4840,768] matrix per
                        # (block,proj) -- 24 pairs total (12 blocks x {q,v}) --
                        # ~357MB of copies (CORRECTED 2026-08-03: the plan
                        # document's original "~714MB" figure was off by ~2x;
                        # recomputed by hand as t*rank*dim*4 bytes *
                        # n_blocks*n_proj = 484*10*768*4*24 = 356,843,520 bytes).
                        # Bandwidth-bound, invisible to MACs (R5). IMPORTANT
                        # CAVEAT, not silently resolved: this
                        # rebuild fires ONCE PER CYCLE (guarded by task_attr,
                        # which only changes when _cur_task advances between
                        # cycles -- NOT once per epoch, unlike TreeLoRA's
                        # analogous per-epoch cache in tree_search). The driver
                        # (models/bounded_memory_mixin.py) profiles epoch 0 of a
                        # sampled cycle and scales by 1/steps_per_epoch, then
                        # Ops_total's formula multiplies back by
                        # n_epochs*steps_per_epoch -- correct for a cost that
                        # truly recurs every epoch, but this one does NOT: it
                        # fires on epoch 0's first step and never again for
                        # epochs 1..19 of the same cycle. Piped through the
                        # standard per-step pipeline, this region's contribution
                        # to Ops_total would be overstated by roughly a factor of
                        # n_epochs (~20x). Flagged rather than silently
                        # "corrected" by an invented scaling rule -- whether to
                        # special-case this region (e.g. a third ledger category
                        # for "once per cycle, not once per epoch") or accept the
                        # overstatement as negligible in absolute terms is an
                        # open question for whoever validates this against a
                        # live run, not resolved here.
                        with ce_region("olora/orth_prev_cache_rebuild"):
                            stacked = torch.cat([A_list[s].weight.detach() for s in range(t)], dim=0)
                        setattr(attn, cache_attr, stacked)
                        setattr(attn, task_attr, t)
                    A_prev = getattr(attn, cache_attr)       # [(t*r), dim] (frozen)
                    # *** UNTESTED as of 2026-08-03 *** -- plan sec 4.2
                    # "orth_penalty_matmul": the FORWARD computation only.
                    # Whether torch.profiler's record_function correctly
                    # attributes the LATER loss.backward() pass (called once,
                    # combining this term with cross-entropy, in the base
                    # BoundedMemoryMixin._bounded_train_epoch loop this method
                    # does not override) back to this scope is UNVERIFIED --
                    # not assumed true. Backward's true cost is exactly the gap
                    # the plan's sec 3.3 differential check exists to resolve;
                    # this tag alone gives only a measured FORWARD floor, not
                    # the "~2x for backward" the old formula assumed.
                    with ce_region("olora/orth_penalty_matmul"):
                        orth = orth + torch.abs(A_t @ A_prev.t()).sum()
        return orth, l2

    # *** UNTESTED as of 2026-08-03 *** -- local GPUs were unavailable
    # (thermal/damage risk) at fix time, so this has NOT been exercised on a
    # live run. Verified only by static tracing of backbone/vit_lora.py: slot
    # indexing (attn._folded_upto, self._cur_task), the fold invariant (slot
    # t-1 is folded when slot t starts training, so t itself stays unfolded
    # for its whole lifetime as "current"), and that frozen_delta_q/v are
    # unconditional register_buffers. No live run has confirmed this executes
    # without error or matches the hand-computed estimate used for the
    # .memcorrected.json files built alongside this fix.
    #
    # FIXED 2026-08-03 (found during the same persistent-memory audit that
    # caught TreeLoRA's bug). O-LoRA previously used the generic fallback
    # (models/lora.py::persistent_state -- "every slot still in the
    # ModuleList", never freed) with the same problem TreeLoRA had: it counts
    # every historical slot as still live and never reads frozen_delta_q/v at
    # all. BUT O-LoRA's correction is NOT a full analog of TreeLoRA's fix --
    # it is only HALF as aggressive, because of a real structural difference:
    # _orth_and_l2 (above) reads every past task's lora_A directly out of the
    # live ModuleList, forever -- old A genuinely cannot be freed or replaced
    # by frozen_delta the way TreeLoRA's/InfLoRA's fully-dead old slots can.
    # Old B, however, IS fully redundant once folded: the fold-enabled forward
    # branch (backbone/vit_lora.py::_lora_delta) only ever reads frozen_delta_*
    # + the current slot, never indexing an old B by position, and nothing
    # else in this file reads old B either. So this override keeps every
    # folded slot's A (the orthogonality penalty's real, unavoidable O(K)
    # dependency) while dropping every folded slot's B (dead weight, replaced
    # by its already-folded contribution inside frozen_delta_q/v), on top of
    # the current (not-yet-folded) slot's own full A+B and the head. This is
    # an ACCOUNTING fix only -- old B tensors are still allocated on the GPU
    # (nothing here calls free_folded_slot, which would also break the
    # orthogonality penalty's old-A read since it frees A and B as a unit) --
    # so this describes what SHOULD be counted as persistent, not a change to
    # what's physically allocated.
    def persistent_state(self):
        net = self._network.module if hasattr(self._network, "module") else self._network
        frozen_delta_bytes = 0
        for attn in net.attn_modules():
            frozen_delta_bytes += attn.frozen_delta_q.numel() * attn.frozen_delta_q.element_size()
            frozen_delta_bytes += attn.frozen_delta_v.numel() * attn.frozen_delta_v.element_size()
        old_a_bytes = 0
        for attn in net.attn_modules():
            for s in range(attn._folded_upto + 1):
                old_a_bytes += attn.lora_A_q[s].weight.numel() * attn.lora_A_q[s].weight.element_size()
                old_a_bytes += attn.lora_A_v[s].weight.numel() * attn.lora_A_v[s].weight.element_size()
        cur_slot_bytes = 0
        for attn in net.attn_modules():
            for mlist in (attn.lora_A_q, attn.lora_B_q, attn.lora_A_v, attn.lora_B_v):
                for p in mlist[self._cur_task].parameters():
                    cur_slot_bytes += p.numel() * p.element_size()
        fc_bytes = sum(p.numel() * p.element_size() for p in net.fc.parameters())
        total_bytes = frozen_delta_bytes + old_a_bytes + cur_slot_bytes + fc_bytes
        return {"params": int(total_bytes // 4), "bytes": int(total_bytes),
                "breakdown": {"frozen_delta": frozen_delta_bytes, "old_lora_A": old_a_bytes,
                             "current_slot": cur_slot_bytes, "fc": fc_bytes}}

    def _train(self, train_loader):
        self._network.to(self._device)
        params = [p for p in self._network.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.min_lr) if self.lr_anneal else None

        lo, hi = self._ce_slice()
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses, ce_run, orth_run, correct, total = 0.0, 0.0, 0.0, 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs, task=self._cur_task, merge=self.train_merge)["logits"]
                local_logits = logits[:, lo:hi]
                local_targets = targets - lo
                ce = F.cross_entropy(local_logits, local_targets)

                orth, l2 = self._orth_and_l2()
                loss = ce + self.lamda_1 * orth + self.lamda_2 * l2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                ce_run += ce.item()
                orth_run += float(orth)

                preds = local_logits.argmax(dim=1)
                correct += preds.eq(local_targets).cpu().sum()
                total += len(targets)
            if scheduler is not None:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            prog_bar.set_description(
                "Task {}, Epoch {}/{} => Loss {:.3f} (CE {:.3f}, Orth {:.3f}), Acc {:.2f}".format(
                    self._cur_task, epoch + 1, self.epochs,
                    losses / len(train_loader), ce_run / len(train_loader),
                    orth_run / len(train_loader), train_acc))
        logging.info("[O-LoRA] Task {} done. Acc {:.2f}, final Orth {:.3f}".format(
            self._cur_task, train_acc, orth_run / len(train_loader)))
