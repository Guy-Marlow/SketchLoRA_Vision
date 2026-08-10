"""O-LoRA (Orthogonal subspace LoRA) for LAMDA-PILOT.

Port of the regularizer from the original O-LoRA repo
(svd_sketching_vision/O-LoRA/src/uie_trainer_lora.py): in addition to the
task-local cross-entropy, the current task's LoRA-A directions are pushed to be
orthogonal to every previous task's (frozen) LoRA-A, and the current LoRA is
L2-regularized:

    loss = CE  +  lambda_1 * sum_layers sum_{s<t} | A_t @ A_s^T |.sum()
               +  lambda_2 * || current-task LoRA ||_2

Here A is applied to the query and value projections (our shared convention).
Inference is the accumulated sum of LoRAs 0..t (CIL) or task-routed (TIL),
reusing the baseline ``LoRAVitNet`` and the TIL eval in ``models/til_base.py``.

2026-08-10: does NOT use frozen-slot dense folding (see backbone/vit_lora.py's
``enable_frozen_folding``). Cross-checked against O-LoRA-Ref
(svd_sketching_vision/O-LoRA/src/peft/tuners/lora.py's ``Linear.forward``):
the reference NEVER materializes a dense delta -- it always sums the frozen
history factor and the current task's factor directly (factored forward), the
same non-fold loop this backbone already provides. Our earlier port opted
into dense folding anyway (a shared LoRA-family convenience, not something
O-LoRA's own algorithm needs); at this project's rank=8-10/dim=768 settings
the crossover where dense folding actually becomes cheaper than factored is
t* = dim/(2r) ~= 38-48 tasks, and every non-streaming O-LoRA split in this
repo (10-20 tasks) runs under that -- so folding was making O-LoRA pay MORE
compute than its own reference algorithm requires, for this whole benchmark
suite except Omnibenchmark-1K. Removed to match the reference and to stop
overcharging O-LoRA's measured CE for a self-inflicted cost. See the
ce_profiling_methodology memory for the fuller investigation.
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
from utils.ce_profiler import ce_region, run_boundary, run_step_narrow
from utils.ce2_profiler import ce2_boundary


class Learner(LoRALearner):
    def __init__(self, args):
        super().__init__(args)
        # 2026-08-10: NOT calling enable_frozen_folding() -- see module
        # docstring. O-LoRA's frozen slots ARE immutable once trained (folding
        # would still be numerically safe), but O-LoRA-Ref itself never folds,
        # and doing so costs more compute than factored forward for every
        # split in this repo except Omnibenchmark-1K's 100 tasks. _lora_delta
        # (backbone/vit_lora.py) falls through to its factored/non-fold loop
        # automatically whenever _fold_enabled stays False (the class
        # default) -- no other code change needed for this to take effect.
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
        # Called from inside models/bounded_memory_mixin.py's _stream_begin_chunk
        # call site, which the driver already wraps in a "boundary_begin"
        # profiling session end-to-end -- no separate wrap needed here (unlike
        # the oracle _train() override below, which has no such outer wrap).
        self._refresh_orth_cache()

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

    def _refresh_orth_cache(self):
        """Rebuild the per-(block,proj) concatenated frozen-A cache used by the
        orthogonality penalty, one [t*r, dim] matrix per (block, projection).

        MOVED here 2026-08-05 (docs/ce_step_boundary_isolation_plan.md sec 3),
        out of a lazy, per-step-guarded rebuild that used to live inside
        _orth_and_l2() (guarded by `if getattr(attn, task_attr, None) != t`,
        checked on every training step). Functionally identical -- same cache
        contents, same values, same construction -- but now fires exactly once,
        explicitly, at task/cycle start, instead of being triggered
        opportunistically on whichever step happens to notice `_cur_task`
        changed. Two reasons this is not just a cosmetic move:

        1. CORRECTNESS: this rebuild is a genuinely once-per-TASK cost (t only
           changes across tasks/cycles, never across epochs within one task) --
           but it used to be captured inside the SAME profiling session as the
           genuinely-per-STEP orth_penalty_matmul/orth_l2_norms costs, both
           landing in `aux_macs_per_step`, which downstream gets scaled by
           n_epochs*steps_per_epoch. That overstates this rebuild's true
           contribution to Ops_total by a factor of n_epochs (~20x at
           tuned_epoch=20) -- see the plan doc's worked example. Making it an
           explicit, separate boundary call lets it be measured and charged
           ONCE per task, not (n_epochs times too many).
        2. PERFORMANCE: removes a per-step attribute-lookup + int-comparison
           from every step for the rest of the task (net improvement, every
           run, profiled or not -- no behavior/cost added to the unprofiled
           path).

        No-op on task 0 (nothing to compare against yet, matching
        _orth_and_l2's own `if t > 0` guard) and a no-op on any later call this
        same task (matches the original guard's idempotence within a task)."""
        t = self._cur_task
        if t <= 0:
            return
        for attn in self._network.attn_modules():
            for proj, A_list in (("q", attn.lora_A_q), ("v", attn.lora_A_v)):
                cache_attr, task_attr = "_orth_prev_" + proj, "_orth_prev_task_" + proj
                if getattr(attn, task_attr, None) != t:
                    # plan sec 4.2 "orth_prev_cache_rebuild": at t=484 this
                    # concatenates a [4840,768] matrix per (block,proj) -- 24
                    # pairs total -- ~357MB of copies (t*rank*dim*4 bytes *
                    # n_blocks*n_proj). Bandwidth-bound, invisible to MACs (R5).
                    with ce_region("olora/orth_prev_cache_rebuild"):
                        stacked = torch.cat([A_list[s].weight.detach() for s in range(t)], dim=0)
                    setattr(attn, cache_attr, stacked)
                    setattr(attn, task_attr, t)

    def _orth_and_l2(self):
        """Orthogonality penalty (current vs all previous tasks) + L2 on the
        current task's LoRA, summed over all attention blocks. Assumes
        _refresh_orth_cache() has already been called this task/cycle (see
        incremental_train's caller / _stream_begin_chunk / _train below) --
        this method no longer rebuilds the cache itself (2026-08-05, see
        _refresh_orth_cache's own docstring for why).

        Vectorised (mirrors svd_sketching_language/tokmem/atomic/multislot_lora.py::
        orthogonality_penalty): the frozen A_{<t} for a given (block, projection) are
        stacked into one [t*r, dim] matrix once per task, so the cross-term is a
        SINGLE matmul+abs+sum instead of a per-s loop. Numerically identical: for the
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
                # plan sec 4.2: two torch.norm calls per block-projection, every
                # step -- genuinely per-step, no cache/rebuild involved.
                with ce_region("olora/orth_l2_norms"):
                    l2 = l2 + torch.norm(A_t, p=2) + torch.norm(B_list[t].weight, p=2)
                if t > 0:
                    cache_attr = "_orth_prev_" + proj
                    A_prev = getattr(attn, cache_attr)       # [(t*r), dim] (frozen)
                    # plan sec 4.2 "orth_penalty_matmul": FORWARD only.
                    # Backward attribution decision (docs/ce_step_boundary_
                    # isolation_plan.md sec 4.2/11.1, resolved 2026-08-05):
                    # forward-only floor, not a throwaway isolated-backward
                    # measurement -- report this as a conservative floor on
                    # the true (forward+backward) cost of this term.
                    with ce_region("olora/orth_penalty_matmul"):
                        orth = orth + torch.abs(A_t @ A_prev.t()).sum()
        return orth, l2

    # 2026-08-10: REMOVED the fold-specific persistent_state() override that
    # used to live here (accounted old-A up to attn._folded_upto, treated old-B
    # as fully dead/folded-away weight). That accounting was correct ONLY under
    # folding: with enable_frozen_folding() no longer called (see __init__ and
    # the module docstring), _folded_upto never advances past its -1 initial
    # value, so that override would silently report ZERO bytes for every
    # historical slot's A and B -- a real undercount, not a harmless no-op,
    # since every slot's A (orthogonality penalty) AND B (factored forward,
    # backbone/vit_lora.py's non-fold loop branch) are now genuinely live and
    # read every step. Falls back to models/lora.py::persistent_state(), which
    # already sums every currently-allocated slot's A+B+head -- exactly the
    # right accounting once nothing gets folded away.
    def _train(self, train_loader):
        # oracle-mode boundary bookkeeping (docs/ce_step_boundary_isolation_plan.md
        # sec 7): trainer.py has no visibility inside incremental_train(), so this
        # method wraps its OWN boundary call using whatever controller trainer.py
        # attached to `self` this task (None when final_metrics/CE-logging is off,
        # in which case run_boundary just calls _refresh_orth_cache() directly --
        # zero added cost on the unprofiled path). Under bounded_memory streaming
        # this call site isn't used at all -- _stream_begin_chunk (above) already
        # calls _refresh_orth_cache() from inside the driver's own outer session.
        with ce2_boundary(self):
            run_boundary(getattr(self, "_ce_boundary_ctrl", None), "boundary", self._refresh_orth_cache)

        self._network.to(self._device)
        params = [p for p in self._network.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.min_lr) if self.lr_anneal else None

        lo, hi = self._ce_slice()
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            # Step-type measurement (docs/ce_step_boundary_isolation_plan.md sec
            # 1a/2/7): only wrap the isolated _orth_and_l2() call -- not the
            # surrounding forward/backward/optimizer.step() -- and only on epoch
            # 0. `step_acc` is a fresh-per-task NarrowAuxAccumulator trainer.py
            # attaches to `self` this task (None when off); run_step_narrow falls
            # back to a direct, unprofiled call otherwise, so epochs 1..N-1 and
            # every unprofiled run are completely unaffected either way.
            step_acc = getattr(self, "_ce_step_acc", None) if epoch == 0 else None
            losses, ce_run, orth_run, correct, total = 0.0, 0.0, 0.0, 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs, task=self._cur_task, merge=self.train_merge)["logits"]
                local_logits = logits[:, lo:hi]
                local_targets = targets - lo
                ce = F.cross_entropy(local_logits, local_targets)

                orth, l2 = run_step_narrow(step_acc, "olora_step", self._orth_and_l2)
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
