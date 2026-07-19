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
                l2 = l2 + torch.norm(A_t, p=2) + torch.norm(B_list[t].weight, p=2)
                if t > 0:
                    cache_attr, task_attr = "_orth_prev_" + proj, "_orth_prev_task_" + proj
                    if getattr(attn, task_attr, None) != t:
                        stacked = torch.cat([A_list[s].weight.detach() for s in range(t)], dim=0)
                        setattr(attn, cache_attr, stacked)
                        setattr(attn, task_attr, t)
                    A_prev = getattr(attn, cache_attr)       # [(t*r), dim] (frozen)
                    orth = orth + torch.abs(A_t @ A_prev.t()).sum()
        return orth, l2

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
