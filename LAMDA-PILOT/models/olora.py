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

    def _orth_and_l2(self):
        """Orthogonality penalty (current vs all previous tasks) + L2 on the
        current task's LoRA, summed over all attention blocks."""
        t = self._cur_task
        orth = 0.0
        l2 = 0.0
        for attn in self._network.attn_modules():
            for A_list, B_list in ((attn.lora_A_q, attn.lora_B_q),
                                   (attn.lora_A_v, attn.lora_B_v)):
                A_t = A_list[t].weight                       # [r, dim]
                l2 = l2 + torch.norm(A_t, p=2) + torch.norm(B_list[t].weight, p=2)
                for s in range(t):
                    A_s = A_list[s].weight                   # [r, dim] (frozen)
                    orth = orth + torch.abs(A_t @ A_s.t()).sum()
        return orth, l2

    def _train(self, train_loader):
        self._network.to(self._device)
        params = [p for p in self._network.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.min_lr)

        lo, hi = self._known_classes, self._total_classes
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
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            prog_bar.set_description(
                "Task {}, Epoch {}/{} => Loss {:.3f} (CE {:.3f}, Orth {:.3f}), Acc {:.2f}".format(
                    self._cur_task, epoch + 1, self.epochs,
                    losses / len(train_loader), ce_run / len(train_loader),
                    orth_run / len(train_loader), train_acc))
        logging.info("[O-LoRA] Task {} done. Acc {:.2f}, final Orth {:.3f}".format(
            self._cur_task, train_acc, orth_run / len(train_loader)))
