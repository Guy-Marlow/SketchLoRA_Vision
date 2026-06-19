"""Baseline per-task LoRA continual learner.

Freezes the pretrained ViT-B/16 backbone and trains one LoRA pair (on the query
and value projections of every block) per task.  Supports both the CIL eval
pipeline of LAMDA-PILOT and the task-incremental (TIL) eval added in
``models/til_base.py``, selected via ``args["scenario"]``.

Training of task t:
  * only task t's LoRA + the classifier head are trainable;
  * the forward routes to task t's LoRA (``merge=False``);
  * cross-entropy is computed on the task-local class slice ``[known, total)``
    so the head is trained per task (no cross-task logit competition).

Inference:
  * CIL  -> merged LoRA (sum 0..t), argmax over the whole grown head;
  * TIL  -> task-routed LoRA, argmax masked to the known task's class slice.
"""

import logging
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.inc_net import LoRAVitNet
from models.til_base import TILLearner
from utils.toolkit import tensor2numpy

num_workers = 8


class Learner(TILLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = LoRAVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args.get("weight_decay") is not None else 5e-4
        self.epochs = args["tuned_epoch"]
        self.min_lr = args["min_lr"] if args.get("min_lr") is not None else 1e-8
        # train forward routing: per-task (False) vs accumulated sum (True)
        self.train_merge = bool(args.get("lora_train_merge", False))

    def after_task(self):
        self._known_classes = self._total_classes

    # -- adapter selection (overridden by SeqLoRA to pin a single adapter) --
    def _train_adapter(self):
        """Which LoRA index to train/route on the current task."""
        return self._cur_task

    def _eval_adapter(self, task):
        """Which LoRA index to route for a sample whose ground-truth task is
        `task` during TIL evaluation."""
        return task

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.data_manager = data_manager
        self._network.update_fc(self._total_classes)
        self._register_task_range()
        self._network.default_task = self._train_adapter()
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        # only the active LoRA + head are trainable
        self._network.freeze_to_task(self._train_adapter())
        for p in self._network.fc.parameters():
            p.requires_grad = True
        self._log_trainable()

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                       shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                      shuffle=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader):
        self._network.to(self._device)
        params = [p for p in self._network.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.min_lr)

        lo, hi = self._known_classes, self._total_classes
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses, correct, total = 0.0, 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs, task=self._train_adapter(), merge=self.train_merge)["logits"]
                # task-local cross-entropy: only this task's class slice
                local_logits = logits[:, lo:hi]
                local_targets = targets - lo
                loss = F.cross_entropy(local_logits, local_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                preds = local_logits.argmax(dim=1)
                correct += preds.eq(local_targets).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            prog_bar.set_description(
                "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task, epoch + 1, self.epochs, losses / len(train_loader), train_acc))
        logging.info("Task {} finished. Train_accy {:.2f}".format(self._cur_task, train_acc))

    # TIL routing: discriminate within a task with that task's own LoRA
    def _forward_task(self, inputs, task):
        net = self._network.module if hasattr(self._network, "module") else self._network
        return net(inputs, task=self._eval_adapter(task), merge=False)

    def _log_trainable(self):
        net = self._network
        total = sum(p.numel() for p in net.parameters())
        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        logging.info("Trainable params this task: {:,} / {:,}".format(trainable, total))
