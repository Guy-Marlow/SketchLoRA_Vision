"""ProgPrompt for LAMDA-PILOT: per-task progressive soft prompts
(backbone/vit_progprompt.py) on a fully-frozen ViT-B/16 backbone -- good-faith port
of Razdaibiedina et al.'s "Progressive Prompts" (T5 origin) onto our ViT CIL
scaffold. See backbone/vit_progprompt.py's docstring for the confirmed reference
details (prefix_len=10, newest-first concatenation) and why lr is NOT transplanted
from the T5 reference's 0.3 (architecture-transfer risk, no validated story for
ViT -- uses this project's own validated L2P/DualPrompt prompt-tuning LR range).

Training: task-routed (route to slot t, which concatenates prompts t,t-1,...,0),
task-local CE (same convention as lora.py).

CIL (deployed): route to the most recent task (full prompt stack, task-id-free
argmax over the grown head -- the method's whole point).
TIL: route to the ground-truth task id directly -- our backbone's `task` argument
IS an integer slot id (no key-matching involved, unlike RainbowPrompt), so this
naturally reproduces "the stack as it was when that task was trained," no bypass
mechanism needed.
"""

import logging

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.inc_net import ProgPromptVitNet
from models.til_base import TILLearner
from models.stream_mixin import StreamMixin
from utils.toolkit import tensor2numpy

num_workers = 8


class Learner(StreamMixin, TILLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = ProgPromptVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args.get("weight_decay") or 5e-4
        self.epochs = args["tuned_epoch"]
        self.min_lr = args.get("min_lr") or 1e-8
        self.lr_anneal = bool(args.get("lr_anneal", True))

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.data_manager = data_manager
        self._network.update_fc(self._total_classes)
        self._register_task_range()
        self._network.default_task = self._cur_task
        logging.info("[ProgPrompt] Learning on {}-{}".format(self._known_classes, self._total_classes))

        for p in self._network.parameters():
            p.requires_grad = False
        self._network.backbone.prompts[self._cur_task].requires_grad_(True)
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
        self._train(self.train_loader)

    def _train(self, train_loader):
        params = [p for p in self._network.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.min_lr) if self.lr_anneal and self.epochs > 1 else None

        lo, hi = self._ce_slice()
        t = self._cur_task
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses, correct, total = 0.0, 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs, task=t)["logits"]
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
            if scheduler is not None:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            prog_bar.set_description(
                "[ProgPrompt] Task {}, Epoch {}/{} => Loss {:.3f}, Acc {:.2f}".format(
                    t, epoch + 1, self.epochs, losses / len(train_loader), train_acc))
        logging.info("[ProgPrompt] Task {} done. Acc {:.2f}".format(t, train_acc))

    def _forward_task(self, inputs, task):
        """TIL routing. Under streaming, `task` (a real task index, from
        _stream_eval's loop) must be remapped to the CHUNK/slot that was
        actually active/deployed for that real task (chunk count generically
        diverges from real task count) -- via the generic _stream_task_to_chunk
        map built by stream_run() (identity outside streaming, since that
        attribute doesn't exist there)."""
        net = self._network.module if hasattr(self._network, "module") else self._network
        slot = getattr(self, "_stream_task_to_chunk", {}).get(task, task)
        return net(inputs, task=slot)

    def persistent_state(self):
        net = self._network.module if hasattr(self._network, "module") else self._network
        prompt_bytes = sum(p.numel() * p.element_size() for p in net.backbone.prompts)
        head_bytes = sum(p.numel() * p.element_size() for p in net.fc.parameters())
        total_bytes = prompt_bytes + head_bytes
        return {"params": int(total_bytes // 4), "bytes": int(total_bytes),
                "breakdown": {"prompts": prompt_bytes, "head": head_bytes}}

    @torch.no_grad()
    def _deployed_forward(self, inputs):
        """The actual deployed CIL forward -- base.py's default _eval_cnn calls
        self._network(inputs) with no task kwarg, which ProgPromptVitNet.forward
        resolves via self.default_task (set to self._cur_task in incremental_train).
        Made explicit here for metrics_logger.py's inference-cost/FLOPs measurement."""
        net = self._network.module if hasattr(self._network, "module") else self._network
        return net(inputs, task=self._cur_task)

    # ==================================================================
    # Boundary-agnostic streaming hooks (models/stream_mixin.py). self.prompts
    # is already preallocated for n_tasks slots (vit_progprompt.py:111-112),
    # identical in spirit to the LoRA scaffold -- routing by CHUNK index instead
    # of real task index is a drop-in change, no resizing needed (chunk count
    # <= real task count given how the memory constraint was derived).
    #
    # Full _stream_train_epoch/_stream_cil_forward overrides are REQUIRED (not
    # just optional faithfulness) because ProgPromptVitNet.forward(x, task=-1)
    # does NOT accept a `merge` kwarg at all -- the generic hooks in
    # stream_mixin.py call self._network(inputs, task=slot, merge=merge), which
    # would raise TypeError against this network. No warmup/extra-loss term
    # needed otherwise (plain CE, same as O-LoRA's simplest case).
    # ==================================================================
    def _stream_init(self):
        self._stream_chunk = -1

    def _stream_slot(self):
        return self._stream_chunk

    def _stream_begin_chunk(self, loader):
        self._stream_chunk += 1
        if self._stream_chunk > 0:
            # slot count is not generically bounded by nb_tasks under this clock --
            # see BOUNDARY_AGNOSTIC_IMPLEMENTATION_LOG.md's "BLOCKING ARCHITECTURAL GAP"
            self._network.backbone.add_task_slot()
        self._cur_task = self._stream_chunk   # persistent_state/logging only
        for p in self._network.parameters():
            p.requires_grad = False
        self._network.backbone.prompts[self._stream_chunk].requires_grad_(True)
        for p in self._network.fc.parameters():
            p.requires_grad = True
        self._stream_new_optimizer()

    def _stream_train_epoch(self, loader, lo, hi):
        self._network.train()
        t = self._stream_chunk
        for _, inputs, targets in loader:
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            logits = self._network(inputs, task=t)["logits"]
            local_logits = logits[:, lo:hi]
            loss = F.cross_entropy(local_logits, targets - lo)
            self._stream_optim.zero_grad()
            loss.backward()
            self._stream_optim.step()
        if self._stream_sched is not None:
            self._stream_sched.step()

    def _stream_cil_forward(self, inputs):
        return self._network(inputs, task=self._stream_chunk)
