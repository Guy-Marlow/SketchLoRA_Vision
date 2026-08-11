"""RainbowPrompt for LAMDA-PILOT (replaces L2P in the final method roster, user
2026-07-16). Port of Hong et al., ICCV 2025 -- see backbone/vit_rainbowprompt.py and
backbone/rainbowprompt_module.py for the mechanism and the two confirmed
simplifications (static self_attn_idx, use_linear=False).

Training: task-routed (train=True; the module (re)computes that task's evolved
prompt each forward), task-local CE + a "pull constraint" on the task-key/CLS-
feature similarity, same sign convention as this project's existing
models/dualprompt.py (`loss -= pull_constraint_coeff * sim_loss`).

CIL (deployed): train=False, no task_id forced -- the module predicts a
BATCH-LEVEL task id via nearest stored task-key match (see rainbowprompt_module.py's
docstring: this is the reference's own behaviour, not per-sample), retrieves that
task's stored evolved prompt, argmax over the full grown head.

TIL: train=False, `known_task=task` -- bypasses key-matching, retrieves the
ground-truth task's stored prompt directly, masks logits to that task's class slice
(via existing til_base.py, unchanged).
"""

import logging

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.inc_net import RainbowPromptVitNet
from models.til_base import TILLearner
from models.stream_mixin import StreamMixin
from utils.toolkit import tensor2numpy

# 2026-08-10: 8->4, see models/lora.py's identical change for rationale.
num_workers = 8


class Learner(StreamMixin, TILLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = RainbowPromptVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args.get("weight_decay") or 5e-4
        self.epochs = args["tuned_epoch"]
        self.min_lr = args.get("min_lr") or 1e-8
        self.pull_constraint_coeff = args.get("pull_constraint_coeff", 0.1)
        self.clip_grad = args.get("clip_grad", 1.0)
        # LR warmup (reference default: warmup_lr=1e-6 -> init_lr over warmup_epochs
        # epoch-equivalents; sched='constant' means NO further schedule after warmup
        # -- confirmed load-bearing live, 2026-07-16: at task 0 (1 epoch/task, so
        # entirely inside the 5-epoch warmup window), applying init_lr=0.03 from
        # step 1 diverges (loss 1.9 -> 17 -> 14.6 -> 26.8 -> 50.9 in 8 steps, WITH
        # grad-clip already applied); a linear ramp from warmup_lr keeps loss in the
        # expected ~1.5-2.7 band throughout. warmup_steps is a fixed budget (not
        # reset per task) tracked via self._global_step across the whole run,
        # approximating the reference's global-epoch-counted warmup with a
        # step-counted one (this codebase's task loop doesn't expose "epoch index
        # across all tasks" the way timm's single-run trainer does).
        self.warmup_lr = args.get("warmup_lr", 1e-6)
        self.warmup_epochs = args.get("warmup_epochs", 5)
        self._global_step = 0
        self._warmup_steps = None   # set on first _train call from that loader's length
        self.data_manager = None

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.data_manager = data_manager
        self._network.update_fc(self._total_classes)
        self._register_task_range()
        self._network.default_task = self._cur_task
        logging.info("[RainbowPrompt] Learning on {}-{}".format(self._known_classes, self._total_classes))

        # only this task's base-knowledge row (+ its key) and the head are trainable
        for p in self._network.parameters():
            p.requires_grad = False
        for p in self._network.fc.parameters():
            p.requires_grad = True
        # nn.Parameter can't be partially frozen by row, so keep each layer's
        # WHOLE base_knowledge tensor (+ base_key) trainable; the forward pass
        # itself only ever reads the current task's row live (prior rows are
        # explicitly .detach()'d in rainbowprompt_module.py's train branch), so no
        # gradient reaches earlier tasks' rows regardless -- matches the reference's
        # own approach of freezing whole components per task, adapted to our
        # single-tensor-per-layer (top_k=1) storage instead of separate modules.
        for bk in self._network.backbone.prompt_module.base_knowledge:
            bk.requires_grad_(True)
        self._network.backbone.prompt_module.base_key.requires_grad_(True)
        # query/key/value_matcher, dense, fc1, fc2 (use_linear=True's Prompt_Evolution
        # sublayers) are shared (non-growing) params. The reference DEFINES a
        # freeze_components() that would lock query/key/value_matcher+dense after
        # task 0, but never calls it anywhere in its own codebase (grepped, zero call
        # sites) -- so its real behavior leaves all six trainable every task; matched
        # here rather than reproducing an orphaned method's unexercised intent.
        pm = self._network.backbone.prompt_module
        if pm.use_linear:
            for module_list in (pm.query_matcher, pm.key_matcher, pm.value_matcher,
                                 pm.dense, pm.fc1, pm.fc2):
                for p in module_list.parameters():
                    p.requires_grad_(True)

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
        # sched='constant' (the reference's own default) means NO decay schedule --
        # only the warmup ramp below, then a flat init_lr.
        optimizer = optim.AdamW(params, lr=self.warmup_lr, weight_decay=self.weight_decay)
        if self._warmup_steps is None:
            self._warmup_steps = self.warmup_epochs * len(train_loader)

        lo, hi = self._ce_slice()
        t = self._cur_task
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses, correct, total = 0.0, 0, 0
            for _, inputs, targets in train_loader:
                lr = self.warmup_lr + (self.init_lr - self.warmup_lr) * min(
                    self._global_step, self._warmup_steps) / self._warmup_steps
                for g in optimizer.param_groups:
                    g["lr"] = lr
                self._global_step += 1

                inputs, targets = inputs.to(self._device), targets.to(self._device)
                out = self._network(inputs, task_id=t, train=True)
                local_logits = out["logits"][:, lo:hi]
                local_targets = targets - lo
                loss = F.cross_entropy(local_logits, local_targets)
                loss = loss - self.pull_constraint_coeff * out["sim_loss"]

                optimizer.zero_grad()
                loss.backward()
                # reference's engine.py:77 clips at every step (--clip-grad,
                # default 1.0). Both this AND the warmup ramp above are load-bearing
                # -- confirmed live, 2026-07-16: init_lr=0.03 from step 1 (even WITH
                # clipping) diverges within 1-2 steps (loss 1.9->17.0->14.6->26.8->
                # 50.9 in an isolated 8-step repro); adding the warmup ramp on top
                # of clipping keeps loss in the expected ~1.4-2.7 band throughout.
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self._network.parameters() if p.requires_grad], self.clip_grad)
                optimizer.step()
                losses += loss.item()

                preds = local_logits.argmax(dim=1)
                correct += preds.eq(local_targets).cpu().sum()
                total += len(targets)
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            prog_bar.set_description(
                "[RainbowPrompt] Task {}, Epoch {}/{} => Loss {:.3f}, Acc {:.2f}".format(
                    t, epoch + 1, self.epochs, losses / len(train_loader), train_acc))
        logging.info("[RainbowPrompt] Task {} done. Acc {:.2f}".format(t, train_acc))

    # -- CIL eval: base.py's _eval_cnn calls self._network(inputs) with no task_id,
    # which RainbowPromptVitNet defaults to self.default_task -- but the deployed
    # RainbowPrompt model predicts task id itself (train=False), so override to
    # force that codepath (task_id argument is unused when train=False and
    # known_task is None -- the module ignores it except to cap the key-matching
    # candidate pool at task_id+1, so pass the LATEST task to search the full
    # history).
    @torch.no_grad()
    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        k = min(self.topk, self._total_classes)
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            logits = self._network(inputs, task_id=self._cur_task, train=False)["logits"]
            predicts = torch.topk(logits, k=k, dim=1, largest=True, sorted=True)[1]
            if k < self.topk:
                pad = predicts[:, -1:].expand(-1, self.topk - k)
                predicts = torch.cat([predicts, pad], dim=1)
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    # TIL: ground-truth task known -- bypass key-matching via known_task
    def _forward_task(self, inputs, task):
        return self._network(inputs, task_id=task, train=False, known_task=task)

    def persistent_state(self):
        net = self._network
        pm = net.backbone.prompt_module
        bk_bytes = sum(bk.numel() * bk.element_size() for bk in pm.base_knowledge)
        key_bytes = pm.base_key.numel() * pm.base_key.element_size()
        stored_bytes = pm.stored_prompts.numel() * pm.stored_prompts.element_size()
        head_bytes = sum(p.numel() * p.element_size() for p in net.fc.parameters())
        evolve_bytes = 0
        if pm.use_linear:
            for module_list in (pm.query_matcher, pm.key_matcher, pm.value_matcher,
                                 pm.dense, pm.fc1, pm.fc2):
                evolve_bytes += sum(p.numel() * p.element_size() for p in module_list.parameters())
        total_bytes = bk_bytes + key_bytes + stored_bytes + head_bytes + evolve_bytes
        total_params = total_bytes // 4
        return {"params": int(total_params), "bytes": int(total_bytes),
                "breakdown": {"base_knowledge": bk_bytes, "base_key": key_bytes,
                             "stored_prompts": stored_bytes, "head": head_bytes,
                             "evolve_sublayers": evolve_bytes}}

    @torch.no_grad()
    def _deployed_forward(self, inputs):
        """The actual deployed CIL forward, verbatim from _eval_cnn (train=False,
        test-time key-matching routing since known_task is not passed), for
        metrics_logger.py's inference-cost/FLOPs measurement."""
        return self._network(inputs, task_id=self._cur_task, train=False)

    # ==================================================================
    # Boundary-agnostic streaming hooks (models/stream_mixin.py). base_knowledge/
    # base_key/stored_prompts are ALREADY preallocated for `n_tasks` slots
    # (backbone/rainbowprompt_module.py:69-74), identical in spirit to the LoRA
    # scaffold's per-task ModuleLists -- so routing by CHUNK index instead of
    # real task index is a drop-in change, no resizing needed (same safe-upper-
    # bound argument as TreeLoRA/HiDeLoRA: chunk count <= real task count given
    # how the memory constraint was derived).
    #
    # The warmup-LR mechanism is NOT touched -- it already tracks a persistent
    # `self._global_step` across the WHOLE run (never reset per task), so it
    # continues to ramp exactly as before under streaming. The generic
    # `_stream_new_optimizer()` helper (flat init_lr from step 1) is NOT used
    # here because the warmup ramp is load-bearing (documented in __init__ --
    # skipping it diverges even with grad-clip); this needs its own optimizer
    # construction + per-step LR schedule, mirroring _train's loop exactly.
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
            self._network.backbone.prompt_module.add_task_slot()
        self._cur_task = self._stream_chunk   # persistent_state/logging only
        for p in self._network.parameters():
            p.requires_grad = False
        for p in self._network.fc.parameters():
            p.requires_grad = True
        for bk in self._network.backbone.prompt_module.base_knowledge:
            bk.requires_grad_(True)
        self._network.backbone.prompt_module.base_key.requires_grad_(True)
        pm = self._network.backbone.prompt_module
        if pm.use_linear:
            for module_list in (pm.query_matcher, pm.key_matcher, pm.value_matcher,
                                 pm.dense, pm.fc1, pm.fc2):
                for p in module_list.parameters():
                    p.requires_grad_(True)
        params = [p for p in self._network.parameters() if p.requires_grad]
        self._stream_optim = optim.AdamW(params, lr=self.warmup_lr, weight_decay=self.weight_decay)
        if self._warmup_steps is None:
            self._warmup_steps = self.warmup_epochs * len(loader)
        self._stream_sched = None   # sched='constant' in the reference -- no decay schedule at all

    def _stream_train_epoch(self, loader, lo, hi):
        """Full override -- mirrors _train's per-batch loop exactly (warmup ramp
        + grad-clip + pull-constraint loss), none of which the generic
        _stream_train_epoch/_stream_extra_loss hooks support."""
        self._network.train()
        t = self._stream_chunk
        for _, inputs, targets in loader:
            lr = self.warmup_lr + (self.init_lr - self.warmup_lr) * min(
                self._global_step, self._warmup_steps) / self._warmup_steps
            for g in self._stream_optim.param_groups:
                g["lr"] = lr
            self._global_step += 1

            inputs, targets = inputs.to(self._device), targets.to(self._device)
            out = self._network(inputs, task_id=t, train=True)
            local_logits = out["logits"][:, lo:hi]
            local_targets = targets - lo
            loss = F.cross_entropy(local_logits, local_targets)
            loss = loss - self.pull_constraint_coeff * out["sim_loss"]

            self._stream_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self._network.parameters() if p.requires_grad], self.clip_grad)
            self._stream_optim.step()

    def _stream_cil_forward(self, inputs):
        """Deployed CIL forward: predicts task id itself (train=False); pass the
        LATEST chunk to search the full key-matching history, same convention
        as the non-streaming _eval_cnn override."""
        return self._network(inputs, task_id=self._stream_chunk, train=False)

    def _forward_task(self, inputs, task):
        """TIL routing: ground-truth ready known, but `known_task` must reference
        the CHUNK/slot that was actually active/deployed for that real task, not
        the real task index itself (chunk count generically diverges from real
        task count) -- remap via the generic _stream_task_to_chunk map built by
        stream_run() (identity outside streaming, since that attribute doesn't
        exist there)."""
        slot = getattr(self, "_stream_task_to_chunk", {}).get(task, task)
        return self._network(inputs, task_id=slot, train=False, known_task=slot)
