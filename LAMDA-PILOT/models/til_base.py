"""Task-incremental (TIL) support for LAMDA-PILOT, implemented non-destructively.

LAMDA-PILOT is a *class-incremental* (CIL) bench: the evaluator in
``models/base.py::BaseLearner._eval_cnn`` takes an argmax over the **whole**
grown head (all classes seen so far) with no notion of task identity at test
time.  Task-incremental evaluation instead assumes the task id is known at test
time and the model only has to discriminate **within** that task's classes.

``TILLearner`` adds this as a parallel code path without editing ``base.py``:

  * It records each task's ``(low, high)`` class range in ``self._task_ranges``.
  * ``eval_task`` dispatches on ``args["scenario"]`` ("cil" | "til" | "both").
  * ``_eval_til`` evaluates each task separately, routes the backbone to that
    task's adapter (``forward(x, task=t)``) and masks the logits to that task's
    class slice before taking the argmax.

Concrete learners just need to (a) subclass ``TILLearner``, (b) store
``self.data_manager`` and call ``self._register_task_range()`` once the task's
class span is known, and (c) accept a ``task`` kwarg in ``network.forward``.
"""

import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.base import BaseLearner

batch_size = 64
num_workers = 8


class TILLearner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.scenario = args.get("scenario", "cil")  # "cil" | "til" | "both"
        self._task_ranges = []                        # list of (low, high) per task
        self.data_manager = None
        # per-task TIL top1 from the most recent _eval_til, for the TIL matrix
        self._til_per_task = None

    # -- bookkeeping ----------------------------------------------------
    def _register_task_range(self):
        """Record the current task's [known, total) class range. Call once per
        task after ``_known_classes``/``_total_classes`` are set."""
        rng = (self._known_classes, self._total_classes)
        if len(self._task_ranges) <= self._cur_task:
            self._task_ranges.append(rng)
        else:
            self._task_ranges[self._cur_task] = rng

    def _task_of_label(self, y):
        for t, (lo, hi) in enumerate(self._task_ranges):
            if lo <= y < hi:
                return t
        return -1

    # -- evaluation dispatch -------------------------------------------
    def eval_task(self):
        cil_accy = None
        til_accy = None
        if self.scenario in ("cil", "both"):
            y_pred, y_true = self._eval_cnn(self.test_loader)
            cil_accy = self._evaluate(y_pred, y_true)
        if self.scenario in ("til", "both"):
            y_pred, y_true = self._eval_til()
            til_accy = self._evaluate(y_pred, y_true)
            logging.info("TIL top1: {}".format(til_accy["top1"]))
        self._til_accy = til_accy   # exposed for utils/metrics_logger.py (eval_task only
                                     # ever returns cnn_accy as "primary"; til_accy was
                                     # otherwise discarded)

        if self.scenario == "til":
            primary = til_accy
        elif self.scenario == "both":
            logging.info("CIL top1: {} | TIL top1: {}".format(cil_accy["top1"], til_accy["top1"]))
            primary = cil_accy
        else:
            primary = cil_accy

        # second return slot is reserved for NME in the CIL pipeline; keep None
        return primary, None

    # -- task-incremental evaluator ------------------------------------
    def _eval_til(self):
        """Evaluate each task with its id known, masking logits to that task's
        class slice.  Returns (y_pred[N, topk], y_true[N]) like ``_eval_cnn``."""
        assert self.data_manager is not None, "TILLearner needs self.data_manager set"
        self._network.eval()
        topk = self.topk
        all_pred, all_true = [], []
        per_task = []                                    # top1 TIL acc per task

        for t, (lo, hi) in enumerate(self._task_ranges):
            test_dataset = self.data_manager.get_dataset(
                np.arange(lo, hi), source="test", mode="test"
            )
            loader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)
            width = hi - lo
            k = min(topk, width)
            t_top1, t_n = 0, 0                            # this task's correct / total
            for _, inputs, targets in loader:
                inputs = inputs.to(self._device)
                with torch.no_grad():
                    logits = self._forward_task(inputs, t)["logits"]
                slice_logits = logits[:, lo:hi]                 # mask to this task
                local = torch.topk(slice_logits, k=k, dim=1, largest=True, sorted=True)[1]
                pred = (local + lo).cpu().numpy()
                if k < topk:                                    # pad to keep [N, topk]
                    pad = np.tile(pred[:, -1:], (1, topk - k))
                    pred = np.concatenate([pred, pad], axis=1)
                all_pred.append(pred)
                all_true.append(targets.cpu().numpy())
                tgt = targets.numpy()
                t_top1 += int((pred[:, 0] == tgt).sum()); t_n += len(tgt)
            per_task.append(round(100.0 * t_top1 / max(t_n, 1), 2))

        self._til_per_task = per_task                    # for the TIL accuracy matrix
        return np.concatenate(all_pred), np.concatenate(all_true)

    def _forward_task(self, inputs, task):
        """Route the network to task `task`. Override if the network's forward
        signature differs."""
        net = self._network.module if hasattr(self._network, "module") else self._network
        return net(inputs, task=task)
