"""LoRA-Subtraction-Sketch, fold-before-eval variant (2026-11-11 user
request).

models/lorasub_sketch.py folds the PREVIOUS task's trained RESIDUAL into
SKETCH at the START of the CURRENT task's incremental_train, before that
task is trained. Since the shared trainer loop runs eval_task() immediately
after incremental_train() returns (before after_task()), this means eval
for task t currently reads the network's own merge=True CIL forward, which
sums SKETCH (folded history through task t-1) + RESIDUAL (task t's own
just-trained, FULL-RANK, not-yet-truncated adapter) -- a strictly more
informative, unbounded representation than what persistent_state() counts
as the actual bank (RESIDUAL is explicitly excluded there as "live,
not-yet-folded").

This variant moves the SAME _sketch_fold() call to the END of
incremental_train instead -- right after self._train() returns, for every
task including task 0, unconditionally -- so eval sees the properly
truncated SKETCH-only state. This matches models/sketchlora.py's own
cadence exactly (its _train() calls self._compress() before returning, so
"Eval ... runs after this returns, before after_task()" always sees the
post-fold sketch -- see that file's own comment at the compress call site).
No eval-routing change is needed to get this: _sketch_fold() already resets
RESIDUAL to (kaiming A, zero B), so the network's merge=True CIL sum
degenerates to SKETCH alone by the time eval runs.

Only incremental_train is overridden here. _sketch_fold() (the QR-SVD merge
+ RESIDUAL reset, including its first-task transplant branch),
persistent_state(), the DRS training loop, and everything else is inherited
unchanged from models/lorasub_sketch.py::Learner."""
import logging

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from models.lorasub_sketch import Learner as LoRASubSketchLearner, num_workers
from models.sketchlora import RESIDUAL


class Learner(LoRASubSketchLearner):
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.data_manager = data_manager
        self._network.update_fc(self._total_classes)
        self._register_task_range()
        self._network.default_task = RESIDUAL
        logging.info("[LoRA-Sub-Sketch-FoldFirst] Learning on {}-{}".format(
            self._known_classes, self._total_classes))

        self._network.freeze_to_task(RESIDUAL)  # train_a=True default: both A,B train
        for p in self._network.fc.parameters():
            p.requires_grad = True
        self._log_trainable()

        train_dataset = self.data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                       shuffle=True, num_workers=num_workers)
        test_dataset = self.data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                      shuffle=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # Fold THIS task's just-trained RESIDUAL into SKETCH now, before
        # eval_task() runs -- unconditionally, every task including task 0
        # (task 0's call is exactly the first-fold transplant branch inside
        # _sketch_fold(), just timed one task earlier than the base class).
        self._sketch_fold()
