"""SD-LoRA-Sketch, fold-before-eval variant (2026-11-11 user request).

models/sdlora_sketch.py folds the PREVIOUS task's trained new-adapter
(direction + magnitude + rescale) into the sketch at the START of the
CURRENT task's incremental_train, before that task is trained. Since the
shared trainer loop runs eval_task() immediately after incremental_train()
returns (before after_task()), this means eval for task t currently reads
sketch_term(x) + new_adapter_term(x) where the new-adapter term is task
t's own just-trained, full-rank-equivalent direction+magnitude -- not yet
baked into the sketch's truncated (U, S, Vh). This is a strictly more
informative representation than what persistent_state() counts as the
actual bank (the new-adapter/rescale/magnitude buffers are accounted
separately from the bounded sketch there).

This variant moves the SAME _fold_sketch() call to the END of
incremental_train instead -- right after self._train() returns, for every
task including task 0, unconditionally -- so eval sees the properly
truncated sketch-only state. This matches models/sketchlora.py's own
cadence exactly (its _train() calls self._compress() before returning, so
"Eval ... runs after this returns, before after_task()" always sees the
post-fold sketch). No eval-routing change is needed: backbone/
vit_sdlora_sketch.py::fold()'s own _new_task_params() resets mag to 0 (and
rescale to 0), so new_adapter_term(x) = scale * B(A(x)) with scale =
mag/||BA||_F = 0/... = 0 exactly -- the sum degenerates to sketch_term(x)
alone by the time eval runs.

Only incremental_train is overridden here. _fold_sketch() (the task-
boundary QR-SVD merge + new-adapter/rescale/magnitude reset),
persistent_state(), the regularized training loop, and everything else is
inherited unchanged from models/sdlora_sketch.py::Learner."""
import logging

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from models.sdlora_sketch import Learner as SDLoRASketchLearner
from models.lora import num_workers


class Learner(SDLoRASketchLearner):
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.data_manager = data_manager
        self._network.update_fc(self._total_classes)
        self._register_task_range()
        self._network.default_task = 0
        logging.info("[SD-LoRA-Sketch-FoldFirst] Learning on {}-{}".format(
            self._known_classes, self._total_classes))

        self._network.freeze_to_task(0)  # task arg is ignored, see backbone's freeze_to_task
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

        # Fold THIS task's just-trained new-adapter into the sketch now,
        # before eval_task() runs -- unconditionally, every task including
        # task 0 (the sketch starts at true zero, so fold()'s general path
        # is already exact for the first fold too -- see backbone/
        # vit_sdlora_sketch.py::fold()'s own docstring).
        self._fold_sketch()
