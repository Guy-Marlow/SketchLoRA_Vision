"""CL-LoRA-Sketch, fold-before-eval variant (2026-11-11 user request).

models/cllora_sketch.py runs its specific-pos fold (net.backbone.
after_task_fold, which folds each SketchQV's RESIDUAL into SKETCH, archives
block_weight/old_adapter_list under bank-cap admission, and resets
RESIDUAL/block_weight for the next task) inside after_task(), which the
shared trainer loop calls AFTER eval_task() -- see models/cllora_sketch.py's
own incremental_train/after_task split. So eval for task t currently reads
sketch_q/v(x) + block_weight * residual_q/v(x) where residual/block_weight
are task t's own just-trained, not-yet-folded values -- a strictly more
informative representation than what persistent_state()/_bank_bytes() count
as the actual bank (residual is excluded there as "live, not-yet-folded").

This variant moves the ENTIRE fold block (net.freeze() + the bank-cap
admission decision + after_task_fold() itself) to the END of
incremental_train instead -- right after self._train() returns, for every
task including task 0, BEFORE add_fc()/replace_fc() -- so both eval and the
NCM prototype computation read the properly folded, post-fold state. This
matches models/sketchlora.py's own cadence (compress() runs before
classifier_alignment, both before incremental_train returns) and is safe
with no eval-routing change needed: backbone/vit_cllora_sketch.py::
SketchQV.fold() replaces RESIDUAL with a fresh Adapter_lora whose lora_A is
zero-initialized (init_option="lora"), so residual_q/v(x) is exactly 0
immediately after fold -- the block_weight reset to fresh random values is
therefore also harmless (any scalar times a zero adapter is still zero).
after_task() itself now only retains the known-classes bookkeeping that has
nothing to do with fold timing.

Only incremental_train/after_task are overridden here. after_task_fold()
itself, persistent_state()/_bank_bytes(), replace_fc(), the training loop,
and everything else is inherited unchanged from
models/cllora_sketch.py::Learner."""
import logging

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from models.cllora_sketch import Learner as CLLoRASketchLearner, num_workers
from utils.ce2_profiler import ce2_boundary


class Learner(CLLoRASketchLearner):
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        self.data_manager = data_manager
        self.train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                       shuffle=True, num_workers=num_workers)

        self.test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size,
                                      shuffle=False, num_workers=num_workers)

        self.train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="test")
        self.train_loader_for_protonet = DataLoader(self.train_dataset_for_protonet, batch_size=self.batch_size,
                                                     shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # Fold THIS task's just-trained specific-pos sketch now, before
        # eval_task() runs -- moved here verbatim from
        # models/cllora_sketch.py::after_task(), which ran it AFTER eval.
        net = self._network.module if hasattr(self._network, "module") else self._network
        net.freeze()
        candidate_bytes = self._bank_admit_candidate_bytes()
        admit = self._bank_admit(candidate_bytes)
        with ce2_boundary(self):
            net.backbone.after_task_fold(admit_bw=admit, admit_kd=admit)

        self._network.add_fc()
        with ce2_boundary(self):
            self.replace_fc(self.train_loader_for_protonet)

    def after_task(self):
        # Fold already happened in incremental_train (above), before eval --
        # only the known-classes bookkeeping remains here.
        self._known_classes = self._total_classes
