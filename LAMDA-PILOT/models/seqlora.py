"""SeqLoRA -- the naive sequential-finetuning lower bound.

A *single* LoRA adapter (index 0) is trained on task 0, then continues training
on each subsequent task with no re-initialisation, no regularisation and no
replay.  That one adapter is used for every class at inference.  This is the
classic "sequential fine-tuning" baseline and is expected to forget
catastrophically -- the representation drifts toward the latest task and earlier
classes are mispredicted.

Contrast with the per-task LoRA in ``models/lora.py`` (one adapter per task,
routed at inference), which is the *strong* reference (its TIL accuracy is close
to an upper bound).  SeqLoRA is its single-adapter analog.

Implementation: identical to the per-task learner except the active adapter is
pinned to 0 for both training and evaluation -- so ``freeze_to_task(0)`` keeps
adapter 0 trainable on every task, the forward always routes to adapter 0, and
TIL "routing" collapses to that same single adapter (hence TIL also forgets,
unlike the per-task method).
"""

from models.lora import Learner as LoRALearner


class Learner(LoRALearner):
    def __init__(self, args):
        super().__init__(args)
        # single adapter -> nothing to merge/accumulate
        self.train_merge = False
        self._network.merge = False

    def _train_adapter(self):
        return 0

    def _eval_adapter(self, task):
        return 0
