"""No-adaptation baseline: the pretrained ViT-B/16 backbone, completely frozen.

LoRA slot 0 stays at its zero-init (kaiming A, zero B) no-op state for the
ENTIRE run -- never touched by an optimizer, so ``net(x, task=0, merge=*)`` is
provably ``W @ x`` on every task, bit-identical to the base backbone alone
(same argument models/sketchlora.py's own docstring makes for why a
zero-residual slot is a mathematically exact no-op at eval time). The
classifier head is fit by a single closed-form nearest-class-mean (NCM)
prototype pass over each task's frozen features -- zero backprop, "no
training" in exactly the sense every other method in this project means it
(no adapter weight and no gradient step on the head is ever taken).

Purpose: the floor every LoRA CL method in this project should beat -- how
much of a method's accuracy is just the pretrained backbone's already
near-linearly-separable features (a "free" zero-shot-ish read of the
classification problem), versus real continual adaptation contributed by
training LoRA at all.

Shares models/lora.py's q,v-LoRA scaffold (backbone_type=
vit_base_patch16_224_lora) and inherits its persistent_state/_forward_task/
_deployed_forward/eval routing unchanged (models/seqlora.py's own minimal-diff
style: pin adapter 0, override only what actually differs) -- NOT the
template repo's models/simplecil.py, which predates this LoRA scaffold, uses a
different network class (SimpleVitNet), and isn't wired into this project's
final_metrics/persistent_state/CE-ledger machinery at all. This way the
baseline is directly, harness-comparably plotted against sketchlora/seqlora/
olora/inflora/treelora rather than living in a separate, incompatible
pipeline. The prototype-fit idea itself is SimpleCIL's (models/simplecil.py::
replace_fc), reimplemented here against LoRAVitNet's forward signature.
"""
import logging

import torch

from models.lora import Learner as LoRALearner


class Learner(LoRALearner):
    def __init__(self, args):
        super().__init__(args)
        # single frozen adapter slot -> nothing to merge/accumulate, mirrors
        # models/seqlora.py's own reasoning verbatim (that slot never trains
        # here either, so this is belt-and-suspenders: B stays exactly zero
        # regardless of the merge flag's value).
        self.train_merge = False
        self._network.merge = False

    def _train_adapter(self):
        return 0

    def _eval_adapter(self, task):
        return 0

    def _train(self, train_loader):
        """Replaces the base gradient-descent loop (models/lora.py::_train)
        entirely -- no optimizer is ever constructed. One frozen-feature
        forward pass (no_grad) over this task's training data, then a
        closed-form per-class mean (NCM prototype) written directly into the
        growing head's weight rows for this task's NEW classes. Prior tasks'
        prototype rows are left untouched -- LoRAVitNet.update_fc (utils/
        inc_net.py) copies old weight/bias rows forward into the grown head
        before this runs, the same assumption models/simplecil.py::replace_fc
        already relies on in this codebase."""
        self._network.to(self._device)
        self._network.eval()
        net = self._network.module if hasattr(self._network, "module") else self._network
        feat_list, label_list = [], []
        with torch.no_grad():
            for _, inputs, targets in train_loader:
                inputs = inputs.to(self._device)
                output = net(inputs, task=self._train_adapter(), merge=self.train_merge)
                feat_list.append(output["features"].cpu())
                label_list.append(targets)
        feats = torch.cat(feat_list, dim=0)
        labels = torch.cat(label_list, dim=0)
        seen = torch.unique(labels).tolist()
        for c in seen:
            proto = feats[labels == c].mean(dim=0)
            net.fc.weight.data[c] = proto.to(net.fc.weight.device)
        logging.info("Task {} finished (no-adaptation baseline: {} class prototypes fit, "
                      "0 gradient steps, 0 LoRA/head parameters trained).".format(
                          self._cur_task, len(seen)))
