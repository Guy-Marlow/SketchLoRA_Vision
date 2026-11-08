"""CL-LoRA-Sketch: CL-LoRA (models/cllora.py) with SketchLoRA compression on
its specific-pos (deep) layers -- full fork, not a modification of
models/cllora.py, per this project's "new experimental mechanism = isolated
codepath" convention (models/inflora_sketch.py's own stated rule). Cannot be
a thin subclass the way inflora_sketch.py is: CL-LoRA's Learner has no
override seams (its after_task/_init_train/replace_fc/incremental_train
inline the specific-pos adapter-list/block-weight logic directly), unlike
InfLoRA's, which was already written with small hook methods to override.

See backbone/vit_cllora_sketch.py's own module docstring for the full
mechanism (SketchQV's fixed 2-slot fold). Summary of what changes vs.
models/cllora.py:

BLOCK-WEIGHT / ORTHOGONALITY MECHANISM: UNCHANGED, deliberately (2026-11-04
user decision -- "we don't want to introduce SketchLoRA's orthogonality
penalty to CL-LoRA... tell me when block-wise weights are trained"). Same
compute_orthogonality_loss, same (3, n_specific) shape including the K row
(confirmed-intentional: costs nothing since CE loss never reads it, gives
the penalty free dimensionality), same per-step training alongside the CE
loss, same freeze-at-task-end timing. The only new thing is what happens to
a frozen {adapter, block_weight} pair AFTER its task ends: instead of
archiving the adapter into a growing per-task bank, SketchQV.fold scales it
by its own (raw, UNNORMALIZED magnitude -- matching exactly how baseline CL-
LoRA applies this scalar in Attention_lora.forward; 2026-11-04 user
decision: don't deviate from how baseline does it) block-weight scalar and
folds it into the running sketch. block_weight_list is kept only as a
reference anchor for future tasks' orthogonality loss, same role
block_weight_list already plays in baseline.

CLASSIFIER: collapsed to a single plain cosine match (utils/inc_net.py::
OurNetSketch), since there's only one feature block now, not N -- see that
class's own docstring. `use_init_ptm`/`use_diagonal`/`moni_adam`/
`adapter_num`/`recalc_sim` are all dropped (tied to the per-task feature
list / dead-code ablation branches this variant doesn't have).
get_A_B_Ahat/solve_similarity/solve_sim_reset are dropped too -- already
dead code in the original (never called), and now also structurally
meaningless (no per-task fc sub-blocks to reweight).

NOT SUPPORTED (scope limitation, not yet requested): `boundary_mode=="budget"`
streaming mode. This variant only implements ordinary oracle (real task-
boundary) training, matching models/inflora_sketch.py's own initial scope.

Config: same general_pos/specfic_pos/msa/use_distillation/ffn_num keys as
CL-LoRA, plus `svd_rank` (sketch's fixed output rank, defaults to `ffn_num`)
and `merge_op` (must be "qrsvd" if present -- no other SketchLoRA merge_op is
wired in here, same scoping-down models/inflora_sketch.py's own docstring
describes for itself).
"""
import logging

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from utils.inc_net import OurNetSketch
from models.base import BaseLearner
from models.bank_cap_mixin import BankCapMixin
from utils.toolkit import tensor2numpy
from utils.ce2_profiler import ce2_boundary

num_workers = 8


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


def compute_orthogonality_loss(previous_weights_list, current_weights, epsilon=1e-8):
    """Unchanged from models/cllora.py -- see this file's own module
    docstring for why this mechanism is deliberately left untouched."""
    total_ortho_loss = 0.0
    current_norm = torch.norm(current_weights.flatten())
    current_normalized = current_weights.flatten() / (current_norm + epsilon)

    for prev_weights in previous_weights_list:
        prev_norm = torch.norm(prev_weights.flatten())
        prev_normalized = prev_weights.flatten() / (prev_norm + epsilon)
        dot_product = torch.abs(torch.sum(prev_normalized * current_normalized))
        total_ortho_loss += dot_product

    if len(previous_weights_list) > 0:
        total_ortho_loss /= len(previous_weights_list)

    return total_ortho_loss


class Learner(BankCapMixin, BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._bank_cap_init(args)
        self._network = OurNetSketch(args, True)

        self.args = args
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.init_cls = args["init_cls"]
        self.inc = args["increment"]

        self.svd_rank = args.get("svd_rank", args["ffn_num"])
        merge_op = args.get("merge_op", "qrsvd")
        assert merge_op == "qrsvd", \
            "CL-LoRA-Sketch only implements merge_op='qrsvd' (exact QR-SVD fold) -- " \
            "got merge_op={}".format(merge_op)

        # Three variants, same file (see backbone/vit_cllora_sketch.py's own
        # module docstring, "THREE VARIANTS"):
        self.use_block_weight_scaling = args.get("use_block_weight_scaling", True)
        self.orth_a_side = args.get("orth_a_side", False)
        self.sketch_orth_weight = args.get("sketchlora_align_weight", 0.5)
        assert self.use_block_weight_scaling or self.orth_a_side, \
            "at least one of use_block_weight_scaling/orth_a_side must be set -- with both " \
            "off this degrades to a bare, unregularized sketch fold, which is mechanically " \
            "available but was never one of the three requested variants."

    def after_task(self):
        self._known_classes = self._total_classes
        net = self._network.module if hasattr(self._network, "module") else self._network
        net.freeze()
        # Single combined admission (not two independent ones) covering both
        # structures that grow per task (block_weight_list, old_adapter_list)
        # -- the sketch itself is fixed-size and never gated (see
        # backbone/vit_cllora_sketch.py::VisionTransformer.after_task_fold's
        # own docstring). Matches this project's "simplest gate" principle
        # (bank_cap_mixin.py's own docstring) since, unlike baseline CL-
        # LoRA's three parallel-indexed structures, nothing here actually
        # requires these two to grow/freeze in lockstep -- a single combined
        # check is just simpler, not a correctness requirement.
        candidate_bytes = self._bank_admit_candidate_bytes()
        admit = self._bank_admit(candidate_bytes)
        with ce2_boundary(self):
            net.backbone.after_task_fold(admit_bw=admit, admit_kd=admit)

    def _bank_admit_candidate_bytes(self):
        net = self._network.module if hasattr(self._network, "module") else self._network
        backbone = net.backbone
        bw_bytes = (backbone.block_weight.numel() * backbone.block_weight.element_size()
                    if backbone.use_block_weight_scaling else 0)
        kd_bytes = (sum(p.numel() * p.element_size() for p in backbone.cur_adapter.parameters())
                    if backbone.use_distillation else 0)
        return bw_bytes + kd_bytes

    @staticmethod
    def _sketch_bytes(backbone):
        total = 0
        for sk in backbone.sketch_list:
            for mod in (sk.sketch_q, sk.sketch_v):
                for p in mod.parameters():
                    total += p.numel() * p.element_size()
        return total

    @staticmethod
    def _residual_bytes(backbone):
        total = 0
        for sk in backbone.sketch_list:
            for mod in (sk.residual_q, sk.residual_v):
                for p in mod.parameters():
                    total += p.numel() * p.element_size()
        return total

    def persistent_state(self):
        net = self._network.module if hasattr(self._network, "module") else self._network
        backbone = net.backbone
        sketch_bytes = self._sketch_bytes(backbone)
        residual_bytes = self._residual_bytes(backbone)
        bw_live_bytes = (backbone.block_weight.numel() * backbone.block_weight.element_size()
                         if backbone.use_block_weight_scaling else 0)
        bw_list_bytes = sum(t.numel() * t.element_size() for t in backbone.block_weight_list)
        general_adapter_bytes = sum(p.numel() * p.element_size() for p in backbone.cur_adapter.parameters())
        old_adapter_bytes = (sum(p.numel() * p.element_size() for p in backbone.old_adapter_list.parameters())
                              if backbone.use_distillation else 0)
        fc_bytes = sum(p.numel() * p.element_size() for p in net.fc.parameters()) if net.fc is not None else 0
        total_bytes = (sketch_bytes + residual_bytes + bw_live_bytes + bw_list_bytes
                       + general_adapter_bytes + old_adapter_bytes + fc_bytes)
        return {"params": int(total_bytes // 4), "bytes": int(total_bytes),
                "breakdown": {"sketch": sketch_bytes, "residual_live": residual_bytes,
                             "block_weight_live": bw_live_bytes, "block_weight_list": bw_list_bytes,
                             "general_adapter": general_adapter_bytes, "old_adapters_kd": old_adapter_bytes,
                             "fc": fc_bytes}}

    def _bank_bytes(self):
        """BankCapMixin hook: sketch (fixed-size, counted here for the same
        reason models/inflora_sketch.py counts its own SKETCH slot despite
        it never growing -- bank_bytes is "current total," not "current
        growth") + block_weight_list + old_adapter_list (the two structures
        that genuinely grow per task). Excludes residual/general_adapter
        (live, continuously-training slots) and fc, same convention as
        every other method in this project."""
        net = self._network.module if hasattr(self._network, "module") else self._network
        backbone = net.backbone
        sketch_bytes = self._sketch_bytes(backbone)
        bw_list_bytes = sum(t.numel() * t.element_size() for t in backbone.block_weight_list)
        old_adapter_bytes = (sum(p.numel() * p.element_size() for p in backbone.old_adapter_list.parameters())
                              if backbone.use_distillation else 0)
        return sketch_bytes + bw_list_bytes + old_adapter_bytes

    @torch.no_grad()
    def _deployed_forward(self, inputs):
        net = self._network.module if hasattr(self._network, "module") else self._network
        return net.forward(inputs, test=True)

    def replace_fc(self, train_loader):
        """Single-pass NCM prototype write -- there's only one feature block
        now (see OurNetSketch's own docstring), unlike baseline CL-LoRA's
        per-adapter-index loop."""
        model = self._network
        model = model.eval()
        net = model.module if hasattr(model, "module") else model

        with torch.no_grad():
            embedding_list, label_list = [], []
            for _, data, label in train_loader:
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = net.backbone(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
            embedding_list = torch.cat(embedding_list, dim=0)
            label_list = torch.cat(label_list, dim=0)

            class_list = np.unique(self.train_dataset_for_protonet.labels)
            for class_index in class_list:
                data_index = (label_list == class_index).nonzero().squeeze(-1)
                proto = embedding_list[data_index].mean(0)
                model.fc.weight.data[class_index, :] = proto

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        self.data_manager = data_manager
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        self.test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        self.train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="test")
        self.train_loader_for_protonet = DataLoader(self.train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._network.add_fc()
        with ce2_boundary(self):
            self.replace_fc(self.train_loader_for_protonet)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        if self._cur_task == 0 or self.init_cls == self.inc:
            optimizer = self.get_optimizer(lr=self.args["init_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["init_epochs"])
        else:
            if "later_lr" not in self.args or self.args["later_lr"] == 0:
                self.args["later_lr"] = self.args["init_lr"]
            if "later_epochs" not in self.args or self.args["later_epochs"] == 0:
                self.args["later_epochs"] = self.args["init_epochs"]

            optimizer = self.get_optimizer(lr=self.args["later_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["later_epochs"])

        self._init_train(train_loader, test_loader, optimizer, scheduler)

    def get_optimizer(self, lr):
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9, lr=lr, weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr, weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr, weight_decay=self.weight_decay)
        return optimizer

    def get_scheduler(self, optimizer, epoch):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch, eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None
        return scheduler

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        if self._cur_task == 0 or self.init_cls == self.inc:
            epochs = self.args['init_epochs']
        else:
            epochs = self.args['later_epochs']

        prog_bar = tqdm(range(epochs))

        for _, epoch in enumerate(prog_bar):
            self._network.train()

            losses = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                lo, hi = self._known_classes, self._total_classes
                aux_targets = targets.clone()
                aux_targets = torch.where((aux_targets >= lo) & (aux_targets < hi), aux_targets - lo, -1)

                # KD phase (general-pos only, unchanged from models/cllora.py)
                # -- own backward+step BEFORE the CE/orth forward below, same
                # ordering fix the original needed (optimizer.step() here
                # updates shared general-pos params in-place, so computing
                # the CE forward beforehand would hit a stale-graph autograd
                # error the moment cur_task>0).
                if self._cur_task > 0 and len(self._network.backbone.old_adapter_list) > 0:
                    kd_ratio = 5.
                    Temperature = 2

                    out_new, out_teacher = self._network.forward_kd(inputs, self._cur_task)
                    out_new_logits = out_new["logits"]
                    out_teacher_logits = out_teacher["logits"]
                    loss_kd = kd_ratio * _KD_loss(out_new_logits, out_teacher_logits, T=Temperature)

                    optimizer.zero_grad()
                    loss_kd.backward()

                    for j in range(len(self._network.backbone.general_pos)):
                        pos = self._network.backbone.adapt_pos.index(self._network.backbone.general_pos[j])
                        for jj in range(len(self._network.backbone.msa)):
                            if self._network.backbone.msa[jj] == 1:
                                temp_weights = 1. * torch.norm(self._network.backbone.old_adapter_list[-1][pos][jj].lora_A.weight, dim=1)
                                temp_weights = 1. * len(temp_weights) * temp_weights / torch.sum(temp_weights)
                                self._network.backbone.cur_adapter[pos][jj].lora_A.weight.grad = temp_weights.unsqueeze(1) * self._network.backbone.cur_adapter[pos][jj].lora_A.weight.grad
                    optimizer.step()

                output = self._network(inputs, test=False)
                logits = output["logits"]
                loss = F.cross_entropy(logits, aux_targets, ignore_index=-1)

                # Block-weight orthogonality penalty -- UNCHANGED from
                # models/cllora.py (2026-11-04 user decision: keep exactly
                # as baseline, do not replace with anything SketchLoRA-
                # specific). Only what happens to block_weight AFTER this
                # task ends differs (see backbone/vit_cllora_sketch.py).
                # Absent entirely (not merely skipped) when
                # use_block_weight_scaling=False -- variant 3.
                if self._cur_task > 0 and self.use_block_weight_scaling:
                    orth_loss_specific = compute_orthogonality_loss(
                        self._network.backbone.block_weight_list, self._network.backbone.block_weight)
                    loss += 0.0001 * orth_loss_specific

                # "orth_a_side" -- the orthogonality penalty already
                # established as part of SketchLoRA (models/sketchlora_align.py
                # ::_orth_loss, itself O-LoRA's _orth_and_l2 orth term
                # verbatim), applied between each specific-pos layer's live
                # RESIDUAL and its frozen SKETCH down-projection. Independent
                # of (and combinable with) the block-weight penalty above --
                # see backbone/vit_cllora_sketch.py's module docstring.
                if self._cur_task > 0 and self.orth_a_side:
                    loss += self.sketch_orth_weight * self._network.backbone.sketch_orth_loss()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)

                correct += preds.eq(aux_targets.expand_as(preds)).cpu().sum()
                total += len(aux_targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)

        logging.info(info)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model.forward(inputs, test=True)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        k = min(self.topk, self._total_classes)
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network.forward(inputs, test=True)["logits"]
            predicts = torch.topk(outputs, k=k, dim=1, largest=True, sorted=True)[1]
            if k < self.topk:
                pad = predicts[:, -1:].expand(-1, self.topk - k)
                predicts = torch.cat([predicts, pad], dim=1)
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)
