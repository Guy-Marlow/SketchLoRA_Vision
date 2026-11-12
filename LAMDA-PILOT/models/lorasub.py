"""LoRA Subtraction for Drift-Resistant Space (CVPR'25) -- see
LoRA_Subtraction_Ref/ (the official repo, read in full before writing this
file: methods/lorasub_drs.py, models/vit_lora.py, models/sinet_lora.py,
optimgrad/adam_proj.py).

WHAT THE REFERENCE DOES
------------------------
Per-task LoRA bank on the K and V attention projections only (Q stays
plain) -- one (A,B) pair per task, PREALLOCATED for the whole run, summed
cumulatively at forward time across every task seen so far
(``weight_k = sum_{t=0}^{task} B_k[t] @ A_k[t]``, same convention this
project's O-LoRA/InfLoRA already use for Q/V, just on K/V here). Training a
new task freezes every past slot and trains ONLY the current task's own
(A,B) -- unlike InfLoRA, A is NOT set analytically; both A and B are
gradient-trained.

The namesake "subtraction" trick: before training task t>0, a probe
forward pass computes the per-block input covariance (``cur_matrix``) with
every past task's K/V contribution subtracted back out (see
backbone/vit_lorasub.py's docstring for the exact mechanism and why it
reduces to a plain frozen-base forward given zero-B-init). That "cleaned"
covariance feeds an InfLoRA-style SVD subspace, used not to set A
analytically but to PROJECT the Adam update applied to A and B each step
(``optimgrad/adam_proj.py::Adam`` -- ported below as ``_ProjectedAdam``,
modernized for current PyTorch in-place-op signatures but otherwise the
same math): keep the top eigenvectors reaching ``svd_thres`` cumulative
variance of the cleaned covariance, build ``transform = basis @ basis.T``,
and project each step's Adam update onto that subspace before applying it.
This constrains the new task's adapter to move only within the directions
that matter most for the UNADAPTED (drift-free) feature distribution --
the actual "drift-resistant space" of the paper's title.

WHAT THIS PORT SIMPLIFIES, AND WHY
------------------------------------
The reference additionally trains a per-task linear head (CE signal only)
and evaluates via NCM against class-mean prototypes plus an auxiliary
triplet loss pulling new features away from old prototypes
(utils/losses.py::AugmentedTripletLoss). This file reuses this project's
standard CIL training/eval path (models/lora.py's shared head +
task-local cross-entropy, growing fc, merge=True argmax at eval) rather
than switching to NCM -- but see PROTOTYPE REPULSION below: the triplet
loss's core effect (pull new-task features away from past-task classes)
is NOT dropped, because it turned out to matter a lot more than the
module docstring originally claimed.

PROTOTYPE REPULSION (added after an initial DRS-only port collapsed badly)
----------------------------------------------------------------------------
An initial version of this file shipped WITHOUT any cross-task feature
repulsion and collapsed hard under CIL eval on ImageNet-R (5 tasks, 10
epochs: 87.1 -> 43.4 -> 24.6 -> ... while per-task train accuracy stayed
healthy, 88 -> 80 -> 75). The diagnosis: unlike O-LoRA (an explicit
orthogonality LOSS between the new task's A and every past task's frozen
A) and InfLoRA (DualGPM's `feature_list`/`feature_mat`, an explicit basis
that ACCUMULATES across every past task and is used to place the new
task's analytic A outside/within a controlled relationship to it), the
DRS projection here (_ProjectedAdam, below) is rebuilt FRESH every task
from that task's OWN data only (self._fea_in is populated and dropped
every task -- no persistent cross-task structure, unlike InfLoRA's
feature_list). It constrains the new adapter to the high-variance INPUT
directions of the current task -- useful for learning that task well, but
nothing about avoiding directions past tasks' adapters already occupy. Two
tasks' (A,B) can land on the same dominant directions and distort each
other in the additive sum Σ B_k[t]@A_k[t], even though each frozen slot is
individually untouched. CL-LoRA avoids this by hard-partitioning (no
summation to interfere); O-LoRA and InfLoRA avoid it with the two
cross-task mechanisms above. This file had none.

The reference's OWN fix for this (in its benchmark, not something this
file had ported) is exactly AugmentedTripletLoss pulling new features away
from `self._protos` -- past-task class-mean prototypes. That IS a
cross-task mechanism, unlike DRS. Added here as a lightweight version:
after each task finishes training, store each of its classes' mean
(normalized) feature as a prototype (`_update_protos`); while training a
LATER task, add a margin hinge loss pushing every training sample's
feature at least `proto_margin` away (Euclidean, on the unit hypersphere)
from its NEAREST past-task prototype, weighted by
`proto_repulsion_weight`. Simpler than the reference's full triplet loss
(no in-batch hardest-positive/negative mining, just repulsion from past
prototypes -- the piece that actually ties training to past-task state)
but ports its central cross-task idea. Config: `use_proto_repulsion`
(default True), `proto_repulsion_weight` (default 0.2, reference's own
`lambada`), `proto_margin` (default 2.0, reference's own `margin_inter`).
Gated off entirely at task 0 (no past prototypes exist yet) and via
`use_proto_repulsion: false` for an ablation back to DRS-only.

This mechanism lives entirely in this shared base class's _train/
_update_protos -- it operates on features and labels only, never touches
the backbone's adapter-bank structure, so it applies unchanged to
models/lorasub_sketch.py (which inherits _train without overriding it)
regardless of whether the bank is the growing per-task kind (this file) or
the fixed 2-slot SketchLoRA-merge kind.

Also simplified: no BankCapMixin / byte-cap gating (explicitly not wanted
for this method). Memory accounting is still tracked and logged every task
(see persistent_state()/_log_memory_accounting()) -- it just never refuses
to grow.

Hyperparameters match this project's convention (tuned_epoch=20,
batch_size=48, lora_rank=10 -- see exps/lorasub/*.json) rather than the
reference's own schedule (its configs use epochs=50, batch_size=128,
init_lr=5e-4 Adam). The optimizer stays Adam (not this project's usual
AdamW/SGD) because the projection math in _ProjectedAdam is derived against
plain Adam's EMA-based update -- that part IS kept faithful to the
reference, per instruction to take the projection code from LoRA
Subtraction's own implementation rather than reusing InfLoRA's (different,
analytic-A) DualGPM pattern.
"""

import logging
import math
from collections import defaultdict

import torch
from torch.nn import functional as F
from tqdm import tqdm

from models.lora import Learner as LoRALearner
from utils.toolkit import tensor2numpy
from utils.rank_diagnostics import log_bank_effective_rank


class _ProjectedAdam(torch.optim.Optimizer):
    """Adam with an optional post-hoc projection of each step's update onto
    a per-parameter subspace. Ported from
    LoRA_Subtraction_Ref/optimgrad/adam_proj.py::Adam -- same math (plain
    Adam's EMA update, then ``update = transform @ update`` or
    ``update @ transform`` depending on which side of `transform`'s square
    shape matches `update`'s, letting ONE [dim,dim] transform apply
    correctly to both A's [rank,dim] and B's [dim,rank] weight, since the
    transform always lives on the `dim` axis both share with the block's
    covariance matrix), modernized for current in-place-op signatures
    (the reference's ``add_(value, tensor)`` form is long removed from
    PyTorch).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, svd=False,
                 thres=1.001, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        svd=svd, thres=thres)
        super().__init__(params, defaults)
        self.eigens = defaultdict(dict)
        self.transforms = {}

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("svd", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            svd = group["svd"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                update = self._get_update(group, p.grad, p)
                if svd and p in self.transforms:
                    transform = self.transforms[p]
                    if transform.shape[0] == update.shape[0]:
                        update = transform @ update
                    else:
                        update = update @ transform
                p.add_(update)
        return loss

    def _get_update(self, group, grad, p):
        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        beta1, beta2 = group["betas"]
        state["step"] += 1

        if group["weight_decay"] != 0:
            grad = grad.add(p, alpha=group["weight_decay"])

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        denom = exp_avg_sq.sqrt().add_(group["eps"])

        bias_correction1 = 1 - beta1 ** state["step"]
        bias_correction2 = 1 - beta2 ** state["step"]
        step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1
        return -step_size * exp_avg / denom

    # -- DRS subspace construction, called once per task>0 before training --
    def get_eigens(self, fea_in):
        """fea_in: {parameter: covariance_matrix}. The SAME [dim,dim]
        covariance is shared across all four of an attention module's
        lora_A_k/lora_B_k/lora_A_v/lora_B_v (one covariance per block, not
        per matrix -- matches the reference exactly)."""
        for group in self.param_groups:
            if not group["svd"]:
                continue
            for p in group["params"]:
                if not p.requires_grad or p not in fea_in:
                    continue
                # symmetric PSD covariance -> U is the eigenbasis (same
                # convention models/inflora.py already uses for this exact
                # kind of matrix).
                U, S, _ = torch.linalg.svd(fea_in[p].double())
                self.eigens[p] = {"eigen_value": S.float(), "eigen_vector": U.float()}

    def get_transforms(self):
        for group in self.param_groups:
            if not group["svd"]:
                continue
            thres = group["thres"]
            for p in group["params"]:
                if not p.requires_grad or thres == 1.0 or p not in self.eigens:
                    continue
                eigen_value = self.eigens[p]["eigen_value"]
                eigen_vector = self.eigens[p]["eigen_vector"]
                cumulative = eigen_value.cumsum(dim=0) / eigen_value.sum()
                num_vectors = int((cumulative >= thres).nonzero(as_tuple=True)[0][0].item()) + 1
                basis = eigen_vector[:, :num_vectors]
                transform = basis @ basis.t()
                self.transforms[p] = transform / torch.norm(transform)
        total = sum(1 for g in self.param_groups if g["svd"] for _ in g["params"])
        logging.debug("[LoRA-Sub] DRS transforms built for {} parameter(s)".format(total))


class Learner(LoRALearner):
    def __init__(self, args):
        super().__init__(args)
        # LoRA Subtraction always forwards with merge=True (sum over every
        # allocated task, current included) -- both for training AND eval;
        # this is not optional/configurable the way O-LoRA's
        # lora_train_merge is, so override unconditionally rather than
        # reading the config key.
        self.train_merge = True
        self.svd_thres = args.get("svd_thres", 0.99)
        self.adam_betas = tuple(args.get("adam_betas", (0.9, 0.999)))
        self.adam_eps = args.get("adam_eps", 1e-8)
        # {parameter: covariance_matrix}, rebuilt fresh each task>0 and
        # dropped immediately after feeding _ProjectedAdam -- transient
        # working memory, never counted in persistent_state().
        self._fea_in = {}
        # Prototype repulsion (see module docstring) -- {class_idx: unit-norm
        # mean feature}, grows by this task's classes at the end of every
        # _train() call. CPU tensors; moved to device only when building the
        # per-task protos_tensor used in the loss.
        self.use_proto_repulsion = bool(args.get("use_proto_repulsion", True))
        self.proto_repulsion_weight = args.get("proto_repulsion_weight", 0.2)
        self.proto_margin = args.get("proto_margin", 2.0)
        self._protos = {}
        # 2026-11-05 user request: summed-bank effective-rank diagnostic
        # (utils/rank_diagnostics.py) -- opt-in, see _log_bank_effective_rank.
        self.log_bank_rank = bool(args.get("log_bank_rank", False))
        self.bank_rank_energy_target = args.get("bank_rank_energy_target", 0.99)

    # -- DRS covariance collection (the "subtraction" probe pass) -------
    @torch.no_grad()
    def _collect_subtracted_covariance(self, train_loader):
        net = self._network
        net.eval()
        net.reset_cur_matrix()
        net.set_collect(True)
        net.set_subtract_mode(True)
        task = self._train_adapter()
        for _, inputs, _t in train_loader:
            net(inputs.to(self._device), task=task, merge=True)
        net.set_collect(False)
        net.set_subtract_mode(False)

        for attn in net.attn_modules():
            cov = attn.cur_matrix.clone()
            self._fea_in[attn.lora_A_k[task].weight] = cov
            self._fea_in[attn.lora_B_k[task].weight] = cov
            self._fea_in[attn.lora_A_v[task].weight] = cov
            self._fea_in[attn.lora_B_v[task].weight] = cov
            attn.reset_cur_matrix()

    # -- prototype repulsion (see module docstring) ----------------------
    @torch.no_grad()
    def _update_protos(self, train_loader):
        """Store the RAW (un-normalized) mean feature per class for the
        JUST-FINISHED task, using the final trained adapter state. Called at
        the end of _train(), after the epoch loop -- these become available
        for the NEXT task's repulsion term, never this task's own
        (current-task samples aren't pushed away from their own classes).

        Matches LoRA_Subtraction_Ref/methods/lorasub_drs.py::_build_protos
        exactly: `class_mean = np.mean(vectors, axis=0)` over RAW feature
        vectors, normalized only lazily at use time (see where
        protos_tensor is built in _train(), below) -- NOT normalizing each
        sample before averaging. A prior version of this file normalized
        every sample first, then averaged, then normalized the mean again;
        when per-sample feature norms vary within a class, that changes
        which samples dominate the resulting direction relative to the
        reference's plain-mean-then-normalize-once convention. Fixed
        2026-11 to match."""
        net = self._network
        net.eval()
        adapter = self._train_adapter()
        feats_by_class = defaultdict(list)
        for _, inputs, targets in train_loader:
            feats = net.extract_vector(inputs.to(self._device), task=adapter, merge=True).cpu()
            for f, y in zip(feats, targets):
                feats_by_class[int(y.item())].append(f)
        for cls, flist in feats_by_class.items():
            self._protos[cls] = torch.stack(flist, dim=0).mean(dim=0)

    def _proto_repulsion_loss(self, features, protos_tensor):
        """features: [B, D] (raw, un-normalized, straight from the
        forward pass). protos_tensor: [P, D], unit-norm, past-task classes
        only. Margin hinge pushing each sample at least proto_margin away
        (Euclidean, on the unit hypersphere) from its nearest past
        prototype."""
        feat_norm = F.normalize(features, dim=-1)
        dists = torch.cdist(feat_norm, protos_tensor)
        min_dist, _ = dists.min(dim=1)
        return F.relu(self.proto_margin - min_dist).mean()

    def _optimizer_param_groups_split(self):
        net = self._network
        task = self._train_adapter()
        feature_params = []
        for attn in net.attn_modules():
            for mlist in (attn.lora_A_k, attn.lora_B_k, attn.lora_A_v, attn.lora_B_v):
                feature_params += [p for p in mlist[task].parameters() if p.requires_grad]
        head_params = [p for p in net.fc.parameters() if p.requires_grad]
        return feature_params, head_params

    def _train(self, train_loader):
        net = self._network
        net.to(self._device)
        # `task` is the REAL task counter (0,1,2,... -- used only to gate the
        # DRS probe/projection setup on "not the first task", and to label
        # logging). `adapter` is the BACKBONE SLOT to route the forward pass
        # to and to index lora_A_k[...]/etc with -- these differ for
        # models/lorasub_sketch.py, which pins adapter to the fixed RESIDUAL
        # slot while task keeps counting real tasks. A previous version of
        # this method conflated the two (used `task` for routing), which was
        # a silent no-op for this class (task == _train_adapter() always,
        # here) but broke the sketch subclass: task 0 never trained RESIDUAL
        # (merge=True with task=0 only summed slot 0), task 1 "worked" only
        # by coincidence (RESIDUAL's index happens to be 1), and task 2
        # crashed indexing a nonexistent 3rd slot.
        task = self._cur_task
        adapter = self._train_adapter()

        if task > 0:
            self._collect_subtracted_covariance(train_loader)

        feature_params, head_params = self._optimizer_param_groups_split()
        optimizer = _ProjectedAdam(
            [{"params": feature_params, "svd": True, "thres": self.svd_thres, "lr": self.init_lr},
             {"params": head_params, "svd": False, "lr": self.init_lr}],
            betas=self.adam_betas, eps=self.adam_eps, weight_decay=self.weight_decay)

        if task > 0:
            optimizer.get_eigens(self._fea_in)
            optimizer.get_transforms()
            self._fea_in = {}  # transient cache, dropped immediately after use

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.min_lr) if self.lr_anneal else None

        # Past-task prototypes for THIS task's repulsion term -- fixed for
        # the whole task (built once here, not re-read from self._protos
        # every batch). None/empty at task 0 (nothing to repel from yet).
        # self._protos holds RAW means (see _update_protos) -- normalized
        # here, once, at use time, matching the reference's own
        # eval_task()/_build_protos lazy-normalization convention.
        use_proto = self.use_proto_repulsion and task > 0 and len(self._protos) > 0
        protos_tensor = (F.normalize(torch.stack(list(self._protos.values()), dim=0), dim=-1).to(self._device)
                          if use_proto else None)

        lo, hi = self._ce_slice()
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            net.train()
            losses, proto_losses, correct, total = 0.0, 0.0, 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                ret = net(inputs, task=adapter, merge=True)
                logits, features = ret["logits"], ret["features"]
                local_logits = logits[:, lo:hi]
                local_targets = targets - lo
                loss = F.cross_entropy(local_logits, local_targets)
                if use_proto:
                    proto_loss = self._proto_repulsion_loss(features, protos_tensor)
                    loss = loss + self.proto_repulsion_weight * proto_loss
                    proto_losses += proto_loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                preds = local_logits.argmax(dim=1)
                correct += preds.eq(local_targets).cpu().sum()
                total += len(targets)
            if scheduler is not None:
                scheduler.step()
            train_acc = tensor2numpy(correct) * 100 / total if total else 0.0
            prog_bar.set_description(
                "Task {}, Epoch {}/{} => Loss {:.3f} (proto {:.3f}), Train_accy {:.2f}".format(
                    task, epoch + 1, self.epochs, losses / len(train_loader),
                    proto_losses / len(train_loader), train_acc))
        logging.info("[LoRA-Sub] Task {} finished. Train_accy {:.2f}".format(task, train_acc))
        self._update_protos(train_loader)
        self._log_memory_accounting()
        self._log_bank_effective_rank()

    # -- memory accounting (the explicit focus for this method) ---------
    def persistent_state(self):
        """K+V adapter bank -- every allocated slot's A+B, all blocks,
        regardless of trainability (never freed once allocated -- same
        "raw allocation" convention models/lora.py already uses for Q/V),
        plus the classifier head AND the prototype-repulsion store
        (self._protos -- one 768-dim vector per class ever seen, never
        freed or compressed, same qualitative unbounded growth as the
        adapter bank itself -- see module docstring). The transient DRS
        covariance cache (self._fea_in, and _ProjectedAdam's own eigens/
        transforms dicts) is deliberately NOT counted here -- it is working
        memory rebuilt fresh each task and dropped immediately after use,
        not part of what persists across the run."""
        net = self._network.module if hasattr(self._network, "module") else self._network
        total_params, total_bytes = 0, 0
        for attn in net.attn_modules():
            for mlist in (attn.lora_A_k, attn.lora_B_k, attn.lora_A_v, attn.lora_B_v):
                for m in mlist:
                    for p in m.parameters():
                        total_params += p.numel()
                        total_bytes += p.numel() * p.element_size()
        lora_bytes = total_bytes
        head_params = sum(p.numel() for p in net.fc.parameters())
        head_bytes = sum(p.numel() * p.element_size() for p in net.fc.parameters())
        proto_params = sum(p.numel() for p in self._protos.values())
        proto_bytes = sum(p.numel() * p.element_size() for p in self._protos.values())
        total_params += head_params + proto_params
        total_bytes += head_bytes + proto_bytes
        return {"params": total_params, "bytes": total_bytes,
                "breakdown": {"lora_slots": lora_bytes, "head": head_bytes, "protos": proto_bytes}}

    def _log_memory_accounting(self):
        net = self._network.module if hasattr(self._network, "module") else self._network
        state = self.persistent_state()
        logging.info(
            "[LoRA-Sub] task {} persistent bank: {} slot(s) allocated (K+V, A+B, every "
            "block) = {:,} bytes (adapters={:,}, head={:,}, protos={:,} [{} classes])".format(
                self._cur_task, net.backbone.n_tasks, state["bytes"],
                state["breakdown"]["lora_slots"], state["breakdown"]["head"],
                state["breakdown"]["protos"], len(self._protos)))

    @torch.no_grad()
    def _log_bank_effective_rank(self):
        """2026-11-05 user request: the summed bank's TRUE combined rank
        (not each slot's own fixed lora_rank). Forms the dense delta =
        sum_{s=0}^{t} B_s @ A_s per (block, K/V) -- the exact uncompressed
        update merge=True's forward applies -- then SVDs it for its
        effective rank. t = self._train_adapter() (the growing per-task
        slot index here; models/lorasub_sketch.py overrides this method to
        a no-op since "summed bank rank" isn't a meaningful question for its
        fixed 2-slot SKETCH+RESIDUAL design -- see that file's override).
        No-op unless log_bank_rank is set."""
        if not self.log_bank_rank:
            return
        net = self._network.module if hasattr(self._network, "module") else self._network
        t = self._train_adapter()
        deltas = {}
        for li, attn in enumerate(net.attn_modules()):
            for proj, (A_list, B_list) in (("k", (attn.lora_A_k, attn.lora_B_k)),
                                            ("v", (attn.lora_A_v, attn.lora_B_v))):
                B_cat = torch.cat([B_list[s].weight for s in range(t + 1)], dim=1).float()
                A_cat = torch.cat([A_list[s].weight for s in range(t + 1)], dim=0).float()
                deltas[(li, proj)] = B_cat @ A_cat
        log_bank_effective_rank("LoRA-Sub", self._cur_task, deltas, self.bank_rank_energy_target)
