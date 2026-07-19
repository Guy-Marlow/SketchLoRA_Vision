"""HiDeLoRA (Hierarchical Decomposition, LoRA instantiation) for LAMDA-PILOT.

Port of HiDeLoRA/engines/hide_lora_wtp_and_tap_engine.py onto the shared per-task
q/v LoRA scaffold (backbone/vit_lora.py; the reference uses k/v -- same convention
shift already used for InfLoRA, models/inflora.py). Verified against the actual
launch script (HiDeLoRA/training_scripts/train_imr_lora.sh), not just the configs
-- lora_momentum=0.1, reg=0.001, crct_epochs=30, ca_lr=0.005,
ca_storage_efficient_method=multi-centroid (n_centroids=10) are the repo's real
values, not guesses.

Three stages per task t:
  1. WTP (within-task prediction): train task-routed adapter slot t (merge=False,
     inherited task-local CE convention) PLUS a contrastive feature-separation term
     `orth_loss` (engine.py:472) -- despite the name, and despite accepting a
     `targets` argument, the reference's orth_loss does NOT use class labels: it
     is a label-agnostic instance-discrimination / prototype-separation loss over
     (stored class-centroid means so far) union (this batch's features), pushing
     every row's self-similarity to dominate its similarity to every other row.
     Ported here verbatim (targets accepted, unused) -- not a simplification on
     our part, that is what the reference actually runs.
  2. Warm start (before training) + momentum (after): slot t's A/B are initialised
     from slot t-1, and after training blended `(1-m)*W_t + m*mean(W_0..t-1)`.
  3. Class statistics: per-class multi-centroid (K-means, n_centroids=10) mean+var
     computed with the task-routed adapter's features.
  4. TAP (task-adaptive prediction, t>0 only): retrain ONLY the head on Gaussian
     samples drawn from every seen class's stored centroid statistics.

Inference is the one genuinely new piece needed beyond the shared scaffold: HiDe
predicts the task id PER SAMPLE (un-adapted backbone forward, task=-1, which our
scaffold already gives for free -- Attention_LoRA._lora_delta returns 0 for
task<0, vit_lora.py:97-98 -- so no second model copy) via nearest-centroid
matching, groups the batch by predicted task, and routes each group through its
own adapter. base.py::_eval_cnn has no notion of per-sample routing (it calls
self._network(inputs) once with one fixed task/merge for the whole batch), so
_eval_cnn is overridden here rather than reused.
"""

import logging
import math

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.lora import Learner as LoRALearner
from utils.toolkit import tensor2numpy

num_workers = 8


class Learner(LoRALearner):
    def __init__(self, args):
        super().__init__(args)
        # HiDeLoRA's warm-start/momentum-blend only ever write to the CURRENT
        # slot (reading, never writing, older frozen slots) -- by the time the
        # NEXT task's freeze_to_task fires (where folding is triggered), this
        # task's blend has already completed, so folding always captures the
        # final post-blend weights. Safe for O(1) merged forward (plan doc §6
        # item 2).
        self._network.enable_frozen_folding()
        self.lora_momentum = args.get("lora_momentum", 0.1)
        # diagnostic-only gate (2026-07-19): lets a streaming ablation isolate
        # warm-start from momentum-blend when investigating fold-frequency
        # sensitivity (see BOUNDARY_AGNOSTIC_IMPLEMENTATION_LOG.md). Defaults to
        # True everywhere -- the non-streaming path and every existing config are
        # completely unaffected unless a config explicitly sets "lora_warmstart":
        # false.
        self.lora_warmstart = args.get("lora_warmstart", True)
        self.reg = args.get("reg", 0.001)
        self.crct_epochs = args.get("crct_epochs", 30)
        self.ca_lr = args.get("ca_lr", 0.005)
        self.n_centroids = args.get("n_centroids", 10)
        # {class_id: [mean_tensor, ...]} / {class_id: [var_tensor, ...]}, n_centroids
        # entries each, populated by _compute_class_stats after every task.
        self._cls_centroids = {}
        self._cls_var = {}
        self._task_of_class = {}          # class_id -> task_id, for TII routing
        self.train_merge = False          # WTP trains task-routed (not merged)

    # -- warm start: slot t <- slot t-1 (verbatim copy, before training) --------
    @torch.no_grad()
    def _warm_start(self):
        net = self._network.module if hasattr(self._network, "module") else self._network
        t = self._cur_task
        for attn in net.attn_modules():
            for A_list, B_list in ((attn.lora_A_q, attn.lora_B_q),
                                   (attn.lora_A_v, attn.lora_B_v)):
                A_list[t].weight.data.copy_(A_list[t - 1].weight.data)
                B_list[t].weight.data.copy_(B_list[t - 1].weight.data)

    # -- momentum: slot t <- (1-m)*slot_t + m*mean(slots 0..t-1) (after training) --
    @torch.no_grad()
    def _momentum_blend(self):
        net = self._network.module if hasattr(self._network, "module") else self._network
        t, m = self._cur_task, self.lora_momentum
        for attn in net.attn_modules():
            for A_list, B_list in ((attn.lora_A_q, attn.lora_B_q),
                                   (attn.lora_A_v, attn.lora_B_v)):
                A_prev_mean = torch.stack([A_list[s].weight.data for s in range(t)]).mean(0)
                B_prev_mean = torch.stack([B_list[s].weight.data for s in range(t)]).mean(0)
                A_list[t].weight.data.mul_(1 - m).add_(A_prev_mean, alpha=m)
                B_list[t].weight.data.mul_(1 - m).add_(B_prev_mean, alpha=m)

    # -- contrastive term (engine.py::orth_loss, verbatim incl. unused targets) --
    def _orth_loss(self, features, targets):
        proto_means = [mu for means in self._cls_centroids.values() for mu in means]
        if proto_means:
            protos = torch.stack(proto_means).to(features.device, features.dtype)
            M = torch.cat([protos, features], dim=0)
        else:
            M = features
        sim = M @ M.t() / 0.8
        labels = torch.arange(sim.shape[0], device=features.device)
        return F.cross_entropy(sim, labels)

    # -- task loop ------------------------------------------------------
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.data_manager = data_manager
        self._network.update_fc(self._total_classes)
        self._register_task_range()
        self._network.default_task = self._cur_task
        logging.info("[HiDeLoRA] Learning on {}-{}".format(self._known_classes, self._total_classes))

        if self._cur_task > 0 and self.lora_warmstart:
            self._warm_start()

        self._network.freeze_to_task(self._cur_task, train_a=True)
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

        self._network.to(self._device)
        self._train(self.train_loader)                       # WTP (+ momentum)
        self._compute_class_stats(data_manager)               # multi-centroid stats
        if self._cur_task > 0:
            self._train_adaptive_prediction()                 # TAP (head only)

    def _train(self, train_loader):
        params = [p for p in self._network.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.min_lr) if self.lr_anneal else None

        lo, hi = self._ce_slice()
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses, correct, total = 0.0, 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                out = self._network(inputs, task=self._cur_task, merge=False)
                logits, features = out["logits"], out["features"]
                local_logits = logits[:, lo:hi]
                local_targets = targets - lo
                ce = F.cross_entropy(local_logits, local_targets)
                orth = self._orth_loss(features, targets)
                loss = ce + self.reg * orth

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
                "[HiDe] Task {}, Epoch {}/{} => Loss {:.3f}, Acc {:.2f}".format(
                    self._cur_task, epoch + 1, self.epochs, losses / len(train_loader), train_acc))
        logging.info("[HiDeLoRA] Task {} WTP done. Train_accy {:.2f}".format(self._cur_task, train_acc))
        if self._cur_task > 0 and self.lora_momentum > 0:
            self._momentum_blend()

    # -- multi-centroid per-class stats (K-means, n_centroids clusters/class) ---
    @torch.no_grad()
    def _compute_class_stats(self, data_manager):
        from sklearn.cluster import KMeans
        net = self._network.module if hasattr(self._network, "module") else self._network
        net.eval()
        for c in range(self._known_classes, self._total_classes):
            self._task_of_class[c] = self._cur_task
            ds = data_manager.get_dataset(np.array([c]), source="train", mode="test")
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=num_workers)
            feats = []
            for _, inputs, _t in loader:
                inputs = inputs.to(self._device)
                out = net(inputs, task=self._cur_task, merge=False)
                feats.append(out["features"].cpu())
            feats = torch.cat(feats, dim=0).numpy()
            k = min(self.n_centroids, len(feats))
            kmeans = KMeans(n_clusters=k, n_init=10).fit(feats)
            means, variances = [], []
            for i in range(k):
                cluster_data = feats[kmeans.labels_ == i]
                means.append(torch.tensor(cluster_data.mean(0), dtype=torch.float32))
                variances.append(torch.tensor(cluster_data.var(0), dtype=torch.float32) + 1e-4)
            self._cls_centroids[c] = means
            self._cls_var[c] = variances

    # -- TAP: retrain only the head on Gaussian-sampled synthetic features ------
    def _train_adaptive_prediction(self):
        net = self._network.module if hasattr(self._network, "module") else self._network
        net.train()
        for p in net.parameters():
            p.requires_grad = False
        for p in net.fc.parameters():
            p.requires_grad = True
        optimizer = optim.SGD(net.fc.parameters(), lr=self.ca_lr, momentum=0.9)
        all_classes = sorted(self._cls_centroids.keys())
        # N synthetic samples/class/epoch (a simplification of the reference's exact
        # batching -- faithful to the ALGORITHM: Gaussian replay of stored per-class
        # centroid stats to recalibrate the head against the full seen label space).
        n_per_class = 4
        for _ in range(self.crct_epochs):
            feats, labels = [], []
            for c in all_classes:
                centroids, variances = self._cls_centroids[c], self._cls_var[c]
                for _n in range(n_per_class):
                    idx = np.random.randint(len(centroids))
                    mean, var = centroids[idx], variances[idx]
                    std = var.clamp(min=1e-6).sqrt()
                    feats.append(mean + std * torch.randn_like(mean))
                    labels.append(c)
            feats = torch.stack(feats).to(self._device)
            labels = torch.tensor(labels, device=self._device)
            logits = net.fc(feats)["logits"]
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logging.info("[HiDeLoRA] TAP done ({} epochs, {} classes).".format(
            self.crct_epochs, len(all_classes)))

    # -- task-inference routing (shared by CIL eval + deployed-latency probe) ---
    @torch.no_grad()
    def _predict_task_ids(self, inputs):
        """Per-sample task id via nearest-centroid match on UN-ADAPTED (task=-1)
        features -- vit_lora.py routes task<0 to a pure backbone forward (no LoRA
        contribution), so this needs no second model copy."""
        net = self._network.module if hasattr(self._network, "module") else self._network
        feats = net(inputs, task=-1, merge=False)["features"]      # [B, d]
        all_means, all_tasks = [], []
        for c, means in self._cls_centroids.items():
            for mu in means:
                all_means.append(mu)
                all_tasks.append(self._task_of_class[c])
        protos = torch.stack(all_means).to(feats.device, feats.dtype)   # [K, d]
        dists = torch.cdist(feats, protos)                              # [B, K]
        nearest = dists.argmin(dim=1).cpu().numpy()
        return np.array([all_tasks[i] for i in nearest])

    @torch.no_grad()
    def _deployed_forward(self, inputs):
        """Task-inference-routed forward (grouped by predicted task, one adapter
        pass per distinct predicted task in the batch) -- the actual deployed CIL
        model. Returns full-head logits, same contract as a plain net(...) call."""
        net = self._network.module if hasattr(self._network, "module") else self._network
        task_ids = self._predict_task_ids(inputs)
        out_logits = torch.zeros(inputs.shape[0], self._total_classes, device=inputs.device)
        for t in np.unique(task_ids):
            mask = task_ids == t
            idx = torch.tensor(np.nonzero(mask)[0], device=inputs.device)
            group = inputs.index_select(0, idx)
            logits = net(group, task=int(t), merge=False)["logits"]
            out_logits.index_copy_(0, idx, logits)
        return {"logits": out_logits}

    # -- CIL eval: base.py::_eval_cnn assumes ONE task/merge per whole batch;
    # HiDe needs per-sample routing, so this is a full override (same return
    # contract: y_pred [N, topk], y_true [N], so _evaluate/forgetting matrices
    # downstream are untouched). ---------------------------------------------
    @torch.no_grad()
    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        k = min(self.topk, self._total_classes)
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            logits = self._deployed_forward(inputs)["logits"]
            predicts = torch.topk(logits, k=k, dim=1, largest=True, sorted=True)[1]
            if k < self.topk:
                pad = predicts[:, -1:].expand(-1, self.topk - k)
                predicts = torch.cat([predicts, pad], dim=1)
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    # TIL routing: ground-truth task known, still task-routed merge=False (WTP's
    # own training-time forward) -- inherited _forward_task (models/lora.py) is
    # already exactly this; no override needed.

    def _log_trainable(self):
        net = self._network
        total = sum(p.numel() for p in net.parameters())
        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        logging.info("[HiDeLoRA] Trainable params this task: {:,} / {:,}".format(trainable, total))

    # -- persistent state: adapter slots (inherited accounting) + centroid stats
    def persistent_state(self):
        base = super().persistent_state()
        centroid_floats = sum(
            len(means) * means[0].numel() * 2   # mean + var, same width
            for means in self._cls_centroids.values() if means)
        centroid_bytes = centroid_floats * 4     # fp32
        return {"params": base["params"] + centroid_floats,
                "bytes": base["bytes"] + centroid_bytes,
                "breakdown": {**base["breakdown"], "centroid_stats": centroid_bytes}}

    # ==================================================================
    # Boundary-agnostic streaming hooks (models/stream_mixin.py).
    #
    # Split into two deliberately DIFFERENT clocks:
    #   - Adapter (LoRA slot) bookkeeping -- warm-start + momentum-blend -- is
    #     CHUNK-scoped (self._stream_chunk), decoupled from real task boundaries,
    #     same as O-LoRA/InfLoRA/TreeLoRA. This is the thing actually being
    #     stress-tested.
    #   - Per-class centroid stats + TAP head recalibration stay REAL-TASK-scoped
    #     (fired from the new _stream_end_task hook, NOT _stream_end_chunk).
    #     Reason: centroids are computed over `range(known_classes, total_classes)`
    #     -- a REAL class range. Real tasks are always delivered as a complete,
    #     uninterrupted block under this design (data delivery is NOT reordered
    #     the way BudgetStreamManager's byte-chunking was), so keeping centroid
    #     computation task-scoped means a class is NEVER split across two
    #     different centroid-computation events -- avoiding the exact
    #     duplicate-class/no-faithful-merge problem that excluded EASE/TUNA/
    #     CL-LoRA from memory-constrained training entirely. Making centroids
    #     chunk-scoped instead would reintroduce that same problem for HiDeLoRA.
    #   `self._cur_task` mirrors the CHUNK index throughout (matching the
    #   existing code's `net(inputs, task=self._cur_task, ...)` calls in
    #   `_compute_class_stats`/`_predict_task_ids`/`_deployed_forward`), so
    #   centroids computed at a real task's end are correctly tagged with
    #   whichever chunk/slot was actually active/deployed at that moment.
    # ==================================================================
    def _stream_init(self):
        self._stream_chunk = -1

    def _stream_slot(self):
        return self._stream_chunk

    def _stream_begin_chunk(self, loader):
        self._stream_chunk += 1
        self._cur_task = self._stream_chunk
        if self._stream_chunk > 0:
            # slot count is not generically bounded by nb_tasks under this clock --
            # see BOUNDARY_AGNOSTIC_IMPLEMENTATION_LOG.md's "BLOCKING ARCHITECTURAL GAP"
            self._network.add_task_slot()
            if self.lora_warmstart:
                self._warm_start()
        self._network.freeze_to_task(self._stream_chunk, train_a=True)
        for p in self._network.fc.parameters():
            p.requires_grad = True
        self._stream_new_optimizer()

    def _stream_end_chunk(self, loader):
        if self._stream_chunk > 0 and self.lora_momentum > 0:
            self._momentum_blend()

    def _stream_train_epoch(self, loader, lo, hi):
        """Full override -- WTP's orth_loss needs the forward pass's `features`
        output, which the generic _stream_train_epoch (logits only) doesn't
        expose. Mirrors _train's per-batch loop exactly."""
        self._network.train()
        t = self._stream_chunk
        for _, inputs, targets in loader:
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            out = self._network(inputs, task=t, merge=False)
            logits, features = out["logits"], out["features"]
            local_logits = logits[:, lo:hi]
            local_targets = targets - lo
            ce = F.cross_entropy(local_logits, local_targets)
            orth = self._orth_loss(features, targets)
            loss = ce + self.reg * orth
            self._stream_optim.zero_grad()
            loss.backward()
            self._stream_optim.step()
        if self._stream_sched is not None:
            self._stream_sched.step()

    def _stream_end_task(self, ct):
        """REAL-task-scoped (not chunk-scoped, see class docstring above)."""
        self._compute_class_stats(self.data_manager)
        if ct > 0:
            self._train_adaptive_prediction()

    def _stream_cil_forward(self, inputs):
        """HiDe's deployed CIL forward is per-sample task-inference routing, not
        a single fixed slot -- override the generic single-slot default."""
        return self._deployed_forward(inputs)

    # TIL routing (_forward_task): inherited default (models/lora.py) already
    # correct here -- HiDeLoRA never overrode it even outside streaming (WTP's
    # own task-routed merge=False forward IS the TIL forward), and the generic
    # _stream_task_to_chunk remap (see models/lora.py::_eval_adapter) makes it
    # correctly chunk-aware under streaming with no HiDeLoRA-specific code.
