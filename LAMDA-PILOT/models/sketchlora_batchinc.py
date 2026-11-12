"""Batch-incremental + KL-drift-detection setting (boundary_mode=
"kl_batch_inc", see trainer.py's new dispatch branch). A full, isolated
codepath per this project's "new experimental mechanism = isolated
codepath" convention -- does not subclass or modify models/sketchlora.py,
models/sketchlora_align.py, models/lora.py, or models/stream_mixin.py.
Uses backbone/vit_sketchlora_batchinc.py exclusively.

See this session's design discussion (memory: new_batch_incremental_
setting_design.md) for the full spec this implements. Summary of the
mechanics this file is responsible for (the backbone owns the adapter
state machine and the fold math; the detector owns the drift signal; this
file owns the stream plan, the two independent clocks, and gluing them
together):

STREAM PLAN: task-major, same real task structure as oracle training (head
grows only at real task boundaries), but each task's own DataLoader is
re-iterated `epochs_per_task` times with a FRESH shuffle every epoch
(automatic: DataLoader(shuffle=True) builds a new RandomSampler on every
fresh __iter__, so a plain `for epoch in range(E): for batch in loader`
already gives this with no extra bookkeeping). The driver never reads
task/epoch/batch position to make its OWN decisions (spawn/fold); it reads
them only to decide head growth and to log.

TWO INDEPENDENT CLOCKS, checked after every single batch:
  - SPAWN: the KL detector (utils/online_kl_drift.py) watches the frozen
    backbone's own CLS feature, recomputed fresh every batch from that
    batch's own inputs with adapters disabled, under no_grad (never
    cached, never precomputed -- see that module's own docstring for why).
    On fire: freeze the live slot into pending (backbone.add_task_slot()),
    rebuild the optimizer and LR schedule from scratch for the new live
    slot (restarts per adapter, never at a task boundary), reset the
    detector.
  - SKETCH: fires on a fixed step interval (`sketch_interval_steps`) OR
    the instant pending reaches its cap, whichever comes first -- checked
    independently of the spawn clock, never triggered by a detector fire.
    The fixed-interval-only ablation (no detector at all) is the same
    code with the detector simply never queried/never able to fire.

OPTIMIZER: rebuilt fresh at every spawn (never warm-started). lr is set by
hand per step (_current_lr_scale()): linear warmup over the first
batchinc_lr_warmup_steps steps of the adapter's life, then held flat at
init_lr indefinitely. There is deliberately no scheduler object anchored
to epochs_per_task -- an earlier version used CosineAnnealingLR(T_max=
epochs_per_task), on the assumption (by analogy with models/stream_mixin.
py's offset_consolidation_run(), which faces the same "real segment length
isn't known in advance" problem) that an adapter outliving T_max would
just sit at the floor LR for the remainder. That assumption was checked
directly against torch's actual implementation (2026-11-12) and is false:
CosineAnnealingLR is periodic with period 2*T_max, not floor-clamped, so
any adapter surviving past one task's worth of epochs saw its LR climb
back toward init_lr and keep oscillating. Hence the flat-after-warmup
scheme above, which has no horizon to get wrong.

HEAD/OPTIMIZER SYNC: BatchIncNet.update_fc swaps in a brand-new fc module
every time the head grows (every real task boundary that introduces new
classes), discarding the old fc's Parameter objects. Whenever this happens
without a spawn also rebuilding the optimizer from scratch, the new fc's
params are added to the existing optimizer as a fresh param group (see the
driver's head-growth block) so they actually receive optimizer steps
rather than silently accumulating unmanaged gradient until the next spawn.

EVAL: happens only at real task boundaries, using whatever the live
deployable state currently is (sketch + pending + live, via the
backbone's own unconditional forward -- there is no separate "merge=False"
mode in this setting, the forward is always everything).
"""
import copy
import logging

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.base import BaseLearner
from backbone.linears import SimpleLinear
from utils.inc_net import BaseNet
from utils.online_kl_drift import KLDriftDetector
from utils.toolkit import tensor2numpy

num_workers = 8


class BatchIncNet(BaseNet):
    """Minimal Net wrapper for backbone/vit_sketchlora_batchinc.py. Deliberately
    does NOT reuse utils/inc_net.py::LoRAVitNet -- that class's forward
    hardcodes threading task/merge kwargs into the backbone call
    (LoRAVitNet.forward -> self.backbone(x, task=task, merge=merge)), but
    this setting's backbone forward takes no such kwargs at all (it always
    includes sketch+pending+live unconditionally, there is no separate
    task-routed mode). BaseNet's own generic forward (`x = self.backbone(x);
    out = self.fc(x); out.update({"features": x})`) already matches this
    backbone's plain forward(x) signature exactly, so only update_fc/
    generate_fc need overriding here -- the same minimal pattern utils/
    inc_net.py::IncrementalNet itself uses for the analogous case."""

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        return SimpleLinear(in_dim, out_dim)


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = BatchIncNet(args, True)

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args.get("weight_decay") or 0.0
        self.min_lr = args.get("min_lr") or 0.0
        self.epochs_per_task = args["tuned_epoch"]

        self.lora_rank = args.get("lora_rank", 10)
        self.svd_rank = args.get("svd_rank", self.lora_rank)
        self.pending_cap = args.get("batchinc_pending_cap", 4)
        self.sketch_interval_steps = args.get("batchinc_sketch_interval_steps", 200)
        self.orth_weight = args.get("batchinc_orth_weight", 0.5)
        # Linear warmup length (in steps) for a freshly spawned adapter's LR --
        # see _current_lr_scale(). Replaces a prior CosineAnnealingLR(T_max=
        # epochs_per_task) that assumed (incorrectly -- verified empirically,
        # 2026-11-12) an adapter never outlives one task; torch's cosine
        # schedule is periodic past T_max, not floor-clamped, so any adapter
        # that survives longer had its LR climb back toward init_lr instead
        # of staying annealed.
        self.lr_warmup_steps = args.get("batchinc_lr_warmup_steps", 20)

        # Detector config -- the caller (this file) is responsible for the
        # samples->batches conversion and the window-vs-shortest-task check
        # per utils/online_kl_drift.py's own docstring; neither is the
        # detector class's job.
        kl_window_samples = args.get("batchinc_kl_window_samples", 256)
        kl_window_batches = max(4, -(-kl_window_samples // self.batch_size))  # ceil div
        self.kl_detector = KLDriftDetector(
            window_batches=kl_window_batches,
            warmup_multiplier=args.get("batchinc_kl_warmup_multiplier", 2),
            tau=args.get("batchinc_kl_tau", 4.0),
            consecutive=args.get("batchinc_kl_consecutive", 3),
            var_eps=args.get("batchinc_kl_var_eps", 1e-4),
            max_chunk_samples=args.get("batchinc_kl_max_chunk_samples"),
        )
        self.kl_window_batches = kl_window_batches
        # Fixed-interval-only ablation: the detector is simply never queried
        # for a spawn decision (spawning instead follows the SAME
        # sketch_interval_steps clock the sketch trigger uses), rather than
        # a separate code path -- per the design's own framing of this as
        # the two clocks coinciding, not a different mechanism.
        self.fixed_interval_only = bool(args.get("batchinc_fixed_interval_only", False))

        self._optimizer = None
        self._global_step = 0
        self._adapter_step_counts = []   # steps trained under each spawned adapter, in spawn order
        self._cur_adapter_steps = 0

    # -- backbone access helper (matches this project's own
    # module-vs-DataParallel-wrapped convention) ---------------------------
    def _net(self):
        return self._network.module if hasattr(self._network, "module") else self._network

    # -- optimizer, rebuilt fresh at every spawn ----------------------------
    # No LR *schedule* object -- a spawned adapter's real lifetime is
    # drift-determined, not known in advance, so there is no safe T_max to
    # anneal against (see _current_lr_scale()). lr is instead set by hand,
    # per step, in the training loop: a short linear warmup then held flat
    # at init_lr for as long as the adapter lives.
    def _new_optimizer_and_schedule(self):
        net = self._net()
        params = [p for p in net.parameters() if p.requires_grad]
        self._optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)
        self._cur_adapter_steps = 0

    def _current_lr_scale(self):
        """1/warmup_steps .. 1.0 linear ramp over the first lr_warmup_steps
        steps of the CURRENT adapter's life, then held flat at 1.0
        indefinitely -- no periodicity, no horizon assumption."""
        if self._cur_adapter_steps >= self.lr_warmup_steps:
            return 1.0
        return (self._cur_adapter_steps + 1) / self.lr_warmup_steps

    def _freeze_to_live(self):
        """Only the live slot's (A,B) and the classifier head are
        trainable -- sketch and every pending slot stay frozen, same
        convention as every other method's freeze_to_task."""
        net = self._net()
        for p in net.backbone.parameters():
            p.requires_grad = False
        live = net.backbone._live_idx
        for attn in net.backbone.attn_modules():
            for mlist in (attn.lora_A_q, attn.lora_B_q, attn.lora_A_v, attn.lora_B_v):
                for p in mlist[live].parameters():
                    p.requires_grad = True
        for p in net.fc.parameters():
            p.requires_grad = True

    # -- KL detector's own frozen-feature probe ----------------------------
    @torch.no_grad()
    def _frozen_cls_features(self, inputs):
        net = self._net()
        net.backbone.set_adapters_disabled(True)
        feats = net.backbone(inputs)
        net.backbone.set_adapters_disabled(False)
        return feats

    # -- the two independent clocks ----------------------------------------
    def _maybe_spawn(self, inputs):
        """Returns True iff a spawn happened this step."""
        if self.fixed_interval_only:
            due = (self._global_step > 0) and (self._global_step % self.sketch_interval_steps == 0)
            if due:
                self._do_spawn()
                return True
            return False
        feats = self._frozen_cls_features(inputs)
        fired = self.kl_detector.step(feats, inputs.shape[0])
        if fired:
            logging.info("[kl_batch_inc] KL detector fired at step {} (reason={}, kl={}, z={})".format(
                self._global_step, self.kl_detector.last_fire_reason,
                self.kl_detector.last_kl, self.kl_detector.last_z))
            self._do_spawn()
            self.kl_detector.reset()
        return fired

    def _do_spawn(self):
        net = self._net()
        self._adapter_step_counts.append(self._cur_adapter_steps)
        net.backbone.add_task_slot()
        self._freeze_to_live()
        self._new_optimizer_and_schedule()
        if len(net.backbone._pending) >= self.pending_cap:
            logging.info("[kl_batch_inc] pending cap ({}) reached immediately after spawn -- "
                         "folding before any training on the new live slot".format(self.pending_cap))
            self._do_fold()

    def _maybe_fold(self):
        net = self._net()
        interval_due = (self._global_step > 0) and (self._global_step % self.sketch_interval_steps == 0)
        cap_due = len(net.backbone._pending) >= self.pending_cap
        if interval_due or cap_due:
            self._do_fold()

    def _do_fold(self):
        net = self._net()
        n_pending = len(net.backbone._pending)
        if n_pending == 0:
            return
        retained = net.backbone.fold_pending()
        vals = list(retained.values())
        logging.info("[kl_batch_inc] fold at step {}: {} pending slot(s) merged, "
                     "retained_energy mean={:.4f} min={:.4f}".format(
                         self._global_step, n_pending, sum(vals) / len(vals), min(vals)))

    # -- eval (real task boundaries only) ----------------------------------
    @torch.no_grad()
    def _eval_cil(self, test_loader, known_classes, total_classes):
        net = self._net()
        net.eval()
        correct, total = 0, 0
        for _, inputs, targets in test_loader:
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            logits = net(inputs)["logits"]
            preds = logits.argmax(dim=1)
            correct += preds.eq(targets).cpu().sum()
            total += len(targets)
        return float(tensor2numpy(correct) * 100 / total) if total else 0.0

    # -- the driver ---------------------------------------------------------
    def kl_batch_inc_run(self, data_manager, args):
        nb_tasks = data_manager.nb_tasks
        n_run = args.get("stop_after_tasks") or nb_tasks
        n_run = min(n_run, nb_tasks)
        if n_run < nb_tasks:
            logging.info("[kl_batch_inc] stop_after_tasks={} (of {} total)".format(n_run, nb_tasks))

        # minimum detectable segment is ~3*window_batches steps (per
        # utils/online_kl_drift.py's own docstring) -- warn, don't crash, if
        # a real task's own step count can't clear that bar at this batch
        # size/epoch count, since that's a config-level choice, not
        # something this driver should silently override. Checked per-task
        # inside the loop below (where each task's actual sample count is
        # already being read to build its DataLoader), NOT estimated here
        # from data_manager.get_task_size(t) -- that returns the task's
        # CLASS count, not its sample count, so a pre-loop estimate from it
        # alone would be nonsense (confirmed: an earlier version of this
        # check used get_task_size(t) directly as a sample count and
        # reported "3 steps" for a 5000-image CIFAR-100 task).
        min_detectable = 3 * self.kl_window_batches

        self._known_classes = 0
        self._total_classes = 0
        self._global_step = 0
        self._adapter_step_counts = []
        self._cur_adapter_steps = 0
        self._net().to(self._device)

        results = []
        known = 0
        for t in range(n_run):
            task_size = data_manager.get_task_size(t)
            lo, hi = known, known + task_size
            known = hi
            if hi > self._total_classes:
                self._total_classes = hi
                self._net().update_fc(self._total_classes)
                self._net().to(self._device)
                # BatchIncNet.update_fc always swaps in a brand-new fc
                # module (del self.fc; self.fc = fc), so any optimizer built
                # before this point still only references the OLD, now-
                # orphaned fc Parameters. Without this, the new fc would
                # receive real gradients every step (it requires_grad=True
                # by construction) but never actually get updated -- its
                # .grad would sit un-zeroed and un-stepped until the next
                # spawn rebuilds the optimizer from scratch, at which point
                # a single, badly-oversized step would fire from whatever
                # gradient had silently accumulated in the meantime. Add the
                # new fc as a fresh param group instead of rebuilding the
                # whole optimizer, so the live adapter's own Adam state and
                # warmup position are left untouched. The old fc's now-dead
                # param group (if any) is left in place -- its .grad stays
                # None forever once orphaned, so AdamW skips it in step();
                # it is fully discarded the next time a spawn rebuilds the
                # optimizer from net.parameters() anyway.
                if self._optimizer is not None:
                    for p in self._net().fc.parameters():
                        p.requires_grad = True
                    self._optimizer.add_param_group(
                        {"params": list(self._net().fc.parameters()), "lr": self.init_lr})

            if t == 0:
                # fc only exists from here on (update_fc just created it) --
                # the initial freeze/optimizer setup for slot 0 (the very
                # first live adapter) must wait until after this point.
                # Every LATER spawn builds its own fresh optimizer via
                # _do_spawn() instead, so this is a one-time bootstrap only.
                self._freeze_to_live()
                self._new_optimizer_and_schedule()

            train_set = data_manager.get_dataset(np.arange(lo, hi), source="train", mode="train")
            train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                      num_workers=num_workers)
            test_set = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
            test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False,
                                     num_workers=num_workers)

            task_steps = len(train_loader) * self.epochs_per_task
            if task_steps < min_detectable:
                logging.warning(
                    "[kl_batch_inc] task {} is only {} steps ({} batches x {} epochs), below the "
                    "detector's own ~{}-step minimum detectable segment (window_batches={}) -- the "
                    "detector may never fire within this task at this config; consider a smaller "
                    "batchinc_kl_window_samples".format(
                        t, task_steps, len(train_loader), self.epochs_per_task,
                        min_detectable, self.kl_window_batches))

            self._net().train()
            for epoch in range(self.epochs_per_task):
                losses, ce_run, orth_run, correct, total = 0.0, 0.0, 0.0, 0, 0
                for _, inputs, targets in train_loader:
                    inputs, targets = inputs.to(self._device), targets.to(self._device)

                    # Spawn check BEFORE this batch's own training step --
                    # the frozen-feature probe reads this exact batch's
                    # inputs, and a fire must land the new live slot before
                    # that same batch trains it (the detector's job is to
                    # decide what trains on the NEXT step, including this
                    # one, not to retroactively judge a step already taken).
                    self._maybe_spawn(inputs)

                    logits = self._net()(inputs)["logits"]
                    local_logits = logits[:, lo:hi]
                    ce = F.cross_entropy(local_logits, targets - lo)
                    orth = self._net().backbone.orth_loss()
                    loss = ce + self.orth_weight * orth

                    lr_scale = self._current_lr_scale()
                    for g in self._optimizer.param_groups:
                        g["lr"] = self.init_lr * lr_scale

                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()

                    self._global_step += 1
                    self._cur_adapter_steps += 1
                    self._maybe_fold()

                    losses += loss.item()
                    ce_run += ce.item()
                    orth_run += float(orth)
                    preds = local_logits.argmax(dim=1)
                    correct += preds.eq(targets - lo).cpu().sum()
                    total += len(targets)
                train_acc = tensor2numpy(correct) * 100 / total if total else 0.0
                logging.info(
                    "[kl_batch_inc] Task {}, Epoch {}/{} => Loss {:.3f} (CE {:.3f}, orth {:.5f}), "
                    "Train_accy {:.2f}, global_step={}".format(
                        t, epoch + 1, self.epochs_per_task, losses / len(train_loader),
                        ce_run / len(train_loader), orth_run / len(train_loader), train_acc,
                        self._global_step))

            with torch.no_grad():
                self._replace_fc_head(train_loader=None)  # no-op placeholder, see method docstring
            cil = self._eval_cil(test_loader, self._known_classes, self._total_classes)
            logging.info("[kl_batch_inc] Task {} boundary eval: CIL top1 = {:.2f} "
                         "(pending={}, live={})".format(
                             t, cil, list(self._net().backbone._pending), self._net().backbone._live_idx))
            results.append({"completed": t + 1, "cil": cil, "til": None,
                            "global_step": self._global_step,
                            "n_pending": len(self._net().backbone._pending),
                            "n_spawned": self._net().backbone.n_tasks})
            self._known_classes = self._total_classes

        self._adapter_step_counts.append(self._cur_adapter_steps)
        logging.info("[kl_batch_inc] FINAL: {} total steps, {} adapters spawned, "
                     "per-adapter step counts: {}".format(
                         self._global_step, self._net().backbone.n_tasks, self._adapter_step_counts))
        assert sum(self._adapter_step_counts) == self._global_step, (
            "per-adapter step counts {} sum to {}, not global_step {} -- step accounting is broken".format(
                self._adapter_step_counts, sum(self._adapter_step_counts), self._global_step))
        return results

    def _replace_fc_head(self, train_loader):
        """Placeholder: OurNet's own fc is a plain gradient-trained linear
        head here (CE loss trains it directly every step, same as every
        other CIL method's growing fc in this project) -- NOT an NCM/
        prototype head, so there is nothing to separately "replace" after
        training. Kept as an explicit no-op (rather than simply omitted)
        so a future variant that DOES want prototype-based eval has an
        obvious, named place to add it without restructuring the driver."""
        return
