"""SD-LoRA + Sketch: SD-LoRA's magnitude/direction decoupling applied to
SketchLoRA's fixed-size, QR-SVD-folded sketch, with the sketch's own
singular values made per-task re-learnable via a small rescaling vector.
Built from a web-claude implementation brief (2026-11-06 user request) --
see backbone/vit_sdlora_sketch.py for the forward/merge mechanics; this
file holds the training loop, the regularizer, and the task-boundary
orchestration.

ALGORITHM. Per adapted projection (Q, V -- independently, see DEVIATION
below), per task t:
  - the sketch (U, S, Vh -- explicit factored SVD, non-trainable buffers,
    fixed size svd_rank) holds the folded history through task t-1.
  - a fresh rescaling vector r (length svd_rank, init 0) proposes a
    per-singular-value multiplicative adjustment: S_eff = S*(1+r).
  - a fresh new-adapter direction (A_new, B_new -- small-nonzero init, see
    INIT DEVIATION below) and a fresh scalar magnitude m (init 0) represent
    this task's own update: m * normalize(B_new @ A_new).
  - forward (both train and eval, same formula, never materializing a
    dense [dim,dim] weight -- see backbone/vit_sdlora_sketch.py::
    Attention_SDLoRASketch._sketch_term/_new_adapter_term):
        delta(x) = sketch_term(x) + (m / ||B_new A_new||_F) * B_new(A_new(x))
  - at the task boundary (start of the NEXT task's incremental_train,
    mirroring models/lorasub_sketch.py's own fold-then-train convention):
    bake S_eff into the sketch, scale the new adapter by its magnitude,
    QR-SVD-merge both into a fresh truncated (U,S,Vh), log retained energy,
    then reset the new-adapter/rescale/magnitude for the next task.
    Optimizer rebuild is automatic: _train() below always constructs a
    fresh optim.AdamW from self._optimizer_param_groups() every call, so
    no stale Adam state from the discarded adapter survives regardless of
    whether the underlying Parameter objects were replaced or just
    reinitialized in place (the latter is what actually happens -- see
    backbone/vit_sdlora_sketch.py::Attention_SDLoRASketch._new_task_params).

DEVIATION FROM SD-LoRA (flagged, per the brief): magnitudes are PER
PROJECTION (one m_q, one m_v, per block), NOT shared across Q/V the way
models/sdlora.py's single global per-task scalar is -- the sketch's Q and V
spectra are unrelated, so forcing them to share a scale has no principled
basis here.

INIT DEVIATION (flagged, per the brief): this method's forward divides by
||B_new @ A_new||_F, undefined at B_new=0 -- standard LoRA's own zero-B
init (used by models/sdlora.py's own add_task_slot) is therefore not usable
here. Both A_new and B_new start small-and-nonzero instead (see backbone/
vit_sdlora_sketch.py::NEW_ADAPTER_INIT_STD); the "no behavior jump at task
start" guarantee the brief asks for comes from the magnitude m starting at
0 (the whole adapter term is then exactly 0 regardless of direction), not
from either factor being zero.

WHY NO WARM-START (unlike models/sdlora.py's own warm-started
task_scales): the brief is explicit -- "discard the adapter, its
magnitude, and the rescaling vector" / keep no per-task state after a
boundary. A deliberate, different design choice from vanilla SD-LoRA's
(also deliberate, separately flagged) warm-start, not an oversight.

REGULARIZER (_rescale_reg_loss): energy-weighted L2 penalty on the rescale
vector. Each singular value's share of the sketch's total energy
(S^2, normalized) is computed fresh every training step directly from the
CURRENT sketch buffers -- those are frozen (non-trainable) for the whole
task, so this is equivalent to "computed once at task start" without
needing to cache it. Weighted by share^p (p=sdsketch_reg_power, default 1)
and summed over every (block, projection) with sdsketch_reg_strength.
Skipped at task 0 (empty sketch: S is all zero, so both the energy-share
normalization and the direct S-dependence would otherwise need a 0/0
guard) -- the brief's own instruction.

SCOPE: oracle (task-boundary) CIL path only, matching this project's other
SketchLoRA-family extensions (models/lorasub_sketch.py, models/
inflora_sketch.py) and SD-LoRA itself (models/sdlora.py). No task-routed
(TIL) forward is specially defined for this method -- the backbone's
_lora_delta always combines sketch + new adapter regardless of any
task/merge flag, so this inherits models/lora.py::Learner._forward_task
unchanged: TIL "accuracy" is the same full merged forward, masked to each
task's own class slice (the same characterization models/sdlora.py's own
module docstring gives its own invented merge=False case).

ABLATION: sdsketch_freeze_rescale=True freezes the rescale vector at 0
forever, giving "plain SketchLoRA with SD-LoRA-style [direction+magnitude]
adapters" (the brief's own description of this ablation) -- see
backbone/vit_sdlora_sketch.py's freeze_rescale plumbing.

TESTS: tests/check_sdlora_sketch.py verifies the 6 properties the brief
asks be checked before any real training run (exact merge / unchanged
output under no truncation, true-vs-reported discarded energy, a known
rescale with a silenced adapter reproducing exactly the rescaled singular
values, sketch orthonormality across repeated synthetic merges, and the
rescale gradient matching a dense-weight reference) -- run it first.
"""

import logging

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.lora import Learner as LoRALearner, num_workers
from utils.toolkit import tensor2numpy

_PROJS = ("q", "v")


class Learner(LoRALearner):
    def __init__(self, args):
        super().__init__(args)
        # This method always combines sketch + new-adapter, in both
        # training and eval -- there is no separate "slot alone" mode (see
        # backbone/vit_sdlora_sketch.py::Attention_SDLoRASketch._lora_delta,
        # which ignores task/merge entirely). Set for consistency with
        # every other CIL method's convention, not because this class's
        # own forward actually branches on it.
        self.train_merge = True
        self._network.merge = True
        self.svd_rank = args.get("svd_rank", args.get("lora_rank", 10))
        self.reg_strength = args.get("sdsketch_reg_strength", 0.0)
        self.reg_power = args.get("sdsketch_reg_power", 1.0)
        self.freeze_rescale = bool(args.get("sdsketch_freeze_rescale", False))

    # -- single reused slot, forever -- see module docstring. ------------
    def _train_adapter(self):
        return 0

    def _eval_adapter(self, task):
        return 0

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.data_manager = data_manager
        self._network.update_fc(self._total_classes)
        self._register_task_range()
        self._network.default_task = 0
        logging.info("[SD-LoRA-Sketch] Learning on {}-{}".format(self._known_classes, self._total_classes))

        # Fold the PREVIOUS task's fully-trained new-adapter into the
        # sketch first (mirrors models/lorasub_sketch.py's own
        # incremental_train ordering exactly), then reset fresh for THIS
        # task's own training. No-op at task 0 (nothing trained yet).
        if self._cur_task > 0:
            self._fold_sketch()

        self._network.freeze_to_task(0)  # see backbone's freeze_to_task -- task arg is ignored
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

    def _rescale_reg_loss(self):
        if self._cur_task <= 0 or self.reg_strength == 0.0:
            return 0.0
        net = self._network.module if hasattr(self._network, "module") else self._network
        reg = 0.0
        for attn in net.attn_modules():
            for proj in _PROJS:
                S = getattr(attn, "sketch_S_{}".format(proj))
                rescale = getattr(attn, "rescale_{}".format(proj))
                energy = S.pow(2)
                share = energy / energy.sum().clamp_min(1e-12)
                reg = reg + (share.pow(self.reg_power) * rescale.pow(2)).sum()
        return reg

    def _train(self, train_loader):
        """models/lora.py::Learner._train's loop, duplicated (same reason
        models/sketchlora_align.py's own _train_core duplicates it rather
        than hooking the shared base class -- see that file's docstring)
        with the rescale regularizer term added."""
        self._network.to(self._device)
        params = self._optimizer_param_groups()
        optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.min_lr) if self.lr_anneal else None

        lo, hi = self._ce_slice()
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses, ce_run, reg_run, correct, total = 0.0, 0.0, 0.0, 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs, task=0, merge=True)["logits"]
                local_logits = logits[:, lo:hi]
                local_targets = targets - lo
                ce = F.cross_entropy(local_logits, local_targets)
                reg = self._rescale_reg_loss()
                loss = ce + self.reg_strength * reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                ce_run += ce.item()
                reg_run += float(reg)

                preds = local_logits.argmax(dim=1)
                correct += preds.eq(local_targets).cpu().sum()
                total += len(targets)
            if scheduler is not None:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            n = len(train_loader)
            prog_bar.set_description(
                "[SD-LoRA-Sketch] Task {}, Epoch {}/{} => Loss {:.3f} (CE {:.3f}, reg {:.5f}), Acc {:.2f}".format(
                    self._cur_task, epoch + 1, self.epochs, losses / n, ce_run / n, reg_run / n, train_acc))
        logging.info("[SD-LoRA-Sketch] Task {} finished. Train_accy {:.2f}".format(self._cur_task, train_acc))
        self._log_memory_accounting()

    @torch.no_grad()
    def _fold_sketch(self):
        net = self._network.module if hasattr(self._network, "module") else self._network
        per_block = net.fold()
        vals = [v for blk in per_block.values() for v in blk.values()]
        logging.info("[SD-LoRA-Sketch] fold at task {}: retained_energy mean={:.4f} min={:.4f}".format(
            self._cur_task, sum(vals) / len(vals), min(vals)))

    # -- memory accounting: sketch (bounded) + new-adapter + rescale/mag
    # (both transient/live, like models/lorasub_sketch.py's own "residual"
    # breakdown key) + head. Same convention as that file's persistent_state.
    def persistent_state(self):
        net = self._network.module if hasattr(self._network, "module") else self._network
        sketch_bytes = adapter_bytes = regmag_bytes = 0
        for attn in net.attn_modules():
            for proj in _PROJS:
                for name in ("sketch_U_{}".format(proj), "sketch_S_{}".format(proj), "sketch_Vh_{}".format(proj)):
                    t = getattr(attn, name)
                    sketch_bytes += t.numel() * t.element_size()
                for mod_name in ("lora_A_{}".format(proj), "lora_B_{}".format(proj)):
                    p = getattr(attn, mod_name)[0].weight
                    adapter_bytes += p.numel() * p.element_size()
                for name in ("rescale_{}".format(proj), "mag_{}".format(proj)):
                    p = getattr(attn, name)
                    regmag_bytes += p.numel() * p.element_size()
        head_bytes = sum(p.numel() * p.element_size() for p in net.fc.parameters())
        total_bytes = sketch_bytes + adapter_bytes + regmag_bytes + head_bytes
        return {"params": int(total_bytes // 4), "bytes": int(total_bytes),
                "breakdown": {"sketch": sketch_bytes, "new_adapter": adapter_bytes,
                             "rescale_mag": regmag_bytes, "head": head_bytes}}

    def _log_memory_accounting(self):
        state = self.persistent_state()
        logging.info(
            "[SD-LoRA-Sketch] task {} persistent bank: FIXED sketch (U+S+Vh, Q+V, every "
            "block) = {:,} bytes (sketch={:,}, new_adapter={:,}, rescale_mag={:,}, "
            "head={:,})".format(
                self._cur_task, state["bytes"], state["breakdown"]["sketch"],
                state["breakdown"]["new_adapter"], state["breakdown"]["rescale_mag"],
                state["breakdown"]["head"]))
