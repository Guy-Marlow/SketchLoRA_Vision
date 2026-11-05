"""LoRA Subtraction + SketchLoRA continual merge (fixed-size bank, no
orthogonality penalty).

models/lorasub.py's per-task K/V adapter bank grows by one slot every
task, forever. This variant bounds it at a FIXED 2 slots -- SKETCH (index
0, the folded history) and RESIDUAL (index 1, the slot actually being
trained) -- using the exact same QR-SVD merge SketchLoRA/InfLoRA-Sketch
already use (models/inflora_sketch.py::_sketch_fold /
backbone/vit_cllora_sketch.py::SketchQV.fold): concatenate SKETCH's and
RESIDUAL's factors, QR-factorize both sides, take a small SVD of the
resulting small matrix, truncate to svd_rank -- the TRUE SVD of the merged
delta, computed without ever forming the dense [dim,dim] delta except
transiently for the retained-energy diagnostic.

This is DIFFERENT from O-LoRA's bounded-bank story, which needs an
explicit orthogonality LOSS term between tasks to keep interference down.
The QR-SVD fold handles overlapping directions numerically (it is an exact
decomposition, not an approximation that needs a penalty to behave well),
so per instruction this variant adds NO orthogonality loss anywhere. The
training loop (CE + the DRS covariance-projected Adam step from
models/lorasub.py) is completely unchanged -- only the bank/fold mechanism
differs, via the two overrides below (_train_adapter routes to RESIDUAL,
incremental_train folds before freezing instead of growing a new slot).

The "subtraction" probe (backbone/vit_lorasub.py's subtract_mode) is
bank-structure-agnostic -- it always just skips ALL LoRA deltas (SKETCH
and RESIDUAL alike) during the probe forward, so it needs no changes here;
it already gives the same "pure frozen-base" covariance whether the live
slot being trained is a growing per-task slot or a fixed RESIDUAL slot.

Requires lora_n_slots=2 (asserted). svd_rank defaults to lora_rank (an
exact, lossless first fold -- same convention as InfLoRA-Sketch).
"""

import logging
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.lorasub import Learner as LoRASubLearner
from models.sketchlora import SKETCH, RESIDUAL

num_workers = 8


class Learner(LoRASubLearner):
    def __init__(self, args):
        super().__init__(args)
        assert args.get("lora_n_slots", 2) == 2, \
            "LoRA-Subtraction-Sketch is a fixed 2-slot method (SKETCH + RESIDUAL) -- " \
            "it never grows a per-task slot bank. Got lora_n_slots={}".format(
                args.get("lora_n_slots", 2))
        self.svd_rank = args.get("svd_rank", args.get("lora_rank", 10))
        self.merge_op = args.get("merge_op", "qrsvd")
        assert self.merge_op == "qrsvd", \
            "LoRA-Subtraction-Sketch only implements merge_op='qrsvd'. Got {}".format(self.merge_op)
        self._sketch_populated = False

    # -- fixed routing: RESIDUAL is always the slot currently being
    # trained, SKETCH is the folded history. --------------------------
    def _train_adapter(self):
        return RESIDUAL

    def _eval_adapter(self, task):
        return SKETCH

    # -- streaming-driver hooks (models/stream_mixin.py) -- lets this method
    # plug into ANY of that file's decoupled-boundary drivers (stream_run,
    # the plateau variants, and offset_consolidation_run) with the same
    # minimal 3-hook contract every other 2-slot sketch method uses (see
    # models/sketchlora.py's own _stream_slot/_stream_begin_chunk/
    # _stream_end_chunk for the template this mirrors). Routing is fixed
    # (always RESIDUAL, same as the oracle path's _train_adapter above) --
    # no period/lazy-merge cycling, since this method's own design is
    # "always fold when the driver says to." The DRS subtraction-probe +
    # projected-Adam machinery (models/lorasub.py's _ProjectedAdam) does
    # NOT run under these drivers -- they use StreamMixin's own generic
    # optimizer/training-step helpers (plain AdamW, plain CE), same
    # simplification every other non-SeqLoRA method already accepts under
    # stream_run(). Flagged, not hidden: this is a real reduction from the
    # oracle path's behavior, acceptable for exercising the fold/bank
    # mechanism under a different boundary schedule, not for reproducing
    # DRS's own accuracy numbers under streaming.
    def _stream_slot(self):
        return RESIDUAL

    def _stream_begin_chunk(self, loader):
        # Only here so _sketch_fold()'s own logging (keyed off self._cur_task)
        # prints a real, advancing segment count under streaming instead of
        # the -1 it would otherwise never leave (same reason
        # models/sketchlora.py's own _stream_begin_chunk does this) --
        # routing/freezing itself is StreamMixin's default, unchanged
        # (freeze_to_task(self._stream_slot()=RESIDUAL, train_a=True) + fresh
        # optimizer); no period/lazy-merge cycling to add on top of that,
        # unlike sketchlora.py's own override.
        self._cur_task += 1
        super()._stream_begin_chunk(loader)

    def _stream_end_chunk(self, loader):
        self._sketch_fold()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.data_manager = data_manager
        self._network.update_fc(self._total_classes)
        self._register_task_range()
        self._network.default_task = RESIDUAL
        logging.info("[LoRA-Sub-Sketch] Learning on {}-{}".format(self._known_classes, self._total_classes))

        # No add_task_slot() -- fixed 2-slot architecture, unlike
        # models/lorasub.py's growing bank. Fold the PREVIOUS task's fully
        # trained RESIDUAL into SKETCH first, then reset RESIDUAL fresh.
        if self._cur_task > 0:
            self._sketch_fold()

        self._network.freeze_to_task(RESIDUAL)  # train_a=True default: both A,B train
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
        # Inherited from models/lorasub.py unchanged: runs the DRS
        # subtraction-probe + projected-Adam training loop exactly as the
        # growing-bank variant does, just routed to RESIDUAL via
        # _train_adapter()/_optimizer_param_groups_split() above.
        self._train(self.train_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    @torch.no_grad()
    def _sketch_fold(self):
        """Exact QR-SVD merge of RESIDUAL into SKETCH -- same algorithm as
        models/inflora_sketch.py::_sketch_fold, applied to K/V instead of
        Q/V. No orthogonality penalty anywhere in this file or in the
        inherited training loop -- the merge is what keeps this bounded,
        not a loss term."""
        net = self._network.module if hasattr(self._network, "module") else self._network
        retained_energies = []
        for attn in net.attn_modules():
            for A_list, B_list in ((attn.lora_A_k, attn.lora_B_k), (attn.lora_A_v, attn.lora_B_v)):
                A_sk, B_sk = A_list[SKETCH].weight, B_list[SKETCH].weight
                A_res, B_res = A_list[RESIDUAL].weight, B_list[RESIDUAL].weight
                dt = B_sk.dtype

                if not self._sketch_populated:
                    # Nothing to merge yet -- transplant task 0's trained
                    # RESIDUAL into SKETCH verbatim (exact, zero loss).
                    assert A_res.shape[0] == self.svd_rank, \
                        "first-fold transplant needs svd_rank == lora_rank for an exact, " \
                        "lossless copy -- got lora_rank={} svd_rank={}".format(
                            A_res.shape[0], self.svd_rank)
                    A_list[SKETCH].weight.data.copy_(A_res)
                    B_list[SKETCH].weight.data.copy_(B_res)
                    retained_energies.append(1.0)
                else:
                    delta_true = (B_sk.float() @ A_sk.float()) + (B_res.float() @ A_res.float())
                    B_cat = torch.cat([B_sk, B_res], dim=1).float()      # [d, r_sk + rank]
                    A_cat = torch.cat([A_sk, A_res], dim=0).float()      # [r_sk + rank, d]
                    Q_B, R_B = torch.linalg.qr(B_cat)
                    Q_A, R_A = torch.linalg.qr(A_cat.t())
                    M_small = R_B @ R_A.t()
                    U_small, S, Vh_small = torch.linalg.svd(M_small)
                    U = Q_B @ U_small
                    Vh = Vh_small @ Q_A.t()

                    r = min(self.svd_rank, S.shape[0])
                    sqrtS = S[:r].sqrt()
                    B_new = (U[:, :r] * sqrtS).to(dt)
                    A_new = (sqrtS.unsqueeze(1) * Vh[:r, :]).to(dt)

                    recon = B_new.float() @ A_new.float()
                    fro_true = torch.linalg.norm(delta_true)
                    retained = 1.0 - (torch.linalg.norm(delta_true - recon) / fro_true.clamp_min(1e-12)) ** 2
                    retained_energies.append(retained.item())

                    A_list[SKETCH].weight.data.copy_(A_new)
                    B_list[SKETCH].weight.data.copy_(B_new)

                nn.init.kaiming_uniform_(A_list[RESIDUAL].weight, a=math.sqrt(5))
                nn.init.zeros_(B_list[RESIDUAL].weight)

        logging.info("[LoRA-Sub-Sketch] fold at task {}: retained_energy mean={:.4f} min={:.4f}".format(
            self._cur_task, sum(retained_energies) / len(retained_energies), min(retained_energies)))
        self._sketch_populated = True

    # -- memory accounting: SKETCH slot only (bounded) + head + protos;
    # RESIDUAL (live, not-yet-folded) excluded -- the whole point of this
    # variant, same convention InfLoRA-Sketch uses for its own
    # persistent_state(). self._protos (prototype repulsion, see
    # models/lorasub.py's module docstring) is inherited unchanged from the
    # base class -- it is bank-structure-agnostic, so it grows here exactly
    # as it does for the growing-bank variant (one 768-dim vector per class
    # ever seen), independent of SKETCH/RESIDUAL being fixed-size. -------
    def persistent_state(self):
        net = self._network.module if hasattr(self._network, "module") else self._network
        sketch_bytes = 0
        for attn in net.attn_modules():
            for mlist in (attn.lora_A_k, attn.lora_B_k, attn.lora_A_v, attn.lora_B_v):
                for p in mlist[SKETCH].parameters():
                    sketch_bytes += p.numel() * p.element_size()
        residual_bytes = 0
        for attn in net.attn_modules():
            for mlist in (attn.lora_A_k, attn.lora_B_k, attn.lora_A_v, attn.lora_B_v):
                for p in mlist[RESIDUAL].parameters():
                    residual_bytes += p.numel() * p.element_size()
        fc_bytes = sum(p.numel() * p.element_size() for p in net.fc.parameters())
        proto_bytes = sum(p.numel() * p.element_size() for p in self._protos.values())
        total_bytes = sketch_bytes + residual_bytes + fc_bytes + proto_bytes
        return {"params": int(total_bytes // 4), "bytes": int(total_bytes),
                "breakdown": {"sketch": sketch_bytes, "residual": residual_bytes,
                             "head": fc_bytes, "protos": proto_bytes}}

    def _log_memory_accounting(self):
        state = self.persistent_state()
        logging.info(
            "[LoRA-Sub-Sketch] task {} persistent bank: FIXED 2 slots (K+V, A+B, every "
            "block) = {:,} bytes (sketch={:,}, residual={:,}, head={:,}, protos={:,} "
            "[{} classes])".format(
                self._cur_task, state["bytes"], state["breakdown"]["sketch"],
                state["breakdown"]["residual"], state["breakdown"]["head"],
                state["breakdown"]["protos"], len(self._protos)))

    # -- 2026-11-05: summed-bank effective-rank diagnostic (utils/
    # rank_diagnostics.py) explicitly does NOT apply here -- deliberate
    # no-op override, not an oversight. The user's request was specifically
    # about the GROWING-BANK variants (O-LoRA, InfLoRA-Bank, LoRA-Sub) to
    # compare how their different cross-task mechanisms affect the summed
    # bank's rank as it grows over many tasks; this variant never has more
    # than 2 live slots (SKETCH+RESIDUAL) to sum, and SKETCH is already a
    # rank-capped (<=svd_rank) compressed representation by construction --
    # "effective rank of the combined update" here would just measure the
    # fixed svd_rank truncation this variant always applies, not a cross-
    # task redundancy/collapse effect, so it isn't the same diagnostic and
    # would be misleading reported under the same log tag as the bank
    # variants. Inherits models/lorasub.py's own flag/method otherwise --
    # this override replaces the base class's version, not log_bank_rank
    # itself, so setting the flag on a sketch config is simply inert (no
    # error, no output) rather than producing a confusing number.
    def _log_bank_effective_rank(self):
        return
