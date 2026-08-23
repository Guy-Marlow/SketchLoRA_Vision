"""Rank-truncation accuracy probe (2026-08-21 user request): trains a
per-task LoRA adapter in true isolation (models/lora.py's base Learner,
lora_merge/lora_train_merge left at their default False -- merge=False
routing means each task's adapter is trained on ONLY that task's own data,
never summed with any other slot, no orthogonality/regularizer term either
-- a fresh Kaiming/zero-init rank-10 adapter per task, no history), then at
the end of each task: exact-SVD-decomposes that task's own (B_t @ A_t) per
wrapped module, and re-evaluates the SAME task's own held-out test set
after truncating the smallest 1..(r-1) singular values (9 truncation
levels for the project's standard rank-10 adapters, plus the untruncated
rank-10 baseline as truncation level 0) -- answering "how much of this
adapter's own classification accuracy survives as its smaller singular
directions are stripped out."

SCOPE, deliberately narrow, matching the user's own confirmed answers:
  - "the adapter update" = this task's own residual alone (NOT SketchLoRA's
    sketch+residual composite, which grows past rank 10 after task 0 and
    would not fit a clean "9 truncations to rank-1" sweep).
  - accuracy = task-local only (this task's own classes, this task's own
    (truncated) adapter alone, merge=False) -- isolates rank-truncation
    fidelity from any continual-learning/forgetting effect.

Truncation is per-module, independent, at a UNIFORM level across all 24
wrapped (block,proj) modules per truncation step -- e.g. "truncation level
3" means every one of the 24 modules has its own 3 smallest singular
values (by that module's OWN spectrum) dropped, matching this project's
existing sketch_diag convention of reporting per-module diagnostics at a
shared truncation level (r_hat/fro/retained_energy), not a single pooled
spectrum across modules.

Truncated (A,B) are zero-padded back to the ORIGINAL rank-r Linear layer
shape (dropped directions' rows/columns set to exactly zero) rather than
reallocating narrower Linear layers -- avoids touching module structure,
and "memory consumed by the sketched factors" is reported analytically
from the KEPT rank only (rank_kept * 2*dim*4 bytes * 24 modules), not from
the (deliberately oversized, zero-padded) tensor's actual allocation.
"""
import json
import logging
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.lora import Learner as LoRALearner, num_workers

MODULES_TOTAL = 24        # 12 blocks x {q, v}
BYTES_PER_RANK_UNIT = 2 * 768 * 4 * MODULES_TOTAL   # A+B, all modules, fp32


class Learner(LoRALearner):
    def __init__(self, args):
        super().__init__(args)
        assert not self.train_merge, \
            "rank-truncation probe requires isolated per-task training (merge=False) -- " \
            "set neither lora_merge nor lora_train_merge in this config"
        self.rank_probe_out = args.get(
            "rank_probe_out", "run_logs/lora_rank_probe/results.json")
        self._rank_probe_results = []

    def _attns(self):
        net = self._network.module if hasattr(self._network, "module") else self._network
        return net.attn_modules()

    def _train(self, train_loader):
        super()._train(train_loader)
        self._run_rank_probe()

    @torch.no_grad()
    def _eval_task_local(self, loader, lo, hi, t):
        """merge=False, routed to slot t, masked to this task's own class
        slice [lo,hi) -- top1/top5 + wall-clock ms/image (CUDA-synced)."""
        net = self._network.module if hasattr(self._network, "module") else self._network
        net.eval()
        correct1 = correct5 = total = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _, inputs, targets in loader:
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            logits = net(inputs, task=t, merge=False)["logits"][:, lo:hi]
            local_targets = targets - lo
            top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
            correct1 += top5[:, :1].eq(local_targets.unsqueeze(1)).sum().item()
            correct5 += top5.eq(local_targets.unsqueeze(1)).any(dim=1).sum().item()
            total += len(targets)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "top1": round(100.0 * correct1 / total, 3),
            "top5": round(100.0 * correct5 / total, 3),
            "ms_per_image": round(elapsed_ms / total, 5),
            "n_images": total,
        }

    @torch.no_grad()
    def _run_rank_probe(self):
        t = self._train_adapter()
        lo, hi = self._ce_slice()
        r = None
        module_state = []   # (A_list, B_list, A_orig, B_orig, U, S, Vh) per wrapped module
        singular_values = []   # 24 x r, per-module spectra

        for attn in self._attns():
            for A_list, B_list in ((attn.lora_A_q, attn.lora_B_q), (attn.lora_A_v, attn.lora_B_v)):
                A = A_list[t].weight.detach().clone()   # [r, dim]
                B = B_list[t].weight.detach().clone()   # [dim, r]
                r = A.shape[0]
                delta = (B.float() @ A.float())          # [dim, dim], exact rank <= r
                U, S, Vh = torch.linalg.svd(delta)
                singular_values.append(S[:r].cpu().tolist())
                module_state.append((A_list, B_list, A, B, U, S, Vh))

        test_dataset = self.data_manager.get_dataset(
            np.arange(lo, hi), source="test", mode="test")
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                  shuffle=False, num_workers=num_workers)

        def apply_truncation(keep):
            fros = []
            for (A_list, B_list, A_orig, B_orig, U, S, Vh) in module_state:
                if keep == r:
                    A_list[t].weight.data.copy_(A_orig)
                    B_list[t].weight.data.copy_(B_orig)
                    fro = (B_orig.float() @ A_orig.float()).norm().item()
                else:
                    root_S = S[:keep].sqrt()
                    B_hat = (U[:, :keep] * root_S.unsqueeze(0))
                    A_hat = (root_S.unsqueeze(1) * Vh[:keep, :])
                    fro = (B_hat @ A_hat).norm().item()
                    A_full = torch.zeros_like(A_orig)
                    B_full = torch.zeros_like(B_orig)
                    A_full[:keep] = A_hat.to(A_orig.dtype)
                    B_full[:, :keep] = B_hat.to(B_orig.dtype)
                    A_list[t].weight.data.copy_(A_full)
                    B_list[t].weight.data.copy_(B_full)
                fros.append(fro)
            return fros

        truncations = []
        for removed in range(0, r):   # 0 = untruncated baseline, 1..r-1 = removed smallest k
            keep = r - removed
            fros = apply_truncation(keep)
            metrics = self._eval_task_local(test_loader, lo, hi, t)
            truncations.append({
                "removed_smallest": removed,
                "rank_kept": keep,
                "accuracy": {"top1": metrics["top1"], "top5": metrics["top5"]},
                "inference_ms_per_image": metrics["ms_per_image"],
                "n_test_images": metrics["n_images"],
                "adapter_weight_frobenius_mean": round(float(np.mean(fros)), 5),
                "adapter_weight_frobenius_per_module": [round(f, 5) for f in fros],
                "memory_mb": round(keep * BYTES_PER_RANK_UNIT / 1024**2, 5),
            })
            logging.info(
                "[RankProbe] task {} removed_smallest={} rank_kept={} top1={:.2f} top5={:.2f} "
                "ms/img={:.4f} mem={:.3f}MB".format(
                    self._cur_task, removed, keep, metrics["top1"], metrics["top5"],
                    metrics["ms_per_image"], truncations[-1]["memory_mb"]))

        # restore the real trained weights before continuing to the next task
        apply_truncation(r)

        self._rank_probe_results.append({
            "task": self._cur_task,
            "class_range": [int(lo), int(hi)],
            "lora_rank": r,
            "singular_values_per_module": singular_values,
            "singular_values_mean": [
                round(float(np.mean([sv[i] for sv in singular_values])), 5) for i in range(r)],
            "truncations": truncations,
        })
        os.makedirs(os.path.dirname(self.rank_probe_out), exist_ok=True)
        json.dump(self._rank_probe_results, open(self.rank_probe_out, "w"), indent=2)
        logging.info("[RankProbe] task {} complete, results written to {}".format(
            self._cur_task, self.rank_probe_out))
