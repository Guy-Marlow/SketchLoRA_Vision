"""Unified metrics logger for the final-experiments harness (Experiments_Timeline.pdf).

Wraps trainer.py's existing task loop (task / sample / budget boundary modes alike)
without replacing its accuracy-matrix bookkeeping -- this module ADDS the metrics the
PDF requires that trainer.py doesn't already compute (timing, peak memory, persistent
storage, FLOPs/latency, boundary overhead) and derives the standard CL summary
statistics (FAA, AIA, forgetting, BWT, last-task accuracy) from the SAME cnn_matrix /
til_matrix trainer.py already assembles, using the identical forgetting formula
trainer.py's own `print_forget` branch uses (trainer.py:150) so numbers agree exactly.

Required per-learner hook: `persistent_state(self) -> dict` with keys
{"params": int, "bytes": int, "breakdown": {...}}. `default_persistent_state` below is a
conservative fallback; every method in the final experiment grid should override this
explicitly with an exact accounting of its own persistent structures (adapter banks,
prompt pools, DualGPM feature lists, class statistics, etc.) -- see models/*.py.

Output: run_logs/final/<family>/metrics_<model>_<dataset>_<regime>_s<seed>.json,
written incrementally after every task/chunk (crash-safe; a run can be resumed/audited
mid-flight) and finalized (status: "done") after the last task.
"""

import io
import json
import logging
import os
import time

import torch

from utils.flops import measure_inference_flops


def default_persistent_state(learner):
    """Fallback persistent_state: total bytes of every parameter that isn't obviously
    part of the frozen backbone (best-effort heuristic on tensor name). Every method in
    the final grid should override this with an exact accounting instead of relying on
    the heuristic -- see the method-specific overrides in models/*.py."""
    net = learner._network.module if hasattr(learner._network, "module") else learner._network
    backbone_markers = ("patch_embed", "cls_token", "pos_embed", "norm.", "q_proj.weight",
                        "k_proj.weight", "v_proj.weight", "proj.weight", "proj.bias",
                        "fc1.", "fc2.")
    total_params, total_bytes = 0, 0
    for name, p in net.named_parameters():
        is_backbone = any(m in name for m in backbone_markers) and "lora" not in name.lower()
        if is_backbone:
            continue
        total_params += p.numel()
        total_bytes += p.numel() * p.element_size()
    return {"params": total_params, "bytes": total_bytes,
            "breakdown": {"heuristic_non_backbone": total_bytes}}


def safe_persistent_state(learner):
    if hasattr(learner, "persistent_state"):
        try:
            return learner.persistent_state()
        except Exception as e:
            logging.warning("persistent_state() failed ({}); falling back to heuristic".format(e))
    return default_persistent_state(learner)


def serialized_bytes(state_dict_like):
    """Actual torch.save-serialized size of a state dict / tensor collection, for the
    'persistent storage' metric (distinct from raw numel*element_size, which ignores
    torch.save's pickle/container overhead -- usually small but this is the honest
    number for a metric literally named 'persistent storage')."""
    buf = io.BytesIO()
    torch.save(state_dict_like, buf)
    return buf.tell()


def compute_cl_summary(matrix, last_task_idx):
    """FAA / AIA / forgetting / BWT / last-task-accuracy from an accuracy MATRIX.

    `matrix` is a list of (task_idx, row) tuples -- a "row" is the per-task-bucket
    accuracy list recorded at a CIL-eval CHECKPOINT right after training task_idx
    (ragged: row has task_idx+1 entries). task_idx need not be contiguous: this is
    sparse-checkpoint safe (e.g. OmniBenchmark-1K's every-5-tasks CIL eval cadence,
    trainer.py's `is_checkpoint` gate) and reduces to the original dense formula
    exactly when every task is a checkpoint (every dataset other than Omni-1K).

    FAA only reads the FINAL checkpoint's row, which is always present (trainer.py
    forces the last task to be a checkpoint regardless of cadence) -- unaffected by
    sparsity. Forgetting's "max ever recorded" only ever takes a max over whatever
    checkpoints exist; a 0-filled not-yet-existing/not-measured cell can never
    spuriously beat a real (non-negative) accuracy under max, so it's sparse-safe
    as-is, just resolution-limited by however many checkpoints actually happened.
    AIA and BWT are NOT naturally sparse-safe and are redefined below:
      - AIA: originally averaged one term per real task (assumed a dense row every
        task); now averages over the CHECKPOINTS that actually happened instead.
      - BWT: originally read the literal diagonal (a task's accuracy measured
        exactly at its own task boundary); under sparse eval that value doesn't
        exist for most tasks, so it's redefined as each task's FIRST-EVER-RECORDED
        accuracy (the earliest checkpoint that happened to include it) -- the
        standard adjustment for sparse-checkpoint CL evaluation. When every task is
        a checkpoint, the first checkpoint >= task j IS task j itself, so this is
        exactly the old diagonal -- no change for non-Omni datasets.
    Returns None fields if fewer than two checkpoints exist (nothing to measure
    forgetting/BWT against yet)."""
    import numpy as np
    n = last_task_idx + 1
    acc = np.zeros((n, n))
    for task_idx, row in matrix:
        acc[task_idx, :len(row)] = np.array(row)
    acc = acc.T  # [task_bucket, checkpoint_task_idx]: acc[j, i] = accuracy of task j at checkpoint i

    faa = float(np.mean(acc[:, -1]))                                    # final checkpoint's row
    aia = float(np.mean([np.mean(row) for _, row in matrix]))           # avg over CHECKPOINTS only
    last_task_acc = float(acc[-1, -1])
    if n > 1 and len(matrix) > 1:
        forgetting = float(np.mean((np.max(acc, axis=1) - acc[:, -1])[:-1]))
        checkpoint_idxs = sorted(idx for idx, _ in matrix)
        first_seen = np.full(n, np.nan)
        for j in range(n - 1):
            for ci in checkpoint_idxs:
                if ci >= j:
                    first_seen[j] = acc[j, ci]
                    break
        valid = ~np.isnan(first_seen[:-1])
        bwt = float(np.mean(acc[:-1, -1][valid] - first_seen[:-1][valid])) if valid.any() else None
    else:
        forgetting = None
        bwt = None
    return {"final_average_accuracy": round(faa, 4),
            "average_incremental_accuracy": round(aia, 4),
            "forgetting": round(forgetting, 4) if forgetting is not None else None,
            "backward_transfer": round(bwt, 4) if bwt is not None else None,
            "last_task_accuracy": round(last_task_acc, 4)}


class MetricsLogger:
    def __init__(self, out_dir, tag, args):
        os.makedirs(out_dir, exist_ok=True)
        self.out_path = os.path.join(out_dir, "metrics_{}.json".format(tag))
        keep = ("model_name", "dataset", "init_cls", "increment", "seed", "scenario",
                "boundary_mode", "budget_mb", "bm_budget_mb", "n_lora_blocks", "lora_rank",
                "lora_alpha", "svd_energy_target", "svd_rank", "svd_period", "merge_op",
                "cs_rank", "lamda_1", "lamda_2", "lamb", "lame")
        self.meta = {
            "args_subset": {k: args.get(k) for k in keep},
            "status": "running",
            "per_task": [],
            "train_seconds_total": 0.0,
            "eval_seconds_total": 0.0,
            "boundary_seconds_total": 0.0,
        }
        # args["device"][0] is already a concrete torch.device (e.g. cuda:4) by
        # this point (trainer.py::_set_device runs well before MetricsLogger is
        # constructed) -- reset/query stats on THIS device explicitly. Found live
        # 2026-07-19: torch.cuda.max_memory_allocated() with no device arg reports
        # cuda:0 (the process-default), which silently returns ~0 for any run
        # pinned to a different GPU -- every prior final_metrics run on a non-zero
        # GPU index has logged peak_vram_mb: 0.0 as a result, not a real reading.
        self._device = args.get("device", [None])[0]
        self._device_stats_ok = False
        if torch.cuda.is_available() and self._device is not None and self._device.type == "cuda":
            try:
                # MetricsLogger is constructed before the model/any tensor has
                # touched this device, so CUDA's lazy per-device context doesn't
                # exist yet -- reset_peak_memory_stats(device) fails with
                # "Invalid device argument: did you call init?" without this.
                # A trivial allocation forces that context into existence first.
                torch.zeros(1, device=self._device)
                torch.cuda.reset_peak_memory_stats(self._device)
                self._device_stats_ok = True
            except Exception as e:
                logging.warning("[metrics] could not init per-device memory stats "
                                "for {} ({}); peak_vram_mb will be null".format(self._device, e))
        self._write()
        self._t_task_start = None
        self._t_train_end = None

    def _write(self):
        with open(self.out_path, "w") as f:
            json.dump(self.meta, f, indent=2, default=str)

    # ---- per-task timing hooks (call around the existing trainer.py loop body) ----
    def begin_task(self):
        self._t_task_start = time.time()
        self._t_train_end = None

    def mark_train_done(self):
        """Call immediately after incremental_train() returns, before eval_task()."""
        self._t_train_end = time.time()

    def record_task(self, learner, task_idx, cnn_accy, til_accy, boundary_seconds=0.0):
        """Call once per task/chunk, right after eval_task() + after_task(). Returns the
        record it wrote (also appended to self.meta['per_task'])."""
        now = time.time()
        train_s = (self._t_train_end - self._t_task_start) if self._t_train_end else None
        eval_s = (now - self._t_train_end) if self._t_train_end else None
        pstate = safe_persistent_state(learner)
        peak_mb = (round(torch.cuda.max_memory_allocated(self._device) / 1024 / 1024, 1)
                  if self._device_stats_ok else None)
        net = learner._network.module if hasattr(learner._network, "module") else learner._network
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        rec = {
            "task": task_idx,
            "train_seconds": round(train_s, 3) if train_s is not None else None,
            "eval_seconds": round(eval_s, 3) if eval_s is not None else None,
            "boundary_seconds": round(boundary_seconds, 4),
            "cil_top1": cnn_accy["top1"] if cnn_accy else None,
            "cil_top5": cnn_accy["top5"] if cnn_accy else None,
            "til_top1": til_accy["top1"] if til_accy else None,
            "til_top5": til_accy["top5"] if til_accy else None,
            # "trainable" = currently-unfrozen parameter count at this task's own
            # training time (a per-task snapshot, distinct from "persistent" below,
            # which is the method's permanently-carried-forward state).
            "trainable_params": int(trainable_params),
            "persistent_params": pstate["params"],
            "persistent_mb": round(pstate["bytes"] / 1024 / 1024, 4),
            "persistent_state_breakdown": pstate.get("breakdown"),
            "peak_vram_mb": peak_mb,
        }
        self.meta["per_task"].append(rec)
        if train_s is not None:
            self.meta["train_seconds_total"] += train_s
        if eval_s is not None:
            self.meta["eval_seconds_total"] += eval_s
        self.meta["boundary_seconds_total"] += boundary_seconds
        self._write()
        return rec

    # ---- inference cost (call once, after the final task) ----
    def record_inference_cost(self, learner, test_loader, n_batches=10):
        """Measured latency over up to n_batches of the deployed eval path, plus
        analytic FLOPs/image via utils/flops.py::measure_inference_flops (PyTorch's
        built-in profiler, a single forward pass -- not empirical timing of anything
        expensive, no per-batch SVDs/rank computations). Requires learner.
        _deployed_forward(inputs) (models/lora.py + models/hidelora.py +
        models/{ease,tuna,rainbowprompt,progprompt}.py all provide it) -- skips
        this metric (logs a warning, does not fail the run) if absent, since
        several non-final-grid methods sharing this trainer.py (icarl/der/foster/
        aper_* etc.) use unrelated forward signatures this must not assume."""
        if not hasattr(learner, "_deployed_forward"):
            logging.warning("[metrics] {} has no _deployed_forward; skipping inference-cost "
                            "measurement".format(type(learner).__name__))
            return
        net = learner._network
        net.eval()
        device = learner._device
        times = []
        n_images = 0
        sample_batch = None
        try:
            with torch.no_grad():
                for i, (_, inputs, _t) in enumerate(test_loader):
                    if i >= n_batches:
                        break
                    inputs = inputs.to(device)
                    if sample_batch is None:
                        sample_batch = inputs
                    if device != "cpu" and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.time()
                    learner._deployed_forward(inputs)
                    if device != "cpu" and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    times.append(time.time() - t0)
                    n_images += inputs.shape[0]
        except Exception as e:
            logging.warning("[metrics] inference-cost measurement failed ({}); skipping".format(e))
            return
        ms_per_image = (sum(times) / max(n_images, 1)) * 1000.0 if times else None
        flops_total = measure_inference_flops(learner, sample_batch) if sample_batch is not None else None
        flops_per_image = (flops_total / sample_batch.shape[0]) if flops_total is not None else None
        self.meta["inference_ms_per_image"] = round(ms_per_image, 4) if ms_per_image else None
        self.meta["inference_flops_per_image"] = round(flops_per_image, 1) if flops_per_image else None
        self._write()

    # ---- finalize (call once, after the loop ends) ----
    def finalize(self, cnn_matrix, til_matrix, last_task_idx):
        if cnn_matrix:
            self.meta["cil_summary"] = compute_cl_summary(cnn_matrix, last_task_idx)
        if til_matrix:
            self.meta["til_summary"] = compute_cl_summary(til_matrix, last_task_idx)
        self.meta["status"] = "done"
        self._write()
        logging.info("[metrics] wrote {}".format(self.out_path))
