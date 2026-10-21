"""Low-rank adapter factor persistence for the olora_sketch_fd_divergence
campaign (2026-10-21 user design): at each task boundary, O-LoRA, SketchLoRA
(merge_op=randsvd), and FD Sum (merge_op=freqdir_dense) each snapshot their
CURRENT (B, A) factors per wrapped (block, projection) module, so the dense
update B@A -- or O-LoRA's much-higher-rank concatenated B_cat@A_cat -- can be
reconstructed offline for the spectral-norm-of-difference and CKA
comparisons, without needing to re-run training.

Stored as a single torch.save payload per boundary per run:
    {"B": {"<block>_<proj>": Tensor[out, r_total]},
     "A": {"<block>_<proj>": Tensor[r_total, in]},
     "meta": {...}}
B/A are detached, moved to CPU, kept at their native (usually fp16/bf16)
dtype -- callers needing float precision for SVD cast on load, not on save
(keeps these files roughly half the size of a float32 dump).
"""
import logging
import os

import torch


def save_adapter_snapshot(out_dir, fname, layer_proj_to_BA, **meta):
    """layer_proj_to_BA: dict {(block_idx, proj): (B_tensor, A_tensor)} --
    B_tensor [out, r_total], A_tensor [r_total, in] for that module. r_total
    may differ from a single slot's rank (O-LoRA's grows with task count;
    SketchLoRA/FD Sum's stays fixed at svd_rank)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    payload = {
        "B": {"{}_{}".format(li, p): B.detach().cpu() for (li, p), (B, A) in layer_proj_to_BA.items()},
        "A": {"{}_{}".format(li, p): A.detach().cpu() for (li, p), (B, A) in layer_proj_to_BA.items()},
        "meta": meta,
    }
    torch.save(payload, path)
    one_shape = next(iter(payload["B"].values())).shape
    logging.info("[adapter_snapshot] wrote {} ({} modules, e.g. B{} )".format(
        path, len(payload["B"]), tuple(one_shape)))
