"""Offline analysis for the olora_sketch_fd_divergence campaign: computes, at
every task boundary, how far SketchLoRA's and FD Sum's fixed-rank sketches
diverge from O-LoRA's unbounded-rank concatenated adapter.

Three metrics per (dataset, seed, boundary):
  1. spectral_norm_diff: mean over (block, proj) modules of
     ||B_hat@A_hat - B_olora_cat@A_olora_cat||_2 (spectral norm of the DENSE
     UPDATE DIFFERENCE -- not a difference of norms), for SketchLoRA and for
     FD Sum each.
  2. cka_cumulative: CKA between {SketchLoRA, O-LoRA} and {FD Sum, O-LoRA}
     final features, evaluated on the union of every task's test set seen so
     far at that boundary (same images, same order, read off the matching
     "<method>_task{r}_boundary{n}.npz" / "sketch_task{r}_boundary{n}.npz"
     dumps for r in 0..n).
  3. cka_acute (SketchLoRA/FD Sum only, no O-LoRA equivalent -- O-LoRA never
     compresses): CKA between "exact_task{n}.npz" (the just-trained,
     uncompressed state) and "sketch_task{n}_boundary{n}.npz" (immediately
     after that boundary's fold), isolating one sketching operation's own
     representational cost from any later training.

Run once per (dataset, seed) after all three methods' runs for that
combination have finished -- reads directly from each method's own
run_logs/olora_sketch_fd_divergence/<method>/<dataset>/{snapshots,features}/
s<seed>/ directory (see scripts/gen_olora_sketch_fd_divergence_configs.py for
exactly where each config points embed_drift_dir/adapter_snapshot_dir).

Usage: python scripts/analyze_olora_sketch_fd_divergence.py --dataset cifar224 --seed 1993
       (writes run_logs/olora_sketch_fd_divergence/results_<dataset>_s<seed>.json)
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, ".")  # run from the project root: python scripts/analyze_olora_sketch_fd_divergence.py

import numpy as np
import torch

from utils.cka import linear_cka

CAMPAIGN_ROOT = "run_logs/olora_sketch_fd_divergence"


def method_dir(method, dataset, seed, kind):
    # kind: "snapshots" or "features"
    return os.path.join(CAMPAIGN_ROOT, method, dataset, kind, "s{}".format(seed))


def load_snapshot(method, dataset, seed, boundary):
    prefix = "olora" if method == "olora" else "sketch"
    path = os.path.join(method_dir(method, dataset, seed, "snapshots"), "{}_boundary{}.pt".format(prefix, boundary))
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu")


def available_boundaries(method, dataset, seed):
    prefix = "olora" if method == "olora" else "sketch"
    paths = glob.glob(os.path.join(method_dir(method, dataset, seed, "snapshots"), "{}_boundary*.pt".format(prefix)))
    out = []
    for p in paths:
        stem = os.path.basename(p).replace(".pt", "")
        out.append(int(stem.split("boundary")[-1]))
    return sorted(out)


def spectral_norm_diff(snap_a, snap_b):
    """Mean, over every (block,proj) module both snapshots share, of the
    spectral norm of (B_a@A_a - B_b@A_b). Upcasts to float32 for the SVD."""
    vals = []
    for key in snap_a["B"]:
        if key not in snap_b["B"]:
            continue
        Ba, Aa = snap_a["B"][key].float(), snap_a["A"][key].float()
        Bb, Ab = snap_b["B"][key].float(), snap_b["A"][key].float()
        dense_a = Ba @ Aa
        dense_b = Bb @ Ab
        diff = dense_a - dense_b
        vals.append(torch.linalg.matrix_norm(diff, ord=2).item())
    return float(np.mean(vals)) if vals else float("nan")


def load_feature_group(method, dataset, seed, fname):
    path = os.path.join(method_dir(method, dataset, seed, "features"), fname)
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return d["features"]


def cumulative_features(method, dataset, seed, boundary, task_prefix):
    """Concatenate task_prefix_task{r}_boundary{boundary}.npz for r in
    0..boundary, in task order -- matches how every method's own cumulative
    dump is laid out (olora_taskR_boundaryN.npz / sketch_taskR_boundaryN.npz)."""
    feats = []
    for r in range(boundary + 1):
        fname = "{}_task{}_boundary{}.npz".format(task_prefix, r, boundary)
        f = load_feature_group(method, dataset, seed, fname)
        if f is None:
            return None
        feats.append(f)
    return np.concatenate(feats, axis=0)


def analyze(dataset, seed):
    olora_boundaries = set(available_boundaries("olora", dataset, seed))
    sketch_boundaries = set(available_boundaries("sketchlora", dataset, seed))
    fdsum_boundaries = set(available_boundaries("fdsum", dataset, seed))
    common = sorted(olora_boundaries & sketch_boundaries & fdsum_boundaries)
    print("common boundaries: {} .. {} (n={})".format(
        common[0] if common else None, common[-1] if common else None, len(common)))

    results = {"dataset": dataset, "seed": seed, "boundaries": []}
    for n in common:
        snap_olora = load_snapshot("olora", dataset, seed, n)
        snap_sketch = load_snapshot("sketchlora", dataset, seed, n)
        snap_fdsum = load_snapshot("fdsum", dataset, seed, n)

        entry = {"boundary": n}
        entry["spectral_norm_diff_sketchlora_vs_olora"] = spectral_norm_diff(snap_sketch, snap_olora)
        entry["spectral_norm_diff_fdsum_vs_olora"] = spectral_norm_diff(snap_fdsum, snap_olora)

        feat_olora = cumulative_features("olora", dataset, seed, n, "olora")
        feat_sketch = cumulative_features("sketchlora", dataset, seed, n, "sketch")
        feat_fdsum = cumulative_features("fdsum", dataset, seed, n, "sketch")
        if feat_olora is not None and feat_sketch is not None:
            entry["cka_cumulative_sketchlora_vs_olora"] = linear_cka(feat_sketch, feat_olora)
        if feat_olora is not None and feat_fdsum is not None:
            entry["cka_cumulative_fdsum_vs_olora"] = linear_cka(feat_fdsum, feat_olora)

        exact_sketch = load_feature_group("sketchlora", dataset, seed, "exact_task{}.npz".format(n))
        post_sketch = load_feature_group("sketchlora", dataset, seed,
                                          "sketch_task{}_boundary{}.npz".format(n, n))
        if exact_sketch is not None and post_sketch is not None:
            entry["cka_acute_sketchlora"] = linear_cka(exact_sketch, post_sketch)

        exact_fdsum = load_feature_group("fdsum", dataset, seed, "exact_task{}.npz".format(n))
        post_fdsum = load_feature_group("fdsum", dataset, seed, "sketch_task{}_boundary{}.npz".format(n, n))
        if exact_fdsum is not None and post_fdsum is not None:
            entry["cka_acute_fdsum"] = linear_cka(exact_fdsum, post_fdsum)

        results["boundaries"].append(entry)
        print(n, {k: round(v, 4) if isinstance(v, float) else v for k, v in entry.items() if k != "boundary"})

    out_path = os.path.join(CAMPAIGN_ROOT, "results_{}_s{}.json".format(dataset, seed))
    os.makedirs(CAMPAIGN_ROOT, exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)
    print("wrote", out_path)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    analyze(args.dataset, args.seed)
