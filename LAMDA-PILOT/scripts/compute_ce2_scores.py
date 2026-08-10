"""Compute per-task and overall CE (computational efficiency) scores relative
to SeqLoRA from CE2 campaign output (utils/ce2_profiler.py / trainer.py's
ce2_enabled path).

CE for a method at a given unit (task, or the whole run) = that method's total
cost / SeqLoRA's total cost at the SAME unit. 1.0 = SeqLoRA parity; >1.0 =
more expensive; <1.0 = cheaper. Reported on two bases:
  - "train_only": just the train bucket (regular back/forward + any recurring
    extra per-step work, e.g. the shared dense-fold's heavier forward/backward
    or O-LoRA's orthogonality penalty) -- isolates cost that recurs every step,
    at the FIXED epoch/batch-size training budget every CE2 config shares.
  - "total": train + eval + boundary -- the method's full per-task footprint,
    including one-off between-task bookkeeping (SketchLoRA's SVD compress,
    InfLoRA's DualGPM passes, TUNA's classifer_align, etc.) and the eval pass.
Each basis is reported in MACs (the project's established CE currency) and in
wall-clock seconds (what a human actually experiences, includes any real
CPU-contention/profiler-overhead noise MACs can't see).

Usage: python scripts/compute_ce2_scores.py --out-dir run_logs/ce2 \
    --methods seqlora sketchlora olora inflora treelora cllora rainbowprompt ease tuna \
    --dataset omnibenchmark1k --seed 1993
"""
import argparse
import json
import os


def load_tasks(out_dir, method, dataset, seed, prefix_template):
    prefix = prefix_template.format(method=method, seed=seed)
    tag = "{}_{}_{}_s{}".format(method, dataset, prefix, seed)
    path = os.path.join(out_dir, method, "ce2_{}.json".format(tag))
    if not os.path.isfile(path):
        return None, path
    with open(path) as f:
        data = json.load(f)
    return data.get("tasks", []), path


def bucket_total(task, basis):
    train = task["train"]
    if basis == "train_only":
        return dict(train)
    out = dict(train)
    for k in out:
        out[k] = out[k] + task["eval"].get(k, 0.0) + task["boundary"].get(k, 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="run_logs/ce2")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--prefix-template", default="ce2_omnibenchmark1k_10t_{method}_s{seed}")
    ap.add_argument("--baseline", default="seqlora")
    args = ap.parse_args()

    tasks_by_method = {}
    missing = []
    for m in args.methods:
        tasks, path = load_tasks(args.out_dir, m, args.dataset, args.seed, args.prefix_template)
        if tasks is None:
            missing.append((m, path))
            continue
        tasks_by_method[m] = tasks

    if missing:
        print("MISSING (not yet run or path mismatch):")
        for m, path in missing:
            print("  {} -> expected {}".format(m, path))
        print()

    if args.baseline not in tasks_by_method:
        print("FATAL: baseline method '{}' has no CE2 data -- cannot score anything.".format(
            args.baseline))
        return
    baseline_tasks = tasks_by_method[args.baseline]
    n_tasks = len(baseline_tasks)
    print("Baseline: {} ({} tasks)\n".format(args.baseline, n_tasks))

    summary = {}
    for m, tasks in tasks_by_method.items():
        if len(tasks) != n_tasks:
            print("WARNING: {} has {} task records, baseline has {} -- "
                  "scoring only the first {}.".format(m, len(tasks), n_tasks,
                                                        min(len(tasks), n_tasks)))
        n = min(len(tasks), n_tasks)

        method_summary = {"per_task": [], "overall": {}}
        for basis in ("train_only", "total"):
            per_task_macs, per_task_wall = [], []
            sum_m_macs = sum_b_macs = 0.0
            sum_m_wall = sum_b_wall = 0.0
            for t in range(n):
                m_bucket = bucket_total(tasks[t], basis)
                b_bucket = bucket_total(baseline_tasks[t], basis)
                sum_m_macs += m_bucket["macs"]
                sum_b_macs += b_bucket["macs"]
                sum_m_wall += m_bucket["wall_seconds"]
                sum_b_wall += b_bucket["wall_seconds"]
                per_task_macs.append(
                    m_bucket["macs"] / b_bucket["macs"] if b_bucket["macs"] > 0 else None)
                per_task_wall.append(
                    m_bucket["wall_seconds"] / b_bucket["wall_seconds"]
                    if b_bucket["wall_seconds"] > 0 else None)
            method_summary["overall"]["{}_macs".format(basis)] = (
                sum_m_macs / sum_b_macs if sum_b_macs > 0 else None)
            method_summary["overall"]["{}_wall".format(basis)] = (
                sum_m_wall / sum_b_wall if sum_b_wall > 0 else None)
            if basis == "train_only":
                method_summary["per_task_train_macs"] = per_task_macs
                method_summary["per_task_train_wall"] = per_task_wall
            else:
                method_summary["per_task_total_macs"] = per_task_macs
                method_summary["per_task_total_wall"] = per_task_wall
        summary[m] = method_summary

    print("{:14s} {:>14s} {:>14s} {:>14s} {:>14s}".format(
        "method", "CE_train_macs", "CE_total_macs", "CE_train_wall", "CE_total_wall"))
    for m in args.methods:
        if m not in summary:
            continue
        o = summary[m]["overall"]

        def fmt(v):
            return "{:.3f}".format(v) if v is not None else "n/a"
        print("{:14s} {:>14s} {:>14s} {:>14s} {:>14s}".format(
            m, fmt(o.get("train_only_macs")), fmt(o.get("total_macs")),
            fmt(o.get("train_only_wall")), fmt(o.get("total_wall"))))

    print("\nPer-task CE_train_macs curve (method gradually incurring more/less "
          "compute relative to SeqLoRA, task by task):")
    for m in args.methods:
        if m not in summary:
            continue
        curve = summary[m]["per_task_train_macs"]
        curve_str = ", ".join("{:.3f}".format(v) if v is not None else "n/a" for v in curve)
        print("  {:14s} [{}]".format(m, curve_str))

    out_path = os.path.join(args.out_dir, "ce2_scores_{}_s{}.json".format(args.dataset, args.seed))
    with open(out_path, "w") as f:
        json.dump(dict(baseline=args.baseline, dataset=args.dataset, seed=args.seed,
                        summary=summary, missing=[m for m, _ in missing]), f, indent=2)
    print("\nWrote {}".format(out_path))


if __name__ == "__main__":
    main()
