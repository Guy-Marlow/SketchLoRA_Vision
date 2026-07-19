import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from utils.metrics_logger import MetricsLogger
import os
import numpy as np


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["backbone_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )

    # Memory-budget streaming codepath (utils/budget_stream.py): "tasks" become
    # class-contiguous memory chunks (Experiments_Timeline.pdf headline result),
    # decoupled from the dataset's init_cls/increment task groups. Wraps
    # data_manager BEFORE model construction so nb_tasks/nb_classes (read by
    # e.g. InfLoRA's DualGPM total_sessions) reflect the chunk count, then reuses
    # the standard per-task loop below unchanged -- every Learner's
    # incremental_train already only depends on the DataManager interface.
    if args.get("boundary_mode") == "budget":
        from utils.budget_stream import BudgetStreamManager
        data_manager = BudgetStreamManager(data_manager, args["budget_mb"], args["seed"])
        logging.info("[budget] dataset={} budget={}MB -> {} memory-chunks (seed {})".format(
            args["dataset"], args["budget_mb"], data_manager.nb_tasks, args["seed"]))

    args["nb_classes"] = data_manager.nb_classes # update args
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)

    # Unique-image memory-budget streaming codepath (models/stream_mixin.py): the
    # full dataset is partitioned into fixed-size chunks of exactly stream_budget_mb
    # worth of UNIQUE images (task-blocked, each task internally shuffled once), with
    # no image ever repeated or skipped across chunks. Classifier head growth and CE
    # loss range are derived from each chunk's own actual class content (which can
    # span a task boundary); only each method's own adapter bookkeeping (fold/new
    # slot/compress/etc) is decoupled from real task boundaries, firing once per
    # chunk. Eval fires whenever a chunk completes one or more real tasks. Leaves the
    # standard per-task loop below untouched. REDESIGNED 2026-07-19 (supersedes the
    # 2026-07-18 epoch-repeated-sample-count clock, which counted training the SAME
    # images across multiple epochs as new "memory usage" -- see BOUNDARY_AGNOSTIC_
    # IMPLEMENTATION_LOG.md for why that made the budget a proxy for compute/time
    # elapsed rather than genuine unique-data volume, and broke down entirely for
    # dense datasets like Food101 whose per-epoch task volume alone exceeded even the
    # larger of two tested budgets).
    if args.get("boundary_mode") == "sample":
        _run_stream(model, data_manager, args)
        return

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix, til_matrix = [], [], []

    # stop_after_tasks: optionally train/eval only the first N tasks of the regime
    # (smoke/LR-probe runs). args["nb_tasks"] stays the FULL count so per-method
    # schedules (e.g. InfLoRA's DualGPM threshold ramp over total_sessions) see the
    # real regime; only the loop is shortened. None/absent = full run.
    _n_run = args.get("stop_after_tasks") or data_manager.nb_tasks
    _n_run = min(_n_run, data_manager.nb_tasks)
    if _n_run < data_manager.nb_tasks:
        logging.info("stop_after_tasks={} (of {} total)".format(_n_run, data_manager.nb_tasks))

    # Final-experiments metrics (Experiments_Timeline.pdf): opt-in via "final_metrics":
    # true in the config, so every OTHER experiment track sharing this trainer.py
    # (icarl/der/foster/aper_*/ranpac/... none of which are part of the final grid) is
    # completely unaffected. Additive only -- writes its own JSON, changes no existing
    # printed/logged output.
    mlog = None
    if args.get("final_metrics"):
        _tag = "{}_{}_{}_s{}".format(
            args["model_name"], args["dataset"],
            args.get("prefix") or (args.get("boundary_mode") or "task"), args["seed"])
        mlog = MetricsLogger(os.path.join("run_logs", "final", args["model_name"]), _tag, args)

    for task in range(_n_run):
        if args.get("boundary_mode") == "budget":
            # Reseed the global RNG deterministically per chunk so every method's
            # unmodified `DataLoader(..., shuffle=True)` draws the SAME reproducible
            # 10-epoch shuffle sequence at a given (seed, budget, dataset, chunk) --
            # the cross-method fairness requirement agreed for this regime (no Learner
            # file needs to change; this only affects which random draws come next).
            torch.manual_seed(args["seed"] * 1_000_003 + task * 97 + 13)
        if mlog:
            mlog.begin_task()
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        if mlog:
            mlog.mark_train_done()
        cnn_accy, nme_accy = model.eval_task()
        # per-task TIL accuracies (task-aware), parallel to the CNN/CIL matrix
        til_row = getattr(model, "_til_per_task", None)
        if til_row is not None:
            til_matrix.append(list(til_row))
        model.after_task()
        if mlog:
            mlog.record_task(model, task, cnn_accy, getattr(model, "_til_accy", None))

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]    
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_values = [nme_accy["grouped"][key] for key in nme_keys]
            nme_matrix.append(nme_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"])/len(nme_curve["top1"]))

            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {} \n".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

    if mlog:
        test_loader = getattr(model, "test_loader", None)
        if test_loader is not None:
            mlog.record_inference_cost(model, test_loader)
        mlog.finalize(cnn_matrix, til_matrix, task)

    if 'print_forget' in args.keys() and args['print_forget'] is True:
        if len(cnn_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(cnn_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            print('Accuracy Matrix (CNN):')
            print(np.array2string(np_acctable, max_line_width=100000, precision=2, suppress_small=True))
            logging.info('Forgetting (CNN): {}'.format(forgetting))
        if len(nme_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(nme_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            print('Accuracy Matrix (NME):')
            print(np.array2string(np_acctable, max_line_width=100000, precision=2, suppress_small=True))
        logging.info('Forgetting (NME): {}'.format(forgetting))
        # Task-incremental accuracy matrix (task id known; logits masked to each
        # task's class slice). Last column = final per-task TIL retention; its
        # mean = the TIL top1. Parallel to the CNN/CIL matrix above.
        if len(til_matrix) > 0:
            np_til = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(til_matrix):
                np_til[idxx, :len(line)] = np.array(line)
            np_til = np_til.T
            til_forgetting = np.mean((np.max(np_til, axis=1) - np_til[:, task])[:task])
            print('Accuracy Matrix (TIL):')
            print(np.array2string(np_til, max_line_width=100000, precision=2, suppress_small=True))
            logging.info('Forgetting (TIL): {}'.format(til_forgetting))


def _run_stream(model, data_manager, args):
    """Drive the sample-boundary streaming run and report CIL/TIL curves over the
    completed-task checkpoints (the model's adapter events are on the sample clock)."""
    results = model.stream_run(data_manager, args)
    comp = [r["completed"] for r in results]
    cil = [r["cil"] for r in results]
    til = [r["til"] for r in results]   # None entries unless "stream_til": true
    logging.info("[stream] completed-task checkpoints: {}".format(comp))
    logging.info("[stream] CIL curve: {}".format(cil))
    print("[stream] checkpoints (completed tasks):", comp)
    print("[stream] CIL:", cil)
    if til[-1] is not None:
        logging.info("[stream] TIL curve: {}".format(til))
        logging.info("[stream] FINAL  CIL {:.2f} | TIL {:.2f}  (over {} tasks)".format(
            cil[-1], til[-1], comp[-1]))
        print("[stream] TIL:", til)
        print("[stream] FINAL CIL {:.2f} | TIL {:.2f}".format(cil[-1], til[-1]))
    else:
        logging.info("[stream] FINAL  CIL {:.2f}  (over {} tasks) -- TIL not computed "
                      "(meaningless in the memory-increment setup)".format(cil[-1], comp[-1]))
        print("[stream] FINAL CIL {:.2f}".format(cil[-1]))
    # persist the per-checkpoint records (incl. per-task TIL) next to the log
    out = "run_logs/stream_{}_{}_s{}.json".format(
        args["model_name"], args.get("prefix", "run"), args["seed"])
    os.makedirs("run_logs", exist_ok=True)
    with open(out, "w") as f:
        import json as _json
        _json.dump({"args_subset": {k: args.get(k) for k in
                    ("model_name", "dataset", "init_cls", "increment", "stream_budget_mb",
                     "n_lora_blocks", "init_lr", "svd_energy_target", "lamda_1", "lamb", "lame")},
                    "results": results}, f, indent=2)
    print("[stream] wrote", out)


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))