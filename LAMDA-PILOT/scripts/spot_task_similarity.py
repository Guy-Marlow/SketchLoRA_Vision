"""SPOT-style task-pair similarity (SPOT/README.md; eq3: sim(a,b) = 1 - L_a(b)/L_a(a)).

Train a ResNet (--backbone, default resnet18) from scratch on task a to
convergence (SGD lr=0.001, no
momentum, early-stopped on held-out loss with patience=20 -- SPOT's own
func_simCIFAR100/util.py::trainES recipe), record its held-out loss. Warm-start
from that state, take one real SGD step (lr=0.001, momentum=0.9, dampening=0.1
-- also lifted verbatim from their probe optimizer) on a batch of task b, read
the loss on a second batch of b. sim(a,b) = 1 - (that loss) / (task a's own
loss). No code path in the reference repo actually runs this literally (their
script's "single step" value gets overwritten every epoch until convergence;
their notebook reads the loss before the step lands) -- this implements eq3 as
stated rather than either.

Unlike the reference (32x32 CIFAR fed into an unmodified, ImageNet-shaped
ResNet50 -- collapses to a 1x1 feature map before the final stage), this uses
the project's own 224x224 pipeline, so the architecture is used as intended.
Best-checkpoint state is kept in memory, not written to a shared file path
(the reference's own checkpoint.pt is a single global path, unsafe to reuse
across the repeated task loop here).

Splits are just DataManager(shuffle=True, seed=seed) -- fully reproducible from
the seed alone. Real training runs keep using --seed unchanged; this script only
characterizes a chosen set of seeds.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet34, resnet50

sys.path.insert(0, ".")
from utils.data_manager import DataManager

BACKBONES = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}

SOURCE_CONFIG = {
    "cifar224": "exps/sketchlora_orth_wave1_datasets/cifar224_s1993.json",
    "imagenetr": "exps/sketchlora_orth_wave1_datasets/imagenetr_s1993.json",
}


def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def set_device(args):
    args["device"] = [torch.device("cpu") if d == -1 else torch.device("cuda:{}".format(d))
                       for d in args["device"]]


def task_bounds(dm):
    bounds = [0]
    for t in range(dm.nb_tasks):
        bounds.append(bounds[-1] + dm.get_task_size(t))
    return bounds


def task_loader(dm, lo, hi, train, batch_size):
    ds = dm.get_dataset(np.arange(lo, hi), source="train" if train else "test",
                         mode="train" if train else "test")
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=0)


def build_resnet(n_cls, device, backbone="resnet18"):
    net = BACKBONES[backbone](weights=None)
    net.fc = nn.Linear(net.fc.in_features, n_cls)
    return net.to(device)


def eval_loss(net, loader, lo, device):
    net.eval()
    losses = []
    with torch.no_grad():
        for _, x, y in loader:
            x, y = x.to(device), y.to(device) - lo
            losses.append(F.cross_entropy(net(x), y).item())
    return float(np.mean(losses))


def train_task(args, train_loader, val_loader, n_cls, lo, device, max_epoch, patience, backbone="resnet18"):
    net = build_resnet(n_cls, device, backbone)
    opt = optim.SGD(net.parameters(), lr=args["init_lr"])
    best_loss, best_state, bad_epochs = float("inf"), None, 0
    for _ in range(max_epoch):
        net.train()
        for _, x, y in train_loader:
            x, y = x.to(device), y.to(device) - lo
            loss = F.cross_entropy(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_loss = eval_loss(net, val_loader, lo, device)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    net.load_state_dict(best_state)
    return net, best_loss


def probe_loss(args, state_a, n_cls, loader, lo, device, backbone="resnet18"):
    net = build_resnet(n_cls, device, backbone)
    net.load_state_dict(state_a)
    it = iter(loader)
    _, x1, y1 = next(it)
    _, x2, y2 = next(it)
    x1, y1 = x1.to(device), y1.to(device) - lo
    x2, y2 = x2.to(device), y2.to(device) - lo

    opt = optim.SGD(net.parameters(), lr=args["init_lr"], momentum=0.9, dampening=0.1)
    net.train()
    loss1 = F.cross_entropy(net(x1), y1)
    opt.zero_grad()
    loss1.backward()
    opt.step()

    return F.cross_entropy(net(x2), y2).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(SOURCE_CONFIG), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--backbone", choices=sorted(BACKBONES), default="resnet18")
    parser.add_argument("--out", default=None)
    cli = parser.parse_args()

    args = json.load(open(SOURCE_CONFIG[cli.dataset]))
    args["seed"] = cli.seed
    set_random(args["seed"])
    set_device(args)
    device = args["device"][0]

    dm = DataManager(args["dataset"], args["shuffle"], args["seed"],
                      args["init_cls"], args["increment"], args)
    bounds = task_bounds(dm)
    n_tasks = dm.nb_tasks
    task_sizes = [dm.get_task_size(t) for t in range(n_tasks)]
    assert len(set(task_sizes)) == 1, "SPOT probe assumes uniform task width (warm-started head)"
    n_cls = task_sizes[0]

    train_loaders = [task_loader(dm, bounds[t], bounds[t + 1], True, args["batch_size"])
                      for t in range(n_tasks)]
    test_loaders = [task_loader(dm, bounds[t], bounds[t + 1], False, args["batch_size"])
                     for t in range(n_tasks)]

    sim = np.zeros((cli.runs, n_tasks, n_tasks))
    t0 = time.time()
    for run in range(cli.runs):
        for a in range(n_tasks):
            ta = time.time()
            net, basic_loss = train_task(args, train_loaders[a], test_loaders[a], n_cls,
                                          bounds[a], device, cli.max_epoch, cli.patience, cli.backbone)
            print("[{:.0f}s] run {}/{} task {}/{} basic_loss={:.4f} train_time={:.0f}s".format(
                time.time() - t0, run + 1, cli.runs, a, n_tasks, basic_loss, time.time() - ta), flush=True)
            state_a = {k: v.detach().clone() for k, v in net.state_dict().items()}
            for b in range(n_tasks):
                if b == a:
                    continue
                pl = probe_loss(args, state_a, n_cls, train_loaders[b], bounds[b], device, cli.backbone)
                sim[run, a, b] = 1 - pl / basic_loss
        print("[{:.0f}s] run {}/{} complete".format(time.time() - t0, run + 1, cli.runs), flush=True)

    sim_mean = sim.mean(axis=0)
    sim_sym = (sim_mean + sim_mean.T) / 2

    out_path = cli.out or "exps/spot_splits/{}_s{}.json".format(cli.dataset, cli.seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump({
        "dataset": cli.dataset, "seed": cli.seed, "runs": cli.runs, "backbone": cli.backbone,
        "class_order": dm._class_order,
        "task_classes": [dm._class_order[bounds[t]:bounds[t + 1]] for t in range(n_tasks)],
        "sim_matrix": sim_mean.tolist(),
        "sim_matrix_symmetric": sim_sym.tolist(),
    }, open(out_path, "w"), indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
