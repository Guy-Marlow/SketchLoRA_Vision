#!/usr/bin/env python3
"""
Download + materialize datasets into ./data in the ImageFolder/torchvision layout
the LAMDA-PILOT loaders (utils/data.py) already expect.  Download-to-local: existing
loaders are untouched.

  CIFAR-100   : auto-downloaded by torchvision at train time -> here we just warm it.
  ImageNet-R  : canonical PILOT pre-split archive (Google Drive) -> ./data/imagenet-r/{train,test}
  ImageNet-A  : ditto -> ./data/imagenet-a/{train,test}
  CUB         : ditto -> ./data/cub/{train,test}
  OmniBenchmark (later) : ditto -> ./data/omnibenchmark/{train,test}

These are the exact pre-split subsets distributed by the PILOT authors
(https://github.com/sun-hailong/LAMDA-PILOT, README "Datasets"), so train/test
splits match the published benchmark. The authors note they are sampled subsets
distributed for research use.

Usage:
  python scripts/data_prep.py --dataset cifar100
  python scripts/data_prep.py --dataset imagenetr            # default ./data
  python scripts/data_prep.py --dataset imagenetr --data_root /scratch/$USER/data
  python scripts/data_prep.py --dataset all
Requires: gdown  (pip install gdown)
"""
import argparse
import os
import sys
import tarfile
import zipfile

# Canonical PILOT distributions (Google Drive file IDs from the LAMDA-PILOT README).
# Each archive unpacks to a <name>/train and <name>/test ImageFolder tree.
GDRIVE = {
    "imagenetr":     "1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR",
    "imageneta":     "19l52ua_vvTtttgVRziCZJjal0TPE9f2p",
    "cub":           "1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb",
    "omnibenchmark": "1AbCP3zBMtv_TDXJypOCnOgX8hJmvJm3u",
    "vtab":          "1xUiwlnx4k0oDhYi26KL5KwrCAya-mvJ_",
}
# directory name each loader in utils/data.py looks for under ./data
TARGET_DIR = {
    "imagenetr": "imagenet-r", "imageneta": "imagenet-a", "cub": "cub",
    "omnibenchmark": "omnibenchmark", "vtab": "vtab",
}


def _extract(archive, dest):
    if archive.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive) as t:
            t.extractall(dest)
    elif archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        raise ValueError(f"Unknown archive type: {archive}")


def prep_cifar100(data_root):
    from torchvision import datasets
    print(f"[cifar100] warming torchvision download into {data_root} ...")
    datasets.cifar.CIFAR100(data_root, train=True, download=True)
    datasets.cifar.CIFAR100(data_root, train=False, download=True)
    print("[cifar100] done.")


def prep_gdrive(name, data_root):
    try:
        import gdown
    except ImportError:
        sys.exit("gdown is required: pip install gdown")
    target = os.path.join(data_root, TARGET_DIR[name])
    if os.path.isdir(os.path.join(target, "train")) and os.path.isdir(os.path.join(target, "test")):
        print(f"[{name}] already present at {target} (train/ + test/). Skipping.")
        return
    os.makedirs(data_root, exist_ok=True)
    archive = os.path.join(data_root, f"{name}.archive")
    print(f"[{name}] downloading from Google Drive id={GDRIVE[name]} ...")
    gdown.download(id=GDRIVE[name], output=archive, quiet=False)
    print(f"[{name}] extracting -> {data_root} ...")
    _extract(archive, data_root)
    os.remove(archive)
    # archives may unpack to a slightly different top dir; warn if expected layout missing
    if not os.path.isdir(os.path.join(target, "train")):
        print(f"[{name}] WARNING: expected {target}/train not found after extract; "
              f"inspect {data_root} and rename to '{TARGET_DIR[name]}' with train/ + test/.")
    else:
        print(f"[{name}] done -> {target}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["cifar100", "imagenetr", "imageneta", "cub", "omnibenchmark", "vtab", "all"])
    p.add_argument("--data_root", default="./data",
                   help="root the loaders read from (LAMDA-PILOT expects ./data)")
    args = p.parse_args()

    if args.dataset in ("cifar100", "all"):
        prep_cifar100(args.data_root)
    todo = ["imagenetr", "imageneta", "cub"] if args.dataset == "all" else \
           ([args.dataset] if args.dataset in GDRIVE else [])
    for name in todo:
        prep_gdrive(name, args.data_root)


if __name__ == "__main__":
    main()
