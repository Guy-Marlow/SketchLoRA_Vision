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

# HF `datasets` repo id + split-name mapping (HF split name -> our train/test dir name).
# ethz/food101 ships its held-out split as "validation", not "test".
HF_IMAGEFOLDER = {
    "sun397": {"repo": "tanganke/sun397", "splits": {"train": "train", "test": "test"}},
    "food101": {"repo": "ethz/food101", "splits": {"train": "train", "test": "validation"}},
}
# Single-tarball streaming datasets (see svd_sketching/utils/data/prepare_omnibench.py for
# the original streaming-extract implementation this mirrors).
HF_TARBALL = {
    "omnibenchmark1k": {"repo": "LMMM2025/OmniBenchmark-1K", "path": "omnibenchmark1k.tar.gz",
                        "archive_root": "omnibenchmark1k"},
}

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
    """Detect archive format from CONTENT (magic bytes via zipfile/tarfile's own
    sniffers), not the filename -- `prep_gdrive` downloads to a generic
    `<name>.archive` name (Google Drive doesn't expose the real extension), so a
    suffix-based dispatch never matched anything and this always raised
    "Unknown archive type" (found 2026-08-06, reproduced on a fresh Windows
    install; `archive.endswith(...)` was checking a name that can never end in
    .zip/.tar.gz/etc). zipfile.is_zipfile/tarfile.is_tarfile read the actual
    file header, so this works regardless of what the file happens to be named."""
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            t.extractall(dest)
    else:
        raise ValueError(f"Unknown archive type (not a zip or tar, by content): {archive}")


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


def prep_hf_imagefolder(name, data_root):
    """Materialize a HF `datasets` image-classification dataset into
    ./data/<name>/{train,test}/<class_name>/<idx>.jpg (ImageFolder layout)."""
    from datasets import load_dataset
    spec = HF_IMAGEFOLDER[name]
    target = os.path.join(data_root, name)
    if os.path.isdir(os.path.join(target, "train")) and os.path.isdir(os.path.join(target, "test")):
        print(f"[{name}] already present at {target} (train/ + test/). Skipping.")
        return
    for our_split, hf_split in spec["splits"].items():
        print(f"[{name}] loading HF split '{hf_split}' from {spec['repo']} ...")
        ds = load_dataset(spec["repo"], split=hf_split)
        names = ds.features["label"].names
        counts = {}
        for i, ex in enumerate(ds):
            cls = names[ex["label"]]
            dest_dir = os.path.join(target, our_split, cls)
            os.makedirs(dest_dir, exist_ok=True)
            counts[cls] = counts.get(cls, 0) + 1
            ex["image"].convert("RGB").save(os.path.join(dest_dir, f"{counts[cls]:05d}.jpg"), quality=95)
            if (i + 1) % 5000 == 0:
                print(f"  [{name}/{our_split}] {i + 1}/{len(ds)} written", flush=True)
        print(f"[{name}] {our_split}: {sum(counts.values())} images, {len(counts)} classes -> "
              f"{os.path.join(target, our_split)}")
    print(f"[{name}] done -> {target}")
    _evict_hf_cache(spec["repo"], repo_type="dataset")


def _evict_hf_cache(repo_id, repo_type="dataset"):
    """Delete a single repo's entry from the HF hub cache once we've fully
    materialized what we need from it -- avoids paying for both the raw
    parquet/arrow cache AND the exploded JPEGs at once (disk is tight on this
    machine). Re-downloads on demand if the script is ever re-run for this repo."""
    from huggingface_hub import scan_cache_dir
    try:
        cache_info = scan_cache_dir()
        to_delete = [r.size_on_disk for r in cache_info.repos
                     if r.repo_id == repo_id and r.repo_type == repo_type]
        if not to_delete:
            return
        revisions = [rev.commit_hash for r in cache_info.repos
                     if r.repo_id == repo_id and r.repo_type == repo_type
                     for rev in r.revisions]
        freed = cache_info.delete_revisions(*revisions).execute()
        print(f"[{repo_id}] evicted HF cache entry ({sum(to_delete) / 1e9:.2f} GB reclaimed)")
    except Exception as e:
        print(f"[{repo_id}] WARNING: cache eviction failed ({e}); "
              f"clear ~/.cache/huggingface manually if disk is tight")


def prep_hf_tarball(name, data_root):
    """Stream a single HF-hosted .tar.gz and extract directly to
    ./data/<name>/{train,test}/<class>/<file> -- mirrors
    svd_sketching/utils/data/prepare_omnibench.py's one-pass streaming approach
    (gzip has no random access, so a single sequential pass is unavoidable)."""
    from huggingface_hub import HfFileSystem
    spec = HF_TARBALL[name]
    target = os.path.join(data_root, name)
    if os.path.isdir(os.path.join(target, "train")) and os.path.isdir(os.path.join(target, "test")):
        print(f"[{name}] already present at {target} (train/ + test/). Skipping.")
        return
    hf_path = f"datasets/{spec['repo']}/{spec['path']}"
    print(f"[{name}] streaming {hf_path} ...")
    fs = HfFileSystem()
    fobj = fs.open(hf_path, "rb")
    tar = tarfile.open(fileobj=fobj, mode="r|gz")
    n_written = 0
    for m in tar:
        if not m.isfile():
            continue
        parts = m.name.split("/")
        if len(parts) != 4 or parts[0] != spec["archive_root"]:
            continue
        split, cls, fname = parts[1], parts[2], parts[3]
        if split not in ("train", "test") or not fname:
            continue
        dest_dir = os.path.join(target, split, cls)
        os.makedirs(dest_dir, exist_ok=True)
        src = tar.extractfile(m)
        if src is None:
            continue
        with open(os.path.join(dest_dir, fname), "wb") as g:
            g.write(src.read())
        n_written += 1
        if n_written % 5000 == 0:
            print(f"  [{name}] {n_written} files written", flush=True)
    tar.close()
    fobj.close()
    print(f"[{name}] done: {n_written} files -> {target}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["cifar100", "imagenetr", "imageneta", "cub", "omnibenchmark",
                            "omnibenchmark1k", "vtab", "sun397", "food101", "all"])
    p.add_argument("--data_root", default="./data",
                   help="root the loaders read from (LAMDA-PILOT expects ./data)")
    args = p.parse_args()

    if args.dataset in ("cifar100", "all"):
        prep_cifar100(args.data_root)
    gdrive_todo = ["imagenetr", "imageneta", "cub"] if args.dataset == "all" else \
                  ([args.dataset] if args.dataset in GDRIVE else [])
    for name in gdrive_todo:
        prep_gdrive(name, args.data_root)
    hf_imagefolder_todo = ["sun397", "food101"] if args.dataset == "all" else \
                          ([args.dataset] if args.dataset in HF_IMAGEFOLDER else [])
    for name in hf_imagefolder_todo:
        prep_hf_imagefolder(name, args.data_root)
    hf_tarball_todo = ["omnibenchmark1k"] if args.dataset == "all" else \
                      ([args.dataset] if args.dataset in HF_TARBALL else [])
    for name in hf_tarball_todo:
        prep_hf_tarball(name, args.data_root)


if __name__ == "__main__":
    main()
