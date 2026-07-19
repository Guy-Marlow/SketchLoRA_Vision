#!/usr/bin/env python3
"""One-time prep: export the HuggingFace-arrow ImageNet-R (used by the sibling
svd_sketching project) into the ImageFolder layout LAMDA-PILOT's iImageNetR expects.

Source : svd_sketching/data/imagenet_r  (load_from_disk; fields image/wnid/class_name/int_label)
Target : LAMDA-PILOT/data/imagenet-r/{train,test}/<wnid>/<idx>.jpg

Class folders are named by wnid (consistent across train/test), so torchvision
ImageFolder assigns a stable 0..199 label order; LAMDA-PILOT then shuffles class
order per seed for the task splits. No LAMDA-PILOT code is modified.
"""
import os
from datasets import load_from_disk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../LAMDA-PILOT
SRC = os.path.join(ROOT, "..", "..", "svd_sketching", "data", "imagenet_r")
DST = os.path.join(ROOT, "data", "imagenet-r")

def main():
    ds = load_from_disk(os.path.abspath(SRC))
    for split in ("train", "test"):
        d = ds[split]
        sub = os.path.join(DST, split)
        print(f"[{split}] {len(d)} examples -> {sub}", flush=True)
        for i, ex in enumerate(d):
            cdir = os.path.join(sub, ex["wnid"])
            os.makedirs(cdir, exist_ok=True)
            ex["image"].convert("RGB").save(os.path.join(cdir, f"{i:06d}.jpg"), "JPEG", quality=95)
            if (i + 1) % 5000 == 0:
                print(f"  {split}: {i+1}/{len(d)}", flush=True)
        ncls = len(os.listdir(sub))
        print(f"[{split}] done: {ncls} class folders", flush=True)
    print("EXPORT COMPLETE", flush=True)

if __name__ == "__main__":
    main()
