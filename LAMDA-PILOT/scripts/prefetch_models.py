#!/usr/bin/env python3
"""
Pre-download the pretrained ViT-B/16 timm checkpoints into the model cache, so
compute nodes (which may be offline at run time) don't try to fetch at training start.

The backbones load via timm.create_model(name, pretrained=True), which caches to
$HF_HOME (or ~/.cache/huggingface) / torch hub. Point HF_HOME at shared/scratch
storage on the cluster before running this, e.g.:

    export HF_HOME=/scratch/$USER/hf_cache
    python scripts/prefetch_models.py

Both checkpoints we use are fetched: the IN1k-finetuned (vit_base_patch16_224,
the default backbone) and the IN21k variant.
"""
import os

MODELS = ["vit_base_patch16_224", "vit_base_patch16_224_in21k"]


def main():
    import timm
    print(f"timm {timm.__version__} | HF_HOME={os.environ.get('HF_HOME', '~/.cache/huggingface')}")
    for name in MODELS:
        print(f"prefetching {name} ...")
        timm.create_model(name, pretrained=True, num_classes=0)
        print(f"  cached {name}")
    print("done. checkpoints are cached for offline training.")


if __name__ == "__main__":
    main()
