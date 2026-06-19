# Cluster setup — svd_sketching_vision

Self-contained CL bench (LAMDA-PILOT + our LoRA/SVD-sketching methods). No data or
model weights are committed; everything is fetched on first use. Reference repos
(InfLoRA/, O-LoRA/, HiDeLoRA/, TreeLoRA/) are gitignored — they are cited/ported in
comments only and never imported. `randsvd` is vendored at
`LAMDA-PILOT/utils/randsvd.py`.

## 1. Environment
```bash
cd LAMDA-PILOT
pip install -r requirements.txt
```
timm is pinned at 0.6.7 — the pretrained ViT weight keys/URLs differ across timm
versions, so do not float this.

## 2. Model checkpoints (ViT-B/16)
Backbones load via `timm.create_model(..., pretrained=True)`, which downloads to the
HF / torch cache on first use. Point the cache at shared/scratch storage and (optionally)
pre-fetch so compute nodes never hit the network mid-run:
```bash
export HF_HOME=/scratch/$USER/hf_cache      # persists the timm/HF download cache
python scripts/prefetch_models.py           # caches vit_base_patch16_224 (+ _in21k)
```

## 3. Datasets (download-to-local; loaders read ./data)
```bash
python scripts/data_prep.py --dataset cifar100     # torchvision auto-download
python scripts/data_prep.py --dataset imagenetr    # canonical PILOT pre-split (gdown)
# or: --dataset all   (cifar100 + imagenetr + imageneta + cub)
# custom location: --data_root /scratch/$USER/data   (then run from there or symlink ./data)
```
- **CIFAR-100**: fully automatic via torchvision.
- **ImageNet-R / A / CUB / OmniBenchmark**: the loaders in `utils/data.py` expect
  `./data/<name>/{train,test}` ImageFolder trees. `data_prep.py` fetches the PILOT
  authors' pre-split archives (Google Drive), so splits match the published benchmark.
  (OmniBenchmark id is wired but deferred for now.)

## 4. Run
```bash
python main.py --config ./exps/sketchlora_r8_l3.json     # CIFAR-100 10-task
python main.py --config ./exps/sketchlora_r8_l3_20t.json # CIFAR-100 20-task
```
Task split is set in the json via `dataset` + `init_cls` + `increment`; LR/batch via
`init_lr` / `batch_size` (not auto-scaled per split). See exps/ for the method configs
(lora / seqlora / olora / inflora / sketchlora).
