# LightPolySegNet — Lightweight Boundary-Aware Transformer-CNN Hybrid for Real-Time Polyp Segmentation

Official reproducibility package. Training follows the **PraNet standard protocol**:
train on Kvasir-SEG (1000) + CVC-ClinicDB (612); test on CVC-300 (60),
CVC-ColonDB (380), ETIS-Larib (196), Kvasir-SEG and CVC-ClinicDB; input
352×352. Metrics: mDice / mIoU (region) + BF score / Boundary IoU (boundary)
+ Params / MACs / FPS (efficiency).

## 0. Main results

| Method | Params | MACs | Kvasir-SEG | CVC-ClinicDB | CVC-300 | CVC-ColonDB | ETIS-Larib | Mean |
|---|---|---|---|---|---|---|---|---|
| PraNet (reported) | 26.7M | — | 0.898 | 0.899 | 0.871 | 0.709 | 0.628 | 0.801 |
| FANet (re-trained, same protocol) | 7.74M | 44.79 | 0.9075 | 0.9186 | 0.8390 | 0.7397 | 0.6613 | 0.8132 |
| **LightPolySegNet (ours)** | **0.98M** | **0.80** | **0.9378** | **0.9389** | 0.8519 | 0.7624 | 0.6995 | **0.8381** |

- Multi-seed: **0.8347 ± 0.0030** mean mDice over seeds 42/43/44.
- Efficiency: **96.6 FPS** on a Tesla T4 (batch=1, fp32) — ~3× real-time
  endoscopy throughput. FANet's official test protocol uses 10 iterative
  forward passes (≈560× our per-inference compute).
- Ablations (paper Table 5): ImageNet pretraining is the decisive enabler
  (+8.8 mean mDice, +21.9 on ETIS-Larib); color-domain augmentation +2.4;
  boundary-aware loss +0.65 with cross-domain robustness; the timm-aligned
  encoder from scratch alone loses 3.7.

## 1. Environment

Any Python ≥ 3.10. CPU torch is sufficient for smoke tests and inference;
GPU required for training.

```bash
python -m venv venv
venv\Scripts\activate            # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt  # CPU torch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

On Colab, torch 2.x + CUDA is preinstalled — only `pip install thop gdown timm` needed.

## 2. Smoke test (CPU, 1-2 min)

```bash
python tests/smoke_test.py
```

Expected output ends with `SMOKE TEST: 12/12 checks passed` and prints the
parameter count (must be < 10 M). This verifies forward shapes, backward
without NaN, and a 2-epoch mini training loop.

## 3. Data download & layout

Full step-by-step guide with links: [`scripts/download_data.md`](scripts/download_data.md).

```bash
pip install gdown
gdown 1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC -O dataset.zip    # official Polyp-PVT bundle, 1.11 GB (verified 2026-10-21)
unzip dataset.zip -d data_raw/
python scripts/build_pranet_split.py --src data_raw/dataset --out data
```

**Important:** the bundle ships the Polyp-PVT authors' *own* split (1,450/798
pairs), **not** the PraNet protocol used by every baseline in the paper.
`build_pranet_split.py` deterministically reassembles the standard split
(1,612 train pairs = Kvasir-SEG + CVC-ClinicDB; ClinicDB = all 384×288 images;
hard count assertions; `build_log.json` emitted) — do not rearrange the
layout by hand.

Final layout:

```
data/
├── TrainDataset/{images/, masks/}          # 1612 pairs (Kvasir-SEG + CVC-ClinicDB)
└── TestDataset/{CVC-300, CVC-ColonDB, ETIS-Larib, Kvasir-SEG, CVC-ClinicDB}/{images/, masks/}
```

## 4. Train the paper model

```bash
python scripts/train.py --config configs/exp3_pretrain.yaml \
    --data-root data/TrainDataset --out-dir runs/main \
    --epochs 100 --batch-size 8 --val-split 0.0 \
    --backup-dir <any persistent folder>
```

- `configs/exp3_pretrain.yaml` = the paper model: timm-aligned encoder
  (`stage_channels [24,48,64,80]`, MobileViT-style blocks `tf_dims [64,80,96]`,
  `tf_depths [2,4,3]`), ImageNet weights from `pretrained/` (94.2% of all
  parameters loaded), color-domain augmentation, boundary-aware loss.
  **0.98 M params / 0.80 GMACs** — asserted < 10 M at startup.
- On a free Colab T4: ~3 h for 100 epochs with AMP.
- `--backup-dir` mirrors `last/best.pth` + `progress.json` every epoch (crash-safe);
  training auto-resumes with `--resume <backup_dir>/last.pth`.
- Other configs: `exp3_seed43/44.yaml` (multi-seed), `ablation_no_boundary.yaml`
  (R5), `exp5_randinit.yaml` (R3), `fanet_baseline.yaml` (same-protocol FANet),
  `ablation_no_color.yaml`, `default.yaml` (v1 baseline).

## 5. Evaluation (paper table)

```bash
python scripts/eval.py --checkpoint runs/main/last.pth \
    --data-root data/TestDataset --out results/main_results.csv --save-vis 8
```

- Sigmoid → resize to each image's original resolution → threshold 0.5
  (standard PraNet protocol) for mDice/mIoU.
- Boundary metrics are reported under two declared conventions:
  original-resolution and 352×352 (paper primary). One transparency CSV per
  convention is written to `results/`.
- `--save-vis N` writes Original | Ground-truth | Prediction triptychs to
  `results/vis/<dataset>/`.

## 6. Efficiency benchmark

```bash
python scripts/benchmark_speed.py --checkpoint runs/main/last.pth
```

Protocol: 352×352 input, batch=1, fp32, 50 warmup + 300 timed iterations,
compute in multiply-accumulates (MACs, via `thop`). The architecture is read
from the checkpoint's embedded config, so any experiment's checkpoint works.
**Benchmark all compared methods on the same machine.**

## 7. Repository layout

```
.
├── README.md, requirements.txt, .gitignore
├── configs/            # default (v1) / exp3_pretrain (paper) / seeds /
│                       # ablations / fanet_baseline
├── src/                # model.py (LightPolySegNet + FANet), datasets.py,
│                       # losses.py, metrics.py, pretrained.py
├── scripts/            # train.py, eval.py, benchmark_speed.py,
│                       # build_pranet_split.py, aggregate_runs.py,
│                       # download_data.md
├── tests/smoke_test.py # 12 CPU checks
├── results/            # all experiment CSVs (8 runs) + fanet_iters_*.csv
└── sn-jnl.cls          # Springer class used by the manuscript
```

## 8. Reproducibility notes

- Ablation rows in the paper are seed-matched single runs (seed 42); the main
  model is additionally reported over three seeds (42/43/44).
- Per-iteration metrics for the FANet feedback-attention protocol are emitted
  as `results/fanet_iters_<dataset>.csv` for transparency (10 iterations are
  FANet's official test protocol, not a choice made by this evaluation).
- Compute is reported in multiply-accumulates (MACs), as computed with `thop`.
- Baseline values quoted from literature are marked in the paper; cells that
  could not be verified against original publications are left empty rather
  than propagated second-hand.

## 9. Citation

Paper under review at the Journal of Real-Time Image Processing (Springer).
Citation block will be finalized upon acceptance.

## 10. License

Code released for research purposes. `src/fanet.py` is a faithful port of the
official [FANet repository](https://github.com/nikhilroxtomar/FANet)
(Tomar et al., IEEE TNNLS 2022), which carries no explicit license file at the
time of redistribution; used here under fair use for comparative evaluation.
