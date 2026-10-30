# LightPolySegNet — Lightweight Boundary-Aware Transformer-CNN Hybrid for Real-Time Polyp Segmentation

Reproducibility package. Training follows the **PraNet standard protocol**:
train on Kvasir-SEG (1000) + CVC-ClinicDB (612); test on CVC-300 (60),
CVC-ColonDB (380), ETIS-Larib (196), Kvasir-SEG and CVC-ClinicDB; input
352×352. Metrics: mDice / mIoU (region) + BF score / Boundary IoU (boundary)
+ Params / FLOPs / FPS (efficiency).

## 0. Main results

| Method | Params | MACs | Kvasir-SEG | CVC-ClinicDB | CVC-300 | CVC-ColonDB | ETIS-Larib | Mean |
|---|---|---|---|---|---|---|---|---|
| PraNet (reported) | 26.7M | — | 0.898 | 0.899 | 0.871 | 0.709 | 0.628 | 0.801 |
| FANet (re-trained, same protocol) | 7.74M | 44.79 | 0.9075 | 0.9186 | 0.8390 | 0.7397 | 0.6613 | 0.8132 |
| **LightPolySegNet (ours)** | **0.98M** | **0.80** | **0.9378** | **0.9389** | 0.8519 | 0.7624 | 0.6995 | 0.8381 |

Multi-seed: 0.8347 ± 0.0030 mean mDice over seeds 42/43/44.
Efficiency: 96.6 FPS on a Tesla T4 (batch=1, fp32) — ~3× real-time endoscopy
throughput.

## 1. Environment

Local (CPU, smoke tests only — Windows shown):

```bash
# any Python >= 3.10 works; example uses a dedicated venv
python -m venv venv
venv\Scripts\activate          # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt   # CPU torch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Colab: torch 2.x + CUDA is preinstalled — only `pip install thop gdown` needed.

## 2. Smoke test (local, ~1–2 min on CPU)

```bash
python tests/smoke_test.py
```

Expected output ends with `SMOKE TEST: N/N checks passed` and prints the
parameter count (must be < 10 M). This verifies forward shapes, backward
without NaN, and a 2-epoch mini training loop.

## 3. Data download & layout

Full step-by-step guide with links: [`scripts/download_data.md`](scripts/download_data.md).
Quick version (recommended path = Polyp-PVT processed bundle):

```bash
pip install gdown
gdown 1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC -O dataset.zip    # official bundle, 1.11 GB (verified 2026-10-21)
unzip dataset.zip -d data_raw/
python scripts/build_pranet_split.py --src data_raw/dataset --out data
```

Note: the bundle stores the Polyp-PVT authors' *own* split (1450/798 pairs).
`build_pranet_split.py` deterministically reassembles the **PraNet protocol**
(1612 train + 5 standard test sets, ETIS folder renamed `ETIS-Larib`) with
hard count assertions — do not rearrange the layout by hand. On Colab,
`notebooks/colab_train.ipynb` cell 3 does download (3 retries + size gate)
→ extract → reassembly → sanity checks automatically.

Final layout:

```
polyp_paper/data/
├── TrainDataset/{images/, masks/}          # 1612 pairs (Kvasir-SEG + CVC-ClinicDB)
└── TestDataset/{CVC-300, CVC-ColonDB, ETIS-Larib, Kvasir-SEG, CVC-ClinicDB}/{images/, masks/}
```

## 4. Training on Colab (free T4)

Open `notebooks/colab_train.ipynb` → `Runtime > Change runtime type > T4 GPU`
→ *Run all*. The notebook: mounts Drive → copies/downloads data → trains →
evaluates → writes `results/main_results.csv` and checkpoints back to Drive.

Command-line equivalent:

```bash
python scripts/train.py --config configs/default.yaml --data-root data/TrainDataset \
    --out-dir runs/exp1 --epochs 100 --batch-size 8
```

Notes:
- **Final paper run**: use `--val-split 0.0` (train on all 1612 images, PraNet
  protocol); the last checkpoint is the final model. The default `val_split=0.1`
  is only for model-selection experiments.
- **Expected T4 time**: ≈ 1612 imgs/epoch ÷ 8 bs ≈ 201 steps; ~1.5–2 min/epoch
  with AMP → **~2.5–3.5 h for 100 epochs**. Set `--epochs 100`.
- Checkpoints: `runs/exp1/{last.pth, best.pth}`, TensorBoard in
  `runs/exp1/tb` (`tensorboard --logdir runs/exp1/tb`).

## 5. Evaluation (paper table)

```bash
python scripts/eval.py --checkpoint runs/exp1/best.pth --data-root data/TestDataset \
    --out results/main_results.csv
```

Writes CSV: one row per test set + Mean row, columns
`Dataset, Images, mDice, mIoU, BF score, Boundary IoU`. Predictions are
resized to each image's original resolution before thresholding (standard
protocol).

## 6. Efficiency benchmark

```bash
python scripts/benchmark_speed.py --checkpoint runs/exp1/best.pth
```

Protocol: 352×352 input, batch=1, fp32, 50 warmup + 300 timed iterations,
thop FLOPs (GMACs). **Run all compared methods on the same machine** and
report the device line printed by the script.

## 7. Hyperparameters (configs/default.yaml)

| group | key | default |
|---|---|---|
| data | image_size | 352 |
| train | epochs / batch_size / lr | 100 / 8 / 1e-4 (AdamW) |
| train | scheduler | 2-epoch warmup + cosine |
| train | loss weights | w_bce=1.0, w_iou=1.0, w_boundary=0.5, w_aux=0.4 |
| model | stage_channels | [32, 56, 112, 176] |
| model | tf_dims / tf_depths | [72, 128, 176] / [2, 2, 2] |

If the parameter budget is exceeded, tighten `tf_dims`, `tf_depths` or
`stage_channels` in the config — `train.py` asserts params < 10M at startup.

## 8. Repository layout

```
polyp_paper/
├── README.md
├── requirements.txt
├── configs/default.yaml        # all hyperparameters
├── scripts/
│   ├── download_data.md        # dataset links + exact layout
│   ├── train.py                # train (resume / best ckpt / TensorBoard)
│   ├── eval.py                 # 5 test sets -> paper-table CSV
│   └── benchmark_speed.py      # Params / FLOPs / FPS
├── src/
│   ├── datasets.py             # Dataset + augmentation (train/test modes)
│   ├── model.py                # LightPolySegNet (hybrid encoder + boundary decoder)
│   ├── losses.py               # BCE + SoftIoU + boundary + deep supervision
│   └── metrics.py              # Dice / IoU / BF score / Boundary IoU
├── notebooks/colab_train.ipynb # one-click Colab training
└── tests/smoke_test.py         # CPU smoke test
```

## 9. Reproducibility notes

- The official Polyp-PVT data bundle ships the authors' *own* split
  (1,450/798), **not** the PraNet protocol used by every baseline compared in
  the paper. `scripts/build_pranet_split.py` deterministically reassembles the
  standard split (ClinicDB = all 384x288 images; hard-asserted counts;
  `build_log.json` emitted) — do not rearrange the layout by hand.
- Ablation rows are seed-matched single runs (seed 42); the main model is
  additionally reported over three seeds (42/43/44) in the paper.
- Per-iteration metrics for the FANet feedback-attention protocol are emitted
  as `results/fanet_iters_<dataset>.csv` for transparency (10 iterations are
  FANet's official test protocol, not a choice made by this evaluation).
- Ablation/efficiency compute is reported in multiply-accumulates (MACs),
  as computed with `thop`.

## 10. Citation

Paper under review at the Journal of Real-Time Image Processing (Springer).
Citation block will be finalized upon acceptance.

## 11. License

Code released for research purposes. `src/fanet.py` is a faithful port of the
official [FANet repository](https://github.com/nikhilroxtomar/FANet)
(Tomar et al., IEEE TNNLS 2022), which carries no explicit license file at the
time of redistribution; used here under fair use for comparative evaluation.
