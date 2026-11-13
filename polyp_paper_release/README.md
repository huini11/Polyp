# LightPolySegNet — Lightweight Transformer–CNN Hybrid for Real-Time Polyp Segmentation

Official code for **"A Lightweight Transformer–CNN Hybrid Network for Real-Time Polyp Segmentation in Colonoscopy"** (Chao Hu, Eulji University).

**Headline:** a **0.97 M**-parameter Transformer–CNN hybrid whose encoder is weight-compatible with timm's `mobilevit_xxs`, so ImageNet pretraining transfers directly. On the standard, hash-verified **1,450/798** PraNet protocol it reaches **0.8190 ± 0.0054** mean mDice over three seeds (**+1.8** over PraNet, **+1.8** over FANet re-trained under the identical protocol) at **65× lower per-forward compute**, running at **111.6 FPS** on a Tesla T4 — while FANet's official ten-pass feedback protocol yields 3.4 FPS.

> **Note on the shipped model.** Our ablation (three seeds, Welch's *t*-tests) shows that the boundary apparatus — a Sobel edge-attention branch and a boundary loss — brings **no measurable region-accuracy benefit** at fixed weights. The released model therefore **omits** it; the boundary *head* is retained as an optional deployment-facing contour output. This negative result is reported openly in the paper (Table 6, Sec. 4.9).

## 0. Main results (mDice, standard PraNet protocol)

| Method | Params | Kvasir-SEG | CVC-ClinicDB | CVC-300 | CVC-ColonDB | ETIS-Larib | Mean |
|---|---|---|---|---|---|---|---|
| PraNet† | 26.67 M | 0.898 | 0.899 | 0.871 | 0.709 | 0.628 | 0.801 |
| FANet§ | 7.74 M | 0.8573 | 0.8511 | 0.8415 | 0.7502 | 0.7051 | 0.8010 |
| HarDNet-MSEG†ᵃ | ≈33 M | **0.912** | 0.932 | 0.887 | 0.731 | 0.677 | 0.8278 |
| Polyp-PVT† | 28.2 M | 0.917 | **0.937** | **0.900** | **0.808** | **0.787** | **0.870** |
| **LightPolySegNet (3 seeds)** | **0.97 M** | 0.8981±0.0087 | 0.8857±0.0093 | 0.8599±0.0095 | 0.7589±0.0089 | 0.6924±0.0188 | 0.8190±0.0054 |

† values from the original publications (PraNet protocol); § FANet re-trained by us under the identical protocol (incl. the same color-domain augmentation) and evaluated with its official 10-pass feedback protocol; †ᵃ HarDNet-MSEG's own PraNet-split experiments (352×352). We do **not** claim an accuracy advantage over HarDNet-MSEG or Polyp-PVT — the contribution is the efficiency–accuracy operating point plus a fully attributed ablation.

## 1. Efficiency (single Tesla T4, batch 1, 352×352, measured by us)

| Method | Params (M) | MACs (G) | FPS fp32 | FPS fp16 |
|---|---|---|---|---|
| PraNet | 26.67 | 13.15 | 53.4 | 34.2 |
| Polyp-PVT | 28.2 | 10.02 | 54.9 | 34.8 |
| HarDNet-MSEG | 33.3 | 11.40 | 61.8 | 48.4 |
| FANet | 7.74 | 44.79 | 34.4 (3.4 with official 10-pass protocol) | – |
| **LightPolySegNet** | **0.97** | **0.69** | **111.6** | 59.9 |

All vendors use 352×352 inputs under the PraNet protocol, so the numbers are directly comparable. Latency is architecture-dependent, not weight-dependent, so competitor benchmarks run with random weights (stated in the paper). Two additional findings we report honestly:

- **Naive autocast-fp16 slows all five models** on the T4 (ours 111.6→59.9; competitors at best 62→48). Realizing fp16 gains requires engine-level optimization (TensorRT-class), not autocast.
- **End-to-end on a desktop CPU** (pre+inference+post, AMD Ryzen): ours **14.5 FPS** vs 1.8–6.5 FPS for the four competitors — a 2.2–8.1× end-to-end advantage.

## 2. What the ablation establishes (Table 6, three seeds + controls)

| Configuration | Params | Mean mDice |
|---|---|---|
| R1 baseline (no color aug, old 3.64M enc.) | — | 0.7289 |
| R2 + color aug | — | 0.7565 |
| R3 + encoder redesign (random init) | — | 0.7113 |
| R4 full design (R3 + ImageNet) | 0.98 M | 0.8133±0.0026 |
| **M = LightPolySegNet (R4 w/o boundary apparatus)** | **0.97 M** | **0.8190±0.0054** |
| M + Sobel attention | 0.98 M | 0.8165±0.0053 |
| M + boundary loss | 0.97 M | 0.8185±0.0017 |
| R7 w/o Transformer blocks (0.11 M) | 0.11 M | 0.6260±0.0166 |
| R9 pretrained MobileNetV2-100 encoder, same decoder | 1.91 M | 0.8209 (single run) |
| R11 pretrained MobileNetV3-Small encoder, same decoder | 1.04 M | 0.8275±0.0038 |

Three transferable findings: **(i)** ImageNet pretraining is the decisive component (**+10.2** mDice, ETIS +18.6; without it the redesigned sub-1M encoder *underfits*, R3 < R1); **(ii)** color-domain augmentation adds **+2.8**, concentrated on the unseen domains; **(iii)** once pretraining is in place, the encoder family matters less than pretraining itself — a capacity-matched pretrained pure-CNN control (R11) is on par with (marginally above, t=2.2) the hybrid. The boundary apparatus never helps (all three addition deltas negative, t ≤ 1.7).

## 3. Repository layout

```
├── configs/                 # 24 experiment configs (paper row ↔ config below)
├── results/                 # curated experiment CSVs + speed/e2e measurements
│   ├── v2_*.csv             #   one file per training run (14-column metrics)
│   ├── ours_speed_gpu.csv   #   our T4 benchmark (thop + flop_counter, fp32/fp16)
│   ├── competitor_speed_*.csv / e2e_cpu.csv  # competitor & CPU measurements
│   └── comparison_table_v2.md  # full aggregation incl. multi-seed stats
├── scripts/
│   ├── build_pranet_split.py   # deterministic 1,450/798 split builder (hash-verified)
│   ├── train.py                # single-GPU trainer (AMP, cosine schedule)
│   ├── eval.py                 # v3 evaluator: Dice/IoU/maxBF/BF@0.5/bIoU (dual
│   │                           #   protocol) + Sm/Em/wFm/MAE + per-image + size-strata
│   ├── benchmark_speed.py      # our model: params/MACs (thop + flop_counter), FPS,
│   │                           #   latency mean±std/P95, fp16 (autocast) → ours_speed.csv
│   ├── benchmark_competitors.py# PraNet/Polyp-PVT/HarDNet/FANet in isolated
│   │   └── benchmark_one.py    #   subprocesses (auto-clones official repos;
│   │                           #   FANet incl. official 10-pass protocol; --e2e mode)
│   ├── make_figures.py         # paper Fig.2/3 with median-Dice sampling (no cherry-picking)
│   ├── check_runcard.py        # preflight: configs ↔ scripts ↔ deps consistency
│   ├── aggregate_runs.py       # multi-seed aggregation helpers
│   └── download_data.md        # where to get the official Polyp-PVT bundle
├── src/                     # model, datasets, losses, metrics, FANet port, timm loader
├── tests/smoke_test.py      # CPU smoke test (build + forward + loss)
├── notebooks/colab_train.ipynb  # end-to-end Colab pipeline (24 experiments)
└── requirements.txt
```

**Paper row ↔ config mapping**

| Paper row | Config |
|---|---|
| M (shipped model) | `configs/exp8_combo.yaml` (+ `exp8_seed43/44.yaml`) |
| R4 full design | `configs/exp3_pretrain.yaml` (+ `exp3_seed43/44.yaml`) |
| M + boundary loss (= full w/o Sobel) | `configs/exp6_no_sobel.yaml` (+ seeds) |
| M + Sobel attention (= full w/o L_b) | `configs/ablation_no_boundary.yaml` (+ seeds) |
| R3 redesign, random init | `configs/exp5_randinit.yaml` |
| R2 / R1 staircase | `configs/default.yaml` / `configs/ablation_no_color.yaml` |
| R7 w/o Transformer blocks | `configs/exp7_no_transformer.yaml` (+ seeds) |
| R9 MobileNetV2-100 encoder | `configs/exp9_mobilenetv2.yaml` |
| R11 MobileNetV3-Small encoder | `configs/exp11_mv075_seed42/43/44.yaml` |
| FPN-decoder control (not in paper tables) | `configs/exp10_fpn.yaml` |
| FANet re-training | `configs/fanet_baseline.yaml` |

## 4. Setup

```bash
python -m pip install -r requirements.txt
# NOTE: py-sod-metrics is installed from GitHub (no PyPI release):
python -m pip install "git+https://github.com/lartpang/PySODMetrics.git"
```

Sanity check (CPU, <1 min): `python tests/smoke_test.py`

## 5. Data preparation

1. Download the official Polyp-PVT data bundle (see `scripts/download_data.md`).
2. Build the split (folder rename + hard-asserted counts + content-hash
   verification that train and held-out test images are disjoint):

```bash
python scripts/build_pranet_split.py --src <bundle_dir> --out data
# -> data/TrainDataset (1,450 pairs), data/TestDataset/{Kvasir-SEG:100,
#    CVC-ClinicDB:62, CVC-300:60, CVC-ColonDB:380, ETIS-Larib:196}
# -> data/TrainDataset/build_log.json (committed protocol evidence)
```

This reproduces the exact split distributed with Polyp-PVT (which matches the
PraNet protocol counts) — do **not** rearrange the folders by hand.

## 6. Training / evaluation / benchmarking

```bash
# train the shipped model (~3 h on a T4)
python scripts/train.py --config configs/exp8_combo.yaml

# evaluate (14-column CSV: Dice/IoU/maxBF/BF@0.5/bIoU at both protocols,
# S-measure, E-measure, wFb, MAE; per-image and size-stratified CSVs)
python scripts/eval.py --checkpoint runs/exp8/last.pth \
    --data-root data/TestDataset --out results/main_results_exp8.csv --save-vis 16

# our efficiency benchmark (thop + flop_counter, latency mean±std/P95, fp16)
python scripts/benchmark_speed.py --config configs/exp8_combo.yaml --fp16

# competitor benchmarks (auto-clones official repos into external_repos/)
python scripts/benchmark_competitors.py --iters 300 --warmup 20 --fp16

# paper figures (median-Dice success cases + worst-Dice failure cases)
python scripts/make_figures.py --checkpoint runs/exp8/last.pth \
    --data-root data/TestDataset --out results/figs
```

Prefer Colab? `notebooks/colab_train.ipynb` drives the full pipeline
(data → train → eval → archive to Drive) for all 24 experiment configs;
run `python scripts/check_runcard.py` first — it verifies configs, scripts,
and dependencies in ~10 s.

## 7. Reproducibility notes

- **Protocol**: 1,450/798 split rebuilt deterministically from the official
  bundle; every count hard-asserted; train/test disjointness verified by
  content hashing; build log released.
- **Seeds**: main model, full design, single-component additions, and the
  capacity-matched control use three seeds (42/43/44) — means ± sample std.
  Staircase rows (R1–R3) and R9 are single-seed; deltas are read against the
  multi-seed noise floor, and multi-seed comparisons use Welch's *t*-tests.
- **Evaluation**: inputs scaled to [0,1] with ImageNet normalization; fixed
  0.5 threshold; no TTA, no post-processing; no validation split (hyper-
  parameters set once before the study — every number is a final-epoch
  test-set result, stated in the paper).
- **Boundary metrics are protocol-dependent**: the 352×352 protocol is
  primary and declared everywhere; original-resolution columns are provided
  for completeness and are *not* comparable across papers.
- **fp16**: autocast slows all five models on the T4 — engine-level
  optimization would be required (see Sec. 4.7 of the paper).

## 8. Citation

Under review at the *Journal of Real-Time Image Processing*. Citation block
will be finalized upon acceptance.

## 9. Acknowledgments & license

- `src/fanet.py` is a faithful port of the official
  [FANet repository](https://github.com/nikhilroxtomar/FANet)
  (Tomar et al., IEEE TNNLS 2022), re-trained here under the identical
  protocol for controlled comparison.
- All datasets are publicly available and de-identified by their providers
  (Kvasir-SEG, CVC-ClinicDB, CVC-300/EndoScene, CVC-ColonDB, ETIS-Larib).
- The timm-aligned MobileViT blocks follow
  [Mehta et al., ICCV 2021](https://github.com/apple/ml-cvnets) and load the
  [timm](https://github.com/huggingface/pytorch-image-models)
  `mobilevit_xxs` ImageNet checkpoint.
- Code released for research purposes.
