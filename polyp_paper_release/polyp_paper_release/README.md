# LightPolySegNet — Lightweight Transformer–CNN Hybrid for Real-Time Polyp Segmentation

Official reproducibility package (code, configs, and all measured results)
for the manuscript *"A Lightweight Transformer–CNN Hybrid Network for
Real-Time Polyp Segmentation in Colonoscopy"* (under review at the
Journal of Real-Time Image Processing).

**Protocol (standard PraNet split, verified leak-free):** train on
900 Kvasir-SEG + 550 CVC-ClinicDB (1,450 images); test on the held-out
Kvasir-SEG (100) + CVC-ClinicDB (62) subsets and three fully unseen
domains — CVC-300 (60), CVC-ColonDB (380), ETIS-Larib (196); input
352×352. Training and held-out test images are verified disjoint at the
file-content (hash) level by `scripts/build_pranet_split.py`.

## 0. Main results (mDice, three seeds 42/43/44, mean ± sample std)

| Method | Params | MACs | Kvasir (100) | ClinicDB (62) | CVC-300 | ColonDB | ETIS | **Mean** |
|---|---|---|---|---|---|---|---|---|
| PraNet (reported) | 26.7M | 13.2 | 0.898 | 0.899 | 0.871 | 0.709 | 0.628 | 0.801 |
| FANet (re-trained, same protocol) | 7.74M | 44.8 | 0.857 | 0.851 | 0.842 | 0.750 | 0.705 | 0.801 |
| HarDNet-MSEG (reported) | 33.3M | 11.4 | 0.912 | 0.932 | 0.887 | 0.731 | 0.677 | 0.828 |
| Polyp-PVT (reported) | 28.2M | 10.0 | 0.917 | 0.937 | 0.900 | 0.808 | 0.787 | 0.870 |
| **LightPolySegNet (ours)** | **0.98M** | **0.80** | 0.8997±0.0009 | 0.8850±0.0085 | 0.8633±0.0151 | 0.7550±0.0102 | 0.6636±0.0118 | **0.8133±0.0026** |

- **+1.2 over PraNet** with clear gains on the two hardest unseen domains
  (ColonDB +4.6, ETIS +3.6); ahead of FANet (re-trained by us under the
  identical protocol, including the same color augmentation) on
  Kvasir/ClinicDB/CVC-300, statistically tied on ColonDB, behind on ETIS.
- Polyp-PVT and HarDNet-MSEG lead on accuracy at 25–34× our parameter
  count; we do not claim an accuracy advantage over them.

## 1. Efficiency (same Tesla T4, batch 1, fp32, 352×352, identical protocol)

| Method | Params | MACs | FPS | Latency (mean±std) |
|---|---|---|---|---|
| PraNet | 26.7M | 13.15 | 53.4 | 18.71±2.08 ms |
| Polyp-PVT | 28.2M | 10.02 | 54.9 | 18.21±1.90 ms |
| HarDNet-MSEG | 33.3M | 11.40 | 61.8 | 16.18±3.26 ms |
| **LightPolySegNet (ours)** | **0.98M** | **0.80** | **96.6** | **10.4 ms** |

Competitor numbers are an architecture-level re-benchmark on the same GPU
(latency depends on architecture and input size, not on weight values, so
vendor models are measured with random initialization); MACs via thop.
The same model reaches 15.4 FPS on an Intel laptop CPU.

## 2. Ablations (paper §4.7; full numbers in `results/comparison_table_v2.md`)

| Row | Configuration | Params | Mean mDice |
|---|---|---|---|
| R1–R3 | staircase: baseline → +color aug → +redesign (rand-init) | 0.73M→0.98M | 0.7289 / 0.7565 / 0.7113 (single runs) |
| R4 | **full model** | 0.98M | **0.8133±0.0026** |
| R5 | w/o boundary loss | 0.98M | 0.8165±0.0053 |
| R6 | w/o Sobel edge attention | 0.97M | 0.8185±0.0017 |
| R7 | w/o Transformer blocks | 0.11M | 0.6260±0.0166 |
| R8 | w/o boundary loss **and** Sobel | 0.97M | 0.8192 |
| R9 | pretrained MobileNetV2-100 encoder + same decoder | 1.91M | 0.8209 |
| R10 | MobileViT encoder + minimal FPN decoder | 0.95M | 0.8119 |

Reading (as reported in the paper): ImageNet pretraining is the decisive
training-side factor (**+9.9**, ETIS +18.1); color augmentation +2.8;
architectural lightweighting without pretraining loses 4.5; the
Transformer blocks are the largest architectural factor (**+18.7**,
with a capacity/pretraining-coverage confound; a capacity-matched
pretrained CNN control, R9, reaches +1.0 at 2× the parameters — the
hybrid's advantage is parameter efficiency). Over three seeds, the
boundary apparatus (Sobel attention + boundary loss) shows **no
statistically established benefit** to region accuracy at fixed weights;
we report this honestly and keep the boundary head as a deployment-facing
contour output with a declared-protocol boundary evaluation.

## 3. Repository layout

```
configs/     19 experiment configs (yaml) — one per ablation / seed
src/         model, datasets, losses, metrics, FANet baseline, pretrained loader
scripts/     build_pranet_split.py   deterministic 1450/798 split + hash verification
             train.py                training entry point
             eval.py                 14-metric evaluation (mDice/mIoU/BF/BF@0.5/bIoU/
                                     Sm/Em/wFm/MAE; per-image CSV; size-stratified Dice)
             benchmark_speed.py      our model: latency mean±std/P95, fp16, dual MACs counters
             benchmark_competitors.py / benchmark_one.py
                                     same-GPU competitor re-benchmark (subprocess isolated)
             make_figures.py         median/worst-Dice sample figures (anti-cherry-picking)
             aggregate_runs.py       multi-seed aggregation
notebooks/   colab_train.ipynb       one-click Colab pipeline (19 experiments)
results/     all 20 measured CSVs (v2_* = leak-free 1450/798 protocol)
             + competitor_speed.csv + comparison_table_v2.md (full aggregation)
tests/       smoke_test.py           CPU-only sanity check
```

## 4. Reproduction

```bash
pip install -r requirements.txt
python tests/smoke_test.py                                  # CPU sanity check
python scripts/build_pranet_split.py --src <official-bundle> --out data
python scripts/train.py --config configs/exp3_pretrain.yaml # full model, seed 42
python scripts/eval.py --checkpoint runs/exp3/last.pth --data-root data/TestDataset
python scripts/benchmark_speed.py                           # efficiency table
python scripts/benchmark_competitors.py                     # competitor FPS (clones 3 repos)
```

The Colab notebook (`notebooks/colab_train.ipynb`) automates download →
split verification → training → evaluation for every experiment in
`configs/`; its dataset cell hard-asserts the 1,450/798 counts.

## 5. Notes

- `scripts/download_data.md` points to the official Polyp-PVT data bundle;
  the split script rebuilds the standard PraNet protocol from it
  deterministically and releases a build log.
- FANet is re-trained from the official implementation under our protocol
  (same color augmentation) and evaluated with its official 10-iteration
  feedback protocol.
- Code released for research purposes. `src/fanet.py` is a faithful port
  of the official FANet repository (Tomar et al., IEEE TNNLS 2022), used
  here under fair use for comparative evaluation.
- Paper under review at the Journal of Real-Time Image Processing;
  citation block to be finalized upon acceptance.
