# Dataset Download Guide

All five datasets follow the PraNet standard protocol. Target layout after
downloading (paths relative to the `polyp_paper/` root):

```
polyp_paper/
└── data/
    ├── TrainDataset/
    │   ├── images/     <- 1,612 images  (Kvasir-SEG 1000 + CVC-ClinicDB 612)
    │   └── masks/      <- 1,612 masks
    └── TestDataset/
        ├── CVC-300/      {images/, masks/}   (60 images, a.k.a. CVC-ClinicDB test set)
        ├── CVC-ColonDB/  {images/, masks/}   (380)
        ├── ETIS-Larib/   {images/, masks/}   (196)
        ├── Kvasir-SEG/   {images/, masks/}   (1000)
        └── CVC-ClinicDB/ {images/, masks/}   (612)
```

Important: in `TrainDataset/images`, **rename all files to be unique** (prefix
`kvasir_` / `clinic_`) — both source sets use generic numbering.

---

## Option A (recommended): official Polyp-PVT bundle + protocol reassembly

Single Google Drive file (README section 4.2; **verified 2026-10-21**, 1.11 GB,
the older PraNet folder link is dead):

- Repo: https://github.com/DengPingFan/Polyp-PVT (README -> 4.2 Data preparation)
- Drive file: https://drive.google.com/file/d/1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC/view  (`dataset.zip`)
- Baidu mirror: https://pan.baidu.com/s/1BTgT27VxvOgKpHrigwm7Bw  (code: sydz)

**Important**: the bundle stores the authors' *own* split, NOT the PraNet
protocol. Verified structure (2026-10-21):

```
dataset/TrainDataset/{images,masks}   1450 pairs  (550 @384x288 = ClinicDB, 900 = Kvasir)
dataset/TestDataset/CVC-300             60 pairs  (PraNet cross-domain set, verbatim)
dataset/TestDataset/CVC-ColonDB        380 pairs  (verbatim)
dataset/TestDataset/ETIS-LaribPolypDB  196 pairs  (verbatim; rename to ETIS-Larib)
dataset/TestDataset/Kvasir             100 pairs  (Kvasir hold-out of the authors' split)
dataset/TestDataset/CVC-ClinicDB        62 pairs  (ClinicDB hold-out, all 384x288)
```

Reassembly rule (verified): ClinicDB = all 384x288 images (550 + 62 = 612),
Kvasir = the rest (900 + 100 = 1000); in-domain test sets coincide with the
training data by the published PraNet protocol.

```bash
pip install gdown
gdown 1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC -O dataset.zip
unzip dataset.zip -d data_raw/
python scripts/build_pranet_split.py --src data_raw/dataset --out data
```

`build_pranet_split.py` hard-asserts every count (550/62/900/100 -> 612/1000)
and writes `data/build_log.json`. On Colab, cell 3 of
`notebooks/colab_train.ipynb` automates download (3 retries, >500 MB size
gate), extraction, reassembly and the sanity check; it also accepts a
manually uploaded `dataset.zip` at `MyDrive/polyp_paper/dataset.zip`.

## Option B: per-dataset official sources

| Dataset | Source | Link |
|---|---|---|
| Kvasir-SEG | Simula official (direct zip, 44MB) | https://datasets.simula.no/downloads/kvasir-seg.zip ; page: https://datasets.simula.no/kvasir-seg/ |
| Kvasir-SEG (mirror) | Kaggle | https://www.kaggle.com/datasets/debeshjha1/kvasirseg |
| CVC-ClinicDB | Giana / CRCV official page | https://www.polyp.cvc.uab.es/ (formerly https://crcv.ucf.edu/data/GIANA.php) [投稿前复核链接] |
| CVC-300 | same hosting as CVC-ClinicDB (test split of ClinicDB) | via Polyp-PVT Drive bundle (Option A) [投稿前复核链接] |
| CVC-ColonDB | http://mv.cvc.uab.es/projects/colon-qa/cvccolondb [投稿前复核链接] |
| ETIS-Larib | ETIS official request page | https://www.h64.org/ETIS/ (may require an access request form) [投稿前复核链接] |

Practical note: several CVC/ETIS pages now sit behind registration forms.
**For reproducibility and licensing simplicity, Option A (Polyp-PVT bundle) is
the recommended path** — it is the same data every baseline paper reports on.
If any individual link has moved, use the Drive bundle; do NOT substitute
different test splits, or the comparison table will not be comparable.

## Sanity checks after download

```bash
python - <<'PY'
import os
for name, n in [("TrainDataset", 1612), ("TestDataset/CVC-300", 60),
                ("TestDataset/CVC-ColonDB", 380), ("TestDataset/ETIS-Larib", 196),
                ("TestDataset/Kvasir-SEG", 1000), ("TestDataset/CVC-ClinicDB", 612)]:
    imgs = os.listdir(f"data/{name}/images")
    msks = os.listdir(f"data/{name}/masks")
    print(f"{name}: images={len(imgs)} masks={len(msks)}", "OK" if len(imgs) == n == len(msks) else "MISMATCH!")
PY
```
