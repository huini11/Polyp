"""
Figure regeneration for the paper (Fig. 2 success rows + Fig. 3 failure rows).

Selects samples by per-image Dice on the NEW 1,450/798 split:
  - Fig.2 (success): for each of the 4 paper datasets, the image whose Dice
    is CLOSEST TO THE DATASET MEDIAN (Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB,
    ETIS-Larib).
  - Fig.3 (failures): the WORST-Dice image of ETIS-Larib and of CVC-ColonDB.

Outputs (under --out, default results/figs):
  fig2_row1_kvasir_median.png      (1056x352 triptych: original | GT | pred)
  fig2_row2_clinic_median.png
  fig2_row3_colondb_median.png
  fig2_row4_etis_median.png
  fig3_row1_etis_worst.png
  fig3_row2_colondb_worst.png
  fig_selection.csv                (dataset, image, dice, role)
  figs.zip                         (everything above, ready to download)

Usage (Colab, after training):
    python scripts/make_figures.py \
        --checkpoint runs/exp3_pretrain/last.pth \
        --data-root data/TestDataset --out results/figs
"""

import argparse
import os
import sys
import zipfile

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import _load_mask, PolypDataset   # noqa: E402
from src.metrics import dice_score                  # noqa: E402
from src.model import build_model_from_cfg          # noqa: E402

FIG2 = [("Kvasir-SEG", "kvasir"), ("CVC-ClinicDB", "clinic"),
        ("CVC-ColonDB", "colondb"), ("ETIS-Larib", "etis")]
FIG3 = [("ETIS-Larib", "etis"), ("CVC-ColonDB", "colondb")]


@torch.no_grad()
def dataset_dice_table(model, device, root, image_size=352):
    """Run inference over a dataset; return [(name, dice, render_fn)]."""
    ds = PolypDataset(root, image_size, mode="test")
    out = []
    pbar = tqdm(range(len(ds)), desc=f"  {os.path.basename(root)}", ncols=90)
    for i in pbar:
        item = ds[i]
        img = item["image"][None].to(device)
        h, w = item["orig_size"].tolist()
        gt = item["mask"][0].numpy()
        prob = torch.sigmoid(model(img)["seg"])[0, 0].cpu().numpy()

        gt352 = (cv2.resize(gt.astype(np.uint8), (image_size, image_size),
                            interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
        dice = dice_score((prob >= 0.5).astype(np.uint8), gt352)

        def render(prob=prob, h=h, w=w, img_path=ds.pairs[i][0],
                   msk_path=ds.pairs[i][1]):
            orig = cv2.imread(img_path, cv2.IMREAD_COLOR)
            orig = cv2.resize(orig, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            gt_v = _load_mask(msk_path)
            gt_v = cv2.resize(gt_v, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
            pred_v = (prob >= 0.5).astype(np.uint8)
            panel_gt, panel_pr = orig.copy(), orig.copy()
            gm, pm = gt_v > 0, pred_v > 0
            panel_gt[gm] = (0.55 * panel_gt[gm] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
            panel_pr[pm] = (0.55 * panel_pr[pm] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)
            contours, _ = cv2.findContours(pred_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(panel_pr, contours, -1, (0, 255, 255), 1)
            return np.hstack([orig, panel_gt, panel_pr])

        out.append((item["name"], float(dice), render))
    return out


def main():
    ap = argparse.ArgumentParser(description="Regenerate paper Fig.2/Fig.3 samples")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default="data/TestDataset")
    ap.add_argument("--out", default="results/figs")
    ap.add_argument("--image-size", type=int, default=352)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config") or {"model": {}}
    model = build_model_from_cfg(cfg, load_pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[figs] loaded {args.checkpoint} on {device}")

    os.makedirs(args.out, exist_ok=True)
    selection = []

    # ---- per-dataset dice tables ----
    tables = {}
    for name, _ in FIG2 + FIG3:
        root = os.path.join(args.data_root, name)
        tables[name] = dataset_dice_table(model, device, root, args.image_size)

    # ---- Fig.2: median-Dice image per dataset ----
    for name, tag in FIG2:
        tab = sorted(tables[name], key=lambda x: x[1])
        med_idx = len(tab) // 2
        j = min(range(len(tab)), key=lambda k: abs(tab[k][1] - tab[med_idx][1]))
        stem, dice, render = tab[j]
        fname = f"fig2_{tag}_median.png"
        cv2.imwrite(os.path.join(args.out, fname), render())
        selection.append((name, stem, dice, f"fig2 median -> {fname}"))
        print(f"[fig2] {name}: median Dice={dice:.4f} image={stem} -> {fname}")

    # ---- Fig.3: worst image per dataset ----
    for name, tag in FIG3:
        tab = sorted(tables[name], key=lambda x: x[1])
        stem, dice, render = tab[0]
        fname = f"fig3_{tag}_worst.png"
        cv2.imwrite(os.path.join(args.out, fname), render())
        selection.append((name, stem, dice, f"fig3 worst -> {fname}"))
        print(f"[fig3] {name}: worst Dice={dice:.4f} image={stem} -> {fname}")

    # ---- selection log ----
    import csv
    with open(os.path.join(args.out, "fig_selection.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "image", "dice", "role"])
        for row in selection:
            w.writerow(row)

    # ---- zip ----
    zip_path = args.out.rstrip("/\\") + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(os.listdir(args.out)):
            zf.write(os.path.join(args.out, f), f)
    print(f"[done] {zip_path} ready to download (contains all fig samples + selection log)")


if __name__ == "__main__":
    main()
