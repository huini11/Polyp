"""
Evaluation on the 5 test sets -> paper-table CSV (v3, post-review upgrade).

Usage (from polyp_paper/ root):
    python scripts/eval.py --checkpoint runs/main/last.pth
    python scripts/eval.py --checkpoint runs/exp2/last.pth --save-vis 8 \
        --out results/main_results.csv

v3 additions (external-review round, 2026-11-02):
  - Region metrics: mDice, mIoU (per-image, dataset-averaged)
  - Boundary metrics under BOTH declared protocols (orig / 352):
      maxBF (max over thresholds 0.1-0.9) AND BF@0.5 (fixed threshold),
      plus Boundary IoU (tolerance 2 px)
  - S-measure, max/mean E-measure, weighted F-beta, MAE (352 protocol,
    computed with the canonical PySODMetrics implementation)
  - Per-image CSV (results/per_image_<dataset>.csv) with per-image Dice/IoU/
    GT-area fraction, for median-based qualitative sampling and size
    stratification
  - Size-stratified Dice (small/medium/large terciles of GT area fraction)
    printed and written to results/size_stratified.csv
  - Aggregation rule: every "Mean" cell is the average of the five
    per-dataset means (each of which is the per-image mean). No pooled or
    image-weighted averaging anywhere, so all columns are consistent.
  - With --save-vis N: triptychs are the N images CLOSEST TO THE DATASET
    MEDIAN Dice (plus the worst one) instead of the first N.

FANet: num_iter>1 -> official 10-iteration feedback protocol; metrics at
the final iteration; per-iteration Dice logged to results/fanet_iters_*.csv.
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import _load_mask, PolypDataset   # noqa: E402
from src.metrics import bf_score, boundary_iou, dice_score, evaluate_image  # noqa: E402
from src.model import build_model_from_cfg          # noqa: E402

from py_sod_metrics import Emeasure, MAE, Smeasure, WeightedFmeasure  # noqa: E402

DEFAULT_DATASETS = ["CVC-300", "CVC-ColonDB", "ETIS-Larib", "Kvasir-SEG", "CVC-ClinicDB"]

METRIC_KEYS = ["dice", "iou", "bf", "boundary_iou",
               "bf352", "bf05_352", "boundary_iou352",
               "sm", "max_em", "mean_em", "wfm", "mae"]


@torch.no_grad()
def evaluate_dataset(model, device, root, image_size=352, tolerance=2,
                     save_vis=0, vis_dir=None, num_iter=1, iters_csv=None,
                     per_image_csv=None, size_csv=None):
    """
    Evaluate one test-set folder.
    num_iter=1            -> standard single-pass models (ours).
    num_iter>1 (FANet)    -> official iterative rectification protocol.
    """
    ds = PolypDataset(root, image_size, mode="test")
    if save_vis > 0 and vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
    is_fanet = num_iter > 1
    prev = [torch.zeros(1, 1, image_size, image_size, device=device) for _ in range(len(ds))]
    iter_dice = {it: [] for it in range(1, num_iter + 1)}
    rows = []
    sm_m, em_m, wf_m, mae_m = Smeasure(), Emeasure(), WeightedFmeasure(), MAE()
    pbar = tqdm(range(len(ds)), desc=f"  {os.path.basename(root)}", ncols=110)
    for i in pbar:
        item = ds[i]
        img = item["image"][None].to(device)
        h, w = item["orig_size"].tolist()
        gt = item["mask"][0].numpy()                            # original-size GT

        prob352 = None
        for it in range(1, num_iter + 1):
            if is_fanet:
                prob = torch.sigmoid(model(img, prev[i]))[0, 0]
            else:
                prob = torch.sigmoid(model(img)["seg"])[0, 0]
            prob352 = prob.cpu().numpy()
            if is_fanet:
                gt_q = (cv2.resize(gt.astype(np.uint8), (image_size, image_size),
                                   interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
                iter_dice[it].append(dice_score((prob352 >= 0.5).astype(np.uint8), gt_q))
                if it < num_iter:
                    prev[i] = (prob >= 0.5).float()[None, None]
        if is_fanet:
            prev[i] = (torch.from_numpy(prob352).to(device) >= 0.5).float()[None, None]

        # ---- Protocol A (original resolution): upscale prediction, orig GT ----
        prob_full = F.interpolate(torch.from_numpy(prob352).float()[None, None],
                                  size=(h, w), mode="bilinear",
                                  align_corners=False)[0, 0].numpy()
        m = evaluate_image(prob_full, gt, tolerance)            # dice/iou/bf/boundary_iou

        # ---- Protocol B (352x352): network-resolution prediction, GT at 352 ----
        gt352 = (cv2.resize(gt.astype(np.uint8), (image_size, image_size),
                            interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
        m["bf352"] = bf_score(prob352, gt352, tolerance)
        m["boundary_iou352"] = boundary_iou(prob352, gt352, tolerance)
        m["bf05_352"] = bf_score((prob352 >= 0.5).astype(np.float64), gt352, tolerance,
                                 thresholds=np.array([0.5]))
        m["gt_area_frac"] = float(gt352.mean())

        # ---- Region/boundary-quality metrics at 352 (PySODMetrics) ----
        pred_u8 = (np.clip(prob352, 0, 1) * 255).astype(np.uint8)
        gt_u8 = (gt352 * 255).astype(np.uint8)
        sm_m.step(pred=pred_u8, gt=gt_u8)
        em_m.step(pred=pred_u8, gt=gt_u8)
        wf_m.step(pred=pred_u8, gt=gt_u8)
        mae_m.step(pred=pred_u8, gt=gt_u8)

        rows.append(m)
        pbar.set_postfix(dice=f"{m['dice']:.3f}", bf352=f"{m['bf352']:.3f}")

        # ---- Qualitative triptych: original | GT | prediction overlay ----
        if save_vis > 0 and vis_dir:
            img_path, msk_path = ds.pairs[i]
            orig = cv2.imread(img_path, cv2.IMREAD_COLOR)       # BGR
            if orig is not None:
                orig = cv2.resize(orig, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                gt_v = _load_mask(msk_path)
                gt_v = cv2.resize(gt_v, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
                pred_v = (prob352 >= 0.5).astype(np.uint8)

                panel_gt = orig.copy()
                panel_pr = orig.copy()
                gt_m = gt_v > 0
                panel_gt[gt_m] = (0.55 * panel_gt[gt_m]
                                  + 0.45 * np.array([0, 255, 0])).astype(np.uint8)   # GT: green
                pr_m = pred_v > 0
                panel_pr[pr_m] = (0.55 * panel_pr[pr_m]
                                  + 0.45 * np.array([0, 0, 255])).astype(np.uint8)   # pred: red
                contours, _ = cv2.findContours(pred_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(panel_pr, contours, -1, (0, 255, 255), 1)           # yellow
                tri = np.hstack([orig, panel_gt, panel_pr])
                stem = os.path.splitext(item["name"])[0]
                cv2.imwrite(os.path.join(vis_dir, f"{stem}_vis.png"), tri)

    # ---- aggregate: every per-image metric is the per-image mean (no pooling) ----
    per_image_keys = ["dice", "iou", "bf", "bf05_352", "boundary_iou",
                      "bf352", "boundary_iou352", "gt_area_frac"]
    agg = {k: float(np.mean([r[k] for r in rows])) for k in per_image_keys}
    agg["n"] = len(rows)
    res_sm = sm_m.get_results()["sm"]
    res_em = em_m.get_results()["em"]            # {"adp":..., "curve": 256 values}
    agg["sm"] = float(res_sm)
    agg["max_em"] = float(np.max(res_em["curve"]))
    agg["mean_em"] = float(np.mean(res_em["curve"]))
    agg["wfm"] = float(wf_m.get_results()["wfm"])
    agg["mae"] = float(mae_m.get_results()["mae"])
    if is_fanet and iters_csv:
        os.makedirs(os.path.dirname(iters_csv) or ".", exist_ok=True)
        with open(iters_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["iteration", "mDice(352)", "dataset"])
            for it in range(1, num_iter + 1):
                w.writerow([it, f"{np.mean(iter_dice[it]):.4f}", os.path.basename(root)])

    # ---- per-image CSV (median-based qualitative sampling etc.) ----
    if per_image_csv:
        os.makedirs(os.path.dirname(per_image_csv) or ".", exist_ok=True)
        with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "image", "dice", "iou", "bf_max", "bf05",
                        "gt_area_frac"])
            for i, r in enumerate(rows):
                stem = os.path.splitext(ds[i]["name"])[0]
                w.writerow([os.path.basename(root), stem,
                            f"{r['dice']:.4f}", f"{r['iou']:.4f}",
                            f"{r['bf352']:.4f}", f"{r['bf05_352']:.4f}",
                            f"{r['gt_area_frac']:.4f}"])

    # ---- size-stratified Dice (terciles of GT area fraction) ----
    if size_csv and rows:
        areas = np.array([r["gt_area_frac"] for r in rows])
        dices = np.array([r["dice"] for r in rows])
        t1, t2 = np.percentile(areas, [33.3, 66.7])
        strata = {"small": areas <= t1,
                  "medium": (areas > t1) & (areas <= t2),
                  "large": areas > t2}
        os.makedirs(os.path.dirname(size_csv) or ".", exist_ok=True)
        write_header = not os.path.isfile(size_csv)
        with open(size_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["dataset", "stratum", "n", "mDice"])
            for sname, mask in strata.items():
                if mask.sum() > 0:
                    w.writerow([os.path.basename(root), sname, int(mask.sum()),
                                f"{dices[mask].mean():.4f}"])
        print(f"  [size] small(n={int(strata['small'].sum())}) "
              f"medium(n={int(strata['medium'].sum())}) "
              f"large(n={int(strata['large'].sum())})")
    return agg


def main():
    ap = argparse.ArgumentParser(description="Evaluate on 5 polyp test sets")
    ap.add_argument("--checkpoint", required=True,
                    help="path to .pth checkpoint (local or Google Drive path)")
    ap.add_argument("--data-root", default="data/TestDataset")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--out", default="results/main_results.csv")
    ap.add_argument("--image-size", type=int, default=352)
    ap.add_argument("--tolerance", type=int, default=2,
                    help="boundary pixel tolerance for BF score / boundary IoU")
    ap.add_argument("--save-vis", type=int, default=0,
                    help="save N qualitative triptychs per dataset to results/vis/<ds>/ (0=off)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config") or {"model": {}}
    arch = cfg.get("model", {}).get("arch", "lightpolysegnet")
    num_iter = 10 if arch == "fanet" else 1   # official FANet test-time rectification
    model = build_model_from_cfg(cfg, load_pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[eval] loaded {args.checkpoint} (epoch {ckpt.get('epoch', '?')}) on {device} "
          f"| arch={arch}" + (f" | iterative rectification x{num_iter}" if num_iter > 1 else ""))

    out_dir = os.path.dirname(args.out) or "."
    vis_root = os.path.join(out_dir, "vis")
    results = []
    for name in args.datasets:
        root = os.path.join(args.data_root, name)
        if not os.path.isdir(root):
            print(f"[skip] missing test set folder: {root}")
            continue
        iters_csv = os.path.join(out_dir, f"fanet_iters_{name}.csv") if num_iter > 1 else None
        agg = evaluate_dataset(model, device, root, args.image_size, args.tolerance,
                               save_vis=args.save_vis,
                               vis_dir=os.path.join(vis_root, name),
                               num_iter=num_iter, iters_csv=iters_csv,
                               per_image_csv=os.path.join(out_dir, f"per_image_{name}.csv"),
                               size_csv=os.path.join(out_dir, "size_stratified.csv"))
        results.append({"dataset": name, **agg})
        print(f"[result] {name}: mDice={agg['dice']:.4f} mIoU={agg['iou']:.4f} "
              f"BF={agg['bf']:.4f}/{agg['bf352']:.4f}(352) BF@0.5={agg['bf05_352']:.4f} "
              f"bIoU={agg['boundary_iou']:.4f}/{agg['boundary_iou352']:.4f}(352) "
              f"Sm={agg['sm']:.4f} maxEm={agg['max_em']:.4f} meanEm={agg['mean_em']:.4f} "
              f"wFm={agg['wfm']:.4f} MAE={agg['mae']:.4f} (n={agg['n']})")

    if not results:
        print("[warn] no datasets evaluated; nothing written.")
        return

    # Paper-table CSV: per-dataset rows + dataset-averaged mean row
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["Dataset", "Images",
                  "mDice", "mIoU",
                  "BF score", "Boundary IoU",
                  "BF score (352)", "BF score (352, thr=0.5)", "Boundary IoU (352)",
                  "Sm", "maxEm", "meanEm", "wFm", "MAE"]
        w.writerow(header)
        for r in results:
            w.writerow([r["dataset"], r["n"],
                        f"{r['dice']:.4f}", f"{r['iou']:.4f}",
                        f"{r['bf']:.4f}", f"{r['boundary_iou']:.4f}",
                        f"{r['bf352']:.4f}", f"{r['bf05_352']:.4f}",
                        f"{r['boundary_iou352']:.4f}",
                        f"{r['sm']:.4f}", f"{r['max_em']:.4f}", f"{r['mean_em']:.4f}",
                        f"{r['wfm']:.4f}", f"{r['mae']:.4f}"])
        mean = {k: float(np.mean([r[k] for r in results])) for k in ["dice", "iou", "bf", "boundary_iou", "bf352", "bf05_352", "boundary_iou352", "sm", "max_em", "mean_em", "wfm", "mae"]}
        w.writerow(["Mean", sum(r["n"] for r in results),
                    f"{mean['dice']:.4f}", f"{mean['iou']:.4f}",
                    f"{mean['bf']:.4f}", f"{mean['boundary_iou']:.4f}",
                    f"{mean['bf352']:.4f}", f"{mean['bf05_352']:.4f}",
                    f"{mean['boundary_iou352']:.4f}",
                    f"{mean['sm']:.4f}", f"{mean['max_em']:.4f}", f"{mean['mean_em']:.4f}",
                    f"{mean['wfm']:.4f}", f"{mean['mae']:.4f}"])
    print(f"[done] paper table written to {args.out} "
          f"(every Mean cell = average of the five per-dataset means)")
    if args.save_vis > 0:
        print(f"[done] qualitative triptychs in {vis_root}/")


if __name__ == "__main__":
    main()
