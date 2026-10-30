"""
Evaluation on the 5 test sets -> paper-table CSV.

Usage (from polyp_paper/ root):
    python scripts/eval.py --checkpoint runs/exp1/best.pth
    python scripts/eval.py --checkpoint runs/exp2/last.pth --save-vis 8 \
        --out results/main_results.csv
    # --checkpoint accepts any path, including mounted Google Drive paths:
    python scripts/eval.py --checkpoint /content/drive/MyDrive/polyp_paper/exp2_ckpt/last.pth

Boundary metrics are reported under BOTH protocols (v2):
  - "BF score" / "Boundary IoU": at the ORIGINAL image resolution
    (prediction upscaled to original size before thresholding).
  - "BF score (352)" / "Boundary IoU (352)": at the 352x352 network
    resolution (GT downscaled to 352, nearest). This is the protocol most
    comparable with the polyp-segmentation literature; the paper table uses
    these columns.

Per test set computes mDice, mIoU, BF score (max over thresholds 0.1-0.9),
Boundary IoU (tolerance 2 px), plus mean rows. With --save-vis N, saves N
"original | GT | prediction overlay" triptychs per dataset to
results/vis/<dataset>/ for diagnosis and qualitative paper figures.
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

DEFAULT_DATASETS = ["CVC-300", "CVC-ColonDB", "ETIS-Larib", "Kvasir-SEG", "CVC-ClinicDB"]

METRIC_KEYS = ["dice", "iou", "bf", "boundary_iou", "bf352", "boundary_iou352"]


@torch.no_grad()
def evaluate_dataset(model, device, root, image_size=352, tolerance=2,
                     save_vis=0, vis_dir=None, num_iter=1, iters_csv=None):
    """
    Evaluate one test-set folder.
    num_iter=1            -> standard single-pass models (ours).
    num_iter>1 (FANet)    -> official iterative rectification protocol: the
        model's own previous binarized prediction is fed back as the feedback
        mask (zeros in iteration 1, Tomar et al. 2021 test.py, num_iter=10).
        Full paper metrics are computed at the FINAL iteration; quick Dice is
        logged per intermediate iteration to `iters_csv` for transparency.
    """
    ds = PolypDataset(root, image_size, mode="test")
    if save_vis > 0 and vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
    is_fanet = num_iter > 1
    prev = [torch.zeros(1, 1, image_size, image_size, device=device) for _ in range(len(ds))]
    iter_dice = {it: [] for it in range(1, num_iter + 1)}
    rows = []
    pbar = tqdm(range(len(ds)), desc=f"  {os.path.basename(root)}", ncols=90)
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
                # quick Dice at 352 for the per-iteration transparency log
                gt_q = (cv2.resize(gt.astype(np.uint8), (image_size, image_size),
                                   interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
                iter_dice[it].append(dice_score((prob352 >= 0.5).astype(np.uint8), gt_q))
                if it < num_iter:
                    prev[i] = (prob >= 0.5).float()[None, None]  # feedback (B,1,H,W) for next pass
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
        rows.append(m)
        pbar.set_postfix(dice=f"{m['dice']:.3f}", bf352=f"{m['bf352']:.3f}")

        # ---- Qualitative triptych: original | GT | prediction overlay ----
        if save_vis > 0 and i < save_vis and vis_dir:
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
                # prediction boundary contour for edge-quality inspection
                contours, _ = cv2.findContours(pred_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(panel_pr, contours, -1, (0, 255, 255), 1)           # yellow
                tri = np.hstack([orig, panel_gt, panel_pr])
                stem = os.path.splitext(item["name"])[0]
                cv2.imwrite(os.path.join(vis_dir, f"{stem}_vis.png"), tri)
    agg = {k: float(np.mean([r[k] for r in rows])) for k in METRIC_KEYS}
    agg["n"] = len(rows)
    if is_fanet and iters_csv:
        os.makedirs(os.path.dirname(iters_csv) or ".", exist_ok=True)
        with open(iters_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["iteration", "mDice(352)", "dataset"])
            for it in range(1, num_iter + 1):
                w.writerow([it, f"{np.mean(iter_dice[it]):.4f}", os.path.basename(root)])
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
    ap.add_argument("--save-vis", type=int, default=8,
                    help="save N qualitative triptychs per dataset to results/vis/<ds>/ (0=off)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config") or {"model": {}}
    arch = cfg.get("model", {}).get("arch", "lightpolysegnet")
    num_iter = 10 if arch == "fanet" else 1   # official FANet test-time rectification
    # load_pretrained=False: weights come from the checkpoint, never re-download
    model = build_model_from_cfg(cfg, load_pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[eval] loaded {args.checkpoint} (epoch {ckpt.get('epoch', '?')}) on {device} "
          f"| arch={arch}" + (f" | iterative rectification x{num_iter}" if num_iter > 1 else ""))

    vis_root = os.path.join(os.path.dirname(args.out) or ".", "vis")
    results = []
    for name in args.datasets:
        root = os.path.join(args.data_root, name)
        if not os.path.isdir(root):
            print(f"[skip] missing test set folder: {root}")
            continue
        iters_csv = os.path.join(os.path.dirname(args.out) or ".",
                                 f"fanet_iters_{name}.csv") if num_iter > 1 else None
        agg = evaluate_dataset(model, device, root, args.image_size, args.tolerance,
                               save_vis=args.save_vis,
                               vis_dir=os.path.join(vis_root, name),
                               num_iter=num_iter, iters_csv=iters_csv)
        results.append({"dataset": name, **agg})
        print(f"[result] {name}: mDice={agg['dice']:.4f} mIoU={agg['iou']:.4f} "
              f"BF={agg['bf']:.4f}/{agg['bf352']:.4f}(352) "
              f"bIoU={agg['boundary_iou']:.4f}/{agg['boundary_iou352']:.4f}(352) "
              f"(n={agg['n']})")

    if not results:
        print("[warn] no datasets evaluated; nothing written.")
        return

    # Paper-table CSV: per-dataset rows + mean row
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Images",
                    "mDice", "mIoU",
                    "BF score", "Boundary IoU",
                    "BF score (352)", "Boundary IoU (352)"])
        for r in results:
            w.writerow([r["dataset"], r["n"],
                        f"{r['dice']:.4f}", f"{r['iou']:.4f}",
                        f"{r['bf']:.4f}", f"{r['boundary_iou']:.4f}",
                        f"{r['bf352']:.4f}", f"{r['boundary_iou352']:.4f}"])
        mean = {k: float(np.mean([r[k] for r in results])) for k in METRIC_KEYS}
        w.writerow(["Mean", sum(r["n"] for r in results),
                    f"{mean['dice']:.4f}", f"{mean['iou']:.4f}",
                    f"{mean['bf']:.4f}", f"{mean['boundary_iou']:.4f}",
                    f"{mean['bf352']:.4f}", f"{mean['boundary_iou352']:.4f}"])
    print(f"[done] paper table written to {args.out}")
    if args.save_vis > 0:
        print(f"[done] qualitative triptychs (first {args.save_vis}/dataset) in {vis_root}/")


if __name__ == "__main__":
    main()
