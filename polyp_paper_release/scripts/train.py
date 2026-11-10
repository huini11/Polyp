"""
Training entry point.

Usage (from the polyp_paper/ root):
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --data-root data/TrainDataset --epochs 100
    python scripts/train.py --resume runs/exp1/last.pth

Features:
  - PraNet-protocol data (Kvasir-SEG + CVC-ClinicDB in one folder)
  - checkpoint resumption, best-checkpoint saving (by val mDice),
    TensorBoard logging, cosine schedule with warmup, optional AMP.
"""

import argparse
import json
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Make `src` importable when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import PolypDataset          # noqa: E402
from src.losses import CompoundLoss            # noqa: E402
from src.metrics import dice_score             # noqa: E402
from src.model import build_model_from_cfg     # noqa: E402


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def lr_lambda_factory(warmup_epochs, total_epochs, lr_min_ratio=0.0):
    """Warmup (linear) then cosine decay to lr_min."""
    def fn(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        t = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cos = 0.5 * (1 + np.cos(np.pi * t))
        return lr_min_ratio + (1 - lr_min_ratio) * cos
    return fn


@torch.no_grad()
def validate(model, loader, device, arch="lightpolysegnet") -> float:
    """Mean Dice on the validation split (binary at 0.5, at input res).
    For FANet, zero feedback masks are used (monitoring only; the final paper
    protocol trains with val_split=0 where validation is not used)."""
    model.eval()
    dices = []
    for batch in loader:
        img = batch["image"].to(device)
        msk = batch["mask"].to(device)
        if arch == "fanet":
            zeros = torch.zeros(img.shape[0], 1, img.shape[-2], img.shape[-1], device=device)
            seg = model(img, zeros)
        else:
            seg = model(img)["seg"]
        pred = (torch.sigmoid(seg) >= 0.5).float()
        for b in range(img.size(0)):
            dices.append(dice_score(pred[b, 0].cpu().numpy().astype(np.uint8),
                                    msk[b, 0].cpu().numpy().astype(np.uint8)))
    return float(np.mean(dices)) if dices else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train LightPolySegNet")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-root", default=None, help="override train data root")
    ap.add_argument("--out-dir", default=None, help="override output dir")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--val-split", type=float, default=None,
                    help="0.0 = train on all images (final paper run)")
    ap.add_argument("--resume", default=None, help="path to last.pth/best.pth")
    ap.add_argument("--backup-dir", default=None,
                    help="mirror last/best.pth + progress.json here every epoch (e.g. a Drive folder)")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tc = cfg["train"]
    # CLI overrides
    if args.data_root is not None: tc["data_root"] = args.data_root
    if args.out_dir is not None: tc["out_dir"] = args.out_dir
    if args.epochs is not None: tc["epochs"] = args.epochs
    if args.batch_size is not None: tc["batch_size"] = args.batch_size
    if args.lr is not None: tc["lr"] = args.lr
    if args.val_split is not None: tc["val_split"] = args.val_split
    if args.backup_dir is not None: tc["backup_dir"] = args.backup_dir

    set_seed(tc["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(tc["out_dir"], exist_ok=True)
    backup_dir = tc.get("backup_dir")
    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        print(f"[backup] mirroring checkpoints every epoch -> {backup_dir}")
    writer = SummaryWriter(log_dir=os.path.join(tc["out_dir"], "tb"))

    # ---------------- Data ----------------
    aug_color = cfg.get("data", {}).get("aug_color", True)
    full_ds = PolypDataset(tc["data_root"], tc["image_size"], mode="train",
                           aug_color=aug_color)
    n_val = int(len(full_ds) * tc["val_split"])
    if n_val > 0:
        g = torch.Generator().manual_seed(tc["seed"])
        idx = torch.randperm(len(full_ds), generator=g).tolist()
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        train_ds = Subset(full_ds, train_idx)
        val_ds = Subset(full_ds, val_idx)
    else:
        train_ds, val_ds = full_ds, None
    train_loader = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True,
                              num_workers=tc["num_workers"], pin_memory=(device.type == "cuda"),
                              drop_last=True)
    val_loader = (DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=False,
                             num_workers=tc["num_workers"], pin_memory=(device.type == "cuda"))
                  if val_ds is not None else None)
    print(f"[data] train={len(train_ds)}  val={len(val_ds) if val_ds is not None else 0}"
          f"  aug_color={aug_color}")

    # ---------------- Model / loss / optim ----------------
    arch = cfg["model"].get("arch", "lightpolysegnet")
    is_fanet = arch == "fanet"
    model = build_model_from_cfg(cfg).to(device)
    n_params = count_params(model)
    print(f"[model] arch={arch}  params: {n_params:.2f} M (requirement < 10M)")
    assert n_params < 10.0, "Parameter budget exceeded: tighten the model config."

    criterion = CompoundLoss(w_bce=tc["w_bce"], w_iou=tc["w_iou"],
                             w_boundary=tc["w_boundary"], w_aux=tc["w_aux"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"],
                                  weight_decay=tc["weight_decay"])
    start_epoch = 0
    best_dice = 0.0

    # Resume
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_dice = ckpt.get("best_dice", 0.0)
        print(f"[resume] from {args.resume} at epoch {start_epoch}")

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(tc["warmup_epochs"], tc["epochs"]))
    for _ in range(start_epoch):  # advance scheduler to resumed epoch
        scheduler.step()

    scaler = torch.amp.GradScaler("cuda", enabled=(tc["amp"] and device.type == "cuda"))

    # ---------------- Training loop ----------------
    # FANet feedback protocol (official train.py): every image has a "previous
    # epoch" mask (zeros initially) fed as hard attention; predictions update
    # it after each epoch. With val_split>0 we follow the official rule
    # (update only when validation improves); with val_split=0 (paper protocol)
    # we update every epoch.
    prev_masks = {}   # name -> (1,H,W) bool tensor (CPU)
    if args.resume and is_fanet and os.path.isfile(args.resume):
        try:
            saved = torch.load(args.resume, map_location="cpu", weights_only=False).get("prev_masks")
            if saved and "packed" in saved:
                import numpy as np
                bits = np.unpackbits(saved["packed"], axis=1)[:, :saved["hw"]]
                for i, n in enumerate(saved["names"]):
                    prev_masks[n] = torch.from_numpy(bits[i].reshape(saved["shape"])).to(torch.bool)
                print(f"[resume] restored {len(prev_masks)} feedback masks")
        except Exception as e:
            print(f"[resume][WARN] feedback masks not restored ({e}) -> zeros")

    global_step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, tc["epochs"]):
        model.train()
        t0 = time.time()
        epoch_preds = {}
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{tc['epochs']}", ncols=90)
        for batch in pbar:
            img = batch["image"].to(device, non_blocking=True)
            msk = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(tc["amp"] and device.type == "cuda")):
                if is_fanet:
                    names = list(batch["name"])
                    # prev_masks always stored on CPU; move to `device` here so
                    # the zero-template fallback and stored masks can never mix
                    # devices (crash fix: GPU zeros + CPU stored masks).
                    zeros = torch.zeros(1, img.shape[-2], img.shape[-1])  # CPU template
                    fb = torch.stack([(prev_masks[n] if n in prev_masks else zeros)
                                      .to(device) for n in names]).float()
                    out = {"seg": model(img, fb)}
                else:
                    out = model(img)
                losses = criterion(out, msk)
            scaler.scale(losses["total"]).backward()
            if tc["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            if is_fanet:
                with torch.no_grad():
                    preds = (torch.sigmoid(out["seg"].float()) > 0.5).cpu()
                    for n, pr in zip(names, preds):
                        epoch_preds[n] = pr

            writer.add_scalar("train/total", losses["total"].item(), global_step)
            writer.add_scalar("train/seg", losses["seg"].item(), global_step)
            if "boundary" in losses:
                writer.add_scalar("train/boundary", losses["boundary"].item(), global_step)
            global_step += 1
            pbar.set_postfix(loss=f"{losses['total'].item():.4f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        scheduler.step()

        # ---------------- Validation / checkpoints ----------------
        val_dice = validate(model, val_loader, device, arch=arch) if val_loader is not None else float("nan")
        writer.add_scalar("val/dice", val_dice, epoch)
        print(f"[epoch {epoch+1}] loss={losses['total'].item():.4f} "
              f"val_dice={val_dice:.4f} time={time.time()-t0:.0f}s")

        # FANet: update the feedback masks (see protocol note above)
        if is_fanet and epoch_preds:
            improved = (val_loader is None) or (val_dice > best_dice)
            if improved:
                prev_masks.update(epoch_preds)
                print(f"[fanet] feedback masks updated ({len(prev_masks)} entries)")

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_dice": max(best_dice, val_dice if val_ds is not None else 0.0),
            "config": cfg,
        }
        if is_fanet:
            # Persist feedback masks compactly (bit-packed via numpy) for faithful resume
            try:
                import numpy as np
                names = sorted(prev_masks)
                shape = list(prev_masks[names[0]].shape)
                bits = np.stack([prev_masks[n].flatten().numpy().astype(np.uint8)
                                 for n in names])
                ckpt["prev_masks"] = {
                    "names": names,
                    "shape": shape,
                    "hw": int(np.prod(shape)),
                    "packed": np.packbits(bits, axis=1),
                }
            except Exception as e:
                print(f"[fanet][WARN] could not pack feedback masks ({e})")
        torch.save(ckpt, os.path.join(tc["out_dir"], "last.pth"))
        if val_ds is not None and val_dice > best_dice:
            best_dice = val_dice
            torch.save(ckpt, os.path.join(tc["out_dir"], "best.pth"))
            print(f"[ckpt] new best val_dice={best_dice:.4f} -> best.pth")
        elif val_ds is None:
            torch.save(ckpt, os.path.join(tc["out_dir"], "best.pth"))  # final model = last

        # Mirror checkpoints to the backup dir (e.g. mounted Drive) every epoch
        if backup_dir:
            try:
                shutil.copyfile(os.path.join(tc["out_dir"], "last.pth"),
                                os.path.join(backup_dir, "last.pth"))
                best_local = os.path.join(tc["out_dir"], "best.pth")
                if os.path.isfile(best_local):
                    shutil.copyfile(best_local, os.path.join(backup_dir, "best.pth"))
                with open(os.path.join(backup_dir, "progress.json"), "w") as f:
                    json.dump({"epoch_done": epoch + 1, "total": tc["epochs"],
                               "val_dice": None if val_dice != val_dice else round(val_dice, 4)},
                              f)
            except OSError as e:
                print(f"[backup] mirror failed this epoch (retrying next epoch): {e}")

    writer.close()
    print(f"[done] checkpoints in {tc['out_dir']} (best val_dice={best_dice:.4f})")


if __name__ == "__main__":
    main()
