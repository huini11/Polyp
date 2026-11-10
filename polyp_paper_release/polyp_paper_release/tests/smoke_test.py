"""
CPU smoke test — verifies the full engineering loop before any real training.

Checks:
  (a) forward pass output shapes are correct (352x352 seg/boundary maps)
  (b) backward pass runs without NaN gradients
  (c) parameter count < 10M (printed)
  (d) a 2-epoch training loop on a mini synthetic dataset completes and the
      loss decreases (no crash, basic convergence signal)

Run from the polyp_paper/ root:
    python tests/smoke_test.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import PolypDataset              # noqa: E402
from src.losses import CompoundLoss, make_boundary_gt  # noqa: E402
from src.metrics import evaluate_image             # noqa: E402
from src.model import LightPolySegNet              # noqa: E402
from torch.utils.data import DataLoader, Dataset   # noqa: E402

PASSED = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    PASSED.append(bool(cond))
    return cond


# ---------------------------------------------------------------------------
# Mini synthetic dataset: 8 random polyp-like images with elliptical masks
# ---------------------------------------------------------------------------

class MiniSynthetic(Dataset):
    def __init__(self, n=8, size=352):
        self.n, self.size = n, size
        rng = np.random.RandomState(0)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = np.random.RandomState(idx)
        size = self.size
        # Random textured background + bright ellipse "polyp"
        img = (rng.rand(size, size, 3) * 0.3).astype(np.float32)
        msk = np.zeros((size, size), np.float32)
        cx, cy = rng.randint(100, size - 100), rng.randint(100, size - 100)
        ax, ay = rng.randint(40, 90), rng.randint(40, 90)
        yy, xx = np.mgrid[0:size, 0:size]
        inside = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 <= 1.0
        img[inside] += 0.6
        msk[inside] = 1.0
        img = (img - img.mean()) / (img.std() + 1e-8)  # roughly normalized
        return {
            "image": torch.from_numpy(img.transpose(2, 0, 1)).float(),
            "mask": torch.from_numpy(msk)[None],
            "orig_size": torch.tensor([size, size], dtype=torch.long),
            "name": f"synthetic_{idx}.png",
        }


def main():
    torch.manual_seed(42)
    device = torch.device("cpu")
    size = 352

    # ------------------------------------------------------------------
    # (c) Parameter count FIRST (fail fast if the budget is blown)
    # ------------------------------------------------------------------
    model = LightPolySegNet().to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n== Parameter count: {n_params:.2f} M ==")
    check("param budget < 10M", n_params < 10.0, f"{n_params:.2f} M")

    # ------------------------------------------------------------------
    # (a) Forward shapes
    # ------------------------------------------------------------------
    x = torch.randn(2, 3, size, size)
    with torch.no_grad():
        out = model(x)
    check("seg output shape", out["seg"].shape == (2, 1, size, size),
          str(tuple(out["seg"].shape)))
    check("boundary output shape", out["boundary"].shape == (2, 1, size, size),
          str(tuple(out["boundary"].shape)))
    check("aux maps present (deep supervision)", len(out["aux"]) == 2,
          str([tuple(a.shape) for a in out["aux"]]))
    check("seg logits finite", torch.isfinite(out["seg"]).all().item())

    # ------------------------------------------------------------------
    # (b) Backward: finite gradients, no NaN
    # ------------------------------------------------------------------
    model.train()
    criterion = CompoundLoss(w_bce=1.0, w_iou=1.0, w_boundary=0.5, w_aux=0.4)
    out = model(x)
    msk = (torch.rand(2, 1, size, size) > 0.7).float()
    losses = criterion(out, msk)
    losses["total"].backward()
    nan_grads = [n for n, p in model.named_parameters()
                 if p.grad is not None and not torch.isfinite(p.grad).all()]
    check("backward without NaN gradients", len(nan_grads) == 0,
          f"loss_total={losses['total'].item():.4f}, bad grads={nan_grads[:3]}")
    check("boundary GT gradient sanity", make_boundary_gt(msk).max().item() <= 1.0)
    model.zero_grad(set_to_none=True)

    # quick metric sanity on synthetic binary maps
    pred = torch.sigmoid(out["seg"].detach())[0, 0].numpy()
    gt = msk[0, 0].numpy()
    m = evaluate_image(pred, gt)
    check("metrics computable", all(np.isfinite(v) for v in m.values()),
          f"dice={m['dice']:.3f} iou={m['iou']:.3f} bf={m['bf']:.3f} bIoU={m['boundary_iou']:.3f}")

    # ------------------------------------------------------------------
    # (d) 2-epoch training loop on mini synthetic dataset
    # ------------------------------------------------------------------
    ds = MiniSynthetic(n=8, size=size)
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    epoch_losses = []
    model.train()
    for epoch in range(2):
        running = []
        for batch in loader:
            img, tgt = batch["image"], batch["mask"]
            optimizer.zero_grad(set_to_none=True)
            out = model(img)
            losses = criterion(out, tgt)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running.append(losses["total"].item())
        epoch_losses.append(float(np.mean(running)))
        print(f"  [mini-train] epoch {epoch + 1}/2  avg loss = {epoch_losses[-1]:.4f}")
    check("mini training 2 epochs completed", True,
          f"loss {epoch_losses[0]:.4f} -> {epoch_losses[-1]:.4f}")
    check("training loss decreased", epoch_losses[-1] < epoch_losses[0],
          f"{epoch_losses[0]:.4f} -> {epoch_losses[-1]:.4f}")

    # ------------------------------------------------------------------
    # Dataset class smoke test (directory convention + augment path)
    # ------------------------------------------------------------------
    try:
        import cv2  # noqa: F401
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_ds")
        os.makedirs(os.path.join(root, "images"), exist_ok=True)
        os.makedirs(os.path.join(root, "masks"), exist_ok=True)
        import cv2
        for i in range(3):
            im = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
            mk = np.zeros((240, 320), np.uint8)
            cv2.circle(mk, (160, 120), 60, 255, -1)
            cv2.imwrite(os.path.join(root, "images", f"img{i}.png"),
                        cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(root, "masks", f"img{i}.png"), mk)
        tr = PolypDataset(root, 352, mode="train")
        te = PolypDataset(root, 352, mode="test")
        s_tr, s_te = tr[0], te[0]
        check("PolypDataset train mode", s_tr["image"].shape == (3, 352, 352)
              and s_tr["mask"].shape == (1, 352, 352))
        check("PolypDataset test mode keeps orig GT size",
              s_te["image"].shape == (3, 352, 352)
              and s_te["mask"].shape == (1, 240, 320)
              and s_te["orig_size"].tolist() == [240, 320])
        import shutil
        shutil.rmtree(root)
    except ImportError:
        print("[skip] cv2 not installed; dataset smoke test skipped")

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    n_ok = sum(PASSED)
    print(f"SMOKE TEST: {n_ok}/{len(PASSED)} checks passed")
    print(f"Model params: {n_params:.2f} M  |  requirement < 10M -> "
          f"{'OK' if n_params < 10 else 'VIOLATION'}")
    print("=" * 60)
    sys.exit(0 if all(PASSED) else 1)


if __name__ == "__main__":
    main()
