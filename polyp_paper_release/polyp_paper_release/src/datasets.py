"""
Datasets for polyp segmentation (PraNet-style directory convention).

Expected layout (each dataset is one folder with `images/` and `masks/`):

    data/
      TrainDataset/          <- Kvasir-SEG(1000) + CVC-ClinicDB(612) merged
        images/*.png|jpg
        masks/*.png
      TestDataset/
        CVC-300/    {images/, masks/}
        CVC-ColonDB/{images/, masks/}
        ETIS-Larib/ {images/, masks/}
        Kvasir-SEG/ {images/, masks/}
        CVC-ClinicDB/{images/, masks/}

Training augmentation (matches common polyp-segmentation practice):
    random horizontal flip, random vertical flip, random rotation (+/-20 deg),
    random scale-crop (scale 0.75-1.25, random crop, resize back to 352).
Optional color-domain augmentation (aug_color, v2): HSV jitter + brightness /
contrast jitter + random gamma, applied with probability 0.5 in train mode.
Targets the color-domain gap between training data (Kvasir/ClinicDB) and
cross-domain test sets (ETIS/CVC-ColonDB).
Test mode: resize to image_size, keep the original size for evaluation-time
rescaling of predictions (standard protocol).
"""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _list_pairs(root: str):
    """Return sorted (image_path, mask_path) pairs; match by filename stem."""
    img_dir = os.path.join(root, "images")
    msk_dir = os.path.join(root, "masks")
    if not os.path.isdir(img_dir) or not os.path.isdir(msk_dir):
        raise FileNotFoundError(
            f"Expected '{root}/images' and '{root}/masks'. Got images={os.path.isdir(img_dir)}, masks={os.path.isdir(msk_dir)}"
        )
    mask_stems = {}
    for f in sorted(os.listdir(msk_dir)):
        if f.lower().endswith(IMG_EXTS):
            mask_stems[os.path.splitext(f)[0]] = os.path.join(msk_dir, f)
    pairs = []
    for f in sorted(os.listdir(img_dir)):
        if f.lower().endswith(IMG_EXTS):
            stem = os.path.splitext(f)[0]
            if stem in mask_stems:
                pairs.append((os.path.join(img_dir, f), mask_stems[stem]))
    if not pairs:
        raise FileNotFoundError(f"No image/mask pairs found under {root}")
    return pairs


def _load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_mask(path: str) -> np.ndarray:
    msk = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if msk is None:
        raise IOError(f"Cannot read mask: {path}")
    return msk


class PolypDataset(Dataset):
    """Polyp segmentation dataset with train/test modes."""

    def __init__(self, root: str, image_size: int = 352, mode: str = "train",
                 aug_color: bool = True):
        assert mode in ("train", "test")
        self.root = root
        self.size = image_size
        self.mode = mode
        self.aug_color = aug_color and (mode == "train")
        self.pairs = _list_pairs(root)

    def __len__(self):
        return len(self.pairs)

    # ------------------------- augmentation -------------------------
    def _augment(self, img, msk):
        # Random horizontal flip
        if random.random() < 0.5:
            img, msk = img[:, ::-1], msk[:, ::-1]
        # Random vertical flip
        if random.random() < 0.5:
            img, msk = img[::-1], msk[::-1]
        # Random rotation (+/-20 deg), border replicated
        if random.random() < 0.5:
            angle = random.uniform(-20, 20)
            h, w = msk.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(np.ascontiguousarray(img), M, (w, h),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            msk = cv2.warpAffine(np.ascontiguousarray(msk), M, (w, h),
                                 flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        # Random scale-crop: scale in [0.75, 1.25], random crop, resize back
        if random.random() < 0.5:
            scale = random.uniform(0.75, 1.25)
            h, w = msk.shape[:2]
            nh, nw = int(h * scale), int(w * scale)
            img = cv2.resize(np.ascontiguousarray(img), (nw, nh), interpolation=cv2.INTER_LINEAR)
            msk = cv2.resize(np.ascontiguousarray(msk), (nw, nh), interpolation=cv2.INTER_NEAREST)
            if nh >= self.size and nw >= self.size:
                y0 = random.randint(0, nh - self.size)
                x0 = random.randint(0, nw - self.size)
                img = img[y0:y0 + self.size, x0:x0 + self.size]
                msk = msk[y0:y0 + self.size, x0:x0 + self.size]
            else:  # too small after downscale -> pad
                ph, pw = max(0, self.size - nh), max(0, self.size - nw)
                img = cv2.copyMakeBorder(np.ascontiguousarray(img), 0, ph, 0, pw,
                                         cv2.BORDER_REFLECT)
                msk = cv2.copyMakeBorder(np.ascontiguousarray(msk), 0, ph, 0, pw,
                                         cv2.BORDER_REPLICATE)
        return img, msk

    # ------------------------- color augmentation (v2) -------------------------
    def _augment_color(self, img):
        """
        Color-domain augmentation on a uint8 RGB image (H,W,3):
          HSV jitter      -> H +/-10 deg, S +/-30%, V +/-30%
          brightness/contrast -> out = alpha * img + beta, alpha in [0.8,1.2],
                                 beta in [-25, 25]
          random gamma    -> out = 255 * (img/255)^(1/gamma), gamma in [0.7, 1.5]
        Applied as a block with probability 0.5 (see __getitem__). Uses only
        cv2/numpy, no new dependencies.
        """
        # --- HSV jitter ---
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
        h_j = random.randint(-10, 10)
        s_j = random.randint(-76, 76)   # +/-30% of 255
        v_j = random.randint(-76, 76)
        hsv[..., 0] = (hsv[..., 0] + h_j) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] + s_j, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] + v_j, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        # --- brightness / contrast jitter ---
        alpha = random.uniform(0.8, 1.2)
        beta = random.uniform(-25.0, 25.0)
        img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
        # --- random gamma ---
        gamma = random.uniform(0.7, 1.5)
        lut = (np.power(np.arange(256, dtype=np.float32) / 255.0, 1.0 / gamma)
               * 255.0).astype(np.uint8)
        img = cv2.LUT(img, lut)
        return img

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]
        img = _load_image(img_path)
        msk = _load_mask(msk_path)

        if self.mode == "train":
            if self.aug_color and random.random() < 0.5:
                img = self._augment_color(np.ascontiguousarray(img))
            img, msk = self._augment(img, msk)
            img = cv2.resize(np.ascontiguousarray(img), (self.size, self.size),
                             interpolation=cv2.INTER_LINEAR)
            msk = cv2.resize(np.ascontiguousarray(msk), (self.size, self.size),
                             interpolation=cv2.INTER_NEAREST)
            orig_size = (self.size, self.size)
        else:
            orig_size = (msk.shape[0], msk.shape[1])  # keep original GT for eval
            img = cv2.resize(np.ascontiguousarray(img), (self.size, self.size),
                             interpolation=cv2.INTER_LINEAR)

        # To tensors: image /255, normalize (ImageNet stats); mask -> {0,1}
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / \
              np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        msk = torch.from_numpy((msk > 127).astype(np.float32))[None]  # (1,H,W)

        return {
            "image": img,
            "mask": msk,
            "orig_size": torch.tensor(orig_size, dtype=torch.long),
            "name": os.path.basename(img_path),
        }
