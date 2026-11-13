#!/usr/bin/env python3
"""Build the standard polyp-segmentation benchmark split from the official
Polyp-PVT dataset bundle.

PROTOCOL (fixed 2026-11-02 after an external review caught a leakage bug):

  The de-facto standard "PraNet protocol" for this benchmark is
      train  = 1,450 pairs  (900 Kvasir-SEG + 550 CVC-ClinicDB)
      test   =   798 pairs  (Kvasir-SEG 100 + CVC-ClinicDB 62 held-out
                             + CVC-300 60 + CVC-ColonDB 380 + ETIS-Larib 196)
  PraNet's original index lists were never published; the split distributed
  with the Polyp-PVT code release (same counts, 1,450/798) is the community
  reference implementation used by Polyp-PVT, UACANet, HarDNet-MSEG and the
  numbers reported in those papers.

  The official bundle already stores exactly this split:
      dataset/TrainDataset/images            1,450  (550 ClinicDB @384x288
                                                     + 900 Kvasir)
      dataset/TestDataset/Kvasir              100    (held-out)
      dataset/TestDataset/CVC-ClinicDB         62    (held-out)
      dataset/TestDataset/CVC-300              60
      dataset/TestDataset/CVC-ColonDB         380
      dataset/TestDataset/ETIS-LaribPolypDB   196
  Therefore this script simply re-labels the bundle's native layout
  (folder renames only; every count hard-asserted).

  HISTORY / DO NOT REGRESS: an earlier version of this script merged all
  1,000 Kvasir + 612 ClinicDB pairs into TrainDataset (1,612) and also used
  the same pairs as the in-domain test sets -- that tests on training images
  (data leakage) and inflates in-domain results. Never merge Train+Test.
"""
import argparse
import json
import os
import shutil
import sys

def list_pairs(img_dir, msk_dir):
    """Return [(img_path, msk_path)] pairing by stem, tolerant to extensions."""
    pairs = []
    for name in sorted(os.listdir(img_dir)):
        stem, _ = os.path.splitext(name)
        for ext in (".png", ".jpg", ".jpeg", ".bmp"):
            m = os.path.join(msk_dir, stem + ext)
            if os.path.isfile(m):
                pairs.append((os.path.join(img_dir, name), m))
                break
        else:
            raise FileNotFoundError(f"mask not found for {name} in {msk_dir}")
    return pairs

def copy_pairs(pairs, out_img_dir, out_msk_dir, prefix=None):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_msk_dir, exist_ok=True)
    for i, (img, msk) in enumerate(pairs, 1):
        stem = os.path.splitext(os.path.basename(img))[0]
        if prefix:
            stem = f"{prefix}_{i:04d}"
        shutil.copy(img, os.path.join(out_img_dir, stem + os.path.splitext(img)[1]))
        shutil.copy(msk, os.path.join(out_msk_dir, stem + os.path.splitext(msk)[1]))

def main():
    ap = argparse.ArgumentParser(description="Build the standard 1450/798 split from the Polyp-PVT bundle")
    ap.add_argument("--src", required=True, help="path to extracted 'dataset' directory")
    ap.add_argument("--out", default="data", help="output root (creates TrainDataset/TestDataset)")
    args = ap.parse_args()
    src = args.src
    assert os.path.isdir(os.path.join(src, "TrainDataset", "images")), f"bad src: {src}"

    # ---- source counts (hard-asserted: the bundle IS the 1450/798 split) ----
    train = list_pairs(os.path.join(src, "TrainDataset", "images"),
                       os.path.join(src, "TrainDataset", "masks"))
    assert len(train) == 1450, f"bundle TrainDataset expected 1450, got {len(train)}"

    sets = {
        "Kvasir-SEG":   ("Kvasir",             100),
        "CVC-ClinicDB": ("CVC-ClinicDB",        62),
        "CVC-300":      ("CVC-300",             60),
        "CVC-ColonDB":  ("CVC-ColonDB",        380),
        "ETIS-Larib":   ("ETIS-LaribPolypDB",  196),
    }

    # ---- output tree ----
    out = args.out
    for d in ["TrainDataset/images", "TrainDataset/masks"] + [
            f"TestDataset/{name}/images" for name in sets] + [
            f"TestDataset/{name}/masks" for name in sets]:
        os.makedirs(os.path.join(out, d), exist_ok=True)

    # ---- train: verbatim copy (1,450) ----
    copy_pairs(train, os.path.join(out, "TrainDataset/images"),
               os.path.join(out, "TrainDataset/masks"))

    # ---- test sets: verbatim copies with renames ----
    log = {"protocol": "standard 1450/798 (PraNet counts; Polyp-PVT bundle split)",
           "train": 1450, "test": {}, "source": os.path.abspath(src)}
    for name, (src_name, expect) in sets.items():
        pairs = list_pairs(os.path.join(src, "TestDataset", src_name, "images"),
                           os.path.join(src, "TestDataset", src_name, "masks"))
        assert len(pairs) == expect, f"{src_name} expected {expect}, got {len(pairs)}"
        copy_pairs(pairs, os.path.join(out, f"TestDataset/{name}/images"),
                   os.path.join(out, f"TestDataset/{name}/masks"))
        log["test"][name] = len(pairs)

    with open(os.path.join(out, "build_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print("[build] standard 1450/798 split created at", os.path.abspath(out))
    print("[build] TrainDataset: 1450 | TestDataset:", log["test"])
    print("[build] NOTE: in-domain test sets (Kvasir-SEG 100, CVC-ClinicDB 62)")
    print("[build]       are HELD-OUT images, disjoint from the 1,450 training pairs.")

if __name__ == "__main__":
    sys.exit(main())
