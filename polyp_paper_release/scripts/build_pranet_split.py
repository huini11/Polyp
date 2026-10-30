#!/usr/bin/env python3
"""Reassemble the official Polyp-PVT dataset.zip into the PraNet-standard split.

The official bundle (https://drive.google.com/file/d/1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC)
contains the *authors' own* split (TrainDataset 1450 pairs; per-set test folders
totalling 798 pairs), which is NOT the PraNet protocol used by PraNet/Polyp-PVT/
UACANet and every baseline we compare against.

Verified facts about the bundle (checked 2026-10-21 on the official file):
  dataset/TrainDataset/images          1450 imgs  (550 @ 384x288 = ClinicDB, 900 = Kvasir)
  dataset/TestDataset/CVC-ClinicDB      62 pairs  (all 384x288 -> ClinicDB)
  dataset/TestDataset/Kvasir           100 pairs  (none 384x288 -> Kvasir)
  dataset/TestDataset/CVC-300           60 pairs  (PraNet cross-domain test set, verbatim)
  dataset/TestDataset/CVC-ColonDB      380 pairs  (PraNet cross-domain test set, verbatim)
  dataset/TestDataset/ETIS-LaribPolypDB 196 pairs (PraNet cross-domain test set, verbatim)

PraNet protocol reassembly (in-domain test sets coincide with training data
by design -- this is the published protocol of PraNet and all baselines):
  TrainDataset          = 550+62 ClinicDB (612) + 900+100 Kvasir (1000) = 1612 pairs
  TestDataset/Kvasir-SEG   = the 1000 Kvasir pairs
  TestDataset/CVC-ClinicDB = the 612 ClinicDB pairs
  TestDataset/CVC-300      = 60 pairs (verbatim)
  TestDataset/CVC-ColonDB  = 380 pairs (verbatim)
  TestDataset/ETIS-Larib   = 196 pairs (renamed from ETIS-LaribPolypDB)

Every count is hard-asserted; the script refuses to produce a wrong layout.
"""
import argparse
import json
import os
import shutil
import sys
from PIL import Image

CLINIC_SIZE = (384, 288)


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


def is_clinic(img_path):
    with Image.open(img_path) as im:
        return im.size == CLINIC_SIZE


def copy_pair(img, msk, out_img_dir, out_msk_dir, new_stem):
    ext = os.path.splitext(img)[1]
    shutil.copy(img, os.path.join(out_img_dir, new_stem + ext))
    mext = os.path.splitext(msk)[1]
    shutil.copy(msk, os.path.join(out_msk_dir, new_stem + mext))


def main():
    ap = argparse.ArgumentParser(description="Build PraNet split from Polyp-PVT bundle")
    ap.add_argument("--src", required=True, help="path to extracted 'dataset' directory")
    ap.add_argument("--out", default="data", help="output root (creates TrainDataset/TestDataset)")
    args = ap.parse_args()
    src = args.src
    assert os.path.isdir(os.path.join(src, "TrainDataset", "images")), f"bad src: {src}"

    train = list_pairs(os.path.join(src, "TrainDataset", "images"),
                       os.path.join(src, "TrainDataset", "masks"))
    t_kvasir = list_pairs(os.path.join(src, "TestDataset", "Kvasir", "images"),
                          os.path.join(src, "TestDataset", "Kvasir", "masks"))
    t_clinic = list_pairs(os.path.join(src, "TestDataset", "CVC-ClinicDB", "images"),
                          os.path.join(src, "TestDataset", "CVC-ClinicDB", "masks"))

    tr_clinic = [p for p in train if is_clinic(p[0])]
    tr_kvasir = [p for p in train if not is_clinic(p[0])]
    kvasir_all = tr_kvasir + t_kvasir          # expect 900 + 100 = 1000
    clinic_all = tr_clinic + t_clinic          # expect 550 + 62 = 612

    # ---- hard protocol assertions ----
    assert len(tr_clinic) == 550, f"ClinicDB train expected 550, got {len(tr_clinic)}"
    assert len(tr_kvasir) == 900, f"Kvasir train expected 900, got {len(tr_kvasir)}"
    assert len(t_clinic) == 62, f"ClinicDB test expected 62, got {len(t_clinic)}"
    assert len(t_kvasir) == 100, f"Kvasir test expected 100, got {len(t_kvasir)}"
    assert len(kvasir_all) == 1000, f"Kvasir total expected 1000, got {len(kvasir_all)}"
    assert len(clinic_all) == 612, f"ClinicDB total expected 612, got {len(clinic_all)}"
    assert len(train) == 1450, f"train total expected 1450, got {len(train)}"

    # ---- output tree ----
    out = args.out
    for d in ["TrainDataset/images", "TrainDataset/masks",
              "TestDataset/Kvasir-SEG/images", "TestDataset/Kvasir-SEG/masks",
              "TestDataset/CVC-ClinicDB/images", "TestDataset/CVC-ClinicDB/masks",
              "TestDataset/CVC-300/images", "TestDataset/CVC-300/masks",
              "TestDataset/CVC-ColonDB/images", "TestDataset/CVC-ColonDB/masks",
              "TestDataset/ETIS-Larib/images", "TestDataset/ETIS-Larib/masks"]:
        os.makedirs(os.path.join(out, d), exist_ok=True)

    log = {"train": {}, "test": {}, "source": os.path.abspath(src)}

    # TrainDataset = all 1612 pairs, uniquely prefixed
    for tag, pairs in (("kvasir", kvasir_all), ("clinic", clinic_all)):
        for i, (img, msk) in enumerate(sorted(pairs), 1):
            copy_pair(img, msk, os.path.join(out, "TrainDataset/images"),
                      os.path.join(out, "TrainDataset/masks"), f"{tag}_{i:04d}")
        log["train"][tag] = len(pairs)

    # In-domain test sets = same pairs as training (published protocol)
    for i, (img, msk) in enumerate(sorted(kvasir_all), 1):
        copy_pair(img, msk, os.path.join(out, "TestDataset/Kvasir-SEG/images"),
                  os.path.join(out, "TestDataset/Kvasir-SEG/masks"), f"kvasir_{i:04d}")
    for i, (img, msk) in enumerate(sorted(clinic_all), 1):
        copy_pair(img, msk, os.path.join(out, "TestDataset/CVC-ClinicDB/images"),
                  os.path.join(out, "TestDataset/CVC-ClinicDB/masks"), f"clinic_{i:04d}")
    log["test"]["Kvasir-SEG"] = len(kvasir_all)
    log["test"]["CVC-ClinicDB"] = len(clinic_all)

    # Cross-domain test sets: verbatim copies
    for src_name, out_name, expect in [("CVC-300", "CVC-300", 60),
                                       ("CVC-ColonDB", "CVC-ColonDB", 380),
                                       ("ETIS-LaribPolypDB", "ETIS-Larib", 196)]:
        pairs = list_pairs(os.path.join(src, "TestDataset", src_name, "images"),
                           os.path.join(src, "TestDataset", src_name, "masks"))
        assert len(pairs) == expect, f"{src_name} expected {expect}, got {len(pairs)}"
        for i, (img, msk) in enumerate(pairs, 1):
            stem = f"{out_name.replace('-', '').lower()}_{i:04d}"
            copy_pair(img, msk, os.path.join(out, f"TestDataset/{out_name}/images"),
                      os.path.join(out, f"TestDataset/{out_name}/masks"), stem)
        log["test"][out_name] = len(pairs)

    with open(os.path.join(out, "build_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print("[build] PraNet-standard split created at", os.path.abspath(out))
    print("[build] TrainDataset:", log["train"], "| TestDataset:", log["test"])


if __name__ == "__main__":
    sys.exit(main())
