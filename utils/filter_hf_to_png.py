"""
filter_hf_to_png.py
===================
Apply the fine-structure filter to an OpenTO HuggingFace/parquet dataset
(each sample carries a `topology` PIL image) and save the KEPT designs as
PNGs on disk, ready for build_consensus_dataset.py.

Same criterion as filter_fine_structures.py: a design is kept if it shows
fine (thin) structure by EITHER metric --
    thin_fraction >= --thin-frac-min   (member within thin_px of a void edge)
    skel_per_area >= --skel-min        (skeleton length / member area)
so a truss (thick hub + thin spokes) is kept and a solid blob is rejected.
The metric itself is imported from filter_fine_structures.py so the two
paths stay identical.

Mirrors the usual snippet:
    ds = load_dataset('parquet', data_files='data/openTO/data/pretraining*.parquet',
                      split='train')
    ds = ds.filter(fine_structure_filter)          # <- this module
    # ... then save each kept topology as PNG

Usage:
    python filter_hf_to_png.py \
        --data-files 'data/openTO/data/pretraining*.parquet' \
        --out-dir ./data/openTO_filtered

    # dry run: just count keeps/rejects, write no PNGs
    python filter_hf_to_png.py --data-files '...*.parquet' --report
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# shared metric so file-based and parquet-based filtering agree exactly
from filter_fine_structures import measure_gray, keeps

# defaults (kept in sync with filter_fine_structures.py CLI defaults)
THIN_PX = 4.0
THIN_FRAC_MIN = 0.15
SKEL_MIN = 0.02
MIN_MEMBER_PX = 50
IMAGE_FIELD = "topology"          # OpenTO field holding the PIL image
MIN_SIZE = 100                    # also enforce the original size gate


def _to_gray(sample, field):
    """Grayscale numpy array for a sample's image field (PIL or array)."""
    img = sample[field]
    if hasattr(img, "convert"):                    # PIL.Image
        return np.array(img.convert("L"), np.float32)
    arr = np.asarray(img, np.float32)
    if arr.ndim == 3:                              # HxWxC -> luminance
        arr = arr[..., :3].mean(-1)
    return arr


def make_filter(field=IMAGE_FIELD, thin_px=THIN_PX,
                thin_frac_min=THIN_FRAC_MIN, skel_min=SKEL_MIN,
                min_member_px=MIN_MEMBER_PX, min_size=MIN_SIZE):
    """Build a predicate usable directly as `dataset.filter(...)`."""
    def fine_structure_filter(sample):
        img = sample[field]
        # size gate (same spirit as the original dataset_filter)
        if hasattr(img, "size"):
            w, h = img.size
            if not (min_size <= w and min_size <= h):
                return False
        g = _to_gray(sample, field)
        m = measure_gray(g, thin_px)
        return keeps(m, thin_frac_min, skel_min, min_member_px)
    return fine_structure_filter


def main():
    ap = argparse.ArgumentParser(
        description="Filter an OpenTO parquet dataset for fine structures "
                    "and save kept designs as PNGs.")
    ap.add_argument("--data-files", required=True,
                    help="parquet glob, e.g. 'data/openTO/data/pretraining*.parquet'")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-dir", default="./data/openTO_filtered",
                    help="where kept PNGs are written")
    ap.add_argument("--field", default=IMAGE_FIELD,
                    help=f"sample key holding the image (default '{IMAGE_FIELD}')")
    ap.add_argument("--thin-px", type=float, default=THIN_PX)
    ap.add_argument("--thin-frac-min", type=float, default=THIN_FRAC_MIN)
    ap.add_argument("--skel-min", type=float, default=SKEL_MIN)
    ap.add_argument("--min-member-px", type=int, default=MIN_MEMBER_PX)
    ap.add_argument("--min-size", type=int, default=MIN_SIZE)
    ap.add_argument("--num-proc", type=int, default=1,
                    help="parallel workers for dataset.filter (default 1)")
    ap.add_argument("--report", action="store_true",
                    help="dry run: count keeps/rejects, write no PNGs")
    ap.add_argument("--save-parquet", default=None,
                    help="also write the filtered dataset to this parquet path")
    a = ap.parse_args()

    from datasets import load_dataset            # imported lazily

    ds = load_dataset("parquet", data_files=a.data_files, split=a.split)
    print(f"loaded {len(ds)} samples from {a.data_files}")

    predicate = make_filter(a.field, a.thin_px, a.thin_frac_min,
                            a.skel_min, a.min_member_px, a.min_size)
    filtered = ds.filter(predicate, num_proc=a.num_proc)
    print(f"kept {len(filtered)} / {len(ds)} "
          f"({len(filtered) / max(len(ds), 1):.1%}) with fine structure")

    if a.save_parquet:
        filtered.to_parquet(a.save_parquet)
        print(f"filtered dataset -> {a.save_parquet}")

    if a.report:
        return 0

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, sample in enumerate(filtered):
        img = sample[a.field]
        pil = img if hasattr(img, "save") else None
        if pil is None:                            # array -> PIL
            from PIL import Image
            pil = Image.fromarray(np.asarray(_to_gray(sample, a.field),
                                             np.uint8), mode="L")
        # keep an id from the sample if present, else the running index
        sid = sample.get("id") or sample.get("name") or f"sample_{i:05d}"
        sid = str(sid).replace("/", "_")
        pil.convert("L").save(out / f"{sid}.png")
        g = _to_gray(sample, a.field)
        rows.append(dict(png=f"{sid}.png", **measure_gray(g, a.thin_px)))

    with open(out / "kept_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["png", "vf", "thin_fraction",
                                          "skel_per_area", "median_thick",
                                          "member_px"])
        w.writeheader(); w.writerows(rows)

    print(f"wrote {len(rows)} PNGs -> {out}")
    print(f"report -> {out / 'kept_report.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())