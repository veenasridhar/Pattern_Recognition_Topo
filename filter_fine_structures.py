"""
filter_fine_structures.py
=========================
Keep OpenTO designs that contain fine (thin) structures; reject solid
"blob" designs that have no fine structure.

Why not plain volume fraction: a solid blob and a thin truss can have
similar member fractions (measured 0.48 vs 0.36 on the two samples), so
vf alone doesn't separate them. What separates them is how much of the
material is THIN:

  * thin_fraction = fraction of member pixels within `thin_px` of a void
    edge (distance transform). A blob is nearly all interior -> tiny thin
    fraction; a truss is nearly all edge -> large thin fraction.
  * skel_per_area = skeleton length / member area. Thin branchy structures
    are "all skeleton" -> high; blobs -> very low.

Measured on the two samples (thin_px=4):
    blob : thin_fraction=0.04  skel_per_area=0.0073   -> REJECT
    truss: thin_fraction=0.33  skel_per_area=0.0689   -> KEEP

An image is KEPT if it shows fine structure by EITHER metric, so a design
with a thick region AND thin members (like the truss) is kept; only a pure
blob with no fine structure is rejected. Tune the thresholds against your
own dataset with --report first (dry run, just prints the stats).

Usage:
  # dry run: print metrics + keep/reject decision, copy nothing
  python filter_fine_structures.py -i ./designs --report

  # copy the KEPT (fine-structure) images into ./designs_filtered
  python filter_fine_structures.py -i ./designs -o ./designs_filtered

  # also drop the rejected ones somewhere for inspection
  python filter_fine_structures.py -i ./designs -o ./keep --rejected-dir ./blobs
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

IMG_EXTS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}


def measure_gray(g, thin_px: float):
    """Metrics for one design given a grayscale array (float or uint8).
    White (>=128) is treated as member. Reused by the HF/parquet filter."""
    g = np.asarray(g, np.float32)
    member = g >= 128                         # 2-class: white = member
    area = int(member.sum())
    if area == 0:
        return dict(vf=0.0, thin_fraction=0.0, skel_per_area=0.0,
                    median_thick=0.0, member_px=0)
    dist = distance_transform_edt(member)
    thin_fraction = float((dist[member] <= thin_px).mean())
    sk_mask = skeletonize(member)
    sk = int(sk_mask.sum())
    median_thick = float(2.0 * np.median(dist[sk_mask])) if sk else 0.0
    return dict(vf=float(member.mean()),
                thin_fraction=round(thin_fraction, 4),
                skel_per_area=round(sk / area, 4),
                median_thick=round(median_thick, 2),
                member_px=area)


def measure(path: Path, thin_px: float):
    """Return (metrics dict) for one design file, or None if unreadable."""
    try:
        g = np.array(Image.open(path).convert("L"), np.float32)
    except Exception as e:
        print(f"[warn] cannot read {path.name}: {e}", file=sys.stderr)
        return None
    return measure_gray(g, thin_px)


def keeps(m, thin_frac_min, skel_min, min_member_px):
    """Keep if the design shows fine structure by EITHER metric.
    Empty / near-empty designs are rejected outright."""
    if m["member_px"] < min_member_px:
        return False
    return (m["thin_fraction"] >= thin_frac_min) or \
           (m["skel_per_area"] >= skel_min)


def main():
    ap = argparse.ArgumentParser(
        description="Keep fine-structure designs, reject solid blobs.")
    ap.add_argument("-i", "--input", required=True, help="folder of PNGs")
    ap.add_argument("-o", "--output", help="copy KEPT images here "
                                           "(omit with --report for a dry run)")
    ap.add_argument("--rejected-dir", help="also copy rejected (blob) images here")
    ap.add_argument("--thin-px", type=float, default=4.0,
                    help="a member pixel is 'thin' if within this many px of "
                         "a void edge (default 4)")
    ap.add_argument("--thin-frac-min", type=float, default=0.35,
                    help="keep if thin_fraction >= this (default 0.15; "
                         "samples: blob 0.04, truss 0.33)")
    ap.add_argument("--skel-min", type=float, default=0.05,
                    help="keep if skel_per_area >= this (default 0.02; "
                         "samples: blob 0.007, truss 0.069)")
    ap.add_argument("--min-member-px", type=int, default=50,
                    help="reject designs with fewer member pixels than this")
    ap.add_argument("--move", action="store_true",
                    help="move instead of copy (default copy)")
    ap.add_argument("--report", action="store_true",
                    help="dry run: print metrics + decision, copy nothing")
    a = ap.parse_args()

    in_dir = Path(a.input)
    images = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        print(f"no images in {in_dir}", file=sys.stderr)
        return 1

    out_dir = Path(a.output) if a.output else None
    rej_dir = Path(a.rejected_dir) if a.rejected_dir else None
    if not a.report:
        if out_dir is None:
            print("give -o/--output (or use --report for a dry run)",
                  file=sys.stderr)
            return 1
        out_dir.mkdir(parents=True, exist_ok=True)
        if rej_dir:
            rej_dir.mkdir(parents=True, exist_ok=True)

    xfer = shutil.move if a.move else shutil.copy2
    rows, n_keep, n_rej = [], 0, 0

    for p in images:
        m = measure(p, a.thin_px)
        if m is None:
            continue
        keep = keeps(m, a.thin_frac_min, a.skel_min, a.min_member_px)
        rows.append(dict(image=p.name, keep=int(keep), **m))
        n_keep += keep
        n_rej += (not keep)

        if a.report:
            tag = "KEEP" if keep else "reject"
            print(f"{tag:6s} {p.name:32s} vf={m['vf']:.3f} "
                  f"thin={m['thin_fraction']:.3f} "
                  f"skel/area={m['skel_per_area']:.4f} "
                  f"med_thick={m['median_thick']:.1f}")
        else:
            if keep:
                xfer(str(p), str(out_dir / p.name))
            elif rej_dir:
                xfer(str(p), str(rej_dir / p.name))

    # always write a CSV report (next to output, or input for dry run)
    report_at = (out_dir if out_dir else in_dir) / "filter_report.csv"
    with open(report_at, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "keep", "vf",
                                          "thin_fraction", "skel_per_area",
                                          "median_thick", "member_px"])
        w.writeheader(); w.writerows(rows)

    print(f"\n{len(rows)} images: {n_keep} kept (fine structure), "
          f"{n_rej} rejected (blob/empty)")
    print(f"report -> {report_at}")
    if not a.report:
        print(f"kept -> {out_dir}" + (f"   rejected -> {rej_dir}" if rej_dir else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())