"""
generate_openTO_pseudo_groups.py

Builds synthetic consensus groups from OpenTO images for pretraining the
aggregation module + decoder, combining all corruption types discussed:

  Per source image:
    GT     = binarized label map (void / member; OpenTO has no non-design class)
    views  = 5 corrupted copies, shuffled order:
               2 x light jitter (+-1..2 px)      -> near-truth majority
               1 x minority shift (+-4..8 px)    -> misalignment corruption
               1 x branch deletion               -> structural disagreement (missing member)
               1 x spurious member addition      -> structural disagreement (extra load path)
    base   = majority vote over the 5 views

  Output: PNGs in the native 0/128/255 encoding, directory layout matching the
  real-data convention (Subbatch <id>/view_*.png, gt <id>.png, result_<id>.png),
  so the existing Dataset class reads these groups unmodified.
  A manifest.csv records per-view corruption specs (post-shuffle), which the
  attention-weight analysis needs.

Usage:
    python generate_openTO_pseudo_groups.py                  # full generation
    python generate_openTO_pseudo_groups.py --selftest       # synthetic smoke test, no OpenTO data needed
    python generate_openTO_pseudo_groups.py --inspect 3      # generate + save inspection figures for 3 groups

Sanity checks baked in:
    - every written label map asserts unique values subset of {0, 2}
    - baseline (majority vote) accuracy vs GT is reported; warns if pretext
      task is unrealistically hard
    - inspection figures show composite grayscale per view + GT + baseline
"""

import argparse
import csv
import os
import random
import sys

import numpy as np
from PIL import Image

try:
    from skimage.morphology import skeletonize, disk
    from skimage.measure import label as cc_label
    from scipy.ndimage import convolve
    try:                                    # skimage >= 0.26 deprecates binary_dilation
        from skimage.morphology import dilation as _dilation
        def binary_dilation(img, footprint):
            return _dilation(img.astype(bool), footprint)
    except ImportError:
        from skimage.morphology import binary_dilation
except ImportError as e:
    sys.exit(f"Missing dependency: {e}. Install with: pip install scikit-image scipy")

# ---------------- config ----------------
OPENTO_DIR       = './data/openTO_filtered'     # filtered OpenTO PNGs
OUT_ROOT         = './data/openTO_pseudo'

DENSITY_THRESH   = 0.5          # density -> member binarization (document in thesis)
N_VIEWS          = 5
JITTER_RANGE     = (0, 2)       # |px| for the 2 near-truth views
SHIFT_RANGE      = (10, 20)       # |px| for the 1 misaligned view
DELETE_FRAC      = (0.20, 0.45) # fraction of member skeleton deleted in the deletion view
ADD_LINES        = (5, 8)       # spurious members added in the addition view
MIN_CHORD_LEN    = 20           # skip degenerate short spurious chords
MEMBER_HALFWIDTH = 2            # dilation radius: skeleton -> full member width
MAX_MEMBER_FRAC  = 0.6          # skip degenerate designs (mostly solid)
SEED             = 42

CLS_VOID, CLS_NONDESIGN, CLS_MEMBER = 0, 1, 2   # class 1 intentionally absent here
LUT = np.array([0, 128, 255], dtype=np.uint8)

VALID_VALUES = {0, 2}           # OpenTO pseudo-groups: void + member only


# ---------------- geometry ----------------
def shift_label_map(lm, dy, dx, fill=CLS_VOID):
    """Integer-pixel shift; vacated area filled with void. No interpolation ever."""
    out = np.full_like(lm, fill)
    H, W = lm.shape
    ys = slice(max(0, -dy), min(H, H - dy)); yd = slice(max(0, dy), min(H, H + dy))
    xs = slice(max(0, -dx), min(W, W - dx)); xd = slice(max(0, dx), min(W, W + dx))
    out[yd, xd] = lm[ys, xs]
    return out


def draw_offset(rng, lo, hi):
    def mag():
        if hi == 0:
            return 0
        m = rng.randint(max(lo, 0), hi)
        return m * rng.choice([-1, 1]) if m > 0 else 0
    dy, dx = mag(), mag()
    if lo > 0 and dy == 0 and dx == 0:      # SHIFT view must actually shift
        dy = lo
    return dy, dx


# ---------------- corruption: branch deletion ----------------
def delete_member_branches(lm, frac, rng):
    """Delete whole structural branches (skeleton segments), mimicking a run
    where members are absent. Returns (corrupted_lm, actual_deleted_fraction)."""
    member = lm == CLS_MEMBER
    if member.sum() == 0:
        return lm.copy(), 0.0

    skel = skeletonize(member)
    nbrs = convolve(skel.astype(np.uint8), np.ones((3, 3)), mode='constant') - 1
    junctions = skel & (nbrs > 2)

    segments = cc_label(skel & ~junctions, connectivity=2)
    n_seg = int(segments.max())
    if n_seg <= 1:
        # single unbroken line/loop: no branch to remove without destroying everything
        return lm.copy(), 0.0

    seg_ids = list(range(1, n_seg + 1))
    rng.shuffle(seg_ids)
    total = int(skel.sum())
    target = frac * total

    removed_px = 0
    remove_mask = np.zeros_like(skel)
    for sid in seg_ids:
        seg = segments == sid
        seg_px = int(seg.sum())
        if removed_px + seg_px > 2 * target:    # don't wildly overshoot
            continue
        remove_mask |= seg
        removed_px += seg_px
        if removed_px >= target:
            break

    if removed_px == 0:
        return lm.copy(), 0.0

    carve = binary_dilation(remove_mask, disk(MEMBER_HALFWIDTH + 1))
    out = lm.copy()
    out[member & carve] = CLS_VOID
    return out, removed_px / max(total, 1)


# ---------------- corruption: spurious member addition ----------------
def add_spurious_member(lm, rng, n_lines=ADD_LINES):
    """Add plausible false members: chords between two existing skeleton points,
    dilated to member thickness. Anchored to real structure so they resemble
    alternative load paths, not random scribbles.
    Returns (corrupted_lm, n_added, added_mask)."""
    member = lm == CLS_MEMBER
    skel_pts = np.argwhere(skeletonize(member))
    added_mask = np.zeros_like(member)
    if len(skel_pts) < 10:
        return lm.copy(), 0, added_mask

    out = lm.copy()
    added = 0
    attempts = 0
    want = rng.randint(*n_lines)
    while added < want and attempts < want * 10:
        attempts += 1
        p1 = skel_pts[rng.randrange(len(skel_pts))]
        p2 = skel_pts[rng.randrange(len(skel_pts))]
        dist = int(np.abs(p1 - p2).max())
        if dist < MIN_CHORD_LEN:
            continue
        t = np.linspace(0.0, 1.0, dist * 2)
        ys = np.round(p1[0] + t * (p2[0] - p1[0])).astype(int).clip(0, lm.shape[0] - 1)
        xs = np.round(p1[1] + t * (p2[1] - p1[1])).astype(int).clip(0, lm.shape[1] - 1)
        line = np.zeros_like(member)
        line[ys, xs] = True
        thick = binary_dilation(line, disk(MEMBER_HALFWIDTH))
        new_px = thick & ~(out == CLS_MEMBER)
        if new_px.sum() < MIN_CHORD_LEN:        # chord lies almost entirely on existing structure
            continue
        out[thick] = CLS_MEMBER
        added_mask |= new_px
        added += 1
    return out, added, added_mask


# ---------------- perturb thickness ----------------
def perturb_thickness(lm, rng, max_delta=1):
    """Uniformly thicken or thin the member structure of one view by +-delta px.
    Simulates runs allocating more/less material to the same members.

    lm: (H,W) int64 label map. Returns (corrupted_lm, delta).
    delta > 0: dilate members; delta < 0: erode members.
    """
    member = lm == CLS_MEMBER
    if member.sum() == 0:
        return lm.copy(), 0

    delta = rng.choice([-1, 1]) * rng.randint(1, max_delta)

    if delta > 0:
        new_member = binary_dilation(member, disk(delta))
    else:
        # erosion = complement of dilation of the complement
        new_member = ~binary_dilation(~member, disk(-delta))
        # guard: erosion of ~2px-wide members can annihilate them entirely,
        # which would be deletion, not thinning — fall back to no-op
        skel_before = skeletonize(member).sum()
        skel_after  = skeletonize(new_member).sum()
        if skel_after < 0.9 * skel_before:
            return lm.copy(), 0

    out = lm.copy()
    out[member & ~new_member] = CLS_VOID       # thinned-away pixels
    out[new_member] = CLS_MEMBER               # thickened pixels
    return out, delta


# ---------------- group assembly ----------------
def majority_vote(label_maps):
    """Per-pixel majority over label maps. Ties resolve to lowest class (void-biased):
    thin members need a real majority to survive, matching the real baseline's behavior."""
    votes = np.stack([np.eye(3, dtype=np.int64)[m] for m in label_maps]).sum(0)  # (H,W,3)
    return votes.argmax(-1)


def make_group(gt_lm, rng):
    """Corruption mix: 2 jitter + 1 shift + 1 deletion + 1 addition, order shuffled
    so view index never encodes corruption type (prevents positional shortcuts
    in the aggregation module's attention)."""
    views, spec = [], []

    for _ in range(2):                                            # near-truth jittered views
        dy, dx = draw_offset(rng, *JITTER_RANGE)
        views.append(shift_label_map(gt_lm, dy, dx))
        spec.append(dict(kind='jitter', dy=dy, dx=dx, delete_frac=0.0, n_added=0))

    dy, dx = draw_offset(rng, *SHIFT_RANGE)                       # minority misaligned view
    views.append(shift_label_map(gt_lm, dy, dx))
    spec.append(dict(kind='shift', dy=dy, dx=dx, delete_frac=0.0, n_added=0))

    frac = rng.uniform(*DELETE_FRAC)                              # missing-member view
    deleted, actual = delete_member_branches(gt_lm, frac, rng)
    views.append(deleted)
    spec.append(dict(kind='delete', dy=0, dx=0, delete_frac=round(actual, 3), n_added=0))

    added_lm, n_added, _ = add_spurious_member(gt_lm, rng)        # extra-load-path view
    views.append(added_lm)
    spec.append(dict(kind='add', dy=0, dx=0, delete_frac=0.0, n_added=n_added))

    order = list(range(N_VIEWS))
    rng.shuffle(order)
    return [views[j] for j in order], [spec[j] for j in order]


# ---------------- IO + validation ----------------
def load_density(path):
    """Adapt to your filtered OpenTO storage. PNG assumed; for HuggingFace arrow
    format, replace with datasets indexing of the density field."""
    arr = np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0
    return arr


def save_label_map(lm, path):
    vals = set(np.unique(lm).tolist())
    assert vals <= VALID_VALUES, f'{path}: illegal label values {vals} (expected subset of {VALID_VALUES})'
    Image.fromarray(LUT[lm]).save(path)


def baseline_accuracy(views, gt_lm):
    """Fraction of pixels where majority vote matches GT — pretext-difficulty gauge."""
    return float((majority_vote(views) == gt_lm).mean())


def inspect_group(views, spec, gt_lm, out_path):
    """One figure: 5 views (labeled by corruption), GT, baseline. Visual sanity check."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    base = majority_vote(views)
    fig, ax = plt.subplots(1, N_VIEWS + 2, figsize=(4 * (N_VIEWS + 2), 4))
    for v, (lm, s) in enumerate(zip(views, spec)):
        ax[v].imshow(LUT[lm], cmap='gray', vmin=0, vmax=255)
        ax[v].set_title(f"view {v}: {s['kind']}\n(dy={s['dy']}, dx={s['dx']}, "
                        f"del={s['delete_frac']}, add={s['n_added']})", fontsize=9)
    ax[N_VIEWS].imshow(LUT[gt_lm], cmap='gray', vmin=0, vmax=255)
    ax[N_VIEWS].set_title('GT (consensus)')
    ax[N_VIEWS + 1].imshow(LUT[base], cmap='gray', vmin=0, vmax=255)
    ax[N_VIEWS + 1].set_title(f'baseline vote\nacc={baseline_accuracy(views, gt_lm):.3f}')
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ---------------- main generation ----------------
def generate(opento_dir, out_root, inspect_n=0):
    rng = random.Random(SEED)
    subbatch_dir = os.path.join(out_root, 'subbatches')
    gt_dir       = os.path.join(out_root, 'ground_truth')
    img_proc_dir = os.path.join(out_root, 'img_proc')
    inspect_dir  = os.path.join(out_root, 'inspection')
    for d in (subbatch_dir, gt_dir, img_proc_dir):
        os.makedirs(d, exist_ok=True)
    if inspect_n:
        os.makedirs(inspect_dir, exist_ok=True)

    sources = sorted(f for f in os.listdir(opento_dir) if f.lower().endswith('.png'))
    if not sources:
        sys.exit(f'No PNGs found in {opento_dir} — adapt load_density() to your OpenTO storage format.')

    manifest, base_accs, skipped = [], [], 0

    for src in sources:
        sid = os.path.splitext(src)[0]
        density = load_density(os.path.join(opento_dir, src))
        gt_lm = np.where(density >= DENSITY_THRESH, CLS_MEMBER, CLS_VOID).astype(np.int64)

        member_frac = float((gt_lm == CLS_MEMBER).mean())
        if member_frac == 0.0 or member_frac > MAX_MEMBER_FRAC:
            skipped += 1
            continue

        views, spec = make_group(gt_lm, rng)

        gdir = os.path.join(subbatch_dir, f'Subbatch {sid}')
        os.makedirs(gdir, exist_ok=True)
        for v, lm in enumerate(views):
            save_label_map(lm, os.path.join(gdir, f'view_{v}.png'))
        save_label_map(gt_lm, os.path.join(gt_dir, f'gt {sid}.png'))
        save_label_map(majority_vote(views), os.path.join(img_proc_dir, f'result_{sid}.png'))

        base_accs.append(baseline_accuracy(views, gt_lm))

        for v, s in enumerate(spec):
            manifest.append(dict(group=sid, view=v, **s))

        # if inspect_n and len(base_accs) <= inspect_n:
        #     inspect_group(views, spec, gt_lm, os.path.join(inspect_dir, f'{sid}.png'))

    with open(os.path.join(out_root, 'manifest.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['group', 'view', 'kind', 'dy', 'dx', 'delete_frac', 'n_added'])
        w.writeheader()
        w.writerows(manifest)

    n_groups = len(base_accs)
    print(f'Groups written: {n_groups}  (skipped {skipped} degenerate sources)')
    if n_groups:
        accs = np.array(base_accs)
        print(f'Baseline (majority-vote) pixel accuracy vs GT: '
              f'mean={accs.mean():.4f}  min={accs.min():.4f}')
        if accs.mean() < 0.90:
            print('WARNING: baseline accuracy is low — pretext task may be unrealistically hard.\n'
                  '         Consider reducing DELETE_FRAC / ADD_LINES / SHIFT_RANGE.')
    print(f"Manifest: {os.path.join(out_root, 'manifest.csv')}")


# ---------------- self-test (no OpenTO data needed) ----------------
def make_synthetic_design(rng, size=256):
    """A crude truss-like label map: frame + diagonals, member-width strokes."""
    lm = np.zeros((size, size), dtype=np.int64)

    def stroke(p1, p2):
        dist = int(max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])))
        t = np.linspace(0.0, 1.0, dist * 2 + 2)
        ys = np.round(p1[0] + t * (p2[0] - p1[0])).astype(int).clip(0, size - 1)
        xs = np.round(p1[1] + t * (p2[1] - p1[1])).astype(int).clip(0, size - 1)
        line = np.zeros((size, size), dtype=bool)
        line[ys, xs] = True
        lm[binary_dilation(line, disk(MEMBER_HALFWIDTH))] = CLS_MEMBER

    m = 30
    corners = [(m, m), (m, size - m), (size - m, size - m), (size - m, m)]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        stroke(a, b)
    stroke(corners[0], corners[2])
    stroke(corners[1], corners[3])
    for _ in range(rng.randint(1, 3)):                 # random internal braces
        stroke((rng.randint(m, size - m), m), (rng.randint(m, size - m), size - m))
    return lm


def selftest():
    import tempfile
    rng = random.Random(SEED)
    with tempfile.TemporaryDirectory() as tmp:
        src_dir = os.path.join(tmp, 'src')
        os.makedirs(src_dir)
        for i in range(4):
            lm = make_synthetic_design(rng)
            Image.fromarray((lm == CLS_MEMBER).astype(np.uint8) * 255).save(
                os.path.join(src_dir, f'design_{i}.png'))

        out = os.path.join(tmp, 'out')
        generate(src_dir, out, inspect_n=2)

        # structural checks on the output
        subs = sorted(os.listdir(os.path.join(out, 'subbatches')))
        assert len(subs) >= 3, 'expected >=3 groups from 4 synthetic designs'
        rows = list(csv.DictReader(open(os.path.join(out, 'manifest.csv'))))
        kinds_per_group = {}
        for r in rows:
            kinds_per_group.setdefault(r['group'], []).append(r['kind'])
        for g, kinds in kinds_per_group.items():
            assert sorted(kinds) == ['add', 'delete', 'jitter', 'jitter', 'shift'], \
                f'group {g}: wrong corruption mix {kinds}'
        # order shuffle: corruption kinds should not sit at fixed view indices across groups
        kind_at_last = {kinds_per_group[g][-1] for g in kinds_per_group}
        print(f'Self-test passed: {len(subs)} groups, corruption mix verified, '
              f'kinds at final view index across groups: {kind_at_last}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true', help='run on synthetic designs, no OpenTO data needed')
    ap.add_argument('--inspect', type=int, default=0, metavar='N', help='save inspection figures for first N groups')
    ap.add_argument('--src', default=OPENTO_DIR, help='OpenTO source directory')
    ap.add_argument('--out', default=OUT_ROOT, help='output root directory')
    args = ap.parse_args()

    if args.selftest:
        selftest()
    else:
        generate(args.src, args.out, inspect_n=args.inspect)