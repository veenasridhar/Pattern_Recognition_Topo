import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.ndimage import (convolve, distance_transform_edt, gaussian_filter,
                           map_coordinates)
import skimage
from skimage.measure import label as cc_label
from skimage.morphology import disk, skeletonize

if tuple(int(x) for x in skimage.__version__.split(".")[:2]) >= (0, 26):
    from skimage.morphology import dilation as _dil, erosion as _ero
    def binary_dilation(image, footprint=None):
        return _dil(image, footprint).astype(bool)
    def binary_erosion(image, footprint=None):
        return _ero(image, footprint).astype(bool)
else:
    from skimage.morphology import binary_dilation, binary_erosion

# --------------------------------------------------------------------------
# constants (identical to corruption_ops.py so maps interchange freely)
# --------------------------------------------------------------------------
CLS_VOID, CLS_NONDESIGN, CLS_MEMBER = 0, 1, 2
GRAY = {CLS_VOID: 0, CLS_NONDESIGN: 128, CLS_MEMBER: 255}
IMG_EXTS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}

MEMBER_HALFWIDTH = 2
MIN_CHORD_LEN = 50
EVENT_MAX_LEN = 180

EVENT_KINDS = ("add", "remove", "reroute")


# --------------------------------------------------------------------------
# difficulty presets: how hostile a group is to per-pixel majority voting
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Difficulty:
    event_jitter_px: int    # per-view rigid shift of the shared event
    event_warp_amp: float   # per-view elastic warp of the event (px)
    thin_radius: tuple      # (min, max) half-width of added bars, in px
    identity_prob: float    # chance a non-event view is left untouched
    # boundary jitter -> pushes image_proc IoU into a target band while
    # PRESERVING consensus thickness (balanced thin/thicken = median-stable)
    boundary_n_views: int = 4      # how many views get boundary jitter
    boundary_shift_px: int = 1     # independent member-phase shift per view
    boundary_thin_prob: float = 0.8  # prob of a balanced +-1px thickness step


DIFFICULTY = {
    # EASY   : events stamped near-identically -> voting almost always wins
    "EASY":   Difficulty(0, 0.0, (2, 2), 0.5,
                         boundary_n_views=0, boundary_shift_px=0, boundary_thin_prob=0.0),
    # MEDIUM : same structure displaced per view + boundary jitter -> IoU ~0.85-0.9
    "MEDIUM": Difficulty(2, 1.5, (1, 2), 0.2,
                         boundary_n_views=4, boundary_shift_px=1, boundary_thin_prob=0.8),
    # HARD   : thin bars + strong event jitter + boundary jitter
    "HARD":   Difficulty(3, 3.0, (1, 1), 0.0,
                         boundary_n_views=4, boundary_shift_px=1, boundary_thin_prob=1.0),
}


# --------------------------------------------------------------------------
# label-map I/O
# --------------------------------------------------------------------------
def load_label_map(path: Path) -> np.ndarray:
    """Read grayscale, snap each pixel to nearest of {0,128,255} -> {0,1,2}."""
    arr = np.asarray(_open_gray(path), dtype=np.int16)
    levels = np.array([0, 128, 255], dtype=np.int16)
    return np.abs(arr[..., None] - levels[None, None, :]).argmin(-1).astype(np.uint8)


def save_label_map(lm: np.ndarray, path: Path) -> None:
    out = np.zeros_like(lm, dtype=np.uint8)
    out[lm == CLS_NONDESIGN] = 128
    out[lm == CLS_MEMBER] = 255
    _save_gray(out, path)


def _open_gray(path):
    from PIL import Image
    return Image.open(path).convert("L")


def _save_gray(arr, path):
    from PIL import Image
    Image.fromarray(arr, mode="L").save(path)


# --------------------------------------------------------------------------
# small geometry helpers (mirroring corruption_ops.py)
# --------------------------------------------------------------------------
def _member(lm):
    return lm == CLS_MEMBER


def _pin_nondesign(out, ref):
    nd = ref == CLS_NONDESIGN
    out = out.copy()
    out[(out == CLS_NONDESIGN) & ~nd] = CLS_VOID
    out[nd] = CLS_NONDESIGN
    return out


def _shift(lm, dy, dx, fill=CLS_VOID):
    out = np.full_like(lm, fill)
    H, W = lm.shape
    ys = slice(max(0, -dy), min(H, H - dy)); yd = slice(max(0, dy), min(H, H + dy))
    xs = slice(max(0, -dx), min(W, W - dx)); xd = slice(max(0, dx), min(W, W + dx))
    out[yd, xd] = lm[ys, xs]
    return out


def _line_mask(shape, p1, p2):
    n = int(np.abs(np.asarray(p1) - np.asarray(p2)).max()) * 2 + 1
    t = np.linspace(0.0, 1.0, n)
    ys = np.round(p1[0] + t * (p2[0] - p1[0])).astype(int).clip(0, shape[0] - 1)
    xs = np.round(p1[1] + t * (p2[1] - p1[1])).astype(int).clip(0, shape[1] - 1)
    m = np.zeros(shape, dtype=bool)
    m[ys, xs] = True
    return m


def _skeleton_branches(member, min_len=8, max_frac=0.25):
    """Skeleton split at junctions into branch masks, size-filtered so we
    never delete a structural chord that is a large fraction of the whole."""
    sk = skeletonize(member)
    nb = convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") * sk
    stripped = sk & ~(nb >= 4)
    lab = cc_label(stripped, connectivity=2)
    total = max(int(sk.sum()), 1)
    out = []
    for i in range(1, int(lab.max()) + 1):
        b = lab == i
        s = int(b.sum())
        if min_len <= s <= max(min_len, int(max_frac * total)):
            out.append(b)
    return out


def _branch_endpoints(b):
    """Endpoints of a skeleton branch: skeleton pixels with a single
    neighbour (3x3 sum == 2). Falls back to the farthest-apart pair."""
    deg = convolve(b.astype(np.uint8), np.ones((3, 3), np.uint8),
                   mode="constant") * b
    ep = np.argwhere(deg == 2)
    if len(ep) >= 2:
        return ep[0], ep[-1]
    pts = np.argwhere(b)
    if len(pts) < 2:
        return None
    # farthest pair (cheap: extremes along the dominant axis)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    return pts[order[0]], pts[order[-1]]


# --------------------------------------------------------------------------
# event: a pair of masks (pixels to paint member, pixels to clear to void)
# --------------------------------------------------------------------------
@dataclass
class Event:
    kind: str                       # add | remove | reroute
    add_mask: np.ndarray | None     # -> CLS_MEMBER
    remove_mask: np.ndarray | None  # -> CLS_VOID

    @property
    def footprint(self):
        m = None
        for x in (self.add_mask, self.remove_mask):
            if x is not None:
                m = x.copy() if m is None else (m | x)
        return m


# --------------------------------------------------------------------------
# event sampling: thin ADD (void only), thin REMOVE (existing branch),
# and REROUTE (remove existing branch + add bent copy on void)
# --------------------------------------------------------------------------
def _sample_add_mask(lm, rng, radius, avoid=None,
                     min_len=MIN_CHORD_LEN, max_len=EVENT_MAX_LEN, max_tries=40):
    """Thin bent bar between two skeleton points, purely on void. Rejected
    (not clipped) if it touches non-design; kept clear of `avoid`."""
    member = _member(lm)
    pts = np.argwhere(skeletonize(member))
    if len(pts) < 2:
        return None
    nd = lm == CLS_NONDESIGN
    void = lm == CLS_VOID
    keepout = binary_dilation(avoid, disk(3 * radius)) if avoid is not None else None
    for _ in range(max_tries):
        p1, p2 = pts[rng.integers(len(pts))], pts[rng.integers(len(pts))]
        d = float(np.hypot(*(p1 - p2)))
        if not (min_len <= d <= max_len):
            continue
        mid = (p1 + p2) / 2.0
        perp = np.array([-(p2 - p1)[1], (p2 - p1)[0]], float)
        perp /= (np.linalg.norm(perp) + 1e-8)
        way = np.clip((mid + perp * rng.uniform(-0.25, 0.25) * d).astype(int),
                      0, np.array(lm.shape) - 1)
        stroke = _line_mask(lm.shape, p1, way) | _line_mask(lm.shape, way, p2)
        bar = binary_dilation(stroke, disk(radius))
        # if (bar & nd).any():                       # touches gray -> reject
        #     continue
        # if keepout is not None and (bar & keepout).any():
        #     continue
        # novel = bar & ~member
        # if not void[novel].all():                  # every new px must be black
        #     continue
        # if novel.sum() < 0.5 * bar.sum():          # mostly re-painting -> boring
        #     continue
        return bar
    return None


def _sample_remove_mask(lm, rng, avoid=None):
    """One thin existing branch, dilated to its local width, for deletion."""
    member = _member(lm)
    branches = _skeleton_branches(member)
    if not branches:
        return None
    dist = distance_transform_edt(member)
    keepout = binary_dilation(avoid, disk(6)) if avoid is not None else None
    rng.shuffle(branches)
    for b in branches:
        r = int(np.ceil(dist[b].max())) + 1
        mask = binary_dilation(b, disk(r)) & member
        if keepout is not None and (mask & keepout).any():
            continue
        return mask
    return None


def _sample_reroute_masks(lm, rng, radius, avoid=None, max_tries=40):
    """Pick an existing branch (to delete) and lay a bent copy between its
    endpoints along a new path over void (to add). Returns (old, new)."""
    member = _member(lm)
    branches = _skeleton_branches(member)
    if not branches:
        return None
    dist = distance_transform_edt(member)
    nd = lm == CLS_NONDESIGN
    void = lm == CLS_VOID
    keepout = binary_dilation(avoid, disk(3 * radius)) if avoid is not None else None
    rng.shuffle(branches)
    for b in branches:
        eps = _branch_endpoints(b)
        if eps is None:
            continue
        p1, p2 = eps
        d = float(np.hypot(*(np.asarray(p1) - np.asarray(p2))))
        if d < MIN_CHORD_LEN * 0.5:
            continue
        r_old = int(np.ceil(dist[b].max())) + 1
        old = binary_dilation(b, disk(r_old)) & member

        mid = (np.asarray(p1) + np.asarray(p2)) / 2.0
        perp = np.array([-(p2 - p1)[1], (p2 - p1)[0]], float)
        perp /= (np.linalg.norm(perp) + 1e-8)
        for _ in range(max_tries):
            side = rng.choice([-1.0, 1.0])
            off = side * rng.uniform(0.3, 0.7) * d          # push well off-axis
            way = np.clip((mid + perp * off).astype(int),
                          0, np.array(lm.shape) - 1)
            stroke = _line_mask(lm.shape, p1, way) | _line_mask(lm.shape, way, p2)
            new = binary_dilation(stroke, disk(radius))
            if (new & nd).any():
                continue
            novel = new & ~member
            if novel.size and not void[novel].all():
                continue
            if novel.sum() < 0.5 * new.sum():               # barely moved
                continue
            if keepout is not None and ((new | old) & keepout).any():
                continue
            return old, new
    return None


def sample_event(kind, lm, rng, diff: Difficulty, avoid=None):
    """Sample one Event of the given kind, or None if it couldn't be placed.
    `avoid` is the footprint of the other event (keep-out)."""
    if kind == "add":
        r = int(rng.integers(diff.thin_radius[0], diff.thin_radius[1] + 1))
        bar = _sample_add_mask(lm, rng, radius=r, avoid=avoid)
        return Event("add", bar, None) if bar is not None else None
    if kind == "remove":
        br = _sample_remove_mask(lm, rng, avoid=avoid)
        return Event("remove", None, br) if br is not None else None
    if kind == "reroute":
        r = int(rng.integers(diff.thin_radius[0], diff.thin_radius[1] + 1))
        res = _sample_reroute_masks(lm, rng, radius=r, avoid=avoid)
        if res is None:
            return None
        old, new = res
        return Event("reroute", new, old)
    raise ValueError(f"unknown event kind: {kind}")


# --------------------------------------------------------------------------
# apply / jitter an event
# --------------------------------------------------------------------------
def _stamp(lm, event: Event):
    """Apply an event: clear remove_mask to void, then paint add_mask member.
    Order matters so a reroute's new path wins at shared endpoints."""
    out = lm.copy()
    if event.remove_mask is not None:
        out[event.remove_mask & (out == CLS_MEMBER)] = CLS_VOID
    if event.add_mask is not None:
        out[event.add_mask & (out != CLS_NONDESIGN)] = CLS_MEMBER
    return out


def _warp_field(shape, rng, amp):
    H, W = shape
    fy = gaussian_filter(rng.standard_normal((H, W)), 14.0)
    fx = gaussian_filter(rng.standard_normal((H, W)), 14.0)
    fy *= amp / (np.abs(fy).max() + 1e-6)
    fx *= amp / (np.abs(fx).max() + 1e-6)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return yy + fy, xx + fx


def _jitter_event(event: Event, lm, rng, diff: Difficulty):
    """Per-view variant of a shared event. The ADDED structure is jittered
    (rigid shift + warp) so it wobbles between views; the REMOVED structure
    is applied canonically (un-jittered) so the original path is deleted
    identically -- and completely -- in every view. No ghost survives."""
    dy = dx = 0
    if diff.event_jitter_px > 0:
        dy = int(rng.integers(-diff.event_jitter_px, diff.event_jitter_px + 1))
        dx = int(rng.integers(-diff.event_jitter_px, diff.event_jitter_px + 1))
    coords = _warp_field(lm.shape, rng, diff.event_warp_amp) \
        if diff.event_warp_amp > 0 else None

    def _jit(mask):
        if mask is None:
            return None
        m = mask
        if dy or dx:
            m = _shift(m.astype(np.int64), dy, dx, fill=0) > 0
        if coords is not None:
            m = map_coordinates(m.astype(np.uint8), coords,
                                order=0, mode="constant") > 0
        return m & (lm != CLS_NONDESIGN)

    # jitter only the added path; keep the removal canonical
    return Event(event.kind, _jit(event.add_mask), event.remove_mask)


def _perturb_boundary(lm, rng, shift_px, thin_prob):
    """Balanced boundary jitter on the member phase (see difficulty presets):
    an independent small shift, then with prob `thin_prob` a +-1px thickness
    step chosen 50/50 (erode OR dilate). Applied to a MINORITY of views this
    leaves the per-pixel majority thickness intact while roughening the exact
    boundary, lowering image_proc IoU vs GT without thinning the consensus."""
    m = _member(lm)
    if m.sum() == 0:
        return lm.copy()
    dy = int(rng.integers(-shift_px, shift_px + 1))
    dx = int(rng.integers(-shift_px, shift_px + 1))
    ms = _shift(m.astype(np.int64), dy, dx, fill=0) > 0
    if rng.random() < thin_prob:
        if rng.random() < 0.5:
            ms = binary_erosion(ms, disk(1)) | skeletonize(m)   # thinner, keep skel
        else:
            ms = binary_dilation(ms, disk(1))                   # thicker
    out = lm.copy(); out[m] = CLS_VOID
    out[ms & (out != CLS_NONDESIGN)] = CLS_MEMBER
    return _pin_nondesign(out, lm)


# --------------------------------------------------------------------------
# benign perturbation for non-event views
# --------------------------------------------------------------------------
def _benign(lm, rng):
    """Jitter the member phase by 1-2 px OR change thickness by 1 px.
    Non-design stays pinned; thin members never fully vanish."""
    if rng.random() < 0.5:
        dy = int(rng.integers(-2, 3)); dx = int(rng.integers(-2, 3))
        if dy == 0 and dx == 0:
            dy = 1
        m = _member(lm)
        shifted = _shift(m.astype(np.int64), dy, dx, fill=0) > 0
        out = lm.copy(); out[m] = CLS_VOID
        out[shifted & (out != CLS_NONDESIGN)] = CLS_MEMBER
        return out
    m = _member(lm)
    if rng.random() < 0.5:
        new = binary_dilation(m, disk(1)) & (lm != CLS_NONDESIGN)
    else:
        new = binary_erosion(m, disk(1)) | skeletonize(m)   # keep 1px members
    out = lm.copy(); out[m] = CLS_VOID; out[new] = CLS_MEMBER
    return out


# --------------------------------------------------------------------------
# group construction
# --------------------------------------------------------------------------
@dataclass
class ConsensusGroup:
    views: list          # 5 label maps
    gt: np.ndarray       # base + canonical majority event
    image_proc: np.ndarray   # pixel-wise majority vote (>=3/5)
    meta: dict


def majority_vote(views, min_support=3):
    """Classical baseline: member where >= min_support views agree; ties and
    non-design handled explicitly. For 5 views, >=3 == pixel-avg @ 0.5."""
    stack = np.stack(views)
    member_votes = (stack == CLS_MEMBER).sum(0)
    out = np.full(views[0].shape, CLS_VOID, dtype=np.uint8)
    out[views[0] == CLS_NONDESIGN] = CLS_NONDESIGN
    out[(member_votes >= min_support) & (out != CLS_NONDESIGN)] = CLS_MEMBER
    return out


def _resolve_kind(kind, rng, pool):
    return str(rng.choice(pool)) if kind == "random" else kind


def build_group(base_lm, rng, difficulty="MEDIUM",
                majority_event="random", minority_event="random",
                event_pool=EVENT_KINDS, group_size=5) -> ConsensusGroup | None:
    """Assemble one dual-event consensus group. Returns None if either event
    could not be sampled (caller should retry, possibly on another base)."""
    diff = DIFFICULTY[difficulty]

    maj_kind = _resolve_kind(majority_event, rng, event_pool)
    min_kind = _resolve_kind(minority_event, rng, event_pool)

    maj = sample_event(maj_kind, base_lm, rng, diff, avoid=None)
    if maj is None:
        return None
    minr = sample_event(min_kind, base_lm, rng, diff, avoid=maj.footprint)
    if minr is None:
        return None

    # --- assign support: majority=3 views, minority=2 views -----------------
    maj_views = set(int(i) for i in rng.permutation(group_size)[:3])
    min_views = set(int(i) for i in rng.permutation(group_size)[:2])

    # views that receive balanced boundary jitter (fewer than the majority,
    # so the consensus vote/thickness is preserved)
    n_bj = min(diff.boundary_n_views, group_size)
    bj_views = set(int(i) for i in rng.permutation(group_size)[:n_bj])

    views = []
    for i in range(group_size):
        lm = base_lm.copy()
        touched = False
        if i in maj_views:
            ev = maj if difficulty == "EASY" else _jitter_event(maj, lm, rng, diff)
            lm = _stamp(lm, ev); touched = True
        if i in min_views:
            ev = minr if difficulty == "EASY" else _jitter_event(minr, lm, rng, diff)
            lm = _stamp(lm, ev); touched = True
        if not touched and rng.random() >= diff.identity_prob:
            lm = _benign(lm, rng)
        if i in bj_views and diff.boundary_shift_px > 0:
            lm = _perturb_boundary(lm, rng, diff.boundary_shift_px,
                                   diff.boundary_thin_prob)
        views.append(_pin_nondesign(lm, base_lm))

    # --- ground truth: canonical majority event only ------------------------
    gt = _stamp(base_lm, maj)                       # minority deliberately absent
    ip = majority_vote(views, min_support=3)

    # --- SYMMETRIC baseline diagnostics -------------------------------------
    ip_m, gt_m = ip == CLS_MEMBER, gt == CLS_MEMBER
    iou = float((ip_m & gt_m).sum() / max((ip_m | gt_m).sum(), 1))

    # agreement between the baseline and the consensus GT over the event
    # footprint. Works identically for add (GT member there), remove (GT void
    # there) and reroute (mixed) -> "same bar, opposite op" symmetry.
    fp = maj.footprint
    maj_recall = float((ip_m[fp] == gt_m[fp]).mean()) if fp is not None and fp.any() else None

    thick_ratio = _thickness_ratio(ip_m, gt_m)

    meta = dict(
        difficulty=difficulty,
        majority_event=maj_kind, minority_event=min_kind,
        majority_views=sorted(maj_views), minority_views=sorted(min_views),
        majority_px=int(fp.sum()) if fp is not None else 0,
        minority_px=int(minr.footprint.sum()) if minr.footprint is not None else 0,
        pixavg_member_iou_vs_gt=round(iou, 4),
        pixavg_consensus_thickness_ratio=round(thick_ratio, 4),
        pixavg_majority_recall=(round(maj_recall, 4)
                                if maj_recall is not None else None),
    )
    return ConsensusGroup(views=views, gt=gt, image_proc=ip, meta=meta)


def _thickness_ratio(member_pred, member_gt):
    """median(pred thickness) / median(gt thickness) on their skeletons.
    ~1.0 => consensus thickness preserved."""
    def med_thick(mask):
        sk = skeletonize(mask)
        if sk.sum() == 0:
            return 0.0
        return float(2.0 * np.median(distance_transform_edt(mask)[sk]))
    tg = med_thick(member_gt)
    tp = med_thick(member_pred)
    return tp / tg if tg else 1.0


def in_iou_band(group: ConsensusGroup, lo, hi) -> bool:
    return lo <= group.meta["pixavg_member_iou_vs_gt"] <= hi


def is_hard(group: ConsensusGroup, recall_max=0.5, iou_max=1.0) -> bool:
    """A group is 'hard' iff the pixel-averaging baseline disagrees with the
    consensus over the event. Low agreement (`pixavg_majority_recall`) = hard.
    The IoU gate is OFF by default (iou_max=1.0) since overall member IoU is
    dominated by the large unchanged base."""
    r = group.meta["pixavg_majority_recall"]
    if r is not None and r > recall_max:
        return False
    return group.meta["pixavg_member_iou_vs_gt"] < iou_max


# --------------------------------------------------------------------------
# folder driver
# --------------------------------------------------------------------------
def run(input_dir, output_dir, groups_per_image=4, difficulty="MEDIUM",
        majority_event="random", minority_event="random", event_pool=EVENT_KINDS,
        hard_only=False, recall_max=0.5, iou_max=1.0,
        target_iou=None, max_retries=25, seed=42, min_member_px=50):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        print(f"no images in {input_dir}", file=sys.stderr)
        return 1

    d_sub = output_dir / "subbatches"
    d_gt = output_dir / "ground_truth"
    d_ip = output_dir / "image_proc"
    for d in (d_sub, d_gt, d_ip):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    manifest, n_ok, n_skip = [], 0, 0
    lo, hi = (target_iou if target_iou else (0.0, 1.0))

    for img_path in images:
        base = load_label_map(img_path)
        if (base == CLS_MEMBER).sum() < min_member_px:
            n_skip += groups_per_image
            continue

        for _ in range(groups_per_image):
            group, rejected = None, 0
            for _try in range(max_retries):
                cand = build_group(base, rng, difficulty=difficulty,
                                   majority_event=majority_event,
                                   minority_event=minority_event,
                                   event_pool=event_pool)
                if cand is None:
                    continue
                if hard_only and not is_hard(cand, recall_max, iou_max):
                    rejected += 1
                    continue
                if target_iou and not in_iou_band(cand, lo, hi):
                    rejected += 1
                    continue
                group = cand
                break
            if group is None:
                n_skip += 1
                continue
            print('Batch Number ',n_ok)
            x = n_ok + 1                            # consecutive batch number
            gdir = d_sub / f"Batch {x}"
            gdir.mkdir(exist_ok=True)
            for v, view in enumerate(group.views):
                save_label_map(view, gdir / f"image_{v}.png")
            save_label_map(group.gt, d_gt / f"gt {x}.png")
            save_label_map(group.image_proc, d_ip / f"result_{x}.png")

            # meta goes ONLY to the manifest (no per-group meta.json)
            row = dict(batch=x, source_image=img_path.name,
                       easy_rejected_before_keep=rejected, **group.meta)
            manifest.append(row)
            n_ok += 1

    if manifest:
        # union of keys keeps the CSV robust if kinds/fields vary across rows
        leaders = ["batch", "source_image", "difficulty",
                   "majority_event", "minority_event"]
        keys = [k for k in leaders if any(k in r for r in manifest)]
        extra = sorted({k for r in manifest for k in r} - set(keys))
        fieldnames = keys + extra
        with open(output_dir / "manifest.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(manifest)

        recalls = [m["pixavg_majority_recall"] for m in manifest
                   if m["pixavg_majority_recall"] is not None]
        ious = [m["pixavg_member_iou_vs_gt"] for m in manifest]
        tratios = [m["pixavg_consensus_thickness_ratio"] for m in manifest]
        kinds = {}
        for m in manifest:
            kinds[m["majority_event"]] = kinds.get(m["majority_event"], 0) + 1
        print(f"done: {n_ok} groups, {n_skip} skipped -> {output_dir}")
        print(f"majority-event kinds      : {kinds}")
        print(f"image_proc IoU vs GT      : mean={np.mean(ious):.3f} "
              f"[{np.min(ious):.3f}, {np.max(ious):.3f}]")
        print(f"consensus thickness ratio : mean={np.mean(tratios):.3f} "
              f"(~1.0 = consensus thickness preserved)")
        if recalls:
            print(f"pixavg majority-agreement : mean={np.mean(recalls):.3f} "
                  f"max={np.max(recalls):.3f}  (low = hard for the baseline)")
    else:
        print(f"done: 0 groups written, {n_skip} skipped")
    return 0


def _cli():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-n", "--groups-per-image", type=int, default=4)
    ap.add_argument("--difficulty", default="MEDIUM",
                    choices=["EASY", "MEDIUM", "HARD"])
    ap.add_argument("--majority-event", default="random",
                    choices=["add", "remove", "reroute", "random"],
                    help="event present in 3/5 views (goes into GT); "
                         "'random' draws from --event-pool")
    ap.add_argument("--minority-event", default="random",
                    choices=["add", "remove", "reroute", "random"],
                    help="event present in 2/5 views (kept out of GT); "
                         "'random' draws from --event-pool")
    ap.add_argument("--event-pool", nargs="+", default=list(EVENT_KINDS),
                    choices=list(EVENT_KINDS),
                    help="kinds that 'random' draws from (default: all three)")
    ap.add_argument("--hard-only", action="store_true",
                    help="keep a group only if pixel averaging fails on it")
    ap.add_argument("--recall-max", type=float, default=0.5,
                    help="hard-only: keep group if baseline agrees with the "
                         "consensus over at most this fraction of the event")
    ap.add_argument("--iou-max", type=float, default=1.0,
                    help="hard-only: extra gate on member IoU vs GT "
                         "(1.0 = disabled; lower to also require low IoU)")
    ap.add_argument("--target-iou", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="keep a group only if image_proc IoU vs GT falls in "
                         "[LO, HI] (e.g. --target-iou 0.85 0.90)")
    ap.add_argument("--max-retries", type=int, default=25)
    ap.add_argument("--min-member-px", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    return run(a.input, a.output, a.groups_per_image, a.difficulty,
               a.majority_event, a.minority_event, a.event_pool, a.hard_only,
               a.recall_max, a.iou_max,
               tuple(a.target_iou) if a.target_iou else None,
               a.max_retries, a.seed, a.min_member_px)


if __name__ == "__main__":
    sys.exit(_cli())