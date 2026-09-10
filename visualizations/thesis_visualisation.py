import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import torch
from PIL import Image
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import os


def make_tpfnfp_overlay(pred_mask, gt_mask, structure_val=1):
    H, W   = pred_mask.shape
    #print(np.unique(gt_mask))
    canvas = np.stack([gt_mask, gt_mask, gt_mask], axis=-1).astype(np.float32) / 2.0
    pred_bin = (pred_mask == structure_val).astype(np.float32)
    gt_bin   = (gt_mask   == structure_val).astype(np.float32)
    canvas[(pred_bin == 1) & (gt_bin == 1)] = [0.18, 0.80, 0.44]
    canvas[(pred_bin == 0) & (gt_bin == 1)] = [0.90, 0.20, 0.20]
    canvas[(pred_bin == 1) & (gt_bin == 0)] = [0.20, 0.45, 0.90]
    return canvas


def mask_to_rgb(mask, structure_val=1):
    H, W   = mask.shape
    #print(mask.shape)
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    #canvas[mask == 1] = [0.45, 0.45, 0.45]
    canvas[mask == 1] = [1.00, 1.00, 1.00]
    return canvas

def insert_gray(model_pred, refernce_img):
    mask = (refernce_img > 0.49) & (refernce_img < 0.51)
    model_pred[mask] = 0.49
    return model_pred


def visualize_comparison_with_overlay(groups, structure_val=2, save_path=None, batch=None):

    n_groups  = len(groups)
    n_rows    = 5        # equal height rows (was 5 with thin metric row)
    n_cols    = 4

    sample = np.array(groups[0]['gt'])
    h, w = sample.shape[:2]
    aspect = h / w

    col_width = 4.0  # wider columns
    row_height = max(aspect * col_width, 1.2)  # minimum 1.2 inches per row
    # prevents tiny rows for wide images

    fig_height = n_groups * (n_rows * row_height + 0.8)
    fig_width = n_cols * col_width

    fig = plt.figure(figsize=(fig_width, fig_height))
    outer_gs = gridspec.GridSpec(n_groups, 1, figure=fig, hspace=0.15)

    for g_idx, grp in enumerate(groups):

        gt_mask      = np.array(grp['gt'])
        pred_mask    = np.array(grp['pred'])
        base_mask    = np.array(grp['baseline'])

        gt_rgb       = mask_to_rgb(gt_mask,   structure_val)
        pred_rgb     = mask_to_rgb(pred_mask, structure_val)
        base_rgb     = mask_to_rgb(base_mask, structure_val)
        pred_overlay = make_tpfnfp_overlay(pred_mask, gt_mask, structure_val)
        base_overlay = make_tpfnfp_overlay(base_mask, gt_mask, structure_val)

        cld_pred  = grp['cld_pred'];  prec_pred = grp['prec_pred'];  rec_pred = grp['rec_pred']
        cld_base  = grp['cld_base'];  prec_base = grp['prec_base'];  rec_base = grp['rec_base']
        # iou_dl = grp['iou_dl']
        # iou_img_prc = grp['iou_img_prc']

        # all 4 rows equal height
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            n_rows, n_cols,
            subplot_spec=outer_gs[g_idx],
            hspace=0.02,
            wspace=0.02,
            height_ratios=[1, 1, 1, 1,1]
        )

        # ── Col 0: inputs — first 4 of the 5 stacked ──
        inputs = grp['inputs']
        for r in range(n_rows):
            ax = fig.add_subplot(inner_gs[r, 0])
            img = inputs[r] if r < len(inputs) else None
            if img is not None:
                if img.shape[0] == 2:
                    img = torch.argmax(img, dim=0)
                    #img = mask_to_rgb(img, structure_val)
                    print('The input image shape in visualisation after mask change',img.shape)
                if hasattr(img, 'detach'):
                    img = img.detach().cpu().numpy()
                img = np.array(img)
                ax.imshow(img, cmap='gray')
            else:
                ax.axis('off')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0 and g_idx == 0:
                ax.set_title("Input Images", fontsize=8, fontweight='bold', pad=4)

        # ── Col 1: GT spans rows 0-1, hidden rows 2-3 ──
        ax_gt = fig.add_subplot(inner_gs[1:3, 1])
        ax_gt.imshow(gt_rgb, cmap='gray')
        ax_gt.set_xticks([]); ax_gt.set_yticks([])
        for spine in ax_gt.spines.values():
            spine.set_visible(False)
        if g_idx == 0:
            ax_gt.set_title("Ground Truth", fontsize=8, fontweight='bold', pad=4)

        for r in range(2, n_rows):
            ax = fig.add_subplot(inner_gs[r, 1])
            ax.axis('off')

        # ── Col 2: pred rows 0-1 | overlay rows 2-3 with metrics as xlabel ──
        ax_pred = fig.add_subplot(inner_gs[0:2, 2])
        ax_pred.imshow(pred_rgb, cmap='gray')
        ax_pred.set_xticks([]); ax_pred.set_yticks([])
        for spine in ax_pred.spines.values():
            spine.set_visible(False)
        if g_idx == 0:
            ax_pred.set_title("Model Prediction", fontsize=8, fontweight='bold', pad=4)

        ax_pred_ov = fig.add_subplot(inner_gs[0:2, 3])
        ax_pred_ov.imshow(pred_overlay)
        ax_pred_ov.set_xticks([]); ax_pred_ov.set_yticks([])
        for spine in ax_pred_ov.spines.values():
            spine.set_visible(False)
        if g_idx == 0:
            ax_pred_ov.set_title("Model Overlay", fontsize=7.5,
                                  fontstyle='italic', pad=3)
        # metrics as xlabel directly under overlay
        ax_pred.set_xlabel(
            f"clDice: {cld_pred:.3f}  |  clP: {prec_pred:.3f}  |  clR: {rec_pred:.3f} ",
            fontsize=10, fontweight='bold',color='#111111', labelpad=3
        )

        # ── Col 3: baseline rows 0-1 | overlay rows 2-3 with metrics as xlabel ──
        ax_base = fig.add_subplot(inner_gs[2:4, 2])
        ax_base.imshow(base_rgb, cmap='gray')
        ax_base.set_xticks([]); ax_base.set_yticks([])
        for spine in ax_base.spines.values():
            spine.set_visible(False)
        if g_idx == 0:
            ax_base.set_title("Baseline", fontsize=8, fontweight='bold', pad=4)

        ax_base_ov = fig.add_subplot(inner_gs[2:4, 3])
        ax_base_ov.imshow(base_overlay)
        ax_base_ov.set_xticks([]); ax_base_ov.set_yticks([])
        for spine in ax_base_ov.spines.values():
            spine.set_visible(False)
        if g_idx == 0:
            ax_base_ov.set_title("Baseline Overlay", fontsize=7.5,
                                  fontstyle='italic', pad=3)
        ax_base.set_xlabel(
            f"clDice: {cld_base:.3f}  |  clP: {prec_base:.3f}  |  clR: {rec_base:.3f}",
            fontsize=10, fontweight='bold', color='#111111', labelpad=3
        )

    fig.suptitle(
        "Model Prediction vs Pixel Averaging Baseline",
        fontsize=10, fontweight='bold', y=1.002
    )

    plt.tight_layout(rect=[0, 0.07, 1, 1])

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out = os.path.join(save_path, f'comparison_batch_{batch}.png')
        plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def _remove_spines(ax):
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def plot2_overlay(
        gt,  # (H,W) uint8
        pred,  # (H,W) uint8
        baseline,  # (H,W) uint8
        metrics_pred=None,
        metrics_base=None,
        title="Overlay: Prediction vs Pixel-Avg",
        save_path=None,
        dpi=150, batch = 0
):

    cld_p, clp_p, clr_p = metrics_pred
    cld_b, clp_b, clr_b = metrics_base

    ov_pred = make_tpfnfp_overlay(pred, gt)
    ov_base = make_tpfnfp_overlay(baseline, gt)

    H, W = gt.shape
    aspect = W / H
    cell_h = 4.0
    cell_w = cell_h * aspect
    n_cols = 3

    fig_w = cell_w * n_cols + 0.6
    fig_h = cell_h + 1.2  # 1 row + title + legend

    fig, axes = plt.subplots(
        1, 3,
        figsize=(fig_w, fig_h),
        facecolor='white',
        gridspec_kw={'wspace': 0.06}
    )

    # ── Col 0: GT ──
    axes[0].imshow(gt, cmap='gray', vmin=0, vmax=1, aspect='auto')
    axes[0].set_title("Ground Truth", fontsize=14, fontweight='bold', pad=5)
    _remove_spines(axes[0])

    # ── Col 1: Prediction overlay ──
    axes[1].imshow(ov_pred, aspect='auto')
    axes[1].set_title("Model Prediction", fontsize=14, fontweight='bold', pad=5)
    axes[1].set_xlabel(
        f"clDice: {cld_p:.3f}  |  clP: {clp_p:.3f}  |  clR: {clr_p:.3f}",
        fontsize=14,fontweight='bold', color='#222222', labelpad=6
    )
    _remove_spines(axes[1])

    # ── Col 2: Baseline overlay ──
    axes[2].imshow(ov_base, aspect='auto')
    axes[2].set_title("Pixel-Avg Baseline", fontsize=14, fontweight='bold', pad=5)
    axes[2].set_xlabel(
        f"clDice: {cld_b:.3f}  |  clP: {clp_b:.3f}  |  clR: {clr_b:.3f}",
        fontsize=14,fontweight='bold', color='#222222', labelpad=6
    )
    _remove_spines(axes[2])

    # ── Legend ──
    patches = [
        mpatches.Patch(color='#00ff00', label='TP — correctly predicted'),
        mpatches.Patch(color='#ff0000', label='FN — missed structure'),
        mpatches.Patch(color='#0066ff', label='FP — extra / misaligned structure'),
    ]
    fig.legend(
        handles=patches,
        loc='lower center',
        ncol=3,
        fontsize=12,
        frameon=True,
        fancybox=True,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.01)
    plt.subplots_adjust(bottom=0.13)

    if save_path:
        fig.savefig(save_path+f'overlay_{batch}.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    return fig

@torch.no_grad()
def visualize_attsets_weights(weights, save_path, batch_index, b=0, upsample=None):
    """weights: (B, N, C, H, W) softmax-over-views. Plots per-view weight maps only."""
    w = weights[b].mean(dim=1).detach().float().cpu()   # (N,H,W): avg over latent channels
    if upsample:
        w = F.interpolate(w.unsqueeze(1), scale_factor=upsample, mode="nearest").squeeze(1)
    N = w.shape[0]

    fig, ax = plt.subplots(1, N, figsize=(3 * N, 3))
    if N == 1:
        ax = [ax]
    for i in range(N):
        im = ax[i].imshow(w[i],cmap="RdBu_r")
        ax[i].set_title(f"view {i}"); ax[i].axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)   # one shared bar for all panels
    plt.savefig(f"{save_path}/weights_{batch_index}.png", dpi=200, bbox_inches="tight")
    return w
