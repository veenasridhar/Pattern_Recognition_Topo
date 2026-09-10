import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import os
import torch
import torch.nn.functional as F


def make_tpfnfp_overlay(pred_mask, gt_mask, structure_val=1):
    H, W   = pred_mask.shape
    print(np.unique(gt_mask))
    canvas = np.stack([gt_mask, gt_mask, gt_mask], axis=-1).astype(np.float32) / 2.0
    pred_bin = (pred_mask == structure_val).astype(np.float32)
    gt_bin   = (gt_mask   == structure_val).astype(np.float32)
    canvas[(pred_bin == 1) & (gt_bin == 1)] = [0.18, 0.80, 0.44]
    canvas[(pred_bin == 0) & (gt_bin == 1)] = [0.90, 0.20, 0.20]
    canvas[(pred_bin == 1) & (gt_bin == 0)] = [0.20, 0.45, 0.90]
    return canvas


def mask_to_rgb(mask, structure_val=2):
    H, W   = mask.shape
    print(mask.shape)
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    #canvas[mask == 1] = [0.45, 0.45, 0.45]
    canvas[mask == 1] = [1.00, 1.00, 1.00]
    return canvas


def visualize_comparison_with_overlay(groups, structure_val=1, save_path=None, batch=None):
    """
    4 rows, all equal height:
      row 0-1 : GT | Model Pred  | Baseline       (raw)
      row 2-3 : -- | Model Ovrl  | Baseline Ovrl  (TP/FN/FP)

    Metrics shown as xlabel below overlay axes — no separate row.
    """
    n_groups  = len(groups)
    n_rows    = 5        # equal height rows (was 5 with thin metric row)
    n_cols    = 4

    sample = np.array(groups[0]['gt'])
    h, w = sample.shape[:2]
    aspect = h / w

    col_width = 4.0  # wider columns
    row_height = max(aspect * col_width, 1.2)  # minimum 1.2 inches per row
    # prevents tiny rows for wide images

    # n_rows = 5
    # n_cols = 3

    fig_height = n_groups * (n_rows * row_height + 0.8)
    fig_width = n_cols * col_width

    fig = plt.figure(figsize=(fig_width, fig_height))
    outer_gs = gridspec.GridSpec(n_groups, 1, figure=fig, hspace=0.15)

    for g_idx, grp in enumerate(groups):

        gt_mask      = np.array(grp['gt'])
        pred_mask    = np.array(grp['pred'])
        print('The prediction mask is ',pred_mask.shape)
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
        # for r in range(n_rows):
        #     ax = fig.add_subplot(inner_gs[r, 0])
        #     img = torch.argmax(inputs[r],dim=0) if r < len(inputs) else None
        #     if img is not None:
        #         if hasattr(img, 'detach'):
        #             img = img.detach().cpu().numpy()
        #         img = np.array(img)
        #         if img.ndim == 2:
        #             img = mask_to_rgb(img, structure_val)
        #         ax.imshow(img, cmap='gray')
        #     else:
        #         ax.axis('off')
        #     ax.set_xticks([]); ax.set_yticks([])
        #     for spine in ax.spines.values():
        #         spine.set_visible(False)
        #     if r == 0 and g_idx == 0:
        #         ax.set_title("Input Images", fontsize=8, fontweight='bold', pad=4)

        # ── Col 1: GT spans rows 0-1, hidden rows 2-3 ──
        ax_gt = fig.add_subplot(inner_gs[1:3, 1])
        ax_gt.imshow(gt_rgb)
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
        ax_pred.imshow(pred_rgb)
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
        ax_base.imshow(base_rgb)
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
        print(f"Saved → {out}")
    
    plt.imsave(os.path.join(save_path, f'overlay_plot{batch}.png'), pred_overlay)
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

    if batch == 16:
        plt.imsave('overlay.png',ov_base)

    if save_path:
        fig.savefig(save_path+f'overlay_{batch}.png', dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"[Plot 2] Saved → {save_path}")

    return fig

# @torch.no_grad()
# def visualize_attsets_weights(weights, save_path, batch_index, b=0, upsample=None):
#     """weights: (B, N, C, H, W) softmax-over-views tensor (e.g. model.agg.last_weights)."""
#     w = weights[b].mean(dim=1).detach().float().cpu()     # (N,H,W): avg over latent channels
#     if upsample:                                          # match full-res inputs for comparison
#         w = F.interpolate(w.unsqueeze(1), scale_factor=upsample, mode="nearest").squeeze(1)
#     N = w.shape[0]
#     dominant = w.argmax(0)                                # which view wins each pixel
#     entropy = -(w * (w + 1e-8).log()).sum(0) / torch.log(torch.tensor(float(N)))

#     fig, ax = plt.subplots(1, N + 2, figsize=(3 * (N + 2), 3))
#     for i in range(N):
#         im = ax[i].imshow(w[i], vmin=0, vmax=1, cmap="magma")
#         ax[i].set_title(f"view {i}"); ax[i].axis("off")
#     ax[N].imshow(dominant, cmap="tab10", vmin=0, vmax=9)
#     ax[N].set_title("dominant view"); ax[N].axis("off")
#     hm = ax[N + 1].imshow(entropy, vmin=0, vmax=1, cmap="viridis")
#     ax[N + 1].set_title("weight entropy\n(bright = blended)"); ax[N + 1].axis("off")
#     fig.colorbar(hm, ax=ax[N + 1], fraction=0.046)
#     plt.tight_layout()
#     plt.savefig(f"{save_path}+entropy_map_{batch_index}.png", dpi=200, bbox_inches="tight")
#     return w, dominant, entropy

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

# # ── Metric helpers ─────────────────────────────────────────────────────────────
#
# def compute_cldice(pred_bin, gt_bin):
#     skel_pred = skeletonize(pred_bin > 0).astype(np.float32)
#     skel_gt = skeletonize(gt_bin > 0).astype(np.float32)
#     cl_prec = np.sum(skel_pred * gt_bin) / (np.sum(skel_pred) + 1e-7)
#     cl_rec = np.sum(skel_gt * pred_bin) / (np.sum(skel_gt) + 1e-7)
#     if cl_prec + cl_rec < 1e-7:
#         return 0.0, 0.0, 0.0
#     cldice = 2 * cl_prec * cl_rec / (cl_prec + cl_rec)
#     return float(cldice), float(cl_prec), float(cl_rec)
#
#
# def make_tpfnfp_overlay(pred_bin, gt_bin, bg_img=None):
#     """
#     Returns an RGBA overlay image showing TP/FN/FP on top of bg_img.
#     pred_bin, gt_bin : 2D binary arrays (0/1)
#     bg_img           : H×W or H×W×3 background (grayscale or RGB), optional
#     """
#     H, W = gt_bin.shape
#
#     # Background: use provided image or white
#     if bg_img is not None:
#         bg = np.array(bg_img, dtype=np.float32)
#         if bg.ndim == 2:
#             bg = np.stack([bg, bg, bg], axis=-1)
#         bg = bg / bg.max() if bg.max() > 1 else bg
#         canvas = (bg * 0.35 + 0.65).clip(0, 1)  # dim background slightly
#     else:
#         canvas = np.ones((H, W, 3), dtype=np.float32)
#
#     rgba = np.concatenate([canvas, np.ones((H, W, 1))], axis=-1)
#
#     tp_mask = (pred_bin == 1) & (gt_bin == 1)
#     fn_mask = (pred_bin == 0) & (gt_bin == 1)
#     fp_mask = (pred_bin == 1) & (gt_bin == 0)
#
#     # TP = green,  FN = red,  FP = blue
#     rgba[tp_mask] = [0.18, 0.80, 0.44, 1.0]
#     rgba[fn_mask] = [0.90, 0.20, 0.20, 1.0]
#     rgba[fp_mask] = [0.20, 0.45, 0.90, 1.0]
#
#     return rgba
#
#
# def binarize(mask, structure_val=255):
#     """Convert 0/128/255 mask to binary on structure class."""
#     arr = np.array(mask)
#     return (arr == structure_val).astype(np.float32)
#
#
# # ── Main visualization ─────────────────────────────────────────────────────────
#
# def visualize_comparison_with_overlay(groups, structure_val=255, save_path=None, batch =0):
#     """
#     groups: list of dicts with keys:
#         'inputs'   : list of 5 arrays/PIL (raw topo images)
#         'gt'       : ground truth mask  (0/128/255)
#         'pred'     : model prediction   (0/128/255)
#         'baseline' : pixel-avg baseline (0/128/255)
#         'label'    : str, e.g. "High Disagreement"
#
#     Columns per group:
#         0: Input images (stacked)
#         1: GT (raw)
#         2: Model prediction overlay  (TP/FN/FP)
#         3: Baseline overlay          (TP/FN/FP)
#     """
#     n_groups = len(groups)
#     n_inp_rows = 5
#     n_cols = 4
#     total_rows = n_groups * n_inp_rows
#
#     fig, axes = plt.subplots(
#         total_rows, n_cols,
#         figsize=(11, n_groups * 13),
#         gridspec_kw={'wspace': 0.04, 'hspace': 0.04}
#     )
#
#     # Make axes always 2D even for single group
#     if n_groups == 1:
#         axes = axes.reshape(n_inp_rows, n_cols)
#
#     col_titles = ["Input Images", "Ground Truth",
#                   "Model Prediction", "Pixel-Avg Baseline"]
#
#     for g_idx, grp in enumerate(groups):
#         row_offset = g_idx * n_inp_rows
#         mid = row_offset + n_inp_rows // 2
#
#         print('Ground truth ',grp['gt'])
#         gt_bin = binarize(grp['gt'], structure_val=2)
#         pred_bin = binarize(grp['pred'], structure_val=2)
#         base_bin = binarize(grp['baseline'], structure_val=2)
#
#         # Metrics
#         # cld_pred, prec_pred, rec_pred = compute_cldice(pred_bin, gt_bin)
#         # cld_base, prec_base, rec_base = compute_cldice(base_bin, gt_bin)
#
#         cld_pred, prec_pred, rec_pred = grp['cld_pred'], grp['prec_pred'], grp['rec_pred']
#         cld_base, prec_base, rec_base = grp['cld_base'], grp['prec_base'], grp['rec_base']
#
#         # Overlays
#         pred_overlay = make_tpfnfp_overlay(pred_bin, gt_bin, bg_img=grp['pred'])
#         base_overlay = make_tpfnfp_overlay(base_bin, gt_bin, bg_img=grp['baseline'])
#
#         # ── Col 0: Input images ──
#         for i, img in enumerate(grp['inputs']):
#             ax = axes[row_offset + i, 0]
#             ax.imshow(img.detach().cpu().numpy(),cmap= 'gray')
#             ax.set_xticks([])
#             ax.set_yticks([])
#             if g_idx == 0 and i == 0:
#                 ax.set_title(col_titles[0], fontsize=10,
#                              fontweight='bold', pad=6)
#             if i == 0:
#                 ax.set_ylabel(grp.get('label', f'Group {g_idx + 1}'),
#                               fontsize=9, fontweight='bold',
#                               rotation=90, labelpad=8)
#
#         # ── Cols 1–3: GT / Pred overlay / Baseline overlay ──
#         result_cols = [
#             (np.array(grp['gt']), col_titles[1], None, None, None, 1),
#             (pred_overlay, col_titles[2], cld_pred, prec_pred, rec_pred, 2),
#             (base_overlay, col_titles[3], cld_base, prec_base, rec_base, 3),
#         ]
#
#         for img, header, cld, prec, rec, col in result_cols:
#             # Hide all rows in this group-column block
#             for r in range(n_inp_rows):
#                 axes[row_offset + r, col].set_visible(False)
#
#             # Show only center row
#             ax = axes[mid, col]
#             ax.set_visible(True)
#             ax.imshow(img,cmap='gray')
#             ax.set_xticks([]);
#             ax.set_yticks([])
#
#             # Column header for first group only
#             if g_idx == 0:
#                 ax.set_title(header, fontsize=10, fontweight='bold', pad=6)
#
#             # Metric label below
#             if cld is not None:
#                 ax.set_xlabel(
#                     f"clDice: {cld:.3f}  \n clPrec: {prec:.3f}  \n  clRec: {rec:.3f}",
#                     fontsize=12, color='#222222', labelpad=5
#                 )
#
#             # Green/red border: which method has higher clDice
#             if col in (2, 3) and cld is not None:
#                 winner_cld = max(cld_pred, cld_base)
#                 border_col = '#2ecc71' if cld == winner_cld else '#e74c3c'
#                 for spine in ax.spines.values():
#                     spine.set_edgecolor(border_col)
#                     spine.set_linewidth(2.5)
#
#     # ── Shared legend ──
#     legend_patches = [
#         mpatches.Patch(color='#2ecc71', label='True Positive (TP)'),
#         mpatches.Patch(color='#e03333', label='False Negative (FN) — missed'),
#         mpatches.Patch(color='#3373e8', label='False Positive (FP) — Additional / Misaligned'),
#     ]
#     fig.legend(
#         handles=legend_patches,
#         loc='lower center',
#         ncol=3,
#         fontsize=9,
#         frameon=True,
#         bbox_to_anchor=(0.5, 0.01),
#         title="Prediction Error Map",
#         title_fontsize=9
#     )
#
#     fig.suptitle("Model vs. Pixel-Averaging Baseline — clDice & Error Overlay",
#                  fontsize=13, fontweight='bold', y=1.002)
#
#     plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave room for legend
#
#     if save_path:
#         plt.savefig(os.path.join(save_path,f'overlay_comp_{batch}.png'), dpi=150, bbox_inches='tight')
#         #print(f"Saved → {save_path}")
#     plt.show()
#
#
# # ── Example usage ──────────────────────────────────────────────────────────────
#
# if __name__ == "__main__":
#     H, W = 128, 128
#
#
#     def dummy_mask(structure_frac=0.3):
#         m = np.zeros((H, W), dtype=np.uint8)
#         m[:] = 128
#         n = int(H * W * structure_frac)
#         idx = np.random.choice(H * W, n, replace=False)
#         flat = m.flatten();
#         flat[idx] = 255;
#         m = flat.reshape(H, W)
#         return m
#
#
#     groups = [
#         {
#             'label': 'High Disagreement',
#             'inputs': [dummy_mask(0.25 + i * 0.02) for i in range(5)],
#             'gt': dummy_mask(0.30),
#             'pred': dummy_mask(0.28),
#             'baseline': dummy_mask(0.32),
#         },
#         {
#             'label': 'Low Disagreement',
#             'inputs': [dummy_mask(0.40) for _ in range(5)],
#             'gt': dummy_mask(0.40),
#             'pred': dummy_mask(0.38),
#             'baseline': dummy_mask(0.41),
#         },
#     ]
#
#     visualize_comparison_with_overlay(groups, structure_val=255,
#                                       save_path="topo_overlay_comparison.png")












# import os

# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import matplotlib.patches as mpatches
# import numpy as np
# from PIL import Image
# from skimage.morphology import skeletonize
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import matplotlib.patches as mpatches
# import numpy as np
# import os


# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import matplotlib.patches as mpatches
# import numpy as np
# import os


# def make_tpfnfp_overlay(pred_mask, gt_mask, structure_val=2):
#     H, W   = pred_mask.shape
#     print(np.unique(gt_mask))
#     canvas = np.stack([gt_mask, gt_mask, gt_mask], axis=-1).astype(np.float32) / 2.0
#     pred_bin = (pred_mask == structure_val).astype(np.float32)
#     gt_bin   = (gt_mask   == structure_val).astype(np.float32)
#     canvas[(pred_bin == 1) & (gt_bin == 1)] = [0.18, 0.80, 0.44]
#     canvas[(pred_bin == 0) & (gt_bin == 1)] = [0.90, 0.20, 0.20]
#     canvas[(pred_bin == 1) & (gt_bin == 0)] = [0.20, 0.45, 0.90]
#     return canvas


# def mask_to_rgb(mask, structure_val=2):
#     H, W   = mask.shape
#     print(mask.shape)
#     canvas = np.zeros((H, W, 3), dtype=np.float32)
#     #canvas[mask == 1] = [0.45, 0.45, 0.45]
#     canvas[mask == 1] = [1.00, 1.00, 1.00]
#     return canvas


# def visualize_comparison_with_overlay(groups, structure_val=2, save_path=None, batch=None):
#     """
#     4 rows, all equal height:
#       row 0-1 : GT | Model Pred  | Baseline       (raw)
#       row 2-3 : -- | Model Ovrl  | Baseline Ovrl  (TP/FN/FP)

#     Metrics shown as xlabel below overlay axes — no separate row.
#     """
#     n_groups  = len(groups)
#     n_rows    = 5        # equal height rows (was 5 with thin metric row)
#     n_cols    = 4

#     sample = np.array(groups[0]['gt'])
#     h, w = sample.shape[:2]
#     aspect = h / w

#     col_width = 4.0  # wider columns
#     row_height = max(aspect * col_width, 1.2)  # minimum 1.2 inches per row
#     # prevents tiny rows for wide images

#     # n_rows = 5
#     # n_cols = 3

#     fig_height = n_groups * (n_rows * row_height + 0.8)
#     fig_width = n_cols * col_width

#     fig = plt.figure(figsize=(fig_width, fig_height))
#     outer_gs = gridspec.GridSpec(n_groups, 1, figure=fig, hspace=0.15)

#     for g_idx, grp in enumerate(groups):

#         gt_mask      = np.array(grp['gt'])
#         pred_mask    = np.array(grp['pred'])
#         base_mask    = np.array(grp['baseline'])

#         gt_rgb       = mask_to_rgb(gt_mask,   structure_val)
#         pred_rgb     = mask_to_rgb(pred_mask, structure_val)
#         base_rgb     = mask_to_rgb(base_mask, structure_val)
#         pred_overlay = make_tpfnfp_overlay(pred_mask, gt_mask, structure_val)
#         base_overlay = make_tpfnfp_overlay(base_mask, gt_mask, structure_val)

#         cld_pred  = grp['cld_pred'];  prec_pred = grp['prec_pred'];  rec_pred = grp['rec_pred']
#         cld_base  = grp['cld_base'];  prec_base = grp['prec_base'];  rec_base = grp['rec_base']
#         iou_dl = grp['iou_dl']
#         iou_img_prc = grp['iou_img_prc']

#         # all 4 rows equal height
#         inner_gs = gridspec.GridSpecFromSubplotSpec(
#             n_rows, n_cols,
#             subplot_spec=outer_gs[g_idx],
#             hspace=0.02,
#             wspace=0.02,
#             height_ratios=[1, 1, 1, 1,1]
#         )

#         # ── Col 0: inputs — first 4 of the 5 stacked ──
#         inputs = grp['inputs']
#         for r in range(n_rows):
#             ax = fig.add_subplot(inner_gs[r, 0])
#             img = inputs[r] if r < len(inputs) else None
#             if img is not None:
#                 if hasattr(img, 'detach'):
#                     img = img.detach().cpu().numpy()
#                 img = np.array(img)
#                 # if img.ndim == 2:
#                 #     img = mask_to_rgb(img, structure_val)
#                 ax.imshow(img, cmap='gray')
#             else:
#                 ax.axis('off')
#             ax.set_xticks([]); ax.set_yticks([])
#             for spine in ax.spines.values():
#                 spine.set_visible(False)
#             if r == 0 and g_idx == 0:
#                 ax.set_title("Input Images", fontsize=8, fontweight='bold', pad=4)

#         # ── Col 1: GT spans rows 0-1, hidden rows 2-3 ──
#         ax_gt = fig.add_subplot(inner_gs[1:3, 1])
#         ax_gt.imshow(gt_rgb)
#         ax_gt.set_xticks([]); ax_gt.set_yticks([])
#         for spine in ax_gt.spines.values():
#             spine.set_visible(False)
#         if g_idx == 0:
#             ax_gt.set_title("Ground Truth", fontsize=8, fontweight='bold', pad=4)

#         for r in range(2, n_rows):
#             ax = fig.add_subplot(inner_gs[r, 1])
#             ax.axis('off')

#         # ── Col 2: pred rows 0-1 | overlay rows 2-3 with metrics as xlabel ──
#         ax_pred = fig.add_subplot(inner_gs[0:2, 2])
#         ax_pred.imshow(pred_rgb)
#         ax_pred.set_xticks([]); ax_pred.set_yticks([])
#         for spine in ax_pred.spines.values():
#             spine.set_visible(False)
#         if g_idx == 0:
#             ax_pred.set_title("Model Prediction", fontsize=8, fontweight='bold', pad=4)

#         ax_pred_ov = fig.add_subplot(inner_gs[0:2, 3])
#         ax_pred_ov.imshow(pred_overlay)
#         ax_pred_ov.set_xticks([]); ax_pred_ov.set_yticks([])
#         for spine in ax_pred_ov.spines.values():
#             spine.set_visible(False)
#         if g_idx == 0:
#             ax_pred_ov.set_title("Model Overlay", fontsize=7.5,
#                                   fontstyle='italic', pad=3)
#         # metrics as xlabel directly under overlay
#         ax_pred.set_xlabel(
#             f"clDice: {cld_pred:.3f}  |  clP: {prec_pred:.3f}  |  clR: {rec_pred:.3f} \n iou:{iou_dl}",
#             fontsize=10, fontweight='bold',color='#111111', labelpad=3
#         )

#         # ── Col 3: baseline rows 0-1 | overlay rows 2-3 with metrics as xlabel ──
#         ax_base = fig.add_subplot(inner_gs[2:4, 2])
#         ax_base.imshow(base_rgb)
#         ax_base.set_xticks([]); ax_base.set_yticks([])
#         for spine in ax_base.spines.values():
#             spine.set_visible(False)
#         if g_idx == 0:
#             ax_base.set_title("Baseline", fontsize=8, fontweight='bold', pad=4)

#         ax_base_ov = fig.add_subplot(inner_gs[2:4, 3])
#         ax_base_ov.imshow(base_overlay)
#         ax_base_ov.set_xticks([]); ax_base_ov.set_yticks([])
#         for spine in ax_base_ov.spines.values():
#             spine.set_visible(False)
#         if g_idx == 0:
#             ax_base_ov.set_title("Baseline Overlay", fontsize=7.5,
#                                   fontstyle='italic', pad=3)
#         ax_base.set_xlabel(
#             f"clDice: {cld_base:.3f}  |  clP: {prec_base:.3f}  |  clR: {rec_base:.3f}\n iou:{iou_img_prc}",
#             fontsize=10, fontweight='bold', color='#111111', labelpad=3
#         )

#     # ── Shared legend ──
#     # patches = [
#     #     mpatches.Patch(color='#39ff14', label='TP — correctly predicted'),
#     #     mpatches.Patch(color='#ff1744', label='FN — missed structure'),
#     #     mpatches.Patch(color='#1e90ff', label='FP — extra / misaligned structure'),
#     # ]
#     # legend = fig.legend(
#     #     handles=patches, loc='lower center', ncol=3,
#     #     fontsize=12, frameon=True,
#     #     bbox_to_anchor=(0.5, 0.0),
#     # )
#     #
#     # fig.canvas.draw()  # force render so legend has size
#     # legend_height = legend.get_window_extent().height  # pixels
#     # fig_height_px = fig.get_size_inches()[1] * fig.dpi  # pixels
#     # bottom_frac = (legend_height + 10) / fig_height_px  # +10px breathing room

#     # plt.subplots_adjust(
#     #     left=0.03,
#     #     right=0.99,
#     #     top=0.97,
#     #     bottom=bottom_frac  # ← dynamically computed, never overlaps
#     # )

#     fig.suptitle(
#         "Model Prediction vs Pixel Averaging Baseline",
#         fontsize=10, fontweight='bold', y=1.002
#     )

#     plt.tight_layout(rect=[0, 0.07, 1, 1])

#     if save_path:
#         os.makedirs(save_path, exist_ok=True)
#         out = os.path.join(save_path, f'comparison_batch_{batch}.png')
#         plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
#         print(f"Saved → {out}")
#     plt.close()


# def _remove_spines(ax):
#     for sp in ax.spines.values():
#         sp.set_visible(False)
#     ax.set_xticks([])
#     ax.set_yticks([])


# def plot2_overlay(
#         gt,  # (H,W) uint8
#         pred,  # (H,W) uint8
#         baseline,  # (H,W) uint8
#         metrics_pred=None,
#         metrics_base=None,
#         title="Overlay: Prediction vs Pixel-Avg",
#         save_path=None,
#         dpi=150, batch = 0
# ):

#     cld_p, clp_p, clr_p = metrics_pred
#     cld_b, clp_b, clr_b = metrics_base

#     ov_pred = make_tpfnfp_overlay(pred, gt)
#     ov_base = make_tpfnfp_overlay(baseline, gt)

#     H, W = gt.shape
#     aspect = W / H
#     cell_h = 4.0
#     cell_w = cell_h * aspect
#     n_cols = 3

#     fig_w = cell_w * n_cols + 0.6
#     fig_h = cell_h + 1.2  # 1 row + title + legend

#     fig, axes = plt.subplots(
#         1, 3,
#         figsize=(fig_w, fig_h),
#         facecolor='white',
#         gridspec_kw={'wspace': 0.06}
#     )

#     # ── Col 0: GT ──
#     axes[0].imshow(gt, cmap='gray', vmin=0, vmax=2, aspect='auto')
#     axes[0].set_title("Ground Truth", fontsize=14, fontweight='bold', pad=5)
#     _remove_spines(axes[0])

#     # ── Col 1: Prediction overlay ──
#     axes[1].imshow(ov_pred, aspect='auto')
#     axes[1].set_title("Model Prediction", fontsize=14, fontweight='bold', pad=5)
#     axes[1].set_xlabel(
#         f"clDice: {cld_p:.3f}  |  clP: {clp_p:.3f}  |  clR: {clr_p:.3f}",
#         fontsize=14,fontweight='bold', color='#222222', labelpad=6
#     )
#     _remove_spines(axes[1])

#     # ── Col 2: Baseline overlay ──
#     axes[2].imshow(ov_base, aspect='auto')
#     axes[2].set_title("Pixel-Avg Baseline", fontsize=14, fontweight='bold', pad=5)
#     axes[2].set_xlabel(
#         f"clDice: {cld_b:.3f}  |  clP: {clp_b:.3f}  |  clR: {clr_b:.3f}",
#         fontsize=14,fontweight='bold', color='#222222', labelpad=6
#     )
#     _remove_spines(axes[2])

#     # ── Legend ──
#     patches = [
#         mpatches.Patch(color='#00ff00', label='TP — correctly predicted'),
#         mpatches.Patch(color='#ff0000', label='FN — missed structure'),
#         mpatches.Patch(color='#0066ff', label='FP — extra / misaligned structure'),
#     ]
#     fig.legend(
#         handles=patches,
#         loc='lower center',
#         ncol=3,
#         fontsize=12,
#         frameon=True,
#         fancybox=True,
#         bbox_to_anchor=(0.5, 0.0),
#     )

#     fig.suptitle(title, fontsize=11, fontweight='bold', y=1.01)
#     plt.subplots_adjust(bottom=0.13)

#     if save_path:
#         fig.savefig(save_path+f'overlay_{batch}.png', dpi=dpi, bbox_inches='tight', facecolor='white')
#         print(f"[Plot 2] Saved → {save_path}")

#     return fig

# # # ── Metric helpers ─────────────────────────────────────────────────────────────
# #
# # def compute_cldice(pred_bin, gt_bin):
# #     skel_pred = skeletonize(pred_bin > 0).astype(np.float32)
# #     skel_gt = skeletonize(gt_bin > 0).astype(np.float32)
# #     cl_prec = np.sum(skel_pred * gt_bin) / (np.sum(skel_pred) + 1e-7)
# #     cl_rec = np.sum(skel_gt * pred_bin) / (np.sum(skel_gt) + 1e-7)
# #     if cl_prec + cl_rec < 1e-7:
# #         return 0.0, 0.0, 0.0
# #     cldice = 2 * cl_prec * cl_rec / (cl_prec + cl_rec)
# #     return float(cldice), float(cl_prec), float(cl_rec)
# #
# #
# # def make_tpfnfp_overlay(pred_bin, gt_bin, bg_img=None):
# #     """
# #     Returns an RGBA overlay image showing TP/FN/FP on top of bg_img.
# #     pred_bin, gt_bin : 2D binary arrays (0/1)
# #     bg_img           : H×W or H×W×3 background (grayscale or RGB), optional
# #     """
# #     H, W = gt_bin.shape
# #
# #     # Background: use provided image or white
# #     if bg_img is not None:
# #         bg = np.array(bg_img, dtype=np.float32)
# #         if bg.ndim == 2:
# #             bg = np.stack([bg, bg, bg], axis=-1)
# #         bg = bg / bg.max() if bg.max() > 1 else bg
# #         canvas = (bg * 0.35 + 0.65).clip(0, 1)  # dim background slightly
# #     else:
# #         canvas = np.ones((H, W, 3), dtype=np.float32)
# #
# #     rgba = np.concatenate([canvas, np.ones((H, W, 1))], axis=-1)
# #
# #     tp_mask = (pred_bin == 1) & (gt_bin == 1)
# #     fn_mask = (pred_bin == 0) & (gt_bin == 1)
# #     fp_mask = (pred_bin == 1) & (gt_bin == 0)
# #
# #     # TP = green,  FN = red,  FP = blue
# #     rgba[tp_mask] = [0.18, 0.80, 0.44, 1.0]
# #     rgba[fn_mask] = [0.90, 0.20, 0.20, 1.0]
# #     rgba[fp_mask] = [0.20, 0.45, 0.90, 1.0]
# #
# #     return rgba
# #
# #
# # def binarize(mask, structure_val=255):
# #     """Convert 0/128/255 mask to binary on structure class."""
# #     arr = np.array(mask)
# #     return (arr == structure_val).astype(np.float32)
# #
# #
# # # ── Main visualization ─────────────────────────────────────────────────────────
# #
# # def visualize_comparison_with_overlay(groups, structure_val=255, save_path=None, batch =0):
# #     """
# #     groups: list of dicts with keys:
# #         'inputs'   : list of 5 arrays/PIL (raw topo images)
# #         'gt'       : ground truth mask  (0/128/255)
# #         'pred'     : model prediction   (0/128/255)
# #         'baseline' : pixel-avg baseline (0/128/255)
# #         'label'    : str, e.g. "High Disagreement"
# #
# #     Columns per group:
# #         0: Input images (stacked)
# #         1: GT (raw)
# #         2: Model prediction overlay  (TP/FN/FP)
# #         3: Baseline overlay          (TP/FN/FP)
# #     """
# #     n_groups = len(groups)
# #     n_inp_rows = 5
# #     n_cols = 4
# #     total_rows = n_groups * n_inp_rows
# #
# #     fig, axes = plt.subplots(
# #         total_rows, n_cols,
# #         figsize=(11, n_groups * 13),
# #         gridspec_kw={'wspace': 0.04, 'hspace': 0.04}
# #     )
# #
# #     # Make axes always 2D even for single group
# #     if n_groups == 1:
# #         axes = axes.reshape(n_inp_rows, n_cols)
# #
# #     col_titles = ["Input Images", "Ground Truth",
# #                   "Model Prediction", "Pixel-Avg Baseline"]
# #
# #     for g_idx, grp in enumerate(groups):
# #         row_offset = g_idx * n_inp_rows
# #         mid = row_offset + n_inp_rows // 2
# #
# #         print('Ground truth ',grp['gt'])
# #         gt_bin = binarize(grp['gt'], structure_val=2)
# #         pred_bin = binarize(grp['pred'], structure_val=2)
# #         base_bin = binarize(grp['baseline'], structure_val=2)
# #
# #         # Metrics
# #         # cld_pred, prec_pred, rec_pred = compute_cldice(pred_bin, gt_bin)
# #         # cld_base, prec_base, rec_base = compute_cldice(base_bin, gt_bin)
# #
# #         cld_pred, prec_pred, rec_pred = grp['cld_pred'], grp['prec_pred'], grp['rec_pred']
# #         cld_base, prec_base, rec_base = grp['cld_base'], grp['prec_base'], grp['rec_base']
# #
# #         # Overlays
# #         pred_overlay = make_tpfnfp_overlay(pred_bin, gt_bin, bg_img=grp['pred'])
# #         base_overlay = make_tpfnfp_overlay(base_bin, gt_bin, bg_img=grp['baseline'])
# #
# #         # ── Col 0: Input images ──
# #         for i, img in enumerate(grp['inputs']):
# #             ax = axes[row_offset + i, 0]
# #             ax.imshow(img.detach().cpu().numpy(),cmap= 'gray')
# #             ax.set_xticks([])
# #             ax.set_yticks([])
# #             if g_idx == 0 and i == 0:
# #                 ax.set_title(col_titles[0], fontsize=10,
# #                              fontweight='bold', pad=6)
# #             if i == 0:
# #                 ax.set_ylabel(grp.get('label', f'Group {g_idx + 1}'),
# #                               fontsize=9, fontweight='bold',
# #                               rotation=90, labelpad=8)
# #
# #         # ── Cols 1–3: GT / Pred overlay / Baseline overlay ──
# #         result_cols = [
# #             (np.array(grp['gt']), col_titles[1], None, None, None, 1),
# #             (pred_overlay, col_titles[2], cld_pred, prec_pred, rec_pred, 2),
# #             (base_overlay, col_titles[3], cld_base, prec_base, rec_base, 3),
# #         ]
# #
# #         for img, header, cld, prec, rec, col in result_cols:
# #             # Hide all rows in this group-column block
# #             for r in range(n_inp_rows):
# #                 axes[row_offset + r, col].set_visible(False)
# #
# #             # Show only center row
# #             ax = axes[mid, col]
# #             ax.set_visible(True)
# #             ax.imshow(img,cmap='gray')
# #             ax.set_xticks([]);
# #             ax.set_yticks([])
# #
# #             # Column header for first group only
# #             if g_idx == 0:
# #                 ax.set_title(header, fontsize=10, fontweight='bold', pad=6)
# #
# #             # Metric label below
# #             if cld is not None:
# #                 ax.set_xlabel(
# #                     f"clDice: {cld:.3f}  \n clPrec: {prec:.3f}  \n  clRec: {rec:.3f}",
# #                     fontsize=12, color='#222222', labelpad=5
# #                 )
# #
# #             # Green/red border: which method has higher clDice
# #             if col in (2, 3) and cld is not None:
# #                 winner_cld = max(cld_pred, cld_base)
# #                 border_col = '#2ecc71' if cld == winner_cld else '#e74c3c'
# #                 for spine in ax.spines.values():
# #                     spine.set_edgecolor(border_col)
# #                     spine.set_linewidth(2.5)
# #
# #     # ── Shared legend ──
# #     legend_patches = [
# #         mpatches.Patch(color='#2ecc71', label='True Positive (TP)'),
# #         mpatches.Patch(color='#e03333', label='False Negative (FN) — missed'),
# #         mpatches.Patch(color='#3373e8', label='False Positive (FP) — Additional / Misaligned'),
# #     ]
# #     fig.legend(
# #         handles=legend_patches,
# #         loc='lower center',
# #         ncol=3,
# #         fontsize=9,
# #         frameon=True,
# #         bbox_to_anchor=(0.5, 0.01),
# #         title="Prediction Error Map",
# #         title_fontsize=9
# #     )
# #
# #     fig.suptitle("Model vs. Pixel-Averaging Baseline — clDice & Error Overlay",
# #                  fontsize=13, fontweight='bold', y=1.002)
# #
# #     plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave room for legend
# #
# #     if save_path:
# #         plt.savefig(os.path.join(save_path,f'overlay_comp_{batch}.png'), dpi=150, bbox_inches='tight')
# #         #print(f"Saved → {save_path}")
# #     plt.show()
# #
# #
# # # ── Example usage ──────────────────────────────────────────────────────────────
# #
# # if __name__ == "__main__":
# #     H, W = 128, 128
# #
# #
# #     def dummy_mask(structure_frac=0.3):
# #         m = np.zeros((H, W), dtype=np.uint8)
# #         m[:] = 128
# #         n = int(H * W * structure_frac)
# #         idx = np.random.choice(H * W, n, replace=False)
# #         flat = m.flatten();
# #         flat[idx] = 255;
# #         m = flat.reshape(H, W)
# #         return m
# #
# #
# #     groups = [
# #         {
# #             'label': 'High Disagreement',
# #             'inputs': [dummy_mask(0.25 + i * 0.02) for i in range(5)],
# #             'gt': dummy_mask(0.30),
# #             'pred': dummy_mask(0.28),
# #             'baseline': dummy_mask(0.32),
# #         },
# #         {
# #             'label': 'Low Disagreement',
# #             'inputs': [dummy_mask(0.40) for _ in range(5)],
# #             'gt': dummy_mask(0.40),
# #             'pred': dummy_mask(0.38),
# #             'baseline': dummy_mask(0.41),
# #         },
# #     ]
# #
# #     visualize_comparison_with_overlay(groups, structure_val=255,
# #                                       save_path="topo_overlay_comparison.png")