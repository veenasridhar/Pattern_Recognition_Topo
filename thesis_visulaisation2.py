import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import os


def make_tpfnfp_overlay(pred_mask, gt_mask, structure_val=2):
    H, W     = pred_mask.shape
    canvas   = np.zeros((H, W, 3), dtype=np.float32)
    pred_bin = (pred_mask == structure_val).astype(np.float32)
    gt_bin   = (gt_mask   == structure_val).astype(np.float32)
    canvas[(pred_bin == 1) & (gt_bin == 1)] = [0.0, 1.0, 0.0]   # green  TP
    canvas[(pred_bin == 0) & (gt_bin == 1)] = [1.0, 0.0, 0.0]   # red    FN
    canvas[(pred_bin == 1) & (gt_bin == 0)] = [0.0, 0.0, 1.0]   # blue   FP
    return canvas


def mask_to_rgb(mask, structure_val=2):
    H, W   = mask.shape
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    canvas[mask == 1] = [0.45, 0.45, 0.45]   # gray  — non-design
    canvas[mask == 2] = [1.00, 1.00, 1.00]   # white — structure
    return canvas


def visualize_comparison_with_overlay(groups, structure_val=2, save_path=None, batch=None):
    """
    Row 0 : 5 input images side by side (horizontal strip)
    Row 1 : GT | Model Pred | Model Overlay | Baseline | Baseline Overlay
    Metrics shown as title above each overlay image.
    """
    n_groups   = len(groups)
    n_inp_cols = 5
    n_res_cols = 5   # GT | pred | pred_ov | base | base_ov
    n_rows     = 2   # input strip + result row

    sample    = np.array(groups[0]['gt'])
    h, w      = sample.shape[:2]
    aspect    = h / w

    # Each result column same width; input strip spans all 5 result cols
    col_width  = max(2.0, w / 100)          # scale with image width
    row_height = max(aspect * col_width * n_res_cols / n_inp_cols, 1.5)

    fig_width  = n_res_cols * col_width
    fig_height = n_groups * (n_rows * row_height + 1.0)

    fig = plt.figure(figsize=(fig_width, fig_height))
    outer_gs = gridspec.GridSpec(n_groups, 1, figure=fig, hspace=0.25)

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

        # 2 rows: input strip + result row
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            n_rows, 1,
            subplot_spec=outer_gs[g_idx],
            hspace=0.12,              # gap between input strip and result row
        )

        # ── Row 0: input strip — 5 images side by side ──
        input_gs = gridspec.GridSpecFromSubplotSpec(
            1, n_inp_cols,
            subplot_spec=inner_gs[0],
            wspace=0.03              # small gap between input images
        )
        inputs = grp['inputs']
        for k in range(n_inp_cols):
            ax  = fig.add_subplot(input_gs[0, k])
            img = inputs[k] if k < len(inputs) else None
            if img is not None:
                if hasattr(img, 'detach'):
                    img = img.detach().cpu().numpy()
                img = np.array(img)
                # if img.ndim == 2:
                #     img = mask_to_rgb(img, structure_val)
                ax.imshow(img,cmap = 'gray')
            else:
                ax.axis('off')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        # single centered title over the whole strip
        fig.add_subplot(inner_gs[0]).set_title(
            "Input Images", fontsize=12, fontweight='bold', pad=6
        ) if False else None   # done via suptitle-style below
        # use a text annotation centered above strip
        strip_ax = fig.add_subplot(inner_gs[0])
        strip_ax.set_title("Input Images", fontsize=12, fontweight='bold', pad=6)
        strip_ax.axis('off')

        # ── Row 1: GT | pred | pred_ov | base | base_ov ──
        result_gs = gridspec.GridSpecFromSubplotSpec(
            1, n_res_cols,
            subplot_spec=inner_gs[1],
            wspace=0.04              # gap between result images
        )

        result_data = [
            (gt_rgb,       "Ground Truth",     None,     None,      None),
            (pred_rgb,     "Model Prediction", None,     None,      None),
            (pred_overlay, "Model Overlay",    cld_pred, prec_pred, rec_pred),
            (base_rgb,     "Baseline",         None,     None,      None),
            (base_overlay, "Baseline Overlay", cld_base, prec_base, rec_base),
        ]

        for col, (img, title, cld, prec, rec) in enumerate(result_data):
            ax = fig.add_subplot(result_gs[0, col])
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if cld is not None:
                # metrics above overlay as title
                ax.set_title(
                    f"{title}\nclDice: {cld:.3f}  clP: {prec:.3f}  clR: {rec:.3f}",
                    fontsize=12, fontweight='bold', pad=3
                )
            else:
                ax.set_title(title, fontsize=8, fontweight='bold', pad=4)

    # ── Shared legend ──
    patches = [
        mpatches.Patch(color='#00FF00', label='TP — correctly predicted'),
        mpatches.Patch(color='#FF0000', label='FN — missed structure'),
        mpatches.Patch(color='#0000FF', label='FP — hallucinated structure'),
    ]
    legend = fig.legend(
        handles=patches, loc='lower center', ncol=3,
        fontsize=8, frameon=True,
        bbox_to_anchor=(0.5, 0.0),
        title="Overlay Legend", title_fontsize=8
    )

    fig.suptitle(
        "Model Prediction vs Pixel-Avg Baseline",
        fontsize=12, fontweight='bold', y=1.002
    )

    fig.canvas.draw()
    legend_height = legend.get_window_extent().height
    fig_height_px = fig.get_size_inches()[1] * fig.dpi
    bottom_frac   = (legend_height + 10) / fig_height_px

    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=bottom_frac)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out = os.path.join(save_path, f'comparison_batch_{batch}.png')
        plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved → {out}")
    plt.close()