import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec
import os


def visualize_cldice_comparison_grid(groups, epoch, batch, fold, save_path):
    """
    Visualize clDice metrics in multi-panel grid layout.

    Args:
        groups: List of dicts, each containing:
            - 'inputs': list of 5 input images (B, N, C, H, W)
            - 'pred': predicted mask (H, W)
            - 'baseline': baseline mask (H, W)
            - 'gt': ground truth mask (H, W)
            - 'cld_pred', 'prec_pred', 'rec_pred': clDice metrics for model
            - 'cld_base', 'prec_base', 'rec_base': clDice metrics for baseline
        epoch: epoch number
        batch: batch number
        fold: fold number
        save_path: path to save figure
    """

    n_groups = len(groups)
    n_rows = 5  # Input rows (first 4 of 5 stacked images)
    n_cols = 4  # Input | GT | Model | Baseline

    sample = np.array(groups[0]['gt'])
    h, w = sample.shape[:2]
    aspect = h / w

    col_width = 4.0  # wider columns
    row_height = max(aspect * col_width, 1.2)  # minimum 1.2 inches per row
    # prevents tiny rows for wide images

    n_rows = 5
    n_cols = 4

    fig_height = n_groups * (n_rows * row_height + 0.8)
    fig_width = n_cols * col_width

    fig = plt.figure(figsize=(fig_width, fig_height))
    outer_gs = gridspec.GridSpec(n_groups, 1, figure=fig, hspace=0.15)

    # # Create outer gridspec (one row per group)
    # fig = plt.figure(figsize=(16, 5 * n_groups))
    # outer_gs = gridspec.GridSpec(
    #     n_groups, 1,
    #     hspace=0.15,  # Space between groups
    # )

    for g_idx, grp in enumerate(groups):

        # Extract data
        # print(grp['gt'])
        # gt_mask = np.array(grp['gt'])
        # pred_mask = np.array(grp['pred'])
        # base_mask = np.array(grp['baseline'])
        # inputs = grp['inputs']  # List of 5 images

        # Compute TP/FN/FP overlays
        pred_overlay = make_tpfnfp_overlay(grp['pred_foreground'], grp['gt_foreground'])
        base_overlay = make_tpfnfp_overlay(grp['baseline_foreground'], grp['gt_foreground'])

        # Extract metrics
        cld_pred = grp.get('cld_pred', 0.0)
        prec_pred = grp.get('prec_pred', 0.0)
        rec_pred = grp.get('rec_pred', 0.0)

        cld_base = grp.get('cld_base', 0.0)
        prec_base = grp.get('prec_base', 0.0)
        rec_base = grp.get('rec_base', 0.0)

        # Inner gridspec for this group
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            n_rows, n_cols,
            subplot_spec=outer_gs[g_idx],
            hspace=0.02,
            wspace=0.02,
            height_ratios=[1, 1, 1, 1, 1]
        )

        # ── Col 0: Input images (first 4 of 5 stacked) ──
        inputs = grp['inputs']
        for r in range(n_rows):
            ax = fig.add_subplot(inner_gs[r, 0])
            if r < len(inputs):
                img = inputs[r]
                if hasattr(img, 'detach'):
                    img = img.detach().cpu().numpy()
                img = np.array(img)

                # Handle multi-channel input
                if img.ndim == 3 and img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                    ax.imshow(img, cmap='gray')
                else:
                    ax.imshow(img, cmap='gray')
            else:
                ax.axis('off')

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if r == 0 and g_idx == 0:
                ax.set_title("Input Images (1-4)", fontsize=9, fontweight='bold', pad=4)

        # ── Col 1: Ground Truth (spans rows 0-1) ──
        ax_gt = fig.add_subplot(inner_gs[1, 1])
        gt_rgb = grp['gt']
        ax_gt.imshow(gt_rgb)
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        for spine in ax_gt.spines.values():
            spine.set_visible(False)
        if g_idx == 0:
            ax_gt.set_title("Ground Truth", fontsize=9, fontweight='bold', pad=4)

        # Hide rows 2-4 in col 1
        for r in range(2, n_rows):
            ax = fig.add_subplot(inner_gs[r, 1])
            ax.axis('off')

        # ── Col 2: Model Prediction (rows 0-1) | Overlay (rows 2-4) ──
        ax_pred = fig.add_subplot(inner_gs[1, 2])
        pred_rgb = grp['pred']
        ax_pred.imshow(pred_rgb)
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        for spine in ax_pred.spines.values():
            spine.set_visible(False)
        if g_idx == 0:
            ax_pred.set_title("Model Prediction", fontsize=9, fontweight='bold', pad=4)

        # Overlay with metrics
        ax_pred_ov = fig.add_subplot(inner_gs[2, 2])
        ax_pred_ov.imshow(pred_overlay)
        ax_pred_ov.set_xticks([])
        ax_pred_ov.set_yticks([])
        for spine in ax_pred_ov.spines.values():
            spine.set_visible(False)
        if g_idx == 0:
            ax_pred_ov.set_title("Model Overlay", fontsize=8, fontstyle='italic', pad=3)

        ax_pred_ov.set_xlabel(
            f"clDice: {cld_pred:.3f}  |  clP: {prec_pred:.3f}  |  clR: {rec_pred:.3f}",
            fontsize=9, color='#111111', labelpad=4
        )

        # ── Col 3: Baseline (rows 0-1) | Overlay (rows 2-4) ──
        ax_base = fig.add_subplot(inner_gs[1, 3])
        base_rgb = grp['baseline']
        ax_base.imshow(base_rgb)
        ax_base.set_xticks([])
        ax_base.set_yticks([])
        for spine in ax_base.spines.values():
            spine.set_visible(False)
        if g_idx == 0:
            ax_base.set_title("Baseline (Pixel Avg)", fontsize=9, fontweight='bold', pad=4)

        # Overlay with metrics
        ax_base_ov = fig.add_subplot(inner_gs[2, 3])
        ax_base_ov.imshow(base_overlay)
        ax_base_ov.set_xticks([])
        ax_base_ov.set_yticks([])
        for spine in ax_base_ov.spines.values():
            spine.set_visible(False)
        if g_idx == 0:
            ax_base_ov.set_title("Baseline Overlay", fontsize=8, fontstyle='italic', pad=3)

        ax_base_ov.set_xlabel(
            f"clDice: {cld_base:.3f}  |  clP: {prec_base:.3f}  |  clR: {rec_base:.3f}",
            fontsize=9, color='#111111', labelpad=4
        )

    # ── Shared legend ──
    patches = [
        mpatches.Patch(color='#00ff00', label='TP — correctly predicted'),
        mpatches.Patch(color='#ff0000', label='FN — missed structure'),
        mpatches.Patch(color='#0066ff', label='FP — hallucinated structure'),
    ]
    legend = fig.legend(
        handles=patches, loc='lower center', ncol=3,
        fontsize=9, frameon=True, fancybox=True,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.canvas.draw()
    legend_height = legend.get_window_extent().height
    fig_height_px = fig.get_size_inches()[1] * fig.dpi
    bottom_frac = (legend_height + 20) / fig_height_px

    plt.subplots_adjust(
        left=0.04,
        right=0.99,
        top=0.96,
        bottom=bottom_frac
    )

    fig.suptitle(
        f"Model Prediction vs Pixelwise Averaging Baseline",
        fontsize=10, fontweight='bold', y=0.995
    )

    # Save
    os.makedirs(save_path, exist_ok=True)
    filepath = f'{save_path}/cldice_batch{batch}.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def make_tpfnfp_overlay(pred_mask, gt_mask):
    """
    Create TP/FN/FP overlay visualization.

    Args:
        pred_mask: (H, W) predicted mask
        gt_mask: (H, W) ground truth mask

    Returns:
        overlay: (H, W, 3) RGB image with colors
    """
    print(pred_mask.shape)
    print(gt_mask.shape)
    print('Prediction Labels',np.unique(pred_mask))
    print('GT Labels',np.unique(gt_mask))
    H, W= pred_mask.shape
    canvas = np.zeros((H, W, 3), dtype=np.float32)

    # Compute TP, FN, FP
    tp = (pred_mask > 0) & (gt_mask > 0)
    fn = (pred_mask == 0) & (gt_mask > 0)
    fp = (pred_mask > 0) & (gt_mask == 0)

    # Bright colors
    canvas[tp] = [0, 255, 0]  # Green: TP
    canvas[fn] = [255, 0, 0]  # Red: FN
    canvas[fp] = [0, 102, 255]  # Blue: FP

    return canvas.astype(np.uint8)


def mask_to_rgb(mask, structure_val=255):
    """
    Convert discrete class mask to RGB visualization.

    Args:
        mask: (H, W) mask with values {0, 128, 255} or similar
        structure_val: value representing structure/design

    Returns:
        rgb: (H, W, 3) RGB image
    """
    if isinstance(mask, np.ndarray) and mask.ndim == 3:
        mask = mask.squeeze()

    H, W = mask.shape
    rgb = np.ones((H, W, 3)) * 200  # Light gray background

    # Color coding
    rgb[mask == 0] = [50, 50, 50]  # Dark gray: void
    rgb[mask == 128] = [150, 150, 150]  # Medium gray: non-design
    rgb[mask == 255] = [0, 200, 0]  # Green: structure

    return rgb.astype(np.uint8)