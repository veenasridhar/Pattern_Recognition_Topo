from skimage.morphology import skeletonize
import numpy as np
import os
import  matplotlib.pyplot as plt
import cv2

def compute_and_store_skeletons(pred_mask, gt_mask, structure_val=2, save_path=None, index=0):
    """
    Computes hard skeletons of prediction and ground truth masks.

    pred_mask, gt_mask : 2D numpy arrays with class values (0/1/2)
    structure_val      : class index for structure
    save_path          : folder to save skeleton images (optional)
    index              : batch/image index for filename

    Returns:
        skel_pred, skel_gt : 2D binary numpy arrays (0/1)
    """
    pred_bin = (pred_mask == structure_val).astype(np.float32)
    gt_bin = (gt_mask == structure_val).astype(np.float32)

    skel_pred = skeletonize(pred_bin > 0).astype(np.float32)
    skel_gt = skeletonize(gt_bin > 0).astype(np.float32)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(skel_gt, cmap='gray',vmin=0, vmax=1);
        axes[0].set_title("GT Skeleton");
        axes[0].axis('off')
        axes[1].imshow(skel_pred, cmap='gray',vmin=0, vmax=1);
        axes[1].set_title("Pred Skeleton");
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'skeleton_{index}.png'), dpi=150, bbox_inches='tight')
        print('PLot saved')
        plt.close()

    return skel_pred, skel_gt


if __name__ == '__main__':
    skel_pred, skel_gt = compute_and_store_skeletons(
        pred_mask=cv2.imread('data/Edited Ground Truth_ Encoded/gt 1.png', cv2.IMREAD_GRAYSCALE),
        gt_mask=cv2.imread('Residual Convolutional Network Results_original size_Fold 1/val/Epoch 50/result_1.png', cv2.IMREAD_GRAYSCALE),
        structure_val=255,
        save_path=f'skeleton.png',
    )
