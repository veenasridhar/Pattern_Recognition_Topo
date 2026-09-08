import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from util import visualise_sdf


def visualize_iou_images(predicted_dilated_structure, target_dilated_structure, image_processing_structures,dilated_overlap, img_prc_dilated_overlap, iou_dilated,iou_dilated_image_processing,save_path,index):
    fig = plt.figure(figsize=(16, 8))
    grid_spec = fig.add_gridspec(1, 5, hspace=0.1, wspace=0.1)

    print('The predicted structures shape', predicted_dilated_structure.shape)
    print('The ground truth structures shape', target_dilated_structure.shape)
    print('The image processing structure shape', image_processing_structures.shape)

    ax_center = fig.add_subplot(grid_spec[0, 1])
    # recon_images = reconstructed_image.detach().cpu().numpy()
    recon_images = predicted_dilated_structure
    ax_center.imshow(recon_images, cmap='gray')
    ax_center.axis('off')
    ax_center.set_title('Reconstructed Image')

    ax_left = fig.add_subplot(grid_spec[0, 0])
    gt_image_visual = target_dilated_structure
    ax_left.imshow(gt_image_visual, cmap='gray')
    ax_left.axis('off')
    ax_left.set_title('Ground Truth')

    ax_left = fig.add_subplot(grid_spec[0, 2])
    # print(image_processing_result.shape)
    img_prc_result = image_processing_structures
    ax_left.imshow(img_prc_result, cmap='gray')
    ax_left.axis('off')
    ax_left.set_title('Pixelwise Averaging Result')

    ax_left = fig.add_subplot(grid_spec[0, 3])
    # img_prc_result = image_processing_result[i].permute(1, 2, 0).detach().cpu().numpy()
    ax_left.imshow(dilated_overlap)
    ax_left.axis('off')
    ax_left.set_title(f'IoU DL: {iou_dilated:.2f}')

    ax_left = fig.add_subplot(grid_spec[0, 4])
    # img_prc_result = image_processing_result[i].permute(1, 2, 0).detach().cpu().numpy()
    ax_left.imshow(img_prc_dilated_overlap)
    ax_left.axis('off')
    ax_left.set_title(f'IoU Pixelwise Avergaing: {iou_dilated_image_processing:.2f}')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'result_{index}.png'), dpi=150)
    plt.close()

def convert_sdf_to_structures(norm_sdf):
    sdf_px = np.zeros_like(norm_sdf)
    pos, neg = norm_sdf > 0, norm_sdf <= 0
    sdf_px[pos] = 0 #norm_sdf[pos] *  194.633 # multiply where you divided
    sdf_px[neg] = 1 #norm_sdf[neg] * 21
    #print(sdf_px.shape)
    #print(sdf_px.min(),sdf_px.max())
    return sdf_px

def extract_foreground_mask(image):

    channels = (image.shape)[-1]
    if channels == 1:
        grayscale_channel = image[:,:,0]
        mask = (grayscale_channel ==1)
    elif channels == 3:
        channel1 = image[:,:,0]
        channel2 = image[:,:,1]
        channel3 = image[:,:,2]
        mask = (channel1 >= 0.7) & (channel2  >= 0.7) & (channel3 >= 0.7)
        # R = image[:, :, 0]
        # G = image[:, :, 1]
        # mask = (R < 0.2)
    return mask.astype(float)

def iou_sdf(predicted_structures,  gt, image_processing_result, save_path, index):

    #Convert from SDF to Binary structures
    predicted_structures = convert_sdf_to_structures(predicted_structures.permute(1, 2, 0).detach().cpu().numpy())
    gt = convert_sdf_to_structures(gt.detach().cpu().numpy())
    #gt = convert_sdf_to_structures(gt.permute(1, 2, 0).detach().cpu().numpy())
    image_processing_result = convert_sdf_to_structures(image_processing_result.detach().cpu().numpy())


    red_extracted_structures = extract_foreground_mask(predicted_structures)
    # red_extracted_structures = extract_foreground_mask(predicted_structures.unsqueeze(-1).detach().cpu().numpy())
    red_gt_structures = extract_foreground_mask(np.expand_dims(gt, axis=-1))
    red_image_processing_structures = extract_foreground_mask(np.expand_dims(image_processing_result, axis=-1))

    # if tolerance > 0:
    #     r = tolerance
    #     y, x = np.ogrid[-r:r + 1, -r:r + 1]
    #     circle = (x ** 2 + y ** 2 <= r ** 2)
    #     predicted_dilated_structure = binary_dilation(red_extracted_structures.astype(bool), structure=circle)
    #     target_dilated_structure = binary_dilation(red_gt_structures.astype(bool), structure=circle)
    #     image_processing_structures = binary_dilation(red_image_processing_structures.astype(bool), structure=circle)
    # else:
    predicted_dilated_structure = red_extracted_structures
    target_dilated_structure = red_gt_structures
    image_processing_structures = red_image_processing_structures

    TP = (predicted_dilated_structure * target_dilated_structure).sum()
    FP = (predicted_dilated_structure * (1 - target_dilated_structure)).sum()
    FN = ((1 - predicted_dilated_structure) * target_dilated_structure).sum()

    dilated_overlap = np.zeros((*red_extracted_structures.shape, 3))
    dilated_overlap[..., 0] = (predicted_dilated_structure > 0) & (target_dilated_structure == 0)  # FP = red
    dilated_overlap[..., 1] = (predicted_dilated_structure > 0) & (target_dilated_structure > 0)  # TP = green
    dilated_overlap[..., 2] = (predicted_dilated_structure == 0) & (target_dilated_structure > 0)  # FN = blue

    iou_dilated = TP / (TP + FP + FN + 1e-8)

    TP = (image_processing_structures * target_dilated_structure).sum()
    FP = (image_processing_structures * (1 - target_dilated_structure)).sum()
    FN = ((1 - image_processing_structures) * target_dilated_structure).sum()

    img_proc_dilated_overlap = np.zeros((*red_extracted_structures.shape, 3))
    img_proc_dilated_overlap[..., 0] = (image_processing_structures > 0) & (target_dilated_structure == 0)  # FP = red
    img_proc_dilated_overlap[..., 1] = (image_processing_structures > 0) & (target_dilated_structure > 0)  # TP = green
    img_proc_dilated_overlap[..., 2] = (image_processing_structures == 0) & (target_dilated_structure > 0)  # FN = blue

    iou_dilated_image_processing = TP / (TP + FP + FN + 1e-8)

    print('Predicted IoU ', iou_dilated)
    print('Image Processing IoU ', iou_dilated_image_processing)

    visualize_iou_images(predicted_dilated_structure,target_dilated_structure,image_processing_structures, dilated_overlap, img_proc_dilated_overlap,iou_dilated,iou_dilated_image_processing,save_path,index)

    return iou_dilated, iou_dilated_image_processing



# for images in os.listdir('data/Final Data/gt_data_sdf/test'):
#     print(images)
#     test_structure = np.load(os.path.join('data/Final Data/gt_data_sdf/test',images))
#     plt.imshow(test_structure,cmap='seismic',vmax=1,vmin=-1)
#     plt.show()
