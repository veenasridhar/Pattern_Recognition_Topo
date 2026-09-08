import cv2
import numpy as np
#from skimage.morphology import binary_dilation
from scipy.ndimage.morphology import binary_dilation
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
from skimage.morphology import skeletonize
#import gudhi


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True)

        self.feature_map1 = nn.Sequential(*list(vgg.features)[:2])
        self.feature_map2 = nn.Sequential(*list(vgg.features)[2:4])
        self.feature_map3 = nn.Sequential(*list(vgg.features)[4:16])
        self.feature_map4 = nn.Sequential(*list(vgg.features)[16:23])

        for p in self.parameters():
            p.requires_grad = False

    def forward(self,real,reconstructed):
        #Normalize the input as per VGG Model expectations
        real = torch.tensor(real,dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
        #print(real.dtype)
        reconstructed = torch.tensor(reconstructed,dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
        print(real.shape)
        print(reconstructed.shape)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        real = (real - mean) / std
        recon = (reconstructed - mean) / std
        mse_loss = torch.nn.MSELoss().to(real.device)

        loss = 0.0
        real_input = real
        recon_input = recon

        # Forward through earlier layers first
        real1 = self.feature_map1(real_input)
        real2 = self.feature_map2(real1)

        recon1 = self.feature_map1(recon_input)
        recon2 = self.feature_map2(recon1)

        # Only calculate perceptual loss on deeper layers
        real3 = self.feature_map3(real2)
        recon3 = self.feature_map3(recon2)

        real4 = self.feature_map4(real3)
        recon4 = self.feature_map4(recon3)

        # for feature_map in [self.feature_map3, self.feature_map4]:
        #     real_input = feature_map(real_input)
        #     print(real_input.shape)
        #     recon_input = feature_map(recon_input)
        #     print(recon_input.shape)
        #     #print(real_input.shape,recon_input.shape)
        #     loss +=  mse_loss(recon_input,real_input.detach())

        # Creating feature importance for weighting the loss
        importance = real1.abs().mean(dim=1, keepdim=True)
        importance = importance / (importance.max() + 1e-6)
        print('Importance shape',importance.shape)

        loss = 0.0
        diff = (recon1 - real1)**2
        diff2 = (recon2 - real2)**2
        loss += (diff * (1 + 9*importance)).mean()
        loss += (diff2 * (1 + 9 * importance)).mean()

        #loss += mse_loss(recon1, real1.detach())
        #loss += mse_loss(recon2, real2.detach())

        similarity = torch.exp(-loss / 100.0)
        return similarity

def extract_red_mask(image):
    print('Image min and max',image.min(),image.max())
    R = image[:,:,0]
    G = image[:,:,1]
    mask = (R <0.2)
    return mask.astype(float)

def extract_foreground_mask(image):
    #image_expanded = np.expand_dims(image,2)
    channels = (image.shape)[-1]
    print(channels)
    if channels == 1:
        grayscale_channel = image[:,:,0]
        mask = (grayscale_channel > 0.7)
    elif channels == 3:
        R = image[:, :, 0]
        G = image[:, :, 1]
        mask = (R < 0.2)
    else:
        R = image
        mask = (R ==1)
    print(mask.shape)
    return mask.astype(float)

def cl_score(skeleton, mask):
    """Fraction of `skeleton` pixels that lie inside `mask`. None if skeleton is empty."""
    denom = skeleton.sum()
    if denom == 0:
        return None
    return float((skeleton & mask).sum()) / float(denom)

def cldice_metrics(extracted_structures, gt):
    # --- binarize BOTH sides the same way (this is where your IoU version drifted) ---
    pred_mask = extracted_structures.astype(bool)
    gt_mask   = gt.astype(bool)        # <-- match this to your actual GT encoding
    # if gt is 3-ch white-on-black: gt_mask = np.asarray(gt)[..., 0] > 0.5

    skel_pred = skeletonize(pred_mask)
    print('The shape of skel_pred',skel_pred.shape)
    plt.imsave('Pred skeleton.png',skel_pred.squeeze(2))
    
    skel_gt   = skeletonize(gt_mask)
    print('The shape of gt_skel',skel_gt.shape)
    plt.imsave('gt skeleton.png',skel_gt)

    clP = cl_score(skel_pred, np.expand_dims(gt_mask,2))        # predicted centerline supported by GT volume
    clR = cl_score(np.expand_dims(skel_gt,2),   pred_mask)      # GT centerline captured by predicted volume

    print('clp',clP,'clr',clR)
    if clP is None and clR is None:
        return 1.0, 1.0, 1.0                  # both empty -> perfect agreement
    clP = 0.0 if clP is None else clP
    clR = 0.0 if clR is None else clR
    if clP + clR == 0:
        return 0.0, clP, clR

    cldice = 2.0 * clP * clR / (clP + clR)
    print('clDice', cldice, '| clP', clP, '| clR', clR)
    return cldice

def iou_metrics(extracted_structures, gt,tolerance=0):

    # #Extrract only the red structures because the background is irrelevant and inflates the metrics
    # #extracted_structures =extracted_structures.astype(float)/255.0
    # red_extracted_structures = extract_foreground_mask(extracted_structures)
    # plt.imsave('Extracted structures.png',red_extracted_structures)
    # #gt = gt.astype(float)/255.0
    # print('The gt shape',gt.shape)
    # red_mask_gt = gt
    # print('The red maks gt',red_mask_gt.shape)
    # plt.imsave('GT Mask.png',red_mask_gt)
    red_extracted_structures = extracted_structures.squeeze(-1)
    print('Extracted Structures shape',red_extracted_structures.shape,np.unique(red_extracted_structures))
    red_mask_gt = gt
    print('GT shape',red_mask_gt.shape,np.unique(red_mask_gt))

    if tolerance>0:
        r = tolerance
        y, x = np.ogrid[-r:r + 1, -r:r + 1]
        circle = (x ** 2 + y ** 2 <= r ** 2)
        predicted_dilated_structure = binary_dilation(red_extracted_structures.astype(bool), structure=circle)
        target_dilated_structure = binary_dilation(red_mask_gt.astype(bool), structure=circle)

    else:
        predicted_dilated_structure = red_extracted_structures
        target_dilated_structure = red_mask_gt

    TP = (predicted_dilated_structure * target_dilated_structure).sum()
    print('TP',TP)
    FP = (predicted_dilated_structure * (1 - target_dilated_structure)).sum()
    print('FP',FP)
    FN = ((1 - predicted_dilated_structure) * target_dilated_structure).sum()
    print('FN',FN)

    # dilated_overlap = np.zeros((*red_extracted_structures.shape, 3))
    # dilated_overlap[..., 0] = (predicted_dilated_structure > 0) & (target_dilated_structure == 0)  # FP = red
    # dilated_overlap[..., 1] = (predicted_dilated_structure > 0) & (target_dilated_structure > 0)  # TP = green
    # dilated_overlap[..., 2] = (predicted_dilated_structure == 0) & (target_dilated_structure > 0)  # FN = blue


    iou_dilated = TP / (TP + FP + FN + 1e-8)
    print('IoU Dilated',iou_dilated)

    cldice = cldice_metrics(extracted_structures,gt)
    print('cldice',cldice)

    return iou_dilated, cldice

def compute_betti_numbers(image):

    foreground_structures = extract_foreground_mask(image.unsqueeze(dim=-1).cpu().numpy())
    filt = np.ascontiguousarray(1.0 - foreground_structures, dtype= np.float64)
    cc = gudhi.CubicalComplex(dimensions= list(filt.shape), top_dimensional_cells = filt.ravel())
    cc.compute_persistence()
    betti = cc.persistent_betti_numbers(0.5,0.5)
    print(betti)

    b0 = betti[0] if len(betti) > 0 else 0
    b1 = betti[1] if len(betti) > 1 else 0
    return b0, b1



def visulaize_iou_results(batch,raw_images,extracted_structures, gt, predicted_dilated_structure,target_dilated_structure,dilated_overlap,overlap, iou_dilated, iou):

    fig = plt.figure(figsize=(16, 8))
    grid_spec = fig.add_gridspec(3, 5, hspace=0.2, wspace=0.2)

    for j in range(5):
        ax = fig.add_subplot(grid_spec[0, j])
        ax.imshow(raw_images[j][:, :, ::-1])
        if j == 2:
            ax.set_title('Input Images')
        ax.axis('off')

    # ax = fig.add_subplot(grid_spec[1, 1])
    # ax.imshow(extracted_structures[:,:,::-1])
    # ax.set_title('Extracted Structures')
    # ax.axis('off')

    ax = fig.add_subplot(grid_spec[1, 2])
    ax.imshow(gt[:,:,::-1])
    ax.set_title('Ground Truth')
    ax.axis('off')

    #for k, threshold in zip(range(len(common_structures)), threshold_perc_scores.keys()):
    # ax = fig.add_subplot(grid_spec[2, k])
    #      ax.imshow(common_structures[k])
    #      ax.set_title(f'Threshold:{threshold}, LPIPS: {threshold_perc_scores[threshold]:0.2f}')
    #      ax.axis('off')

    # ax = fig.add_subplot(grid_spec[1, 3])
    # ax.imshow(predicted_dilated_structure,cmap='gray')
    # ax.set_title('Predicted Dilated Structure')
    # ax.axis('off')
    #
    # ax = fig.add_subplot(grid_spec[1, 4])
    # ax.imshow(target_dilated_structure,cmap='gray')
    # ax.set_title('Target Dilated Structure')
    # ax.axis('off')

    # ax = fig.add_subplot(grid_spec[2, 1])
    # ax.imshow(dilated_overlap)
    # ax.set_title(f'Dilated Overlap, IOU: {iou_dilated:0.2f}')
    # ax.axis('off')
    #
    # ax = fig.add_subplot(grid_spec[2, 2])
    # ax.imshow(overlap)
    # ax.set_title(f'IOU: {iou:0.2f}')
    # ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'Latent_averaging_Results/IoU Metric/spatial_agg_result{batch}.png')
    plt.close()

def visualise_perc_results(batch,raw_images,gt_image, common_structures,threshold,threshold_perc_scores):
    fig = plt.figure(figsize=(16, 8))
    grid_spec = fig.add_gridspec(3, 5, hspace=0.1, wspace=0.1)

    for j in range(5):
        ax = fig.add_subplot(grid_spec[0, j])
        ax.imshow(raw_images[j][:, :, ::-1])
        if j ==2:
            ax.set_title('Input Images')
        ax.axis('off')

    ax = fig.add_subplot(grid_spec[1, 2])
    ax.imshow(gt_image[:,:,::-1])
    ax.set_title('Ground Truth Image')
    ax.axis('off')

    # ax = fig.add_subplot(grid_spec[1, 3])
    # ax.imshow(common_structures[:,:,::-1])
    # ax.set_title(f'Threshold:{threshold}, SSIM: {threshold_perc_scores:0.2f}')
    # ax.axis('off')


    for k,threshold in zip(range(len(common_structures)),threshold_perc_scores.keys()):
         ax = fig.add_subplot(grid_spec[2, k])
         ax.imshow(common_structures[k])
         ax.set_title(f'Threshold:{threshold}, LPIPS: {threshold_perc_scores[threshold]:0.2f}')
         ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'Image Processing Results/LPIPS Metric_changed/train_images_w_threshold/threshold_result{batch}.png')
    plt.close()


def visualise_results_all(batch,raw_images,norm_gt_image,norm_extracted_common_structures,norm_spatial_aggregated_image,norm_latent_avg_image, iou_dilated_metric_spatial, dilated_overlap_spatial, iou_dilated_metric_latent_avg, dilated_overlap_latent_avg,iou_dilated_metric_img_prc, dilated_overlap_img_pr):
    fig = plt.figure(figsize=(16, 10))
    grid_spec = fig.add_gridspec(3, 5, hspace=0.2, wspace=0.1)


    for j in range(5):
        ax = fig.add_subplot(grid_spec[0, j])
        ax.imshow(raw_images[j][:, :, ::-1])
        if j ==2:
            ax.set_title('Input Images')
        ax.axis('off')

    ax = fig.add_subplot(grid_spec[1, 0])
    ax.imshow(norm_gt_image[:,:,::-1])
    ax.set_title('Ground Truth Image')
    ax.axis('off')

    ax = fig.add_subplot(grid_spec[1, 1])
    ax.imshow(norm_extracted_common_structures)
    ax.set_title(f'Pixel Averaging \n IoU: {iou_dilated_metric_img_prc:0.2f}')
    ax.axis('off')

    ax = fig.add_subplot(grid_spec[1, 2])
    ax.imshow(norm_latent_avg_image[:, :, ::-1])
    ax.set_title(f'Latent Averaging \n IoU: {iou_dilated_metric_latent_avg:0.2f}')
    ax.axis('off')

    ax = fig.add_subplot(grid_spec[1, 3])
    ax.imshow(norm_spatial_aggregated_image[:, :, ::-1])
    ax.set_title(f'Spatial Aggregation \n IoU: {iou_dilated_metric_spatial:0.2f}')
    ax.axis('off')

    ax = fig.add_subplot(grid_spec[2, 1])
    ax.imshow(dilated_overlap_img_pr)
    ax.set_title(f'Pixel Averaging \n IoU: {iou_dilated_metric_img_prc:0.2f}')
    ax.axis('off')

    ax = fig.add_subplot(grid_spec[2, 2])
    ax.imshow(dilated_overlap_latent_avg)
    ax.set_title(f'Latent Averaging \n IoU: {iou_dilated_metric_latent_avg:0.2f}')
    ax.axis('off')

    ax = fig.add_subplot(grid_spec[2, 3])
    ax.imshow(dilated_overlap_spatial)
    ax.set_title(f'Spatial Aggregation \n IoU: {iou_dilated_metric_spatial:0.2f}')
    ax.axis('off')


    # for k,threshold in zip(range(len(common_structures)),threshold_perc_scores.keys()):
    #      ax = fig.add_subplot(grid_spec[2, k])
    #      ax.imshow(common_structures[k])
    #      ax.set_title(f'Threshold:{threshold}, LPIPS: {threshold_perc_scores[threshold]:0.2f}')
    #      ax.axis('off')
    #
    plt.tight_layout()
    plt.savefig(f'./Compiled Results/iou_metric/test/image_{batch}.png')
    plt.close()

def visulaize_iou_thresh(batch,raw_images,gt,common_structures,threshold_perc_scores):

    fig = plt.figure(figsize=(16, 8))
    grid_spec = fig.add_gridspec(3, 5, hspace=0.2, wspace=0.2)

    for j in range(5):
        ax = fig.add_subplot(grid_spec[0, j])
        ax.imshow(raw_images[j][:, :, ::-1])
        if j == 2:
            ax.set_title('Input Images')
        ax.axis('off')


    ax = fig.add_subplot(grid_spec[1, 2])
    ax.imshow(gt[:,:,::-1])
    ax.set_title('Ground Truth')
    ax.axis('off')

    for k, threshold in zip(range(len(common_structures)), threshold_perc_scores.keys()):
        ax = fig.add_subplot(grid_spec[2, k])
        ax.imshow(common_structures[k])
        ax.set_title(f'Threshold:{threshold} \n MSE:{threshold_perc_scores[threshold].item():0.2f}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'Image Processing Results/IoU Metric/train_diff_threshold/result{batch}.png')
    plt.close()

