import os
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
import matplotlib.pyplot as plt
from metrics import *

def image_averaging(raw_images, threshold, normalised=False):

    #Normalising the uint8 values in the RGB images
    if normalised:
        imgs = [np.array(raw_image) for raw_image in raw_images]   # Normalized Images and values range between 0 and 1.
    else:
        imgs = [np.array(raw_image/255.0) for raw_image in raw_images]

    #Aggregation Method - Mean across all the images
    avg_image = np.mean(imgs,axis=0)
    # print('Average image is', avg_image.shape)
    # print('Min average image value', np.min(avg_image))
    # print('Max average image value', np.max(avg_image))

    # Averaging along the channel axis to create a greyscale image. The mask has only (H,W) dimensions only.
    avg_mask = avg_image.mean(axis=2)
    # print('Average image is', avg_image.shape)
    # print('Min average image value', np.min(avg_image))
    # print('Max average image value', np.max(avg_image))

    #Create a threshold mask where the pixel values greater than threshold are True and below are False (BOOLEAN MASK).
    threshold_mask = avg_mask > threshold
    # print('Threshold mask is', threshold_mask.shape)
    # print('Min average image value', np.min(threshold_mask))
    # print('Max average image value', np.max(threshold_mask))

    # Convert it to int (0s and 1s) before visualising
    extracted_structures = np.zeros_like(avg_image)
    white_mask = np.all(avg_image==1,axis=-1)   #Find all non design areas.

    #extracted_structures[threshold_mask == 0] = avg_image[threshold_mask == 0] # To get a original colors
    extracted_structures[threshold_mask == 0]= [1,0,0]              # Threshold mask with dark areas (False values) are actually our structures of interest. Red Structures
    extracted_structures[threshold_mask == 1] = [0.51, 0.51, 0.51]
    extracted_structures[white_mask] = [1,1,1]

    return extracted_structures, avg_image, avg_mask


def classical_approach(batch_directory,image_stack=None,gt=None,whole_directory = False):

    results = {}
    results2={}
    perceptual_metric = VGGPerceptualLoss()
    mse_metric = torch.nn.MSELoss()
    if whole_directory:
        #for threshold in [0.25,0.3,0.35,0.4,0.45,0.5,0.55]:
        for threshold in [0.42]:
            sum_ssim_score = 0
            sum_perceptual_metric = 0
            sum_dreamsim_metric = 0
            sum_mse_metric_latent_avg = 0.0
            sum_mse_metric_img_prc = 0.0
            sum_mse_metric_spatial = 0.0
            sum_iou_d_score = 0
            sum_iou =0
            for batches in os.listdir(batch_directory):
                image_stack = []
                common_structures = []
                #threshold_ssim_scores = {}
                for image in os.listdir(os.path.join(batch_directory, batches)):
                    image = cv2.imread(os.path.join(batch_directory, batches, image))
                    image = cv2.resize(image, (256, 256))
                    image_stack.append(image)
                batch = batches.split(' ')[-1]
                ground_truth_image = cv2.imread(os.path.join(f'./data/Ground Truth Images Resized_red/test/gt {batch}.png'))
                norm_gt_image = np.array(ground_truth_image / 255.0)  # Normalize the ground truth image to keep the images in the same data range.

                #for threshold in [0.3, 0.35, 0.4, 0.45, 0.5]:
                norm_extracted_common_structures,avg_img,avg_mask = image_averaging(image_stack, threshold, normalised=False)
                print(norm_extracted_common_structures.min(),norm_extracted_common_structures.max())
                # norm_extracted_common_structures = norm_extracted_common_structures*255                 # For saving purposes
                # cv2.imwrite(f'Image Processing Results/test/result_{batch}.png',norm_extracted_common_structures[:,:,::-1])

                #norm_extracted_common_structures = np.array(extracted_common_structures / 255.0)

                #common_structures.append(norm_extracted_common_structures)

                spatial_aggregated_image = cv2.imread(os.path.join(f'Spatial Aggregation Results/test_images/output_test_images_{batch}.png'))
                norm_spatial_aggregated_image = np.array(spatial_aggregated_image / 255.0)
                # print(norm_spatial_aggregated_image.min(), norm_spatial_aggregated_image.max())
                #
                latent_averaging_image = cv2.imread(os.path.join(f'Latent_averaging_Results/test_images/output_test_images_{batch}.png'))
                norm_latent_avg_image = np.array(latent_averaging_image / 255.0)
                # print(norm_latent_avg_image.min(), norm_latent_avg_image.max())
                #
                metric = 'iou'
                if metric == 'ssim':
                    ssim_score = ssim(norm_spatial_aggregated_image, norm_gt_image, data_range=1.0, channel_axis=-1)
                    #threshold_ssim_scores[threshold] = ssim_score
                    #visualise_results(batch,image_stack,ground_truth_image,avg_img,avg_mask,common_structures,threshold_ssim_scores)
                    sum_ssim_score += ssim_score
                elif metric == 'iou':
                    #iou_dilated_metric, iou_metric = iou_metrics(image_stack,norm_extracted_common_structures,norm_gt_image,batch)
                    #results[threshold] = iou_dilated_metric
                    iou_dilated_metric_spatial, dilated_overlap_spatial = iou_metrics(image_stack,norm_spatial_aggregated_image[:,:,::-1],norm_gt_image,batch)
                    iou_dilated_metric_latent_avg, dilated_overlap_latent_avg = iou_metrics(image_stack,norm_latent_avg_image[:,:,::-1],norm_gt_image, batch)
                    iou_dilated_metric_img_prc, dilated_overlap_img_prc = iou_metrics(image_stack,norm_extracted_common_structures,norm_gt_image, batch)

                    #threshold_iou_metric = iou_metric
                    #sum_iou_d_score += iou_dilated_metric
                    #sum_iou+= iou_metric
                elif metric == 'perceptual':
                    #predicted_red_structures = extract_red_mask(extracted_common_structures[:,:,::-1])
                    percp_metric = perceptual_metric(norm_extracted_common_structures,norm_gt_image)
                    results[threshold] = percp_metric
                    sum_perceptual_metric += percp_metric
                    #sum_percp_metric += percp_metric
                elif metric == 'dreamsim':
                    sim_metric_spatial = dreamsim_metric(norm_gt_image, norm_spatial_aggregated_image).item()
                    sim_metric_latent = dreamsim_metric(norm_gt_image,norm_latent_avg_image).item()
                    sim_metric_img_proc = dreamsim_metric(norm_gt_image,norm_extracted_common_structures).item()
                    #results[threshold] = sim_metric
                elif metric == 'mse':
                    sim_metric_spatial = mse_metric(torch.tensor(norm_spatial_aggregated_image),torch.tensor(norm_gt_image))
                    sim_metric_latent = mse_metric(torch.tensor(norm_latent_avg_image),torch.tensor(norm_gt_image))
                    sim_metric_img_proc = mse_metric(torch.tensor(norm_extracted_common_structures),torch.tensor(norm_gt_image))

                # sum_mse_metric_latent_avg += sim_metric_latent
                # sum_mse_metric_img_prc += sim_metric_img_proc
                # sum_mse_metric_spatial += sim_metric_spatial
                #

                #print(results)
                #visulaize_iou_thresh(batch, image_stack, norm_gt_image, common_structures, results)

                    #visualise_perc_results(batch, image_stack, norm_gt_image, common_structures, threshold,results)
                visualise_results_all(batch,image_stack,norm_gt_image,norm_extracted_common_structures,norm_spatial_aggregated_image,norm_latent_avg_image,iou_dilated_metric_spatial,dilated_overlap_spatial,iou_dilated_metric_latent_avg,dilated_overlap_latent_avg,iou_dilated_metric_img_prc,dilated_overlap_img_prc)
            #
            # print('The dreamsim metric for image processing',sum_mse_metric_img_prc)
            # print('The dreamsim metric for latent avg',sum_mse_metric_latent_avg)
            # print('The dreamsim metric of spatial aggre',sum_mse_metric_spatial)
            #print(f'The averaged iou score (dilated) at threshold {threshold} is {sum_perceptual_metric / len(os.listdir(batch_directory))}')
            #print(f'The averaged iou score  at threshold {threshold} is {sum_iou / len(os.listdir(batch_directory))}')
            #results[threshold] = sum_perceptual_metric / len(os.listdir(batch_directory))
            #results2[threshold] = sum_iou / len(os.listdir(batch_directory))

        # plt.plot(results.keys(), results.values(), 'bo-')
        # #plt.plot(results2.keys(), results2.values(), 'ro-')
        # plt.title('Threshold Vs Avg Perceptual Score')
        # plt.show()

    else:
        #image_stack = image_stack.cpu().permute(0, 2, 3, 1).numpy()
        extracted_common_structures = image_averaging(image_stack, threshold=0.4, normalised=False)
        print(extracted_common_structures.shape, gt.shape)
        #ssim_score = ssim(extracted_common_structures, gt.cpu().permute(1, 2, 0).numpy(), data_range=1.0,channel_axis=-1)
        #print('THE SSIM score seems to be like ', ssim_score)
        return extracted_common_structures


def visualise_results(batch,raw_images,gt_image, avg_image,avg_mask,common_structures,threshold_ssim_scores):
    fig = plt.figure(figsize=(16, 8))
    grid_spec = fig.add_gridspec(3, 5, hspace=0.1, wspace=0.1)

    for j in range(5):
        ax = fig.add_subplot(grid_spec[0, j])
        ax.imshow(raw_images[j][:, :, ::-1])
        if j ==2:
            ax.set_title('Input Images')
        ax.axis('off')


    ax = fig.add_subplot(grid_spec[1, 1])
    ax.imshow(avg_image[:, :, ::-1])
    ax.set_title('Average Image')
    ax.axis('off')

    ax = fig.add_subplot(grid_spec[1, 2])
    ax.imshow(avg_mask, cmap='gray')
    ax.set_title('Average Mask')
    ax.axis('off')

    ax = fig.add_subplot(grid_spec[1, 3])
    ax.imshow(gt_image[:,:,::-1])
    ax.set_title('Ground Truth Image')
    ax.axis('off')

    for i in [0,4]:
        ax = fig.add_subplot(grid_spec[1, i])
        ax.axis('off')

    for k,threshold in zip(range(len(common_structures)),threshold_ssim_scores.keys()):

        ax = fig.add_subplot(grid_spec[2, k])
        ax.imshow(common_structures[k])
        ax.set_title(f'Threshold:{threshold}, SSIM: {threshold_ssim_scores[threshold]:0.2f}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'Image Processing Results/train/threshold_result{batch}.png')


def find_common_structures(image_stack, method='frequency', threshold=0.6, apply_smoothing=False, filter_radius=1):
    design_stack = np.stack(image_stack, axis=0)
    print(design_stack.shape)

    #if method == 'strict':
    # Intersection - appears in ALL designs
    print("Using STRICT intersection (all designs must agree)")
    binary_stack = design_stack > 0.2
    print(binary_stack.shape)
    print(binary_stack[0].dtype)
    binary_stack = binary_stack * 255
    common = np.all(binary_stack, axis=0).astype(float)  # .astype(float) gives red and yellow colour

    cv2.imshow('Common Image - STRICT', common)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #elif method == 'majority':
    # # Majority voting - appears in >= threshold fraction
    min_count = int(np.ceil(threshold * 5))
    print(f"Using MAJORITY voting (≥{min_count}/{5} designs)")
    # binary_stack = design_stack > 0.6
    # plt.figure(figsize=(15, 3))
    # for i in range(5):
    #     plt.subplot(1, 5, i + 1)
    #     plt.imshow(binary_stack[i].astype(np.uint8), cmap='gray')
    #     plt.title(f'Image {i} binary')
    #     plt.axis('off')
    # plt.show()
    #
    # print(binary_stack.shape)
    count = np.sum(binary_stack, axis=0)
    print(count.shape)
    common = (count >= min_count).astype(float)
    #
    # cv2.imshow('Common Image - MAJORITY', common)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #

    fig = plt.figure(figsize=(16, 8))
    grid_spec = fig.add_gridspec(2, 5, hspace=0.1, wspace=0.1)

    for j in range(5):
        ax = fig.add_subplot(grid_spec[0, j])
        ax.imshow(image_stack[j][:, :, ::-1])
        ax.axis('off')

        '''ax = fig.add_subplot(grid_spec[1, j])
        ax.imshow(binary_stack[j][:,:,::-1])
        ax.axis('off')'''

    ax = fig.add_subplot(grid_spec[1, 1])
    ax.imshow(common)
    ax.axis('off')
    #
    # ax = fig.add_subplot(grid_spec[1, 2])
    # ax.imshow(threshold_mask, cmap='gray')
    # ax.axis('off')
    #
    # ax = fig.add_subplot(grid_spec[1, 3])
    # ax.imshow(extracted_structures)
    # ax.axis('off')

    plt.show()


    # # elif method == 'weighted':
    # # Weighted average
    # print(f"Using WEIGHTED average (threshold={threshold})")
    # avg_density = np.mean(design_stack, axis=0)
    # common = (avg_density > threshold).astype(float)
    #
    # cv2.imshow('Common Image - WEIGHTED', common)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # # elif method == 'frequency':
    # # Keep elements proportional to frequency of appearance
    # print(f"Using FREQUENCY-based (threshold={threshold})")
    # binary_stack = design_stack
    # frequency = np.mean(binary_stack, axis=0)
    # common = (frequency > threshold).astype(float)
    #
    # # else:
    # #    raise ValueError(f"Unknown method: {method}")
    #
    # cv2.imshow('Common Image - FREQUENCY', common)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # # Apply smoothing
    # if apply_smoothing and filter_radius > 0:
    #     print(f"Applying Gaussian smoothing (radius={filter_radius})")
    #     common = gaussian_filter(common, sigma=filter_radius)
    #     common = (common > 0.5).astype(float)  # Re-binarize
    #
    # cv2.imshow('After Smoothing', common)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # # Post-processing: Remove small disconnected regions
    # print("Cleaning up small disconnected regions...")
    # common = remove_small_regions(common, min_size=1)
    # cv2.imshow('After removing small objects', common)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # # Morphological operations to ensure connectivity
    # print("Applying morphological closing...")
    # # common = binary_closing(common).astype(float)
    # # cv2.imshow('After binary closing',common)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()