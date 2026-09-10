import os
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import disk, skeletonize

def compute_clDice(member_pred, member_gt):
    """clDice (centreline Dice) of the baseline member mask vs GT, in [0,1].
    Harmonic mean of
      Tprec = |skeleton(pred) inside GT mask| / |skeleton(pred)|   and
      Tsens = |skeleton(GT)   inside pred mask| / |skeleton(GT)|.
    Rewards getting the centreline/connectivity right, which IoU misses on
    thin members. 1.0 => both skeletons fully covered."""
    member_gt = np.asarray(member_gt).astype(bool)
    member_pred = np.asarray(member_pred).astype(bool)
    sk_p = skeletonize(member_pred)
    sk_g = skeletonize(member_gt)
    n_p, n_g = int(sk_p.sum()), int(sk_g.sum())
    if n_p == 0 or n_g == 0:
        return 1.0 if (n_p == 0 and n_g == 0) else 0.0
    tprec = float((sk_p & member_gt).sum()) / n_p
    tsens = float((sk_g & member_pred).sum()) / n_g
    if tprec + tsens == 0.0:
        return 0.0
    return 2.0 * tprec * tsens / (tprec + tsens)


def extract_foreground_mask(image):

    channel1 = image[:,:,0]
    channel2 = image[:,:,1]
    channel3 = image[:,:,2]
    #print('The unique values of channel 1 ',np.unique(channel1))
    mask = (channel1 >= 0.9) & (channel2  >= 0.9) & (channel3 >= 0.9)
    return mask.astype(float)


def image_averaging(raw_images, threshold, normalised=False):

    #Normalising the uint8 values in the RGB images
    if normalised:
        imgs = [np.array(raw_image) for raw_image in raw_images]   # Normalized Images and values range between 0 and 1.
    else:
        imgs = [np.array(raw_image/255.0) for raw_image in raw_images]

    #Aggregation Method - Mean across all the images
    avg_image = np.mean(imgs,axis=0)

    # Averaging along the channel axis to create a greyscale image. The mask has only (H,W) dimensions only.
    avg_mask = avg_image.mean(axis=2)
    #print('the unique avg mask',np.unique(avg_mask,return_counts=True))
    
    #Create a threshold mask where the pixel values greater than threshold are True and below are False (BOOLEAN MASK).
    threshold_mask = avg_mask > threshold
    
    # Convert it to int (0s and 1s) before visualising
    extracted_structures = np.zeros_like(avg_image)
    gray_mask = np.all((avg_image > 0.49) & (avg_image < 0.51), axis=-1) 
    extracted_structures[threshold_mask == 0] = 0
    extracted_structures[threshold_mask ==1] = 1
    extracted_structures[gray_mask] = 0.5
    
    return extracted_structures



def classical_approach(batch_directory,threshold=0.401):
            
    cldice_total =0.0
    n_batches = 0
    if not os.path.exists(f'Image Processing Results_{threshold}'):
        os.makedirs(f'Image Processing Results_{threshold}')

    for batches in os.listdir(batch_directory):
        for subbatches in os.listdir(os.path.join(batch_directory, batches)):
            image_stack = []
            n_batches += 1
        
            for image in os.listdir(os.path.join(batch_directory, batches, subbatches)):
                image = cv2.imread(os.path.join(batch_directory, batches, subbatches, image))
                image_stack.append(image)
            batch = subbatches.split(' ')[-1]

            ground_truth_image = cv2.imread(os.path.join(f'C:/Users/veena/Desktop/Pattern_Recognition_Topo/data/Ground_Truth/gt {batch}.png'))
            norm_gt_image = np.array(ground_truth_image / 255.0)  # Normalize the ground truth image to keep the images in the same data range.
            
            norm_extracted_common_structures = image_averaging(image_stack, threshold, normalised=False)
            cldice = compute_clDice(extract_foreground_mask(norm_extracted_common_structures),extract_foreground_mask(norm_gt_image))
            cldice_total += cldice

            cv2.imwrite(f'Image Processing Results_{threshold}/result_{batch}.png',norm_extracted_common_structures*255.0)
    
    print('The total cldice for threshold ', threshold, ' : ', cldice_total/n_batches)
    return cldice_total/n_batches

def plot_cldice_over_thresholds(cldice_over_t):
    thresholds = list(cldice_over_t.keys())
    cldice_scores = list(cldice_over_t.values())

    plt.figure(figsize=(5, 3))
    plt.plot(thresholds, cldice_scores, marker='o', linestyle='-', color='b')
    plt.title('clDice Scores over Different Thresholds',fontsize=15,fontweight='bold')
    plt.xlabel('Threshold',fontsize=16,fontweight='bold')
    plt.ylabel('clDice Score',fontsize=16,fontweight='bold')
    plt.tick_params(axis='both', labelsize=16)
    plt.xticks(thresholds)
    plt.savefig('cldice_scores_over_thresholds.pdf', bbox_inches='tight')
    plt.show()


def visualise_threshold_grid(batch,raw_images,gt_image, avg_image,avg_mask,common_structures,threshold_ssim_scores):
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

    for k,threshold in zip(range(len(common_structures)),cldice_over_t.keys()):

        ax = fig.add_subplot(grid_spec[2, k])
        ax.imshow(common_structures[k])
        ax.set_title(f'Threshold:{threshold}, cldice: {cldice_over_t[threshold]:0.2f}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'Image Processing Results_aug/train/result_{batch}.png')

if __name__ =='__main__':

    cldice_over_t = {}

    for threshold in [0.2, 0.3, 0.4, 0.5, 0.6,0.7, 0.8]:
        cldice = classical_approach('C:/Users/veena/Desktop/Pattern_Recognition_Topo/data/Batches', threshold)
        cldice_over_t[threshold] = cldice

    print('The cldice scores over thresholds are : ', cldice_over_t)
    plot_cldice_over_thresholds(cldice_over_t)