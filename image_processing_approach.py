import numpy as np
import scipy
import skimage
from skimage.morphology import binary_closing
from scipy import ndimage
from scipy.ndimage import distance_transform_edt,label,gaussian_filter
from scipy.spatial.transform import RigidTransform
from skimage.graph import route_through_array
import cv2
import os
import torchvision
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import torch
from torchio import ElasticDeformation, RandomElasticDeformation


def create_ground_truth(batch_directory,batch):
   '''
   Currently generating binary images where red indicates the presence of material
   
   :param batch_directory: Description
   :param batch: Description
   '''
   image_stack = []
   for image in os.listdir(batch_directory):
        image_stack.append(cv2.imread(os.path.join(batch_directory,image)))
   #image_averaging(image_stack,batch)
   find_common_structures(image_stack)

def visualize_images(original_images, ground_truth, baseline_output):

    fig = plt.figure(figsize=(16, 8))
    grid_spec = fig.add_gridspec(2, 5, hspace=0.1, wspace=0.1)

    for i in range(len(original_images)):
        ax = fig.add_subplot(grid_spec[0, i])
        orig_img = original_images[i]
        ax.imshow(orig_img)
        ax.axis('off')

    print('Shape of ground truth ',ground_truth.shape)
    #ssim_loss = structural_similarity(baseline_output, ground_truth,channel_axis=-1, data_range = max(baseline_output.max(),ground_truth.max()) - min(baseline_output.min(),ground_truth.min()))
    ssim = StructuralSimilarityIndexMeasure()
    base_image_tensor = torch.tensor(baseline_output).permute(2,0,1).unsqueeze(0)
    ground_truth_tensor = torch.tensor(ground_truth).permute(2,0,1).unsqueeze(0)
    print(base_image_tensor.shape,ground_truth_tensor.shape)
    ssim_metric = ssim(base_image_tensor,base_image_tensor)
    print(ssim_metric)

    ax_center = fig.add_subplot(grid_spec[1, 2])
    ax_center.imshow(baseline_output)
    ax_center.axis('off')
    ax_center.set_title(f'SSIM: {ssim_metric:.2f}')


    ax_left = fig.add_subplot(grid_spec[1, 1])
    ax_left.imshow(ground_truth)
    ax_left.axis('off')
    ax_left.set_title('Ground Truth')

    plt.show()

# def classical_approach(batch_directory,image_stack=None,gt=None,whole_directory = False):
#
#     results = {}
#     if whole_directory:
#         for threshold in [0.55,0.6]:
#             sum_ssim_score =0
#             for batches in os.listdir(batch_directory):
#                 image_stack = []
#                 for image in os.listdir(os.path.join(batch_directory,batches)):
#                     image = cv2.imread(os.path.join(batch_directory, batches,image))
#                     image = cv2.resize(image,(256,256))
#                     image_stack.append(image)
#                 extracted_common_structures = image_averaging(image_stack, threshold)
#                 batch = batches.split(' ')[-1]
#                 ground_truth_image = cv2.imread(os.path.join(f'./data/Ground Truth Images Resized_red/train/gt {batch}.png'))
#                 #Normalize ground truth image
#                 norm_gt_image = np.array(ground_truth_image/255.)
#                 extracted_common_structures = np.array(extracted_common_structures)*255
#                 cv2.imwrite(f'Image_Processing_Results/Threshold_{threshold}/image_{batch}.png',extracted_common_structures[:,:,::-1])
#                 ssim_score = ssim(extracted_common_structures,norm_gt_image,data_range=1.0,channel_axis=-1)
#                 sum_ssim_score +=ssim_score
#                 fig,ax = plt.subplots(2,5,figsize=(16,8))
#                 for i in range(len(image_stack)):
#                     ax[0][i].imshow(image_stack[i][:,:,::-1])
#                     ax[0][i].axis('off')
#
#                 ax[1][2].imshow(ground_truth_image[:,:,::-1])
#                 ax[1][2].axis('off')
#
#                 ax[1][3].imshow(extracted_common_structures)
#                 ax[1][3].set_title(f'SSIM: {ssim_score:.2f}')
#                 ax[1][3].axis('off')
#
#                 for k in [0,1,4]:
#                     ax[1][k].axis('off')
#
#                 plt.savefig(f'./image_processing_results/common_structure {batch}.png')
#                 plt.close()
#             print('The averaged ssim score is', sum_ssim_score/len(os.listdir(batch_directory)))
#             results[threshold] = sum_ssim_score/len(os.listdir(batch_directory))
#
#         plt.plot(results.keys(),results.values())
#         plt.show()
#
#     else:
#         image_stack = image_stack.cpu().permute(0,2,3,1).numpy()
#         extracted_common_structures = image_averaging(image_stack, threshold=0.4, normalised=True)
#         print(extracted_common_structures.shape,gt.shape)
#         ssim_score = ssim(extracted_common_structures, gt.cpu().permute(1,2,0).numpy(), data_range=1.0, channel_axis=-1)
#         print('THE SSIM score seems to be like ',ssim_score)
#         return extracted_common_structures

def image_averaging(raw_images, threshold,normalised=False):

    #Normalising the uint8 values in the RGB images
    if normalised:
        imgs = [np.array(raw_image) for raw_image in raw_images]   # Normalized Images and values range between 0 and 1.
    else:
        imgs = [np.array(raw_image/255.0) for raw_image in raw_images]

        #Aggregation Method - Mean across all the images
    avg_image = np.mean(imgs,axis=0)
    '''cv2.imshow('Avg Image',avg_image)
    cv2.waitKey()
    cv2.destroyAllWindows()'''

    # Threshold mask is a boolean output. The mask has only (H,W) dimensions only.
    #print(avg_image.mean(axis=2) )
    threshold_mask = avg_image.mean(axis=2) > threshold
    #print(threshold_mask.shape)
    # cv2.imshow('Threshold',np.array(threshold_mask).astype(np.uint8)*255)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # Convert it to int (0s and 1s) before visualising
    extracted_structures = np.zeros_like(avg_image)
    white_mask = np.all(avg_image==1,axis=-1)               #Find all non design areas.

    #extracted_structures[threshold_mask == 0] = avg_image[threshold_mask == 0] # To get a original colors 
    extracted_structures[threshold_mask == 0]= [1,0,0]
    extracted_structures[threshold_mask == 1] = [0.51, 0.51, 0.51]
    extracted_structures[white_mask] = [1,1,1]

    #extracted_structures = np.array(extracted_structures) * 255
    # cv2.imshow('Extracted Structures', extracted_structures[:,:,::-1])
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return extracted_structures


def remove_small_regions(binary_image, min_size=3):
        """Remove small disconnected regions"""
        labeled, num_features = label(binary_image)
        
        for region_id in range(1, num_features + 1):
            region = (labeled == region_id)
            if np.sum(region) < min_size:
                binary_image[region] = 0
        
        return binary_image

def find_common_structures(image_stack, method='frequency',threshold=0.7,apply_smoothing=False,filter_radius =1):
    """
        Find common structures across all designs
        
        Parameters:
        -----------
        method : str
            'strict' - Intersection (appears in ALL designs)
            'majority' - Appears in >= threshold fraction of designs
            'weighted' - Weighted average with threshold
            'frequency' - Based on appearance frequency
        threshold : float
            For 'majority' and 'weighted' methods (0.0 to 1.0)
        apply_smoothing : bool
            Apply Gaussian filter to final result
        filter_radius : float
            Smoothing radius
    """
    design_stack = np.stack(image_stack,axis=0)

    #if method == 'strict':
    # Intersection - appears in ALL designs
    print("Using STRICT intersection (all designs must agree)")
    binary_stack = design_stack
    common = np.all(binary_stack, axis=0).astype(float)       #.astype(float) gives red and yellow colour

    cv2.imshow('Common Image - STRICT',common)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


   #elif method == 'majority':
    # Majority voting - appears in >= threshold fraction
    min_count = int(np.ceil(threshold * 5))
    print(f"Using MAJORITY voting (≥{min_count}/{5} designs)")
    binary_stack = design_stack > 0.6
    plt.figure(figsize=(15, 3))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(binary_stack[i].astype(np.uint8), cmap='gray')
        plt.title(f'Image {i} binary')
        plt.axis('off')
    plt.show()

    print(binary_stack.shape)
    count = np.sum(binary_stack, axis=0)
    print(count.shape)
    common = (count >= min_count).astype(float)

    cv2.imshow('Common Image - MAJORITY',common)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

        
    
    #elif method == 'weighted':
    # Weighted average
    print(f"Using WEIGHTED average (threshold={threshold})")
    avg_density = np.mean(design_stack, axis=0)
    common = (avg_density > threshold).astype(float)
    
    cv2.imshow('Common Image - WEIGHTED',common)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

        
    #elif method == 'frequency':
    # Keep elements proportional to frequency of appearance
    print(f"Using FREQUENCY-based (threshold={threshold})")
    binary_stack = design_stack
    frequency = np.mean(binary_stack, axis=0)
    common = (frequency > threshold).astype(float)
    
    #else:
    #    raise ValueError(f"Unknown method: {method}")
    
    cv2.imshow('Common Image - FREQUENCY',common)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Apply smoothing
    if apply_smoothing and filter_radius > 0:
        print(f"Applying Gaussian smoothing (radius={filter_radius})")
        common = gaussian_filter(common, sigma=filter_radius)
        common = (common > 0.5).astype(float)  # Re-binarize
    
    cv2.imshow('After Smoothing',common)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Post-processing: Remove small disconnected regions
    print("Cleaning up small disconnected regions...")
    common = remove_small_regions(common, min_size=1)
    cv2.imshow('After removing small objects',common)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Morphological operations to ensure connectivity
    print("Applying morphological closing...")
    # common = binary_closing(common).astype(float)
    # cv2.imshow('After binary closing',common)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def data_augmentation():
    pass
    # transform = v2.ElasticTransform(alpha=35.0)
    # transform = RandomElasticDeformation(num_control_points=(7,7,7),locked_borders=2)
    # transformed_image = transform(torch.tensor(image).permute(2,0,1).unsqueeze(0))

def analyze_commonality(self):
    """Analyze how features are distributed across designs"""
    
    design_stack = np.stack(self.individual_designs, axis=0)
    binary_stack = (design_stack > 0.5).astype(int)
    
    # Count appearances of each element
    appearance_count = np.sum(binary_stack, axis=0)
    
    print("\n" + "="*60)
    print("COMMONALITY ANALYSIS")
    print("="*60)
    
    n_designs = len(self.individual_designs)
    for count in range(n_designs + 1):
        n_elements = np.sum(appearance_count == count)
        pct = n_elements / appearance_count.size * 100
        
        if count == 0:
            print(f"Appears in {count} designs (void): {n_elements:6d} elements ({pct:5.2f}%)")
        elif count == n_designs:
            print(f"Appears in {count} designs (common): {n_elements:6d} elements ({pct:5.2f}%)")
        else:
            print(f"Appears in {count} designs:         {n_elements:6d} elements ({pct:5.2f}%)")
    
    # Visualize frequency map
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(appearance_count, cmap='YlOrRd', origin='lower', 
                    vmin=0, vmax=n_designs)
    ax.set_title('Feature Frequency Map\n(How many designs have material at each location)')
    ax.set_xlabel('X elements')
    ax.set_ylabel('Y elements')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of designs', rotation=270, labelpad=20)
    cbar.set_ticks(range(n_designs + 1))
    
    plt.tight_layout()
    plt.savefig('frequency_map.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved frequency map: frequency_map.png")
    plt.show()

def image_preprocessing(source_directory,destination_directory,batch):
    for images in os.listdir(os.path.join(source_directory,batch)):
        image = cv2.imread(os.path.join(source_directory,batch, images))
        #image = image[25:435,77:1418] test Batch 3
        #image = image[18:265,17:1482] test Batch 4
        #image = image[29:255,17:1482] test Batch 5,6,7
        #image = image[53:230,17:1482] test Batch 8
        #image = image[10:245,7:254] test Batch 9
        #image = image[10:175,16:1484] test Batch 10
        image = image[20:1000,20:740]
        plt.imshow(image)
        plt.show()
        image = cv2.resize(image, (256, 256))
        #image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_REPLICATE,(255,255,255))
        if not os.path.exists(os.path.join(destination_directory,batch)):
            os.mkdir(os.path.join(destination_directory,batch))
        cv2.imwrite(os.path.join(destination_directory, batch,images), image)

def gt_image_preprocessing(source_directory,destination_directory,batch):
    image = cv2.imread(os.path.join(source_directory,f'gt {batch}.png'))
    #image = image[3:593, 10:530] # Batch 12
    # image = image[:588,:525] #Batch 13
    #image = image[15:880,3:1236] # Batch 14
    #image = image[12:878, 3:1233] # Batch 15
    #image = image[8:598,3:522] #Batch 18
    #image = image[5:288,12:1843] #Batch 36
    #image = image[20:305,8:1841] Batch 37
    #image = image[15:298,10:1843] Batch 38
    #image = image[13:298,3:1843] #Batch 39
    #image = image[18:303, 5:1838] #Batch 40
    #image = image[15:300,5:1838] Batch 41
    #image = image[18:300,12:1843] Batch 42
    #image = image[18:302,5:1841] Batch 43
    #image = image[3:795,:735] Batch 50
    #image = image[3:793,5:738] Batch 51,54
    #image = image[3:793,5:740] #Batch 52,53
    #image = image[3:796,5:743] #Batch 55,56,
    #image = image[12:805, 3:738] Batch 57
    #image = image[3:795,5:745] Batch 58
    #image = image[10:803, 5:738] Batch 59
    #image = image[:,15:1843] Batch 60
    #image = image[:,5:] Batch 61
    #image = image[:,3:1833] #Batch 62,63
    #image = image[:,5:1835] #Batch 64
    #image = image[3:,3:1833] # Batch 65
    #image = image[:,5:1832] Batch 66
    #image = image[:,3:] #Batch 67,68,69
    #image = image[2:,5:1837] Batch 71
    #image = image[1:,1:1830] #Batch 82
    #image = image[:,:1820]# Batch 83
    #image = image[1:203,2:1830] #Batch 84,85,86
    #image = image[2:205, 1:1822]  # Batch 87
    #image = image[2:205,1:1825] #Batch 88
    #image = image[5:209,2:1828]# Batch 89
    #image = image[5:210,:1823] #Batch 90
    #image = image[11:175,85:1736]# test batch 0
    #image = image[9:318,20:1848] #test batch 1
    #image = image[6:315,12:1843] #test batch 2
    #image = image[17:530,92:1750] #test Batch 3
    #image = image[8:313,12:1836] #test Batch 4
    #image = image[24:302,15:1840] #test Batch 5,
    #image = image[22:298, 12:1838]  # test Batch 6
    #image = image[27:307,10:1835] #test Batch 7
    #image = image[37:253,4:1828] #test Batch 8
    image = image[1:294,:] #test Batch 9
    # image = image[10:175,16:1484] test Batch 10
    plt.imshow(image)
    plt.show()
    image = cv2.resize(image, (236, 236))
    image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value =(255, 255, 255))
    #plt.imshow(image)
    #plt.show()
    cv2.imwrite(os.path.join(destination_directory,f'gt {batch}.png'), image)

def image_resizing(source_directory,batch):
    '''i=0
    for images in os.listdir(os.path.join(source_directory, batch)):'''

        #image = cv2.imread(os.path.join(source_directory))
    image  = cv2.imread(source_directory)
    image = cv2.resize(image, (1500, 300), interpolation=cv2.INTER_AREA)
    plt.imshow(image)
    plt.show()
    cv2.imwrite(f'./resized_image.png', image)
    #i = i+1



# def preprocess_gt(image_directory):

    #ground_truth_images = []
    
#     test_transformations = v2.Compose([
#             v2.ToDtype(torch.float32, scale=True),
#             v2.Resize((256,256)),
#             v2.ToTensor(),
#             #v2.Normalize(mean=(0.5,), std= (1.0,)),
#         ])

#     for images in os.listdir(image_directory):
#         image = torchvision.io.decode_image(os.path.join(image_directory,images))
#         image = test_transformations(image)
#         ground_truth_images.append(image)
    
#     print(len(ground_truth_images))
#     '''cv2.imshow('Image',ground_truth_images[0].permute(1, 2, 0).detach().cpu().numpy())
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()'''
    
    #return ground_truth_images

