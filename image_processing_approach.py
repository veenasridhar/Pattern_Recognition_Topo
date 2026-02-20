import numpy as np
import scipy
import skimage
from skimage.morphology import binary_closing
from scipy import ndimage
from scipy.ndimage import distance_transform_edt,label,gaussian_filter
from skimage.graph import route_through_array
import cv2
import os
import torchvision
from torchvision.transforms import v2
import matplotlib.pyplot as plt


def best_classical_method(designs):
    """
    Most sophisticated classical approach possible
    """
    
    # Step 1: Extract skeletons (topological essence)
    from skimage.morphology import skeletonize, medial_axis
    skeletons = [skeletonize(d > 0.2) for d in designs]

    for sk in skeletons:
        image = (sk.astype('uint8')) * 255
        cv2.imshow('Skeletonized Image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    
    # Step 2: Use DTW (Dynamic Time Warping) or similar
    # to align structures before comparison
    from scipy.spatial.distance import directed_hausdorff
    
    # Find reference design (median of all pairwise distances)
    distances = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            distances[i,j] = directed_hausdorff(
                np.argwhere(skeletons[i]), 
                np.argwhere(skeletons[j])
            )[0]
    
    reference_idx = np.argmin(np.sum(distances, axis=1))
    reference = skeletons[reference_idx]
    print(reference)
    
    # Step 4: Weighted frequency voting
    stack = np.stack(skeletons, axis=0)
    frequency = np.mean(stack, axis=0)
    
    common = (frequency > 0.2).astype(float)
    image = common * 255
    cv2.imshow('Frequency Image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

     # Step 5: Morphological smoothing
    '''common = ndimage.binary_closing(common, iterations=10).astype(float)
    #image = common * 255
    cv2.imshow('After Binary Closing Image',common)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    common = skimage.morphology.remove_small_objects(common, min_size=10)
    image = common * 255
    cv2.imshow('After Removing small objects',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    
    # Step 6: Dilate skeleton back to thickness
    common = ndimage.binary_dilation(common, iterations=3)
    common = (common.astype('uint8')) * 255
    print(type(common),common.dtype)
    cv2.imshow('Common Image',common)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Step 3: Align all designs to reference
    '''from skimage.registration import phase_cross_correlation
    aligned = []
    for skeleton in skeletons:
        shift = phase_cross_correlation(reference, skeleton)[0]
        aligned.append(scipy.ndimage.shift(skeleton, shift))'''
    '''for align_images in aligned:
        image = (align_images.astype('uint8')) * 255
        cv2.imshow('Aligned Image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''
    
    return common

def create_ground_truth(batch_directory,batch):
   '''
   Currently generating binary images where red indicates the presence of material
   
   :param batch_directory: Description
   :param batch: Description
   '''
   image_stack = []
   for image in os.listdir(batch_directory):
        image_stack.append(cv2.imread(os.path.join(batch_directory,image)))
   image_averaging(image_stack,batch)
   #find_common_structures(image_stack)

def image_averaging(raw_images,batch):
    
    #Normalising the uint8 values in the RGB images 
    imgs = [np.array(raw_image/255.0) for raw_image in raw_images]   # Normalized Images and values range between 0 and 1.
    cv2.imshow('Raw Image',imgs[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    #Aggregation Method - Mean across all the images
    avg_image = np.mean(imgs,axis=0)
    print('Data type of the averaged images',avg_image.dtype)
    print('Shape of averaged iamge',avg_image.shape)
    cv2.imshow('Averaged Images', avg_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    threshold_mask = avg_image.mean(axis=2) > 0.4
    #Why the threshold of 0.4? ====> Because it created the most non-overlapping structures. 
    #Threshold mask is a boolean output. The mask has only (H,W) dimensions only. 
    # Convert it to int (0s and 1s) before visualising
    cv2.imshow('Threshold Mask', (threshold_mask.astype(np.uint8))*255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    extracted_structures = np.zeros_like(avg_image)
    print('Shape of extractedh structures',extracted_structures.shape)
    #extracted_structures[threshold_mask == 0] = avg_image[threshold_mask == 0] # To get a original colors 
    extracted_structures[threshold_mask == 0] = [0,0,1]
    extracted_structures[threshold_mask == 1] = [0.64, 0.64, 0.64]

    cv2.imshow('Extracted Structures', extracted_structures)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def remove_small_regions(binary_image, min_size=3):
        """Remove small disconnected regions"""
        labeled, num_features = label(binary_image)
        
        for region_id in range(1, num_features + 1):
            region = (labeled == region_id)
            if np.sum(region) < min_size:
                binary_image[region] = 0
        
        return binary_image

def find_common_structures(image_stack, method='frequency',threshold=0.6,apply_smoothing=False,filter_radius =1):
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
    binary_stack = design_stack > 0.5
    common = np.all(binary_stack, axis=0).astype(float)       #.astype(float) gives red and yellow colour

    cv2.imshow('Common Image - STRICT',common)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


   #elif method == 'majority':
    # Majority voting - appears in >= threshold fraction
    min_count = int(np.ceil(threshold * 5))
    print(f"Using MAJORITY voting (≥{min_count}/{5} designs)")
    binary_stack = design_stack > 0.5
    count = np.sum(binary_stack, axis=0)
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
    binary_stack = design_stack > 0.5
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

