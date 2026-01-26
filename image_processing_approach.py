import numpy as np

def best_classical_method(designs):
    """
    Most sophisticated classical approach possible
    Still won't beat your VAE, but at least it's trying
    """
    
    # Step 1: Extract skeletons (topological essence)
    from skimage.morphology import skeletonize, medial_axis
    skeletons = [skeletonize(d > 0.5) for d in designs]
    
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
    
    # Step 3: Align all designs to reference
    from skimage.registration import phase_cross_correlation
    aligned = []
    for skeleton in skeletons:
        shift = phase_cross_correlation(reference, skeleton)[0]
        aligned.append(scipy.ndimage.shift(skeleton, shift))
    
    # Step 4: Weighted frequency voting
    stack = np.stack(aligned, axis=0)
    frequency = np.mean(stack, axis=0)
    
    # Step 5: Morphological smoothing
    common = (frequency > 0.6).astype(float)
    common = binary_closing(common, iterations=2)
    common = remove_small_objects(common, min_size=10)
    
    # Step 6: Dilate skeleton back to thickness
    common = binary_dilation(common, iterations=3)
    
    return common