from dreamsim import dreamsim
import cv2
import torch
from PIL import Image
import numpy as np

def dreamsim_metric(gt,extracted_structures):

    model, preprocess = dreamsim(pretrained=True,dreamsim_type='dino_vitb16',device='cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    gt = (gt * 255).astype(np.uint8)
    gt = Image.fromarray(gt)
    extracted_structures = (extracted_structures * 255).astype(np.uint8)
    extracted_structures = Image.fromarray(extracted_structures)

    preprocessed_gt = preprocess(gt)
    preprocessed_gt = preprocessed_gt.to(device)
    preprocessed_extracted_structures = preprocess(extracted_structures)
    preprocessed_extracted_structures = preprocessed_extracted_structures.to(device)
    print(model.device)
    print(preprocessed_extracted_structures.device)
    print(preprocessed_gt.device)
    dist = model(preprocessed_gt, preprocessed_extracted_structures)
    return dist

# gt = Image.open('data/Ground Truth Images Resized_red/train/gt 2.png')
# extracted = Image.open('Image_Processing_Results/Threshold_0.3/image_2.png')
# extracted2 = Image.open('Image_Processing_Results/Threshold_0.4/image_2.png')
# extracted3 = Image.open('Image_Processing_Results/Threshold_0.45/image_2.png')
# predicted = Image.open('output_images_0/output_images_0_0_0.png')
# print('Threshold: 0.3', dreamsim_metric(gt,extracted))
# print('Threshold: 0.4', dreamsim_metric(gt,extracted2))
# print('Threshold: 0.45', dreamsim_metric(gt,extracted3))
# print('Predicted', dreamsim_metric(gt,predicted))

