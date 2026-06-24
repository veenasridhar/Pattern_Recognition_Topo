import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np

pool = nn.Sequential(torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
)

def downsample_div8(a, m=8):
    # a: 2D numpy array (H, W) in {0,128,255}
    t = torch.from_numpy(a).float().permute(2,0,1)[None]   # [1,1,H,W]
    print(t.shape)
    t = pool(t)                                  # plain 2x2 max-pool, aspect preserved
    print(t.shape)
    H, W = t.shape[-2:]
    ph, pw = (-H) % m, (-W) % m
    t = F.pad(t, (pw//2, pw-pw//2, ph//2, ph-ph//2), value=127)         # 0 = void, reach multiple of m
    return t.squeeze(0).permute(1,2,0).numpy()

main_dir = 'data/Bigger Batches of Data_original_size'
for batches in os.listdir(main_dir):
    for subbatch in os.listdir(os.path.join(main_dir, batches)):
        for imagepath in os.listdir(os.path.join(main_dir, batches,subbatch)):
            image = cv2.imread(os.path.join(main_dir,batches,subbatch,imagepath))
            print(os.path.join(main_dir,batches,imagepath))
            print(image.shape)
            resized_image = downsample_div8(image) /255.0
            print(np.unique(resized_image,return_counts=True))
            if not os.path.isdir(f'data/Batches/{batches}'):
                os.mkdir(f'data/Batches/{batches}')
            if not os.path.isdir(f'data/Batches/{batches}/{subbatch}'):
                os.mkdir(f'data/Batches/{batches}/{subbatch}')
            plt.imsave(os.path.join(f'data/Batches/{batches}/{subbatch}',imagepath), resized_image)
            plt.close()

# main_dir = 'data/Edited Ground Truth_ Encoded'
# for imagepath in os.listdir(main_dir):
#     image = cv2.imread(main_dir+'/'+imagepath)
#     resized_image = downsample_div8(image) / 255.0
#     print(np.unique(resized_image, return_counts=True))
#     plt.imsave(os.path.join(f'data/Ground_Truth_resized', imagepath), resized_image)
#     plt.close()