import collections
import random
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import torchvision
import os
import json
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import KFold,train_test_split
import numpy as np
import glob
from torchvision import tv_tensors

def build_dataset_items(data_directory):
    batches_list , items = [], []
    for dirpath, dirnames, filenames in os.walk(data_directory):
        if any(f.endswith(".png") for f in filenames):                              # a leaf that holds data = a sub-batch
            rel = os.path.relpath(dirpath, data_directory).split(os.sep)
            view = os.path.join(*rel[:1])                                           # the main-batch
            items.append(dirpath)                                                   # Items have the full address of all batches from data directory to final leaf batch
            batches_list.append(view)
    uniq = {v: k for k, v in enumerate(sorted(set(batches_list)))}                  #Dictionary of Main batches as keys and a number as the value

    batches_list = np.array([uniq[v] for v in batches_list])                        # Convert the list of batch names (string) into number.
    assert len(items) > 0, f"no .png-containing folders under {data_directory}"
    return batches_list, items



def input_pipeline_discretevals(data_directory, ground_truth_dir, image_processing_dir, fold, single_image_reconstruction = False):

    batches_list, items = build_dataset_items(data_directory)

    print('Total Sub Batches ', len(items))
    print("number of views:", len(set(batches_list)))
    print("sub-batches per view:", collections.Counter(batches_list.tolist()))

    train_idx, val_idx, test_idx = grouped_split(batches_list, fold, ratios=[0.7, 0.2, 0.1])

    print('Train Index', train_idx)
    print('Test Index', test_idx)
    print('Val Index', val_idx)

    if single_image_reconstruction:
        train_dataset = ImageDataset(items, train_idx, augment=True)
        val_dataset = ImageDataset(items, val_idx,augment=False)
        test_dataset = ImageDataset(items, test_idx, augment=False)
    else:
        train_dataset = TopoDataset_Augmented(items, train_idx, ground_truth_dir, image_processing_dir, augment=True)
        val_dataset = TopoDataset_Augmented(items, val_idx, ground_truth_dir, image_processing_dir, augment=False)
        test_dataset = TopoDataset_Augmented(items, test_idx, ground_truth_dir, image_processing_dir, augment=False)

    return train_dataset,val_dataset,test_dataset


class OpenTODatasetdisc(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.ds = hf_dataset
        self.transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.ToTensor(),
        ])
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img = self.ds[idx]["topology"]
        if img.mode != "L":     #Checks if the images are grayscale
            img = img.convert("L")
        image_tensor = self.transform(img)
        print('The shape of the image tensor', image_tensor.shape)
        image_tensor = torch.nn.functional.one_hot(image_tensor.long(), num_classes=3).permute(2, 0, 1).float()
        print('The shape of the image tensor', image_tensor.shape)
        #print('The typeof image', type(image_tensor))
        return image_tensor, image_tensor
