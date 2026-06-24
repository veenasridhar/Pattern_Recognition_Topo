import collections

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

def input_pipeline(data_directory, ground_truth_dir, image_processing_dir, fold):

    batches_list, items = build_dataset_items(data_directory)

    print('Total Sub Batches ', len(items))
    print("number of views:", len(set(batches_list)))
    print("sub-batches per view:", collections.Counter(batches_list.tolist()))

    train_idx, val_idx, test_idx = grouped_split(batches_list, fold, ratios=[0.7, 0.2, 0.1])

    print('Train Index', train_idx)
    print('Test Index', test_idx)
    print('Val Index', val_idx)

    train_dataset = TopoDatasetImg(items, train_idx, ground_truth_dir, image_processing_dir, augment=True)
    val_dataset = TopoDatasetImg(items, val_idx, ground_truth_dir, image_processing_dir, augment=False)
    test_dataset = TopoDatasetImg(items, test_idx, ground_truth_dir, image_processing_dir, augment=False)

    return train_dataset,val_dataset,test_dataset


def grouped_split(group_of_item, fold, ratios=(0.7, 0.2, 0.1), seed_base=0):
    group_of_item = np.asarray(group_of_item)
    groups = np.unique(group_of_item)
    sizes = {g: int((group_of_item == g).sum()) for g in groups}
    total = len(group_of_item)
    targets = np.array(ratios) * total            # target sample counts: train/test/val [84, 21, 10.5]

    rng = np.random.default_rng(seed_base + fold)  # Range Generator
    order = rng.permutation(groups)             #[ 2 10  3 12  0  4  7  5 16 13 14 11  6  9 17  8  1 15] - different main batch order for  each fold

    assigned = {0: set(), 1: set(), 2: set()}      # 0=train, 1=test, 2=val
    counts = np.zeros(3)
    for g in order:
        deficit = (targets - counts) / targets     # who's most under quota
        s = int(np.argmax(deficit))
        assigned[s].add(g)
        counts[s] += sizes[g]

    idx = lambda s: np.array([i for i, g in enumerate(group_of_item) if g in assigned[s]])
    train_idx, test_idx, val_idx = idx(0), idx(1), idx(2)  # train, val,test

    id_to_group = {0: 'Anowas V4 Bottom', 1: 'Anowas V4 Side View' ,2:'Anowas V4 Top', 3:'Bottom Inside View V3', 4:'Bottom View MW',
                   5: 'Bottom View V3',6: 'Front View V3', 7:'Intermediate Floor V3', 8: 'Jermias Side', 9:'Jermias_bottom', 10: 'Rear View MW',
                   11: 'Side View MW', 12: 'Side View V3', 13: 'Top View V3', 14: 'gen WK Front', 15: 'gen WK Side', 16: 'gen WK Top', 17: 'gn WK Bottom'}
    for name, split_idx in [("train", train_idx), ("test", test_idx), ("val", val_idx)]:
        ids = sorted(set(group_of_item[split_idx].tolist()))
        print(f"{name}: {len(split_idx)} sub-batches across {len(ids)} views")
        for i in ids:
             print(f"    {id_to_group[i]}")
        split_log = {name: [id_to_group[i] for i in sorted(set(group_of_item[idx].tolist()))]
                     for name, idx in [("train", train_idx), ("test", test_idx), ("val", val_idx)]}
    with open(f"split_fold{fold}.json", "w") as f:
        json.dump(split_log, f, indent=2)

    assert not (set(group_of_item[train_idx]) & set(group_of_item[test_idx]))
    assert not (set(group_of_item[train_idx]) & set(group_of_item[val_idx]))
    assert not (set(group_of_item[test_idx]) & set(group_of_item[val_idx]))

    return train_idx, test_idx, val_idx

def permute_views(x):
    return x.index_select(-3, torch.randperm(x.shape[-3]))

class TopoDatasetImg(Dataset):
    def __init__(self, subbatch_dirs, idx, ground_truth_dir, img_proc_dir ,augment):

        self.subbatch_dirs = subbatch_dirs
        self.indices = idx
        self.ground_truth_dir = ground_truth_dir
        self.img_proc_dir = img_proc_dir
        self.augment = augment
        # Image Transformations
        self.train_transformations = v2.Compose([
            v2.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.float32}, scale=True),  # Scale = true already scales the images between 0,1.
            v2.RandomHorizontalFlip(p=0.5),
            v2.Lambda(permute_views, tv_tensors.Image),
            #v2.Resize((256, 256)),
            #v2.Grayscale(num_output_channels=3),                    # Only for VAE approach
            v2.ToTensor(),
        ])
        self.test_transformations = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            #v2.Resize((256, 256)),
            #v2.Grayscale(num_output_channels=3),                # Only for VAE approach
            v2.ToTensor(),
            # v2.Normalize(mean=(0.5,), std= (1.0,)),
        ])

    def __len__(self):
        return len(self.indices)


    def convert_labels(self,input):

        input = input.clone()
        input[input < 115.0] = 0
        input[input > 200.0] = 2
        input[(input >= 115.0) & (input <= 200.0)] =1  #since floating point precision might vary
        return input.long()

    def __getitem__(self, i):
        index = self.indices[i]
        sub_dir = self.subbatch_dirs[index]
        sub = (sub_dir.split('/')[-1]).split(' ')[-1]
        image_files = sorted(glob.glob(os.path.join(sub_dir, f'*.png')))  # the 5 images
        images = torch.cat([torchvision.io.decode_image(f,torchvision.io.ImageReadMode.GRAY) for f in image_files], dim=0)  # [5,H,W]
        images = tv_tensors.Image(images)
        labels = torchvision.io.decode_image(os.path.join(self.ground_truth_dir, f'gt {sub}.png'),torchvision.io.ImageReadMode.GRAY)
        labels = tv_tensors.Mask(labels)
        avg_output = torchvision.io.decode_image(os.path.join(self.img_proc_dir, f'result_{sub}.png'),torchvision.io.ImageReadMode.GRAY)
        avg_output = tv_tensors.Mask(avg_output)
        if self.augment:
            images, labels,avg_output = self.train_transformations(images,labels,avg_output)
        else:
            images, labels, avg_output = self.test_transformations(images, labels, avg_output)
        print('Labels unique', labels.unique())
        print('Avg Output', avg_output.unique())
        labels = self.convert_labels(labels)
        avg_output = self.convert_labels(avg_output)
        print('Labels unique',labels.unique())
        print('Avg Output',avg_output.unique() )
        #sdf = torch.from_numpy(np.load(os.path.join('data/Final Data/Batches Image Processing', f'sdf_{sub}.npy'))).float()
        return images, labels, avg_output



class TopoDataset(Dataset):
    def __init__(self, subbatch_dirs, idx, ground_truth_dir, img_proc_dir ,augment):

        self.subbatch_dirs = subbatch_dirs
        self.indices = idx
        self.ground_truth_dir = ground_truth_dir
        self.img_proc_dir = img_proc_dir
        self.augment = augment
        # Image Transformations
        self.train_transformations = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),  # Scale = true already scales the images between 0,1.
            #v2.RandomHorizontalFlip(p=0.5),
            #v2.Lambda(permute_views, tv_tensors.Image),
            #v2.Resize((256, 256)),
            v2.Grayscale(num_output_channels=3),                    # Only for VAE approach
            #v2.ToTensor(),
        ])
        self.test_transformations = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            #v2.Resize((256, 256)),
            v2.Grayscale(num_output_channels=3),                # Only for VAE approach
            #v2.ToTensor(),
            # v2.Normalize(mean=(0.5,), std= (1.0,)),
        ])

    def __len__(self):
        return len(self.indices)


    def convert_labels(self,input):

        input = input.clone()
        input[input < 0.4] = 0
        input[input > 0.75] = 2
        input[(input >= 0.4) & (input <=0.75)] =1  #since floating point precision might vary
        return input.long()

    def __getitem__(self, i):
        index = self.indices[i]
        sub_dir = self.subbatch_dirs[index]
        sub = (sub_dir.split('/')[-1]).split(' ')[-1]
        image_files = sorted(glob.glob(os.path.join(sub_dir, f'*.png')))  # the 5 images
        images = torch.stack([torchvision.io.decode_image(f,torchvision.io.ImageReadMode.GRAY) for f in image_files], dim=0)  # [5,H,W]
        #images = tv_tensors.Image(images)
        labels = torchvision.io.decode_image(os.path.join(self.ground_truth_dir, f'gt {sub}.png'),torchvision.io.ImageReadMode.GRAY)
        #labels = tv_tensors.Mask(labels)
        avg_output = torchvision.io.decode_image(os.path.join(self.img_proc_dir, f'result_{sub}.png'),torchvision.io.ImageReadMode.GRAY)
        #avg_output = tv_tensors.Mask(avg_output)
        if self.augment:
            images= self.train_transformations(images)
        else:
            images = self.test_transformations(images)

        labels = self.train_transformations(labels)
        avg_output = self.train_transformations(avg_output)
        print('Labels unique', labels.unique())
        print('Avg Output', avg_output.unique())
        #labels = self.convert_labels(labels)
        #avg_output = self.convert_labels(avg_output)
        # print('Labels unique',labels.unique())
        # print('Avg Output',avg_output.unique() )
        #sdf = torch.from_numpy(np.load(os.path.join('data/Final Data/Batches Image Processing', f'sdf_{sub}.npy'))).float()
        return images, labels, avg_output

class ImageDataset(Dataset):
    def __init__(self, subbatch_dirs, idx,augment):
        self.subbatch_dirs = subbatch_dirs
        self.indices = idx
        self.image_files = []
        for index in idx:  # SAME indices as stage 2
            sub_dir = subbatch_dirs[index]
            self.image_files += sorted(glob.glob(os.path.join(sub_dir, '*.png')))

        self.train_transformations = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Grayscale(num_output_channels=3),
        ])
        self.test_transformations = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Grayscale(num_output_channels=3),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, i):
        img = torchvision.io.decode_image(self.image_files[i], torchvision.io.ImageReadMode.GRAY)  # [1,H,W]
        tf = self.train_transformations
        img = tf(img)  # [3,H,W]
        return img, img


def input_pipeline_vae(data_directory, ground_truth_dir, image_processing_dir, fold, single_image_reconstruction = False):

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
        train_dataset = TopoDataset(items, train_idx, ground_truth_dir, image_processing_dir, augment=True)
        val_dataset = TopoDataset(items, val_idx, ground_truth_dir, image_processing_dir, augment=False)
        test_dataset = TopoDataset(items, test_idx, ground_truth_dir, image_processing_dir, augment=False)

    return train_dataset,val_dataset,test_dataset










