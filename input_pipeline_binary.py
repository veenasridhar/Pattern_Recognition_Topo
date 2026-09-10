import collections
from logging import config
import logging
import random
from xml.parsers.expat import model
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
import torch.nn.functional as F
from torch.utils.data import ConcatDataset

#MODEL = config.get('model', 'CNN')  # Default to CNN if not specified in config
MODEL = 'VAE'  # Default to CNN if not specified in config

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

def natural_sort_key(s):
    # Splits "Batch 11" into ["Batch ", 11] so sorting is numeric-aware
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]

def create_individual_images(image_directory):
    image_groups = []
    if image_directory:
        for sub_dir in sorted(os.listdir(image_directory), key=natural_sort_key):
            for images in sorted(os.listdir(os.path.join(image_directory, sub_dir)), key=natural_sort_key):
                if str(images).endswith('.png'):
                    image_groups.append(os.path.join(image_directory, sub_dir, images))
                    # print(image_group)
    return image_groups


class OpenTOImageDataset(Dataset):
    def __init__(self, openTO_individual_dataset , augment):
        
        self.individual_img_dirs = create_individual_images(openTO_individual_dataset)
        self.augment = augment
        self.train_transformations = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
        ])
    
    def __len__(self):
        return len(self.individual_img_dirs)

    def convert_labels(self,input):
        input = input.clone()
        input[input < 115.0] = 0
        input[input > 200.0] = 1
        input[(input >= 115.0) & (input <= 200.0)] =0  #since floating point precision might vary
        return input

    def __getitem__(self, i):
        img = torchvision.io.decode_image(self.individual_img_dirs[i], torchvision.io.ImageReadMode.GRAY)  # [1,H,W]
        binary = self.convert_labels(img)
        images = binary.long()
        images = F.one_hot(images, num_classes=2)        # [1, H, W, 2]
        images = images.permute(0, 3, 1, 2).squeeze(0).float()      # [1, 2, H, W]
        labels = binary.squeeze(0).long()        
        return images, labels

def build_subbatch_items(data_directory):
    """Multi-image: one item per sub-batch (leaf dir of .png views)."""
    items, view_names = [], []
    for dirpath, _, filenames in os.walk(data_directory):
        if any(f.endswith(".png") for f in filenames):
            rel = os.path.relpath(dirpath, data_directory).split(os.sep)
            items.append(dirpath)
            view_names.append(rel[0])          # main-batch = view name (string)
    assert items, f"no .png-containing folders under {data_directory}"
    return items, view_names

def assign_views(view_names, fold, ratios=(0.7, 0.2, 0.1), seed_base=0):
    """Deterministic per fold. Returns assigned = {0:{views}, 1:{...}, 2:{...}}
    with 0=train, 1=val, 2=test. Balances by sub-batch count of the
    directory it's computed on."""
    view_names = np.asarray(view_names)
    views = np.unique(view_names)                        # sorted -> deterministic
    sizes = {v: int((view_names == v).sum()) for v in views}
    total = len(view_names)
    targets = np.array(ratios) * total

    rng   = np.random.default_rng(seed_base + fold)
    order = rng.permutation(views)

    assigned = {0: set(), 1: set(), 2: set()}
    counts = np.zeros(3)
    for v in order:
        deficit = (targets - counts) / targets
        s = int(np.argmax(deficit))
        assigned[s].add(v)
        counts[s] += sizes[v]
    return assigned

def indices_for_views(view_names, assigned):
    view_names = np.asarray(view_names)
    idx = lambda s: np.array([i for i, v in enumerate(view_names) if v in assigned[s]])
    return idx(0), idx(1), idx(2)                        # train, val, test

def build_image_items(data_directory):
    """Single-image: one item per individual .png, labelled by its view."""
    items, view_names = [], []
    for dirpath, _, filenames in os.walk(data_directory):
        pngs = [f for f in filenames if f.endswith(".png")]
        if not pngs:
            continue
        view = os.path.relpath(dirpath, data_directory).split(os.sep)[0]
        for f in sorted(pngs, key=natural_sort_key):
            items.append(os.path.join(dirpath, f))
            view_names.append(view)
    assert items, f"no .png under {data_directory}"
    return items, view_names


def input_pipeline_discretevals(data_directory, individual_directory,
                                ground_truth_dir, image_processing_dir, fold,single_image_reconstruction,augmented_directory=None,
                                ratios=(0.7, 0.2, 0.1), seed_base=0):

    if single_image_reconstruction:
        # single-image dir reuses the SAME view assignment
        items_s, views_s = build_image_items(individual_directory)
        unknown = set(views_s) - set().union(*assigned.values())
        assert not unknown, f"views in individual dir not in batched dir: {unknown}"
        tr_s, va_s, te_s = indices_for_views(views_s, assigned)

        # split log (per fold)
        split_log = {name: sorted(assigned[s]) for name, s in
                    [("train", 0), ("val", 1), ("test", 2)]}
        with open(f"split_fold{fold}_individual.json", "w") as f:
            json.dump(split_log, f, indent=2)
            
        # single-image datasets (same views, per-image granularity)
        train = ImageDataset(items_s, tr_s, augment=True)
        val = ImageDataset(items_s, va_s, augment=False)
        test = ImageDataset(items_s, te_s, augment=False)
            

    else:
        # multi-image (batched) dir defines the canonical view split
        items_m, views_m = build_subbatch_items(data_directory)
        assigned = assign_views(views_m, fold, ratios, seed_base)   # {0:train,1:val,2:test}
        tr_m, va_m, te_m = indices_for_views(views_m, assigned)

        # split log (per fold)
        split_log = {name: sorted(assigned[s]) for name, s in
                    [("train", 0), ("val", 1), ("test", 2)]}
        with open(f"split_fold{fold}.json", "w") as f:
            json.dump(split_log, f, indent=2)

        # multi-image datasets
        train = TopoDataset_Augmented(items_m, tr_m, ground_truth_dir, image_processing_dir, augment=True)
        val   = TopoDataset_Augmented(items_m, va_m, ground_truth_dir, image_processing_dir, augment=False)
        test  = TopoDataset_Augmented(items_m, te_m, ground_truth_dir, image_processing_dir, augment=False)

         # ---- add offline-augmented batches to TRAIN ONLY ----
        if augmented_directory is not None:
            items_a, views_a = build_subbatch_items(augmented_directory)

            # same view assignment -> augmented copy of a val/test batch
            # can NEVER end up in train
            unknown_a = set(views_a) - set().union(*assigned.values())
            assert not unknown_a, f"views in augmented dir not in batched dir: {unknown_a}"

            tr_a, _, _ = indices_for_views(views_a, assigned)   # keep only train views

            train_aug = TopoDataset_Augmented(
                items_a, tr_a,
                ground_truth_dir,
                image_processing_dir,
                augment=True)

            train = ConcatDataset([train, train_aug])   # val/test untouched

    return train,val,test

class ImageDataset(Dataset):
    def __init__(self, subbatch_dirs, idx, augment,
                 p_flip=0.5,            # prob of group-consistent horizontal flip (GT flipped too)
                 p_corrupt=0.5,         # prob of corrupting a minority of views (GT unchanged)
                 max_corrupted_views=2, # never corrupt more than 2 of 5 -> majority stays intact
                 n_patches=(1, 3),      # patches per corrupted view (min, max)
                 patch_frac=(0.08, 0.25),  # patch side length as fraction of H/W (min, max)
                 noise_std=(0.05, 0.20)):  # Gaussian noise strength range
        self.subbatch_dirs = subbatch_dirs
        self.indices = idx
        self.augment = augment
 
        self.p_flip = p_flip
        self.p_corrupt = p_corrupt
        self.max_corrupted_views = max_corrupted_views
        self.n_patches = n_patches
        self.patch_frac = patch_frac
        self.noise_std = noise_std
 
    def __len__(self):
        return len(self.indices)
    
    def to_label_map(self,gray):                    
        g = gray.squeeze(-3).long()             
        lm = torch.zeros_like(g)                 # default: non-design (0)
        lm[g < 64]   = 0                         # void
        lm[g >= 192] = 1                         # structural member
        return lm
    
    def _group_flip(self, images, labels, avg_output):
        """Horizontal flip applied consistently to ALL views + GT + avg output."""
        images = torch.flip(images, dims=[-1])
        return images
 
    # ------------------------------------------------------------------
    def __getitem__(self, i):
        index = self.indices[i]
        sub_dir = self.subbatch_dirs[index]
        sub = (sub_dir.split('/')[-1]).split(' ')[-1]
 
        img = torchvision.io.decode_image(self.subbatch_dirs[index], torchvision.io.ImageReadMode.GRAY)  # [1,H,W]
       
        image_labels  = self.to_label_map(img)        # [1, H, W] long

        target = image_labels
        
         # ---- encoder input: one-hot, scaled to [-1, 1] ----
        images = F.one_hot(image_labels, num_classes=2)        # [5, H, W, 3]
        images = images.permute(2,0,1).float()      # [5, 3, H, W]
        if self.augment:
            # 1) Group-consistent geometric transform: GT must follow
            if random.random() < self.p_flip:
                images = self._group_flip(images, target, avg_output)
 
        return images, target

    
class TopoDataset_Augmented(Dataset):
    def __init__(self, subbatch_dirs, idx, ground_truth_dir, img_proc_dir , augment,
                 p_flip=0.5,            # prob of group-consistent horizontal flip (GT flipped too)
                 p_permute=0.5,):         # prob of shuffling view order (GT unchanged)
                #  p_corrupt=0.5,         # prob of corrupting a minority of views (GT unchanged)
                #  max_corrupted_views=2, # never corrupt more than 2 of 5 -> majority stays intact
                #  n_patches=(1, 3),      # patches per corrupted view (min, max)
                #  patch_frac=(0.08, 0.25),  # patch side length as fraction of H/W (min, max)
                #  noise_std=(0.05, 0.20)):  # Gaussian noise strength range
        self.subbatch_dirs = subbatch_dirs
        self.indices = idx
        self.ground_truth_dir = ground_truth_dir
        self.img_proc_dir = img_proc_dir
        self.augment = augment
        self.model = MODEL
 
        self.p_flip = p_flip
        self.p_permute = p_permute
        # self.p_corrupt = p_corrupt
        # self.max_corrupted_views = max_corrupted_views
        # self.n_patches = n_patches
        # self.patch_frac = patch_frac
        # self.noise_std = noise_std
 
        # Base transforms (applied to images, labels, avg_output alike).
        # All random augmentations are done manually in __getitem__ so that
        # group-consistency (flip) vs. per-view (corruption) is explicit.
        self.base_transformations = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),   # scales to [0,1]
            #v2.Grayscale(num_output_channels=3),     # Only for VAE approach
        ])
 
    def __len__(self):
        return len(self.indices)
 
    def convert_labels(self, input):
        input = input.clone()
        input[input < 0.4] = 0
        input[input > 0.75] = 1
        input[(input >= 0.4) & (input <= 0.75)] = 1
        return input.long()
    
    def to_label_map(self,gray):                     # gray: [..., 1, H, W] uint8
        g = gray.squeeze(-3).long()             # [..., H, W]
        lm = torch.zeros_like(g)                 # default: non-design (0)
        lm[g < 64]   = 0                        # void
        lm[g >= 192] = 1                        # structural member
        return lm

    # Augmentations
    def _group_flip(self, images, labels, avg_output):
        """Horizontal flip applied consistently to ALL views + GT + avg output."""
        images = torch.flip(images, dims=[-1])
        labels = torch.flip(labels, dims=[-1])
        avg_output = torch.flip(avg_output, dims=[-1])
        return images, labels, avg_output
 
    def _permute_views(self, images):
        """Shuffle the order of the 5 views. Consensus is order-invariant,
        so GT stays unchanged."""
        perm = torch.randperm(images.shape[0])
        return images[perm]

    # def permute_views(x):
    #     return x.index_select(-3, torch.randperm(x.shape[-3]))
 
    # ------------------------------------------------------------------
    def __getitem__(self, i):
        index = self.indices[i]
        sub_dir = self.subbatch_dirs[index]
        sub = (sub_dir.split('/')[-1]).split(' ')[-1]
 
        image_files = sorted(glob.glob(os.path.join(sub_dir, '*.png')))  # the 5 images
        if self.model == 'VAE':
            images = torch.stack(
                [torchvision.io.decode_image(f, torchvision.io.ImageReadMode.GRAY)
                for f in image_files], dim=0)  # [5, 1, H, W]
        elif self.model == 'CNN':
            images = torch.cat(
                [torchvision.io.decode_image(f, torchvision.io.ImageReadMode.GRAY)
                for f in image_files], dim=0)  # [5, H, W]
        labels = torchvision.io.decode_image(
            os.path.join(self.ground_truth_dir, f'gt {sub}.png'),
            torchvision.io.ImageReadMode.GRAY)
        avg_output = torchvision.io.decode_image(
            os.path.join(self.img_proc_dir, f'result_{sub}.png'),
            torchvision.io.ImageReadMode.GRAY)
        
        if self.augment:
            # 1) Group-consistent geometric transform: GT must follow
            if random.random() < self.p_flip:
                images, labels, avg_output = self._group_flip(images, labels, avg_output)
 
            # 2) View permutation: GT unchanged (consensus is a set operation)
            if random.random() < self.p_permute:
                images = self._permute_views(images)
 
        image_labels  = self.to_label_map(images)        # [5, H, W] long
        label_map    = self.to_label_map(labels)         # [H, W]    long
        avg_label    = self.to_label_map(avg_output)     # [H, W]    long

        if self.model == 'CNN':
            images = image_labels.float()
        elif self.model == 'VAE':
            images = F.one_hot(image_labels, num_classes=2)        # [5, H, W, 2]
            images = images.permute(0, 3, 1, 2).float()      # [5, 2, H, W]
    
        return images, label_map, avg_label

def create_image_groups(image_directory):
    image_groups = []
    if image_directory:
        for sub_dir in sorted(os.listdir(image_directory), key=natural_sort_key):
            image_group = []
            for images in sorted(os.listdir(os.path.join(image_directory, sub_dir)), key=natural_sort_key):
                if str(images).endswith('.png'):
                    image_group.append(os.path.join(image_directory, sub_dir, images))
                    # print(image_group)
            image_groups.append(image_group)
        # print(image_groups)
    return image_groups


class OpenTOConsensusDatasetVAEbinary(Dataset):
    def __init__(self,image_groups, ground_truth_dir, img_prc_dir, augment):
        self.subbatch_dirs = image_groups
        self.ground_truth_dir = ground_truth_dir
        self.img_proc_dir = img_prc_dir
        self.augment = augment
        self.base_transformations = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),   # scales to [0,1]
            v2.Grayscale(num_output_channels=3),     # Only for VAE approach
        ])
    
    def __len__(self):
        return len(self.subbatch_dirs)
    
    def to_label_map(self,gray):                     # gray: [..., 1, H, W] uint8
            g = gray.squeeze(-3).long()             # [..., H, W]
            lm = torch.zeros_like(g)                 # default: non-design (1)
            lm[g < 64]   = 0                        # void
            lm[g >= 192] = 1                        # structural member
            return lm

    def __getitem__(self, i):
        sub_dir = self.subbatch_dirs[i]
        #print(sub_dir)
        sub = (sub_dir[0].split('/')[-2]).split(' ')[-1]

        #image_files = sorted(glob.glob(os.path.join(sub_dir, '*.png')))  # the 5 images
        if model == 'VAE':
            images = torch.stack(
                [torchvision.io.decode_image(f, torchvision.io.ImageReadMode.GRAY)
                 for f in sub_dir], dim=0)  # [5, 1, H, W]
        elif model == 'CNN':
            images = torch.cat(
                    [torchvision.io.decode_image(f, torchvision.io.ImageReadMode.GRAY)
                     for f in sub_dir], dim=0)  # [5, H, W]
        labels = torchvision.io.decode_image(
            os.path.join(self.ground_truth_dir, f'gt {sub}.png'),
            torchvision.io.ImageReadMode.GRAY)
        avg_output = torchvision.io.decode_image(
            os.path.join(self.img_proc_dir, f'result_{sub}.png'),
            torchvision.io.ImageReadMode.GRAY)
        
        #print('TO Dataset labels',labels.unique())
        
        image_labels  = self.to_label_map(images)        # [5, H, W] long
        label_map    = self.to_label_map(labels)                 # [H, W]    long
        avg_label    = self.to_label_map(avg_output)             # [H, W]    long

        images = image_labels
        
        #---- encoder input: one-hot, scaled to [-1, 1] ----
        images = F.one_hot(image_labels, num_classes=2)        # [5, H, W, 2]
        images = images.permute(0, 3, 1, 2).float()      # [5, 2, H, W]
        
        return images, label_map , avg_label

class TopoDatasetImg(Dataset):
    def __init__(self, subbatch_dirs, idx, ground_truth_dir, img_proc_dir ,augment):

        self.subbatch_dirs = subbatch_dirs
        self.indices = idx
        self.ground_truth_dir = ground_truth_dir
        self.img_proc_dir = img_proc_dir
        self.augment = augment

        # Image Transformations
        self.train_transformations = v2.Compose([
            #v2.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.float32}, scale=True),  # Scale = true already scales the images between 0,1.
            v2.RandomHorizontalFlip(p=0.5),
            v2.Lambda(permute_views, tv_tensors.Image),
            v2.ToTensor(),
        ])
        self.test_transformations = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.ToTensor(),
        ])

    def __len__(self):
        return len(self.indices)

    def convert_labels(self,input):
        input = input.clone()
        input[input < 115.0] = 0
        input[input > 200.0] = 1
        input[(input >= 115.0) & (input <= 200.0)] = 0  #since floating point precision might vary
        return input.long()

    def __getitem__(self, i):
        index = self.indices[i]
        sub_dir = self.subbatch_dirs[index]
        sub = (sub_dir.split('/')[-1]).split(' ')[-1]
        image_files = sorted(glob.glob(os.path.join(sub_dir, f'*.png')))  # the 5 images
       
        images = torch.cat([self.convert_labels(torchvision.io.decode_image(f,torchvision.io.ImageReadMode.GRAY)) for f in image_files], dim=0)  # [5,H,W]
        #images = tv_tensors.Image(images)
        labels = torchvision.io.decode_image(os.path.join(self.ground_truth_dir, f'gt {sub}.png'),torchvision.io.ImageReadMode.GRAY)
        #labels = tv_tensors.Mask(labels)
        avg_output = torchvision.io.decode_image(os.path.join(self.img_proc_dir, f'result_{sub}.png'),torchvision.io.ImageReadMode.GRAY)
        #avg_output = tv_tensors.Mask(avg_output)
        if self.augment:
            images, labels,avg_output = self.train_transformations(images,labels,avg_output)
        else:
            images, labels, avg_output = self.test_transformations(images, labels, avg_output)
       
        images = images.float()
        logger.info('In TopoDatasetImg The unique values in images %d', images.unique())
        logger.info('In TopoDatasetImg The unique values in labels %d', labels.unique())
        logger.info('In TopoDatasetImg The shape of images %s', images.shape)
        labels = self.convert_labels(labels)
        avg_output = self.convert_labels(avg_output)
        
        #sdf = torch.from_numpy(np.load(os.path.join('data/Final Data/Batches Image Processing', f'sdf_{sub}.npy'))).float()
        return images, labels, avg_output



# def cnn_input_pipeline():
#         if dataset == 'RailwayTO':
#             batches_list, items = build_dataset_items(data_dir)                   # batches_list is the list of batches as main batch indices [7,7,11,11,11,11,....]
                                        
#             # build augmented items ONCE (same main-batch ids as batches_list)
#             aug_batches_list, aug_items = build_dataset_items(aug_data_dir)
            
#             train_idx, val_idx, test_idx = grouped_split(batches_list, fold=fold, ratios=[train_ratio, val_ratio, test_ratio])

#             # which MAIN BATCHES are in train this fold
#             train_groups = set(np.asarray(batches_list)[train_idx].tolist())

#             # augmented sub-batches whose main batch is a train batch -> train only
#             aug_train_idx = np.array([i for i, g in enumerate(aug_batches_list) if g in train_groups])

#             # safety: no augmented val/test batch can leak into train
#             val_groups  = set(np.asarray(batches_list)[val_idx].tolist())
#             test_groups = set(np.asarray(batches_list)[test_idx].tolist())
#             assert not (set(aug_batches_list[aug_train_idx].tolist()) & (val_groups | test_groups)), \
#                 "augmented train batches overlap val/test!"

#             train_dataset = TopoDatasetImg(items,train_idx,ground_truth_dir= ground_truth_dir, img_proc_dir = image_processing_dir, augment= True)
#             val_dataset = TopoDatasetImg(items,val_idx,ground_truth_dir= ground_truth_dir, img_proc_dir = image_processing_dir, augment=False)
#             test_dataset = TopoDatasetImg(items,test_idx,ground_truth_dir= ground_truth_dir, img_proc_dir = image_processing_dir,augment=False)

#             # augmented dataset -> TRAIN ONLY, points at augmented GT/result dirs
#             aug_train_dataset = TopoDatasetImg(aug_items, aug_train_idx,
#                                             ground_truth_dir=ground_truth_dir,
#                                             img_proc_dir=image_processing_dir,
#                                             augment=True)
            
#             train_combined = ConcatDataset([train_dataset,aug_train_dataset])
                             
                        
#         elif dataset == 'OpenTO Consenus':

#             opento_dataset_dir = create_image_groups(opento_batches_dir)
            
#             openTO_train_dataset = OpenTOConsensusDatasetbinary(opento_dataset_dir,opento_gt_dir,opento_img_prc_dir, augment = False)

#             dataset_size = len(openTO_train_dataset)
#             train_size = int(train_ratio * dataset_size)
#             val_size = int(val_ratio * dataset_size)
#             test_size = dataset_size - train_size - val_size

#             # Reproducible split
#             generator = torch.Generator().manual_seed(42)

#             train_dataset, val_dataset, test_dataset = random_split(openTO_train_dataset, [train_size, val_size, test_size], generator=generator)
            
        
#         train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
#         val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=4,pin_memory=True)
#         test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=4,pin_memory=True)



# class OpenTOConsensusDatasetbinary(Dataset):                            ## For CNN Architecture
#     def __init__(self,image_groups, ground_truth_dir, img_prc_dir, augment):
#         self.subbatch_dirs = image_groups
#         self.ground_truth_dir = ground_truth_dir
#         self.img_proc_dir = img_prc_dir
#         self.augment = augment
#         self.base_transformations = v2.Compose([
#             v2.ToDtype(torch.float32, scale=True),   # scales to [0,1]
#             v2.Grayscale(num_output_channels=3),     # Only for VAE approach
#         ])
    
#     def __len__(self):
#         return len(self.subbatch_dirs)
    
#     def to_label_map(self,gray):                     # gray: [..., 1, H, W] uint8
#             g = gray.squeeze(-3).long()             # [..., H, W]
#             lm = torch.ones_like(g)                 # default: non-design (1)
#             lm[g < 64]   = 0                        # void
#             lm[g >= 192] = 1                        # structural member
#             return lm

#     def __getitem__(self, i):
#         sub_dir = self.subbatch_dirs[i]
#         print(sub_dir)
#         sub = (sub_dir[0].split('/')[-2]).split(' ')[-1]
#         print(sub)
#         #sub = i

#         print(sub_dir, sub)
#         #image_files = sorted(glob.glob(os.path.join(sub_dir, '*.png')))  # the 5 images
#         images = torch.cat(
#             [torchvision.io.decode_image(f, torchvision.io.ImageReadMode.GRAY)
#              for f in sub_dir], dim=0)  # [5, 1, H, W]
#         labels = torchvision.io.decode_image(
#             os.path.join(self.ground_truth_dir, f'gt {sub}.png'),
#             torchvision.io.ImageReadMode.GRAY)
#         avg_output = torchvision.io.decode_image(
#             os.path.join(self.img_proc_dir, f'result_{sub}.png'),
#             torchvision.io.ImageReadMode.GRAY)
        
#         print('TO Dataset labels',labels.unique())
        
#         image_labels  = self.to_label_map(images).float()        # [5, H, W] long
#         label_map    = self.to_label_map(labels)                 # [H, W]    long
#         avg_label    = self.to_label_map(avg_output)             # [H, W]    long

#         #images = image_labels
        
#         # ---- encoder input: one-hot, scaled to [-1, 1] ----
#         #images = F.one_hot(image_labels, num_classes=3)        # [5, H, W, 3]
#         #images = images.permute(0, 3, 1, 2).float()      # [5, 3, H, W]
#         # print('The shape of the images', images.shape)
#         # print('The type of the images', images.dtype)
        
#         # Deterministic base transforms for everything
#         # images = self.base_transformations(images)
#         # labels = self.base_transformations(labels)
#         # avg_output = self.base_transformations(avg_output)
 
#         # if self.augment:
#         #     # 1) Group-consistent geometric transform: GT must follow
#         #     if random.random() < self.p_flip:
#         #         images, labels, avg_output = self._group_flip(images, labels, avg_output)
 
#         #     # 2) View permutation: GT unchanged (consensus is a set operation)
#         #     if random.random() < self.p_permute:
#         #         images = self._permute_views(images)
 
#         #     # 3) Minority-view corruption: GT unchanged (majority preserved)
#         #     if random.random() < self.p_corrupt:
#         #         images = self._corrupt_minority_views(images)
 
#         return image_labels, label_map , avg_label

# def grouped_split(group_of_item, fold, ratios=(0.7, 0.2, 0.1), seed_base=0):
#     group_of_item = np.asarray(group_of_item)
#     groups = np.unique(group_of_item)
#     sizes = {g: int((group_of_item == g).sum()) for g in groups}
#     total = len(group_of_item)
#     targets = np.array(ratios) * total            # target sample counts: train/test/val [84, 21, 10.5]

#     rng = np.random.default_rng(seed_base + fold)  # Range Generator
#     order = rng.permutation(groups)             #[ 2 10  3 12  0  4  7  5 16 13 14 11  6  9 17  8  1 15] - different main batch order for  each fold

#     assigned = {0: set(), 1: set(), 2: set()}      # 0=train, 1=test, 2=val
#     counts = np.zeros(3)
#     for g in order:
#         deficit = (targets - counts) / targets     # who's most under quota
#         s = int(np.argmax(deficit))
#         assigned[s].add(g)
#         counts[s] += sizes[g]

#     idx = lambda s: np.array([i for i, g in enumerate(group_of_item) if g in assigned[s]])
#     train_idx, test_idx, val_idx = idx(0), idx(1), idx(2)  # train, val,test

#     id_to_group = {0: 'Anowas V4 Bottom', 1: 'Anowas V4 Side View' ,2:'Anowas V4 Top', 3:'Bottom Inside View V3', 4:'Bottom View MW',
#                    5: 'Bottom View V3',6: 'Front View V3', 7:'Intermediate Floor V3', 8: 'Jermias Side', 9:'Jermias_bottom', 10: 'Rear View MW',
#                    11: 'Side View MW', 12: 'Side View V3', 13: 'Top View V3', 14: 'gen WK Front', 15: 'gen WK Side', 16: 'gen WK Top', 17: 'gn WK Bottom'}
#     for name, split_idx in [("train", train_idx), ("test", test_idx), ("val", val_idx)]:
#         ids = sorted(set(group_of_item[split_idx].tolist()))
#         print(f"{name}: {len(split_idx)} sub-batches across {len(ids)} views")
#         for i in ids:
#              print(f"    {id_to_group[i]}")
#         split_log = {name: [id_to_group[i] for i in sorted(set(group_of_item[idx].tolist()))]
#                      for name, idx in [("train", train_idx), ("test", test_idx), ("val", val_idx)]}
#     with open(f"split_fold{fold}.json", "w") as f:
#         json.dump(split_log, f, indent=2)

#     assert not (set(group_of_item[train_idx]) & set(group_of_item[test_idx]))
#     assert not (set(group_of_item[train_idx]) & set(group_of_item[val_idx]))
#     assert not (set(group_of_item[test_idx]) & set(group_of_item[val_idx]))

#     return train_idx, test_idx, val_idx





























