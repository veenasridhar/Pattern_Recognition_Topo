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
import torch.nn.functional as F


def natural_sort_key(s):
    # Splits "Batch 11" into ["Batch ", 11] so sorting is numeric-aware
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]

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
 
        # Base transforms (applied to images, labels, avg_output alike).
        # All random augmentations are done manually in __getitem__ so that
        # group-consistency (flip) vs. per-view (corruption) is explicit.
        self.base_transformations = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),   # scales to [0,1]
            #v2.Grayscale(num_output_channels=3),     # Only for VAE approach
        ])
 
    def __len__(self):
        return len(self.indices)
    
    def to_label_map(self,gray):                     # gray: [..., 1, H, W] uint8
        g = gray.squeeze(-3).long()             # [..., H, W]
        lm = torch.ones_like(g)                 # default: non-design (1)
        lm[g < 64]   = 0                        # void
        lm[g >= 192] = 2                        # structural member
        return lm

    # ------------------------------------------------------------------
    # Augmentations
    # ------------------------------------------------------------------
    def _group_flip(self, images, labels, avg_output):
        """Horizontal flip applied consistently to ALL views + GT + avg output."""
        images = torch.flip(images, dims=[-1])
        return images
 
    def _corrupt_minority_views(self, images):
        """Corrupt 1..max_corrupted_views randomly chosen views with random
        patch masking or Gaussian noise. GT stays unchanged: with <=2 of 5
        views corrupted, the (majority) consensus is preserved.
        images: [N, C, H, W] float in [0,1]."""
        n_views = images.shape[0]
        k = random.randint(1, self.max_corrupted_views)
        view_ids = torch.randperm(n_views)[:k].tolist()
        H, W = images.shape[-2:]
 
        for v_idx in view_ids:
            mode = random.choice(['mask', 'noise'])
            if mode == 'mask':
                for _ in range(random.randint(*self.n_patches)):
                    ph = max(1, int(H * random.uniform(*self.patch_frac)))
                    pw = max(1, int(W * random.uniform(*self.patch_frac)))
                    y = random.randint(0, H - ph)
                    x = random.randint(0, W - pw)
                    # fill 0.0 = delete material (background),
                    # fill 1.0 = add fake material (structural member)
                    fill = random.choice([0.0, 1.0])
                    images[v_idx, :, y:y + ph, x:x + pw] = fill
            else:  # 'noise'
                std = random.uniform(*self.noise_std)
                noise = torch.randn_like(images[v_idx]) * std
                images[v_idx] = (images[v_idx] + noise).clamp(0.0, 1.0)
        return images
 
    # ------------------------------------------------------------------
    def __getitem__(self, i):
        index = self.indices[i]
        sub_dir = self.subbatch_dirs[index]
        print(sub_dir)
        sub = (sub_dir.split('/')[-1]).split(' ')[-1]
 
        img = torchvision.io.decode_image(self.subbatch_dirs[index], torchvision.io.ImageReadMode.GRAY)  # [1,H,W]
       
        image_labels  = self.to_label_map(img)        # [1, H, W] long

        target = image_labels
        
         # ---- encoder input: one-hot, scaled to [-1, 1] ----
        images = F.one_hot(image_labels, num_classes=3)        # [5, H, W, 3]
        images = images.permute(2,0,1).float()      # [5, 3, H, W]
        # if self.augment:
        #     # 1) Group-consistent geometric transform: GT must follow
        #     if random.random() < self.p_flip:
        #         images = self._group_flip(images, labels, avg_output)
 
            # # 3) Minority-view corruption: GT unchanged (majority preserved)
            # if random.random() < self.p_corrupt:
            #     images = self._corrupt_minority_views(images)
 
        return images, target

def input_pipeline_discretevals(data_directory, individual_directory,
                                ground_truth_dir, image_processing_dir, fold,single_image_reconstruction,
                                ratios=(0.7, 0.2, 0.1), seed_base=0):

    # --- multi-image (batched) dir defines the canonical view split ---
    items_m, views_m = build_subbatch_items(data_directory)
    assigned = assign_views(views_m, fold, ratios, seed_base)   # {0:train,1:val,2:test}
    tr_m, va_m, te_m = indices_for_views(views_m, assigned)

    # --- single-image dir reuses the SAME view assignment ---
    items_s, views_s = build_image_items(individual_directory)
    unknown = set(views_s) - set().union(*assigned.values())
    assert not unknown, f"views in individual dir not in batched dir: {unknown}"
    tr_s, va_s, te_s = indices_for_views(views_s, assigned)

    # split log (per fold)
    split_log = {name: sorted(assigned[s]) for name, s in
                 [("train", 0), ("val", 1), ("test", 2)]}
    with open(f"split_fold{fold}.json", "w") as f:
        json.dump(split_log, f, indent=2)
    
    if single_image_reconstruction:
        # single-image datasets (same views, per-image granularity)
        train = ImageDataset(items_s, tr_s, augment=True)
        val = ImageDataset(items_s, va_s, augment=False)
        test = ImageDataset(items_s, te_s, augment=False)
    
    else:
        # multi-image datasets
        train = TopoDataset_Augmented(items_m, tr_m, ground_truth_dir, image_processing_dir, augment=True)
        val   = TopoDataset_Augmented(items_m, va_m, ground_truth_dir, image_processing_dir, augment=False)
        test  = TopoDataset_Augmented(items_m, te_m, ground_truth_dir, image_processing_dir, augment=False)

    return train,val,test