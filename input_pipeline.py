import cv2
import torch
from numba.core import target_extension
from sentry_sdk.ai.utils import truncate_and_annotate_embedding_inputs
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


class SimpleCustomBatch(Dataset):

    def __init__(self, image_directory,gt_directory, output_file=None, transform=None,target_transform = None):

        self.image_directory = image_directory
        self.gt_directory = gt_directory
        self.transform =transform
        

    # Override the __getitem__() method
    def __getitem__(self,index):

        image_path = os.path.join(self.image_directory,f'image {index}'+'.png')
        image = torchvision.io.decode_image(image_path,torchvision.io.ImageReadMode.RGB)
        if self.transform:
            image = self.transform(image)
        if self.gt_directory:
            gt_image = torchvision.io.decode_image(os.path.join(self.gt_directory, f'gt {index}.png'),
                                                   torchvision.io.ImageReadMode.RGB)
            gt_image = self.transform(gt_image)

        return image,gt_image

        #Just to check the input images to the model
        '''image2 = image.permute(1,2,0)
        image2 = image2.clip(0,1)
        #print('The shape of the image is ',image2.shape)        
        plt.imsave(f'transformed_train_images\\Img_{index}.png',image2)'''
        #Parallely stack all the images

    def __len__(self):
        #print(len(os.listdir(self.image_directory)))
        return len(os.listdir(self.image_directory))
    

def natural_sort_key(s):
    # Splits "Batch 11" into ["Batch ", 11] so sorting is numeric-aware
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]


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


class PreBatchedImageDataset(Dataset):

    def __init__(self, image_groups, gt_directory, transform):
        self.image_groups = image_groups
        self.gt_directory = gt_directory
        self.transform = transform

    def __getitem__(self, index):

        image_paths = self.image_groups[index]
        if index > 71:
            index += 4
        gt_image = None
        if self.gt_directory:
            gt_image = torchvision.io.decode_image(os.path.join(self.gt_directory, f'gt {index}.png'),
                                                   torchvision.io.ImageReadMode.RGB)
            gt_image = self.transform(gt_image)
        # Array of images instead of a single image
        images = []

        for imagepath in image_paths:
            image = torchvision.io.decode_image(imagepath, torchvision.io.ImageReadMode.RGB)
            if self.transform:
                image = self.transform(image)
            images.append(image)

        return torch.stack(images), gt_image  # This created a array of images (num_images,C,H,W)

    def __len__(self):
        return len(self.image_groups)

def input_pipeline(single_image_reconstruction = False, n_splits=5):
    '''config_file = open('./config.json')
    config_data = json.load(config_file)'''

    # Image Transformations
    train_transformations = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # Scale = true already scales the images between 0,1.
        v2.Resize((256, 256)),
        v2.ToTensor(),
        # v2.Normalize(mean=(0.5,), std= (1.0,)),
    ])

    test_transformations = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((256, 256)),
        v2.ToTensor(),
        # v2.Normalize(mean=(0.5,), std= (1.0,)),
    ])

    if single_image_reconstruction:
        # Create a Dataset
        train_dataset = SimpleCustomBatch(image_directory='data/unbatched_resized/train', gt_directory='data/gt_unbatched_resized/train',transform = train_transformations)
        test_dataset = SimpleCustomBatch(image_directory='data/unbatched_resized/test',gt_directory='data/gt_unbatched_resized/test', transform=test_transformations)
    else:
        train_image_groups = create_image_groups('./data/batched_data_resized_red/train')
        train_dataset = PreBatchedImageDataset(image_groups=train_image_groups,gt_directory='./data/Ground Truth Images Resized_red/train',transform=train_transformations)
        # train_subset = torch.utils.data.Subset(train_dataset,range(10))
        test_image_groups = create_image_groups('./data/batched_data_resized_red/test')
        test_dataset = PreBatchedImageDataset(image_groups=test_image_groups, gt_directory='./data/Ground Truth Images Resized_red/test',transform=test_transformations)

    return train_dataset,test_dataset

    # Elastic Transform check
    '''elastic_transform = v2.ElasticTransform(alpha=50.0)'''

class SimpleCustomBatchGrayscale(Dataset):

    def __init__(self, image_directory, gt_directory, output_file=None, transform=None, target_transform=None):

        self.image_directory = image_directory
        self.gt_directory = gt_directory
        self.transform = transform
        self.target_transform = target_transform

    def convert_labels(self, input):

        input = input.clone()
        input[input == 0] = 0
        input[input == 1] = 2
        input[(input >= 0.25) & (input < 0.75)] = 1  # since floating point precision might vary

        return input.long()

    # Override the __getitem__() method
    def __getitem__(self, index):

        image_path = os.path.join(self.image_directory, f'image {index}' + '.png')
        image = torchvision.io.decode_image(image_path, torchvision.io.ImageReadMode.GRAY)
        if self.transform:
            image = self.transform(image)
        if self.gt_directory:
            gt_image = torchvision.io.decode_image(os.path.join(self.gt_directory, f'image {index}.png'),
                                                   torchvision.io.ImageReadMode.GRAY)
            gt_image = self.target_transform(gt_image)
            #gt_image = self.convert_labels(gt_image)

        return image, gt_image

        # Just to check the input images to the model
        '''image2 = image.permute(1,2,0)
        image2 = image2.clip(0,1)
        #print('The shape of the image is ',image2.shape)        
        plt.imsave(f'transformed_train_images\\Img_{index}.png',image2)'''
        # Parallely stack all the images

    def __len__(self):
        # print(len(os.listdir(self.image_directory)))
        return len(os.listdir(self.image_directory))


def show_image(image, label):
    image_show = image.permute(1, 2, 0)
    plt.figure(figsize=(6, 8))
    plt.title(label)
    plt.imshow(image_show)
    plt.show()

class PreBatchedGrayscaleImageDataset(Dataset):

    def __init__(self, image_groups, gt_directory,image_processing_directory, transform,target_transform=None):
        self.image_groups = image_groups
        self.gt_directory = gt_directory
        self.transform = transform
        self.target_transform = target_transform
        self.image_processing_directory = image_processing_directory

    def convert_labels(self,input):

        input = input.clone()
        input[input == 0.0] = 0
        input[input == 1.0] = 2
        input[(input >= 0.25) & (input < 0.75)] =1  #since floating point precision might vary

        return input.long()

    def __getitem__(self, index):

        image_paths = self.image_groups[index]
        if index > 71:
             index += 4
        gt_image = None
        if self.gt_directory:
            gt_image = torchvision.io.decode_image(os.path.join(self.gt_directory, f'gt {index}.png'),
                                                   torchvision.io.ImageReadMode.GRAY)
            #print(gt_image.unique())
            gt_image = self.target_transform(gt_image)
        gt_image = self.convert_labels(gt_image)
        #print(gt_image.shape)
        # Array of images instead of a single image
        images = []

        image_processing_result = torchvision.io.decode_image(os.path.join(self.image_processing_directory, f'result_{index}.png'),
                                                   torchvision.io.ImageReadMode.GRAY)
        image_processing_result = self.target_transform(image_processing_result)

        for imagepath in image_paths:
            image = torchvision.io.decode_image(imagepath, torchvision.io.ImageReadMode.GRAY)
            if self.transform:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images)                     # For VAE approach, stack images with a extra dimsenion 1 for channel (GRAYSCALE)
        #images = torch.cat(images)                      # For simple conv approach, concatenates images without an extra dimension for a channel
        # perm = torch.randperm(images.size(0))
        # print(perm)
        # images_permuted = images[perm]
        # print(images_permuted.shape)

        return  torch.tensor(images), gt_image, image_processing_result  # This created a array of images (num_images,C,H,W)

    def __len__(self):
        return len(self.image_groups)


def show_image(image, label):
    image_show = image.permute(1, 2, 0)
    plt.figure(figsize=(6, 8))
    plt.title(label)
    plt.imshow(image_show)
    plt.show()


def input_pipeline_grayscale(single_image_reconstruction = False, n_splits=5):
    '''config_file = open('./config.json')
    config_data = json.load(config_file)'''

    # Image Transformations
    train_transformations = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # Scale = true already scales the images between 0,1.
        v2.Resize((256, 256)),
        v2.Grayscale(num_output_channels=3),                    # Only for VAE approach
        v2.ToTensor(),
        # v2.Normalize(mean=(0.5,), std= (1.0,)),
    ])

    target_transformations = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # Scale = true already scales the images between 0,1.
        v2.Resize((256, 256)),
        #v2.Grayscale(num_output_channels=3),
        v2.ToTensor(),
        # v2.Normalize(mean=(0.5,), std= (1.0,)),
    ])

    test_transformations = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((256, 256)),
        v2.Grayscale(num_output_channels=3),                # Only for VAE approach
        v2.ToTensor(),
        # v2.Normalize(mean=(0.5,), std= (1.0,)),
    ])

    if single_image_reconstruction:
        # Create a Dataset
        train_dataset = SimpleCustomBatchGrayscale(image_directory='data/Final Data/unbatched_data_grayscale/train', gt_directory='data/Final Data/unbatched_data_grayscale/train',transform = train_transformations,target_transform=target_transformations)
        test_dataset = SimpleCustomBatchGrayscale(image_directory='data/Final Data/unbatched_data_grayscale/test',gt_directory='data/Final Data/unbatched_data_grayscale/test', transform=test_transformations,target_transform=target_transformations)
    else:
        train_image_groups = create_image_groups('data/Final Data/batched_data_grayscale_resized/train')
        train_dataset = PreBatchedGrayscaleImageDataset(image_groups=train_image_groups,gt_directory='data/Final Data/gt_grayscale_resized/train',image_processing_directory='data/Final Data/Image Processing Results Grayscale/train',transform=train_transformations,target_transform=target_transformations)
        # train_subset = torch.utils.data.Subset(train_dataset,range(10))
        test_image_groups = create_image_groups('data/Final Data/batched_data_grayscale_resized/test')
        test_dataset = PreBatchedGrayscaleImageDataset(image_groups=test_image_groups, gt_directory='data/Final Data/gt_grayscale_resized/test',image_processing_directory='data/Final Data/Image Processing Results Grayscale/test',transform=test_transformations,target_transform=target_transformations)

    return train_dataset,test_dataset


class ResidualGrayscaleImageDataset(Dataset):

    def __init__(self, image_groups, gt_directory, image_processing_directory, sdf_directory, image_processing_sdf_directory, transform, target_transform=None):
        self.image_groups = image_groups
        self.gt_directory = gt_directory
        self.image_processing_directory = image_processing_directory
        self.transform = transform
        self.target_transform = target_transform
        self.sdf_directory = sdf_directory
        self.img_proc_sdf_directory = image_processing_sdf_directory

    def convert_labels(self,input):

        input = input.clone()
        input[input == 0.0] = 0
        input[input == 1.0] = 2
        input[(input >= 0.25) & (input < 0.75)] =1  #since floating point precision might vary
        return input.long()

    def __getitem__(self, index):

        image_paths = self.image_groups[index]
        if index > 71:
             index += 4

        gt_image = None
        if self.gt_directory:
            gt_image = torchvision.io.decode_image(os.path.join(self.gt_directory, f'gt {index}.png'),
                                                   torchvision.io.ImageReadMode.GRAY)
            #print(gt_image.unique())
            gt_image = self.target_transform(gt_image)
        gt_image = self.convert_labels(gt_image)
        #print(gt_image.shape)

        # Array of images instead of a single image
        images = []
        for imagepath in image_paths:
            image = torchvision.io.decode_image(imagepath, torchvision.io.ImageReadMode.GRAY)
            if self.transform:
                image = self.transform(image)
            images.append(image)

        images = torch.cat(images)

        image_preprocessing_result = torchvision.io.decode_image(os.path.join(self.image_processing_directory, f'result_{index}.png'),torchvision.io.ImageReadMode.GRAY)
        image_preprocessing_result = self.target_transform(image_preprocessing_result)
        image_processing_result = self.convert_labels(image_preprocessing_result)


        gt_sdf = np.load(os.path.join(self.sdf_directory, f'gt_{index}.npy'))
        sdf = torch.from_numpy(gt_sdf)

        image_processing_sdf = np.load(os.path.join(self.img_proc_sdf_directory, f'image_{index}.npy'))
        image_processing_sdf = torch.from_numpy(image_processing_sdf)

        #images = torch.cat(images)             #For simple conv approach, concatenates images without an extra dimension for a channel
        #images = torch.stack(images)            #For VAE approach, stack images with a extra dimsenion 1 for channel (GRAYSCALE)
        #print(images.shape)

        return  (torch.tensor(images), gt_image,image_processing_result)
                 #sdf,image_processing_sdf)  # This created a array of images (num_images,C,H,W)

    def __len__(self):
        return len(self.image_groups)


def input_pipeline_residual_grayscale(single_image_reconstruction = False, n_splits=5):
    '''config_file = open('./config.json')
    config_data = json.load(config_file)'''

    # Image Transformations
    train_transformations = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # Scale = true already scales the images between 0,1.
        v2.Resize((256, 256)),
        #v2.Grayscale(num_output_channels=3),                    # Only for VAE approach
        v2.ToTensor(),
        # v2.Normalize(mean=(0.5,), std= (1.0,)),
    ])

    target_transformations = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # Scale = true already scales the images between 0,1.
        v2.Resize((256, 256)),
        v2.ToTensor(),
        # v2.Normalize(mean=(0.5,), std= (1.0,)),
    ])

    test_transformations = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((256, 256)),
        #v2.Grayscale(num_output_channels=3),                # Only for VAE approach
        v2.ToTensor(),
        # v2.Normalize(mean=(0.5,), std= (1.0,)),
    ])

    train_image_groups = create_image_groups('data/Final Data/batched_data_grayscale_resized/train')
    train_dataset = ResidualGrayscaleImageDataset(image_groups=train_image_groups,gt_directory='data/Final Data/gt_grayscale_resized/train',image_processing_directory='data/Final Data/Image Processing Results Grayscale/train',sdf_directory= 'data/Final Data/gt_data_sdf/train',
                                                  image_processing_sdf_directory ='data/Final Data/image_processing_data_sdf/train',transform= train_transformations,target_transform=target_transformations)
    # train_subset = torch.utils.data.Subset(train_dataset,range(10))
    test_image_groups = create_image_groups('data/Final Data/batched_data_grayscale_resized/test')
    test_dataset = ResidualGrayscaleImageDataset(image_groups=test_image_groups, gt_directory='data/Final Data/gt_grayscale_resized/test',image_processing_directory='data/Final Data/Image Processing Results Grayscale/test',sdf_directory= 'data/Final Data/gt_data_sdf/test',
                                                 image_processing_sdf_directory ='data/Final Data/image_processing_data_sdf/test',transform=test_transformations,target_transform=target_transformations)

    return train_dataset,test_dataset

def input_pipeline_original_size(single_image_reconstruction = False, n_splits=5):
    '''config_file = open('./config.json')
    config_data = json.load(config_file)'''

    # Image Transformations
    train_transformations = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # Scale = true already scales the images between 0,1.
        #v2.Resize((256, 256)),
        #v2.Grayscale(num_output_channels=3),                    # Only for VAE approach
        v2.ToTensor(),
        # v2.Normalize(mean=(0.5,), std= (1.0,)),
    ])

    target_transformations = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # Scale = true already scales the images between 0,1.
        #v2.Resize((256, 256)),
        v2.ToTensor(),
        # v2.Normalize(mean=(0.5,), std= (1.0,)),
    ])

    test_transformations = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        #v2.Resize((256, 256)),
        #v2.Grayscale(num_output_channels=3),                # Only for VAE approach
        v2.ToTensor(),
        # v2.Normalize(mean=(0.5,), std= (1.0,)),
    ])

    train_image_groups = create_image_groups('data/Final Data/batched_data_grayscale_original size/train')
    train_dataset = ResidualGrayscaleImageDataset(image_groups=train_image_groups,gt_directory='data/Final Data/gt_grayscale_original size/train',image_processing_directory='data/Final Data/Image Processing Original Size Grayscale/train',sdf_directory= 'data/Final Data/gt_data_sdf/train',
                                                  image_processing_sdf_directory ='data/Final Data/image_processing_data_sdf/train',transform= train_transformations,target_transform=target_transformations)
    # train_subset = torch.utils.data.Subset(train_dataset,range(10))
    test_image_groups = create_image_groups('data/Final Data/batched_data_grayscale_original size/test')
    test_dataset = ResidualGrayscaleImageDataset(image_groups=test_image_groups, gt_directory='data/Final Data/gt_grayscale_original size/test',image_processing_directory='data/Final Data/Image Processing Original Size Grayscale/test',sdf_directory= 'data/Final Data/gt_data_sdf/test',
                                                 image_processing_sdf_directory ='data/Final Data/image_processing_data_sdf/test',transform=test_transformations,target_transform=target_transformations)

    return train_dataset,test_dataset



def natural_sort_key(s):
    # Splits "Batch 11" into ["Batch ", 11] so sorting is numeric-aware
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]

import numpy as np

# def grouped_split(group_of_item, fold, ratios=(0.7, 0.2, 0.1), seed_base=0):
#     group_of_item = np.asarray(group_of_item)
#     groups = np.unique(group_of_item)
#     sizes = {g: int((group_of_item == g).sum()) for g in groups}
#     total = len(group_of_item)
#     targets = np.array(ratios) * total            # target sample counts: train/test/val
#
#     rng = np.random.default_rng(seed_base + fold)  # different order each fold
#     order = rng.permutation(groups)
#
#     assigned = {0: set(), 1: set(), 2: set()}      # 0=train, 1=test, 2=val
#     counts = np.zeros(3)
#     for g in order:
#         deficit = (targets - counts) / targets     # who's most under quota
#         s = int(np.argmax(deficit))
#         assigned[s].add(g)
#         counts[s] += sizes[g]
#
#     idx = lambda s: np.array([i for i, g in enumerate(group_of_item) if g in assigned[s]])
#     return idx(0), idx(1), idx(2)                   # train, test, val


def create_numpy_groups(image_directory):
    image_groups = []
    if image_directory:
        for sub_dir in sorted(os.listdir(image_directory), key=natural_sort_key):
            image_group = []
            for images in sorted(os.listdir(os.path.join(image_directory, sub_dir)), key=natural_sort_key):
                if str(images).endswith('.npy'):
                    image_group.append(os.path.join(image_directory, sub_dir, images))
                    # print(image_group)
            image_groups.append(image_group)
        # print(image_groups)
    return image_groups

import numpy as np

def grouped_split(group_of_item, fold, ratios=(0.7, 0.2, 0.1), seed_base=0):
    group_of_item = np.asarray(group_of_item)
    print('group_of_item',group_of_item)
    groups = np.unique(group_of_item)
    print('Groups:',groups)
    sizes = {g: int((group_of_item == g).sum()) for g in groups}
    print('Sizes: ',sizes)
    total = len(group_of_item)
    print('Total: ',total)
    targets = np.array(ratios) * total            # target sample counts: train/test/val
    print('Target Ratios',targets)

    rng = np.random.default_rng(seed_base + fold)  # different order each fold
    print ('Range',rng)
    order = rng.permutation(groups)
    print('Order', order)

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

class SDFDataset(Dataset):

    def __init__(self, image_groups, gt_directory, image_processing_directory, sdf_directory ,transform, target_transform=None):
        self.image_groups = image_groups
        self.gt_directory = gt_directory
        self.image_processing_directory = image_processing_directory
        self.sdf_directory = sdf_directory

    def __getitem__(self, index):

        image_paths = self.image_groups[index]
        if index > 71:
             index += 4
        gt_image = None
        if self.gt_directory:
            gt_image = torch.from_numpy(np.load(os.path.join(self.gt_directory, f'gt_{index}.npy')))
        images = []
        for imagepath in image_paths:
            image = torch.from_numpy(np.load(imagepath))
            images.append(image)
        images = np.stack(images)
        image_preprocessing_result = torch.from_numpy(np.load(os.path.join(self.image_processing_directory, f'image_{index}.npy')))
        #print('Image ', image_preprocessing_result.min(), image_preprocessing_result.max())

        return  torch.tensor(images), gt_image, image_preprocessing_result  # This created a array of images (num_images,C,H,W)

    def __len__(self):
        return len(self.image_groups)


def input_pipeline_sdf(fold):
    '''config_file = open('./config.json')
    config_data = json.load(config_file)'''

    # Image Transformations
    train_transformations = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),                  # Scale = true already scales the images between 0,1.
        v2.Resize((256, 256)),
        #v2.Grayscale(num_output_channels=3),                    # Only for VAE approach
        v2.ToTensor(),
        # v2.Normalize(mean=(0.5,), std= (1.0,)),
    ])

    target_transformations = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # Scale = true already scales the images between 0,1.
        #v2.Resize((256, 256)),
        v2.ToTensor(),
        # v2.Normalize(mean=(0.5,), std= (1.0,)),
    ])

    test_transformations = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((256, 256)),
        #v2.Grayscale(num_output_channels=3),                # Only for VAE approach
        v2.ToTensor(),
        # v2.Normalize(mean=(0.5,), std= (1.0,)),
    ])


    train_image_groups = create_numpy_groups('data/Final Data/batched_data_sdf/train')
    train_dataset = SDFDataset(image_groups=train_image_groups,gt_directory='data/Final Data/gt_data_sdf/train',image_processing_directory='data/Final Data/image_processing_data_sdf/train',sdf_directory= 'data/Final Data/sdf_dir/train',transform= train_transformations,target_transform=target_transformations)
    # train_subset = torch.utils.data.Subset(train_dataset,range(10))
    test_image_groups = create_numpy_groups('data/Final Data/batched_data_sdf/test')
    test_dataset = SDFDataset(image_groups=test_image_groups, gt_directory='data/Final Data/gt_data_sdf/test',image_processing_directory='data/Final Data/image_processing_data_sdf/test',sdf_directory= 'data/Final Data/sdf_dir/test',transform=test_transformations,target_transform=target_transformations)

    return train_dataset, test_dataset

def list_dirs(p):
    return sorted(d for d in glob.glob(os.path.join(p, '*')) if os.path.isdir(d))

class TopoDataset(Dataset):
    def __init__(self, subbatch_dirs, group_ids):
        self.items, self.group_of_item = [], []
        for dirpath, dirnames, filenames in os.walk('data/Final Data/Bigger Batches of Data SDF'):
            if any(f.endswith(".npy") for f in filenames):  # a leaf that holds data = a sub-batch
                rel = os.path.relpath(dirpath, 'data/Final Data/Bigger Batches of Data SDF').split(os.sep)
                view = os.path.join(*rel[:1])  # the main-batch / view id
                self.items.append(dirpath)
                self.group_of_item.append(view)
        uniq = {v: k for k, v in enumerate(sorted(set(self.group_of_item)))}
        print('Unique',uniq)
        self.group_of_item = np.array([uniq[v] for v in self.group_of_item])
        assert len(self.items) > 0, f"no .npy-containing folders under {'data/Final Data/Bigger Batches of Data SDF'}"
    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sub_dir = self.items[i]
        print('Sub Directory',sub_dir)
        sub = (sub_dir.split('/')[-1]).split(' ')[-1]
        image_files = sorted(glob.glob(os.path.join(sub_dir, f'*.npy')))  # the 5 images
        print('Image Files',image_files)
        images = torch.stack([torch.from_numpy(np.load(f)).float() for f in image_files], dim=0)  # [5,H,W]

        labels = torch.from_numpy(np.load(os.path.join('data/Final Data/Batches GT SDF', f'gt_{sub}.npy')))
        avg_output = torch.from_numpy(np.load(os.path.join('data/Final Data/Batches Image Processing SDF', f'image_{sub}.npy')))
        #sdf = torch.from_numpy(np.load(os.path.join('data/Final Data/Batches Image Processing', f'sdf_{sub}.npy'))).float()
        return images, labels, avg_output


class TopoDatasetImg(Dataset):
    def __init__(self, subbatch_dirs, group_ids):
        self.items, self.group_of_item = [], []
        for dirpath, dirnames, filenames in os.walk('data/Final Data/Bigger Batches of Data'):
            if any(f.endswith(".png") for f in filenames):  # a leaf that holds data = a sub-batch
                rel = os.path.relpath(dirpath, 'data/Final Data/Bigger Batches of Data').split(os.sep)
                view = os.path.join(*rel[:1])  # the main-batch / view id
                self.items.append(dirpath)
                self.group_of_item.append(view)
        uniq = {v: k for k, v in enumerate(sorted(set(self.group_of_item)))}
        print('Unique',uniq)
        self.group_of_item = np.array([uniq[v] for v in self.group_of_item])
        assert len(self.items) > 0, f"no .png-containing folders under {'data/Final Data/Bigger Batches of Data'}"
    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sub_dir = self.items[i]
        print('Sub Directory',sub_dir)
        sub = (sub_dir.split('/')[-1]).split(' ')[-1]
        image_files = sorted(glob.glob(os.path.join(sub_dir, f'*.png')))  # the 5 images
        print('Image Files',image_files)
        images = torch.stack([torchvision.io.decode_image(f,torchvision.io.ImageReadMode.GRAY) for f in image_files], dim=0)  # [5,H,W]

        labels = torch.from_numpy(np.load(os.path.join('data/Final Data/Batches GT SDF', f'gt_{sub}.npy')))
        avg_output = torch.from_numpy(np.load(os.path.join('data/Final Data/Batches Image Processing SDF', f'image_{sub}.npy')))
        #sdf = torch.from_numpy(np.load(os.path.join('data/Final Data/Batches Image Processing', f'sdf_{sub}.npy'))).float()
        return images, labels, avg_output
















