import os

import numpy as np
import skimage.metrics
from safetensors.torch import save_file, load_file
from input_pipeline import *

#from utils import *
#from model_training.model import *
#from transfer_learning import *
#from VQ_VAE_transfer_learning import *
#from transformers import AutoModel
import shutil
import cv2
#from image_processing_approach import *
#from UNetEncoder import *
from sklearn.model_selection import KFold
from input_pipeline import *
#from vae_gan import *
#from vae_gan_trail import *
from diffusers import AutoencoderKL
from baseline_approaches import *
#from permutation_invariance_evaluation import *
from VAE_Perceptual_Loss import *
from datasets import load_dataset
from torch.utils.data import ConcatDataset
#from attSets import *
#from util import encode_densities
#from VAE_GAN_training import *
#from discriminator import *
#from vae_gan_ssim_sharpness import *
#from vae_latent_visualisation import *
#from vae_iai import *
from conv_on_original_size import simple_Conv

def dataset_filter(data_sample, min_size = 256, min_vf=0.2, max_vf= 0.7):
    img = data_sample['topology']
    w,h = img.size
    #volume_fraction = data_sample['volume fraction']

    valid_size =  min_size <= w and min_size <= h
    #valid_vf = min_vf <= volume_fraction <= max_vf

    return valid_size

def main():
    baseline = 'Train'
    model = 'CNN'
    if baseline == 'Baseline':
        #create_ground_truth('data/batched_data_resized_red/train/Batch 17',0)
        classical_approach('data/batched_data_resized_red/test/',None,None,True)

    elif baseline == 'Train':

        batch_size =1
        data_dir = 'data/Batches'
        ground_truth_dir = 'data/Ground_Truth_resized'
        image_processing_dir = 'data/image_processing_results'
        single_image_reconstruction = True
        wandb.login()
        #
        for fold in range(5):
              #pretrained_dataset = load_dataset('parquet', data_files='./data/pretraining*.parquet', split='train')
              filtered_dataset = load_dataset('parquet',data_files = './data/pretraining_hf.parquet',split='train')
              # filtered_dataset = pretrained_dataset.filter(dataset_filter)
              # print('Length of filtered dataset', len(filtered_dataset))
              #
              # # # Save as one file
              # filtered_dataset.to_parquet("./data/pretraining_hf.parquet")

              # # # Or save as HF arrow format (faster to reload, preserves image types)
              # filtered_dataset.save_to_disk("./data/pretraining_hf")

              ds_train_test = filtered_dataset.train_test_split(test_size=0.3, seed=42)
              ds_val_test = ds_train_test["test"].train_test_split(test_size=0.2, seed=42)

              ds_train = ds_train_test["train"]
              ds_val = ds_val_test["train"]
              ds_test = ds_val_test["test"]

              print(f"Train: {len(ds_train)}, Val: {len(ds_val)}, Test: {len(ds_test)}")

              openTO_train = OpenTODataset(ds_train)
              openTO_val = OpenTODataset(ds_val)
              openTO_test = OpenTODataset(ds_test)


              train_dataset, val_dataset, test_dataset = input_pipeline_vae(data_dir, ground_truth_dir, image_processing_dir, fold, single_image_reconstruction=single_image_reconstruction)

              train_combined = ConcatDataset([train_dataset, openTO_train])
              val_combined = ConcatDataset([val_dataset, openTO_val])
              test_combined = ConcatDataset([test_dataset, openTO_test])

              # Transform the Dataset into DataLoader
              train_dataloader = DataLoader(dataset=train_combined, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
              val_dataloader = DataLoader(dataset=val_combined, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
              test_dataloader = DataLoader(dataset=test_combined, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

              print(len(train_dataloader))
              print(len(val_dataloader))
              print(len(test_dataloader))

              #vae_tuner = VAEGANFineTuner(model_name='./finetuned_vae/individually_trained_models_grayscale',multi_image_mode=True)
              vae_tuner = VAEGANFineTuner(model_name='stabilityai/sd-vae-ft-mse',multi_image_mode=False)
              #soft_aggregator = SetAggregation()

              print(f'*--------------Fold {fold}--------------------------')
              vae_tuner.finetune(train_dataloader,val_dataloader,test_dataloader,fold=1)
        #     #soft_aggregator(train_dataloader, None, test_dataloader,fold)
    elif baseline == 'Eval':
        if model == 'CNN':
            model = torch.load('convolutional_network_chkpoint/convolutional_chkpoint.pth',weights_only=False)
            eval(model)
        elif model == 'VAE':
            model = AutoencoderKL.from_pretrained('./finetuned_vae/individually_trained_models',use_safetensors=True)
            eval(model)




if __name__ == "__main__":
    main()
