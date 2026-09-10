import os
import numpy as np
from simple_conv_approach import train_grayscale
import skimage.metrics
from safetensors.torch import save_file, load_file
import shutil
import cv2
from sklearn.model_selection import KFold,train_test_split
from torch.utils.data import random_split, DataLoader
from classical_image_processsing_approaches.pixelwise_avg import *
from input_pipeline_binary import *
from diffusers import AutoencoderKL
from Custom_VAE import *
from torch.utils.data import ConcatDataset
from models import *
import json

batch_size =1
data_dir = 'data/Batches'
ground_truth_dir = 'data/Ground_Truth'
image_processing_dir = 'data/Image Processing Results'
single_data_dir = 'data/Individual data_wo_ground_truth'
openTO_individual_dataset = 'OpenTO_Consensus_Dataset_reduced_thickness_cldice/subbatches_resized'
aug_data_dir = None             #'data/Augmented TO Data'
single_image_reconstruction = False
dataset = 'RailwayTO' # 'OpenTO' or 'RailwayTO'
mode = 'Train'
nn_model = 'VAE'
chkpoint_pth = None  # None or path to checkpoint
latent_technique = 'learned_aggregation'
# # # # Split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

    

def main():
    
    if mode == 'Baseline':
        classical_approach(data_dir)

    elif mode == 'Train':

        wandb.login()

        for fold in range(5):
            if single_image_reconstruction:

                if dataset == 'OpenTO':
                   
                    openTO_dataset = OpenTOImageDataset(openTO_individual_dataset , augment = True)    # For Custom VAE with single image reconstruction (binary output)

                    dataset_size = len(openTO_dataset)
                    train_size = int(train_ratio * dataset_size)
                    val_size = int(val_ratio * dataset_size)
                    test_size = dataset_size - train_size - val_size
                    # Reproducible split
                    generator = torch.Generator().manual_seed(42)
                    train_dataset, val_dataset, test_dataset = random_split(
                        openTO_dataset,
                        [train_size, val_size, test_size],
                        generator=generator
                    )
                elif dataset == 'RailwayTO':

                    train_dataset, val_dataset, test_dataset = input_pipeline_discretevals(data_dir, single_data_dir,
                    ground_truth_dir, image_processing_dir, fold, single_image_reconstruction=single_image_reconstruction)  # For Custom VAE with binary (2 channels)

                # #train_dataset, val_dataset, test_dataset = input_pipeline_discretevals(data_dir,single_data_dir, ground_truth_dir = '', 
                # #image_processing_dir='', fold=fold, single_image_reconstruction=single_image_reconstruction) # Stability AI VAE

                train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
                val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
                test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

            else:
                if dataset == 'OpenTO':

                    opento_dataset_dir = create_image_groups('OpenTO_Consensus_Dataset_reduced_thickness_cldice/subbatches_resized')
                    opento_gt_dir = 'OpenTO_Consensus_Dataset_reduced_thickness_cldice/ground_truth_resized'
                    opento_img_prc_dir = 'OpenTO_Consensus_Dataset_reduced_thickness_cldice/image_proc_resized' 
                    
                    # # # # For discrete label inputs 
                    # # #openTO_train_dataset = OpenTOConsensusDatasetdisc(opento_dataset_dir,opento_gt_dir,opento_img_prc_dir, augment = True)
                    openTO_train_dataset = OpenTOConsensusDatasetVAEbinary(opento_dataset_dir,opento_gt_dir,opento_img_prc_dir, augment = True)

                    
                    dataset_size = len(openTO_train_dataset)
                    train_size = int(train_ratio * dataset_size)
                    val_size = int(val_ratio * dataset_size)
                    test_size = dataset_size - train_size - val_size

                    # # Reproducible split
                    generator = torch.Generator().manual_seed(42)

                    train_dataset, val_dataset, test_dataset = random_split(
                        openTO_train_dataset,
                        [train_size, val_size, test_size],
                        generator=generator
                    )

                elif dataset == 'RailwayTO':
                    train_dataset, val_dataset, test_dataset = input_pipeline_discretevals(data_dir, single_data_dir,
                    ground_truth_dir, image_processing_dir, fold, single_image_reconstruction=single_image_reconstruction,augmented_directory=aug_data_dir)  # For Custom VAE with binary (2 channels)
                

                # Transform the Dataset into DataLoader
                train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, )
                val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,)
                test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, )

            if nn_model == 'CNN':
                model = simple_Conv(in_channels=5, out_channels=2)
                train_grayscale(model,train_dataloader,val_dataloader,test_dataloader,fold=fold)

            elif nn_model == 'VAE':
                if chkpoint_pth:
                    model = torch.load(chkpoint_pth,weights_only=False)
                    #model = torch.load(f'finetuned_vae/Custom VAE_RailwayTO dataset_individual_checkpoint_ce_cldice_{fold}.pth',weights_only=False)
                else:
                    model = TOVAE()
                    vae_tuner = CustomVAE(model, multi_image_mode = True)
                    vae_tuner.finetune(train_dataloader,val_dataloader,test_dataloader,fold=fold,latent_technique = 'learned_aggregation')

            # Old models for VAE
            #vae_tuner = VAEGANFineTuner(model_name='finetuned_vae/individually_trained_models_ce_cldice',multi_image_mode=True)
            #vae_tuner = VAEGANFineTuner(model_name='stabilityai/sd-vae-ft-mse',multi_image_mode= not single_image_reconstruction)
            #model = torch.load('finetuned_vae/Custom VAE_OpenTO dataset_individual_checkpoint_ce_cldice_{fold}.pth',weights_only=False)
        
    elif mode == 'Eval':
        if nn_model == 'CNN':
            model = torch.load('convolutional_network_chkpoint/convolutional_chkpoint.pth',weights_only=False)
            eval(model)
        elif nn_model == 'VAE':
            model = torch.load('Custom VAE Results Final/Checkpoints/Custom_VAE_RailwayTO_common_structures_checkpoint_learned_aggregation_0.pth',weights_only=False)
            eval(model)

if __name__ == "__main__":
    main()
