import os
import numpy as np
import skimage.metrics
from safetensors.torch import save_file, load_file
#from input_pipeline import *
import shutil
import cv2
from sklearn.model_selection import KFold,train_test_split
from torch.utils.data import random_split, DataLoader
from input_pipeline_binary import *
#from input_pipeline import *
#from multi_input_pipeline import *
from diffusers import AutoencoderKL
from baseline_approaches import *
#from VAE_Perceptual_Loss import *
from VAE_cldice_celoss import *
from Custom_VAE import *
from datasets import load_dataset
from torch.utils.data import ConcatDataset
from conv_on_original_size import simple_Conv
from concurrent.futures import ThreadPoolExecutor


def main():
    mode = 'Train'
    model = 'CNN'
    if mode == 'Baseline':
        #create_ground_truth('data/batched_data_resized_red/train/Batch 17',0)
        classical_approach('data/batched_data_resized_red/test/',None,None,True)

    elif mode == 'Train':

        batch_size =1
        data_dir = 'data/Batches'# with augmented data
        ground_truth_dir = 'data/Ground_Truth'#_augmented'
        image_processing_dir = 'data/Image Processing Results'
        single_data_dir = 'data/Individual data_wo_ground_truth'
        openTO_individual_dataset = 'OpenTO_Consensus_Dataset_reduced_thickness_cldice/subbatches_resized'
        aug_data_dir = 'data/Augmented TO Data'
        single_image_reconstruction = False
        wandb.login()
        #
        for fold in range(5):
            if single_image_reconstruction:
                
                #openTO_dataset = OpenTOImageDataset(openTO_individual_dataset , augment = True)    # For Custom VAE with single image reconstruction (binary output)

                #openTO_dataset = OpenTOVAEImageDataset(openTO_individual_dataset, augment = True)    # For StabilityAI VAE with single image reconstruction (3 class output)
                
                #print(f"Length of openTO_dataset: {len(openTO_dataset)}")

                #Split ratios
                # train_ratio = 0.7
                # val_ratio = 0.2
                # test_ratio = 0.1

                # dataset_size = len(openTO_dataset)
                # train_size = int(train_ratio * dataset_size)
                # val_size = int(val_ratio * dataset_size)
                # test_size = dataset_size - train_size - val_size

                # # Reproducible split
                # generator = torch.Generator().manual_seed(42)

                # train_dataset, val_dataset, test_dataset = random_split(
                #     openTO_dataset,
                #     [train_size, val_size, test_size],
                #     generator=generator
                # )

                train_dataset, val_dataset, test_dataset = input_pipeline_discretevals(data_dir, single_data_dir,
                ground_truth_dir, image_processing_dir, fold, single_image_reconstruction=single_image_reconstruction)  # For Custom VAE with binary (2 channels)

                # #train_dataset, val_dataset, test_dataset = input_pipeline_discretevals(data_dir,single_data_dir, ground_truth_dir = '', 
                # #image_processing_dir='', fold=fold, single_image_reconstruction=single_image_reconstruction) # Stability AI VAE

                train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
                val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
                test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

            else:
                # opento_dataset_dir = create_image_groups('OpenTO_Consensus_Dataset_reduced_thickness_cldice/subbatches_resized')
                # opento_gt_dir = 'OpenTO_Consensus_Dataset_reduced_thickness_cldice/ground_truth_resized'
                # opento_img_prc_dir = 'OpenTO_Consensus_Dataset_reduced_thickness_cldice/image_proc_resized' 
                
                # # # # # For discrete label inputs 
                # # # #openTO_train_dataset = OpenTOConsensusDatasetdisc(opento_dataset_dir,opento_gt_dir,opento_img_prc_dir, augment = True)
                # openTO_train_dataset = OpenTOConsensusDatasetVAEbinary(opento_dataset_dir,opento_gt_dir,opento_img_prc_dir, augment = True)

                # # # # # Split ratios
                # train_ratio = 0.7
                # val_ratio = 0.2
                # test_ratio = 0.1

                # dataset_size = len(openTO_train_dataset)
                # train_size = int(train_ratio * dataset_size)
                # val_size = int(val_ratio * dataset_size)
                # test_size = dataset_size - train_size - val_size

                # # # Reproducible split
                # generator = torch.Generator().manual_seed(42)

                # train_dataset, val_dataset, test_dataset = random_split(
                #     openTO_train_dataset,
                #     [train_size, val_size, test_size],
                #     generator=generator
                # )
                
                #train_dataset, val_dataset, test_dataset = input_pipeline_discretevals(data_dir, ground_truth_dir, image_processing_dir, fold, single_image_reconstruction=single_image_reconstruction)

                # For VAE Perceptual Loss
                #openTO_train_dataset = OpenTOConsensusDataset(opento_dataset_dir,opento_gt_dir,opento_img_prc_dir, augment = True)
                # train_dataset, val_dataset, test_dataset = input_pipeline_vae(data_dir, ground_truth_dir, image_processing_dir, fold, single_image_reconstruction=single_image_reconstruction)

                # train_dataset, val_dataset, test_dataset = input_pipeline_discretevals(data_dir,single_data_dir, ground_truth_dir, 
                # image_processing_dir, fold=fold, single_image_reconstruction=single_image_reconstruction)

                # train_combined = ConcatDataset([train_dataset, openTO_train_dataset])

                train_dataset, val_dataset, test_dataset = input_pipeline_discretevals(data_dir, single_data_dir,
                ground_truth_dir, image_processing_dir, fold, single_image_reconstruction=single_image_reconstruction,augmented_directory=aug_data_dir)  # For Custom VAE with binary (2 channels)
                
                # Transform the Dataset into DataLoader
                train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, )
                val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,)
                test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, )

            print(len(train_dataloader))
            print(len(val_dataloader))
            print(len(test_dataloader))

            #vae_tuner = VAEGANFineTuner(model_name='finetuned_vae/individually_trained_models_ce_cldice',multi_image_mode=True)
            #vae_tuner = VAEGANFineTuner(model_name='stabilityai/sd-vae-ft-mse',multi_image_mode= not single_image_reconstruction)
            #model = torch.load('finetuned_vae/Custom VAE_OpenTO dataset_individual_checkpoint_ce_cldice_{fold}.pth',weights_only=False)
            model = torch.load(f'finetuned_vae/Custom VAE_RailwayTO dataset_individual_checkpoint_ce_cldice_{fold}.pth',weights_only=False)
            #model = TOVae2x()
            vae_tuner = CustomVAE(model, multi_image_mode = True)
            #soft_aggregator = SetAggregation()

            print(f'*--------------Fold {fold}--------------------------')
            vae_tuner.finetune(train_dataloader,val_dataloader,test_dataloader,fold=fold,latent_technique = 'learned_aggregation')
    #       #soft_aggregator(train_dataloader, None, test_dataloader,fold)
    elif mode == 'Eval':
        if model == 'CNN':
            model = torch.load('convolutional_network_chkpoint/convolutional_chkpoint.pth',weights_only=False)
            eval(model)
        elif model == 'VAE':
            model = torch.load('Custom VAE Results Final/Checkpoints/Custom_VAE_RailwayTO_common_structures_checkpoint_learned_aggregation_0.pth',weights_only=False)
            eval(model)




if __name__ == "__main__":
    main()
