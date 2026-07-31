import os
import numpy as np
import skimage.metrics
from safetensors.torch import save_file, load_file
#from input_pipeline import *
import shutil
import cv2
from sklearn.model_selection import KFold
#from input_pipeline import *
#from input_pipeline_discrete import *
from multi_input_pipeline import *
from diffusers import AutoencoderKL
from baseline_approaches import *
#from VAE_Perceptual_Loss import *
from VAE_cldice_celoss import *
#from Custom_VAE import *
from datasets import load_dataset
from torch.utils.data import ConcatDataset
from conv_on_original_size import simple_Conv
from concurrent.futures import ThreadPoolExecutor

def dataset_filter(data_sample, min_size = 72, min_vf=0.2, max_vf= 0.7):
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
        ground_truth_dir = 'data/Ground_Truth'
        image_processing_dir = 'data/Image Processing Results'
        single_data_dir = 'data/Individual data'
        single_image_reconstruction = True
        wandb.login()
        #
        for fold in range(5):
            if single_image_reconstruction:
               
                # ds_train_test = filtered_dataset.train_test_split(test_size=0.3, seed=42)
                # ds_val_test = ds_train_test["test"].train_test_split(test_size=0.2, seed=42)

                # ds_train = ds_train_test["train"]
                # ds_val = ds_val_test["train"]
                # ds_test = ds_val_test["test"]

                # print(f"Train: {len(ds_train)}, Val: {len(ds_val)}, Test: {len(ds_test)}")

                # # openTO_train = OpenTODataset(ds_train)
                # # openTO_val = OpenTODataset(ds_val)
                # # openTO_test = OpenTODataset(ds_test)

                # openTO_train = OpenTODatasetdisc(ds_train)
                # openTO_val = OpenTODatasetdisc(ds_val)
                # openTO_test = OpenTODatasetdisc(ds_test)


                #train_dataset, val_dataset, test_dataset = input_pipeline_vae(data_dir, ground_truth_dir, image_processing_dir, fold, single_image_reconstruction=single_image_reconstruction)

                train_dataset, val_dataset, test_dataset = input_pipeline_discretevals(data_dir,single_data_dir, ground_truth_dir = '', 
                image_processing_dir='', fold=fold, single_image_reconstruction=single_image_reconstruction)

                # train_combined = ConcatDataset([train_dataset, openTO_train])
                # val_combined = ConcatDataset([val_dataset, openTO_val])
                # test_combined = ConcatDataset([test_dataset, openTO_test])

                # Transform the Dataset into DataLoader
                train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
                val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
                test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

            else:
                opento_dataset_dir = create_image_groups('data/Consensus Dataset/subbatches')
                opento_gt_dir = 'data/Consensus Dataset/ground_truth'
                opento_img_prc_dir = 'data/Consensus Dataset/image_proc' 
                
                # For discrete label inputs
                openTO_train_dataset = OpenTOConsensusDatasetdisc(opento_dataset_dir,opento_gt_dir,opento_img_prc_dir, augment = True)
                train_dataset, val_dataset, test_dataset = input_pipeline_discretevals(data_dir, ground_truth_dir, image_processing_dir, fold, single_image_reconstruction=single_image_reconstruction)

                # For VAE Perceptual Loss
                # openTO_train_dataset = OpenTOConsensusDataset(opento_dataset_dir,opento_gt_dir,opento_img_prc_dir, augment = True)
                # train_dataset, val_dataset, test_dataset = input_pipeline_vae(data_dir, ground_truth_dir, image_processing_dir, fold, single_image_reconstruction=single_image_reconstruction)

                train_combined = ConcatDataset([train_dataset, openTO_train_dataset])
                
                # Transform the Dataset into DataLoader
                train_dataloader = DataLoader(dataset=train_combined, batch_size=batch_size, shuffle=True, )
                val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,)
                test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, )

            print(len(train_dataloader))
            print(len(val_dataloader))
            print(len(test_dataloader))

            #vae_tuner = VAEGANFineTuner(model_name='./finetuned_vae/individually_trained_models',multi_image_mode=True)
            vae_tuner = VAEGANFineTuner(model_name='stabilityai/sd-vae-ft-mse',multi_image_mode= not single_image_reconstruction)
            # model = TOVae2x()
            # vae_tuner = CustomVAE(model, multi_image_mode = True)
            #soft_aggregator = SetAggregation()

            print(f'*--------------Fold {fold}--------------------------')
            vae_tuner.finetune(train_dataloader,val_dataloader,test_dataloader,fold=fold)
    #       #soft_aggregator(train_dataloader, None, test_dataloader,fold)
    elif baseline == 'Eval':
        if model == 'CNN':
            model = torch.load('convolutional_network_chkpoint/convolutional_chkpoint.pth',weights_only=False)
            eval(model)
        elif model == 'VAE':
            model = AutoencoderKL.from_pretrained('./finetuned_vae/individually_trained_models',use_safetensors=True)
            eval(model)




if __name__ == "__main__":
    main()
