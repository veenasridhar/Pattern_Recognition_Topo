import os

import numpy as np
import skimage.metrics
from safetensors.torch import save_file, load_file
from input_pipeline import *

#from utils import *
#from model_training.model import *
#from transfer_learning import *
from torch.utils.tensorboard import SummaryWriter
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
from permutation_invariance_evaluation import *
from VAE_Perceptual_Loss import *
#from attSets import *
#from util import encode_densities
#from VAE_GAN_training import *
#from discriminator import *
#from vae_gan_ssim_sharpness import *
#from vae_latent_visualisation import *
#from vae_iai import *


def main():
    baseline = 'Train'
    if baseline == 'Baseline':
        #create_ground_truth('data/batched_data_resized_red/train/Batch 17',0)
        classical_approach('data/batched_data_resized_red/test/',None,None,True)

    elif baseline == 'Train':

        batch_size =4
        data_dir = 'data/Final Data/Bigger Batches of Data'
        ground_truth_dir = 'data/Final Data/gt_grayscale_resized'
        image_processing_dir = 'data/Final Data/image processing resized'

        wandb.login()

        for fold in range(5):

            train_dataset, val_dataset, test_dataset = input_pipeline(data_dir, ground_truth_dir, image_processing_dir, fold)

            # Transform the Dataset into DataLoader
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
            val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


            vae_tuner = VAEGANFineTuner(model_name='stabilityai/sd-vae-ft-mse',multi_image_mode=True)
            #soft_aggregator = SetAggregation()

            print(f'*--------------Fold {fold}--------------------------')
            vae_tuner.finetune(train_dataloader,val_dataloader,test_dataloader,fold)
            #soft_aggregator(train_dataloader, None, test_dataloader,fold)
    elif baseline == 'Eval':
        model = AutoencoderKL.from_pretrained('./finetuned_vae/individually_trained_models',use_safetensors=True)
        eval(model)




if __name__ == "__main__":
    main()
