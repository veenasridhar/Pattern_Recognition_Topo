import os.path
from os.path import exists

import cv2
import torch
import wandb.util
import wandb
from diffusers import AutoencoderKL
import torch.nn as nn
from torchvision.transforms import v2
from torchinfo import summary
from attSets import AttSets
from input_pipeline import *
from tqdm import tqdm
from torchmetrics import StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt
import torchvision.models as models
from visualizations import *
#from attSets import AttSets
from resnet_network import *


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True)

        self.feature_map1 = nn.Sequential(*list(vgg.features)[:4])
        self.feature_map2 = nn.Sequential(*list(vgg.features)[4:9])
        self.feature_map3 = nn.Sequential(*list(vgg.features)[9:16])
        self.feature_map4 = nn.Sequential(*list(vgg.features)[16:23])

        for p in self.parameters():
            p.requires_grad = False

    def forward(self,real,reconstructed):
        #Normalize the input as per VGG Model expectations
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(real.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(real.device)
        real = (real - mean) / std
        recon = (reconstructed - mean) / std
        mse_loss = torch.nn.MSELoss().to(real.device)

        loss = 0.0
        real_input = real
        recon_input = recon

        for feature_map in [self.feature_map1, self.feature_map2, self.feature_map3, self.feature_map4]:
            real_input = feature_map(real_input)
            recon_input = feature_map(recon_input)
            #print(real_input.shape,recon_input.shape)
            loss += mse_loss(recon_input,real_input.detach())

        return loss

class VAEGANFineTuner:

    def __init__(self, model_name='stabilityai/sd-vae-ft-mse', multi_image_mode=True, device=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multi_image_mode = multi_image_mode
        self.vae = AutoencoderKL.from_pretrained(model_name)
        #self.vae = AutoencoderKL.from_pretrained('./finetuned_vae/individually_trained_models',use_safetensors=True) # Forces the model to use pretrained weights
        self.aggregation_module = AttSets(latent_channels=4)
        self.vae.to(self.device)
        #self.PROJECT_NAME = 'Individual Reconstruction Cross Validation'
        self.PROJECT_NAME = 'Topology Optimization'
        #self.GROUP_NAME = 'experiment' + wandb.util.generate_id()
        #print(summary(self.vae,(8,3,256,256)))

        #self.discriminator = PatchDiscriminator(in_channels=3).to(self.device)
        #print(f"Model parameters: {sum(p.numel() for p in self.vae.parameters()):,}")
        #print('Running on ', self.device)
        #self.disc_warmup = 5

    def sharpness_loss(self,reconstructed):

        t = [(1.0, 0.0, 0.0),  # full red
             (0.5, 0.5, 0.5),  # full grey
             (1.0, 1.0, 1.0), ]  # full white
        t = torch.tensor(t,dtype= torch.float,device=self.device)  # The shape of the tensor is (T,C)
        #print(t.shape)
        reconstructed = reconstructed.permute(0,2,3,1).unsqueeze(-2)            # This reshapes the tensor to be (B,H,W,1,C)
        print(reconstructed.shape)

        dist = ((reconstructed - t) ** 2).sum(dim=-1)  # This first subtracts (H,W,1,C) - (T,C) and the result is summed along dim(-1) thus resulting in (H,W,T)
        '''
        [[[0.9,0,0],[0,0,0],[0,0,0],[0,0,0]],
        [[[0.9,0,0],[0,0,0],[0,0,0],[0,0,0]],
        [[0.5,0,0],[0.5,0,0],[0,0,0],[0,0,0]],
        [[0,0,0],[0,0,0],[0.9,0.9,0.9],[0,0,0]],
        [[0,0,0],[0,0,0],[0.9,0.9,0.9],[0,0,0]]]

        This is a 4x4 image with 3 channels. Calculating distance would mean subtracting each pixel [0.9,0,0] with the target values (red,grey and white). This why unsqueeze(-2) is important to match dimensions for subtraction.
        Sum along dimension -1  will ensure that the red, grey and white each have 1 value instream of 3 channels. The shape of dist is [B,H,W,T] - T is the distance to the each of the target color '''
        minimum_dist = dist.min(-1).values.mean()
        ''' min(-1) selects the  minimum distance to the target value and is averaged across all pixels of all images. This is equal to calculating for each images individually and then averaging.
        '''
        return minimum_dist


    def finetune(self, train_loader, val_loader, test_loader,fold, epochs=51, lr=0.0001, kl_weight=1e-6, latent_technique = 'averaging', save_path="./finetuned_vae"):

        # Start a new wandb run to track this script.
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="veenasridhar96-university-of-stuttgart",
            # Set the wandb project where this run will be logged.
            #project=self.PROJECT_NAME,
            #group=self.GROUP_NAME,
            name = 'Topology Optimization',
            #job_type= f'run- {fold}',
            # Track hyperparameters and run metadata.
            config={
                "learning_rate": lr,
                'kl_weight': kl_weight,
                #'SSIM Weight': '0.00001',
                #'latent_technique': 'Averaging',
                'loss': ' KL + Perceptual Loss ',
                "architecture": "VAE",
                "dataset": "TO",
                "epochs": epochs,
                "fold": fold,
            }, )

        #torch.autograd.set_detect_anomaly(True)
        if self.multi_image_mode:
            decoder_optimizer = torch.optim.AdamW(self.vae.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(decoder_optimizer,gamma=0.095)
        else:
            optimizer = torch.optim.AdamW(self.vae.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        perceptual_loss = VGGPerceptualLoss().to(self.device)
        #cldice_loss = soft_dice_cldice()
        mse_loss = torch.nn.MSELoss()
        ce_weights = torch.tensor([0.25, 0.35, 0.5],device=self.device)
        ce_loss = torch.nn.CrossEntropyLoss(weight= ce_weights)
        best_loss = float('inf')

        for epoch in range(epochs):
            self.vae.train()
            train_vae_loss = 0.0
            train_mse_loss = 0.0
            train_kl_loss = 0.0
            train_iou_dl = 0.0
            train_iou_img_prc = 0.0

            train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
            print(f'The learning rate is {decoder_optimizer.param_groups[0]['lr']:.4f}')
            #kl_weight = min(1.0,epoch/40) *1e-5     #KL Annealing
            index =0

            if self.multi_image_mode:
                for batch, input in enumerate(train_progress_bar):
                    images, gt, avg_result = input
                    images = images.to(self.device)         # (B,N,C,H,W) = (2,5,3,256,256)
                    print('Images Unique', images[0][0].permute(1, 2, 0).unique())
                    target = gt.to(self.device)             # (B,C,H,W) = (2,3,256,256)
                    print('Target UNique',target[0].permute(1, 2, 0).unique())

                    batch_size, num_imgs = images.shape[0], images.shape[1]
                    combinedimages = images.reshape(-1, *images.shape[2:]).to(self.device)          #(B*N,C,H,W) = (10,3,256,256)
                    # The above line reshapes the input images from (B,N,C,H,W) to (B*N,C,H,W) where *images.shape[2:] (the * is called the argument unpacking unpacks (C,H,W) and the -1
                    # in the first indes allows python to infer the dimensions on its own while maintaining the data count so for the first indes B*N is done.)

                    # Sampling happens from Diffusers library
                    #combinedimages = combinedimages.unsqueeze(dim=1)            #Only for grayscale images
                    print('The shape of combined images',combinedimages.shape)
                    posterior = self.vae.encode(combinedimages).latent_dist        # Posterior - N(mu,std), latent_dist - Encoded outputs of Encoder represented as the mean and logvar of DiagonalGaussianDistribution. DiagonalGaussianDistribution allows for sampling latents from the distribution
                    #target_posterior = self.vae.encode(target).latent_dist

                    individual_mu = posterior.mean       # Individual mu is mean represented using 4 latent variables (10,4,32,32)
                    log_var = posterior.logvar          # Log variance represent the uncertainty or spread of the distribution. (log for stability)

                    # Reshape it to (B,N,latent_dimension,H,W)
                    mu = individual_mu.reshape(batch_size, num_imgs, *individual_mu.shape[1:])  # (B,N,latent_dimension,H,W) = (2,5,4,32,32)
                    log_var = log_var.reshape(batch_size, num_imgs, *log_var.shape[1:])         # (B,N,latent_dimension,H,W) = (2,5,4,32,32)

                    '''with torch.no_grad():
                        z_individual = posterior.sample()
                        original_reconstruction = self.vae.decode(z_individual).sample
                        original_reconstruction = original_reconstruction.view(batch_size, num_imgs, *original_reconstruction.shape[1:])'''

                    #visualize_interpolation(images, mu,self.vae)

                    #target_mu = target_posterior.mean           # (B,latent_dimension,H,W) = (2,4,32,32)
                    #target_logvar = target_posterior.logvar     # (B,latent_dimension,H,W) = (2,4,32,32)

                    if latent_technique == 'product_of_gaussians':
                        # Apply Product of Gaussians across the N images (dim=1)
                        # Calculate the precision (B,N,latent_dim, H,W)
                        precision = torch.exp(-log_var)
                        combined_precision = precision.sum(dim=1)  # Precision summed along the number of images (B,latent_dim,H,W)
                        combined_mu = (mu * precision).sum(dim=1) / combined_precision
                        combined_log_variance = -torch.log(combined_precision)

                        combined_std_dev = (0.5 * combined_log_variance).exp()                              # Standard deviation
                        z_averaged = combined_mu + combined_std_dev * torch.randn_like(combined_mu)

                    elif latent_technique == 'averaging':
                        # Averaging the distributions of images in the latent space
                        averaged_mu = mu.mean(dim=1)                #(B,C,H,W) = (2,4,32,32)
                        averaged_logvar = log_var.mean(dim=1)       #(B,C,H,W) = (2,4,32,32)
                        averaged_std_dev = (0.5 * averaged_logvar).exp()  # Converting the log variance $square(\sigma)$ into standard deviation (just sigma) - (B,C,H,W)

                        # Sample a Z from the mean distribution using the REPARAMETERIZATION TRICK.  Z = \mu + \std * \epsilon (noise sampled from N(0,1)
                        z_averaged = averaged_mu + averaged_std_dev * torch.randn_like(averaged_mu)           # (B,C,H,W)

                    elif latent_technique == 'attention_based':
                        z_averaged =  self.aggregation_module(posterior, num_imgs, batch_size)


                    '''if epoch%5 ==0 and batch < 10:
                        visualize_latent_space(individual_mu, combined_mu,target_mu,)'''

                    reconstructed = self.vae.decode(z_averaged).sample
                    print('Reconstructed UNique',reconstructed[0].permute(1,2,0).unique())
                    #visulaize_images(self.multi_image_mode, batch, epoch, reconstructed, images, target,f'./output_images_{fold}', False)

                        #latent_space_exploration(individual_mu, combined_mu, target_mu, self.vae)

                    #visualize_feature_maps(images,mu,reconstructed, averaged_mu)

                    reconstruction_loss = mse_loss(reconstructed,target)

                    target = target.squeeze(dim=1)

                    print("reconstructed:", reconstructed.shape, reconstructed.dtype)
                    print("target:", target.shape, target.dtype)

                    #reconstruction_loss = ce_loss(reconstructed,target)

                    #reconstruction_loss = perceptual_loss(target,reconstructed)

                    # if latent_technique == 'product_of_gaussians':
                    #     kl_loss = -0.5 *(1+combined_log_variance - combined_mu.pow(2)- combined_log_variance.exp()).mean()  # KL Divergence is calculated using closed form equation insetead of posterior.kl().mean() which might work for individual image decoding
                    # elif latent_technique == 'averaging':
                    #     kl_loss = -0.5 * (1 + averaged_logvar  - averaged_mu.pow(2) - averaged_logvar.exp()).mean()

                    loss = reconstruction_loss

                    #Backpropogation based on every step in the epoch
                    decoder_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(),1.0)
                    decoder_optimizer.step()

                    #Aggregating the loss values for each step in the epoch to output finally as a epoch loss values
                    train_vae_loss +=loss.item()
                    train_mse_loss +=reconstruction_loss.item()
                    #train_kl_loss +=kl_loss.item()
                    #train_sharpness_loss += shp_loss.item()
                    #reconstructed = torch.argmax(reconstructed, dim=1)

                    if epoch % 5 == 0:
                        if not os.path.exists(os.path.join(f'./output_images_{fold}/train/Epoch {epoch}')):
                            os.mkdir(os.path.join(f'./output_images_{fold}/train/Epoch {epoch}'))
                        for i in range(len(reconstructed)):
                            iou_dilated,iou_dilated_image_processing,dilated_overlap,img_proc_dilated_overlap = visulaize_iou_metrics(images[i], epoch, batch, reconstructed[i], gt[i], avg_result[i], tolerance=0,
                                              save_path=f'./output_images_{fold}/train/Epoch {epoch}')
                            train_iou_dl += iou_dilated
                            train_iou_img_prc += iou_dilated_image_processing
                            index += 1
                            visulaize_images(self.multi_image_mode, batch, epoch, reconstructed[i], images[i], target[i], avg_result[i], iou_dilated,iou_dilated_image_processing,dilated_overlap,
                                         img_proc_dilated_overlap, f'./output_images_{fold}/train/Epoch {epoch}', index, False)


                    # Printing the step loss values in the progress bar
                    train_progress_bar.set_postfix({
                        'Total loss': loss.item(),
                        'reconstruction_loss': reconstruction_loss.item(),
                        #'KL_loss': kl_loss.item(),
                        #'Sharpness Loss': shp_loss.item(),
                    })

            ###################################
            ### Single Image Mode
            ###################################
            else:
                for batch, (images,target) in enumerate(train_progress_bar):

                    images = images.to(self.device)
                    target = target.to(self.device)

                    print(images.shape)
                    print('The learning rate is',lr)

                    posterior = self.vae.encode(images).latent_dist      #  Posterior - N(mu,std), latent_dist - Encoded outputs of Encoder represented as the mean and logvar of DiagonalGaussianDistribution. DiagonalGaussianDistribution allows for sampling latents from the distribution
                    individual_mu = posterior.mean

                    '''if epoch > 10 and batch <3:
                        visualize_feature_maps(images,individual_mu,False)
                        latent_space_exploration(individual_mu,None,None,self.vae)'''

                    z = posterior.sample()                              # Reparametrization happens internally as it is a function from DiagonalGaussianDistribution class based on the Stable Diffusion VAE model used.
                    '''if epoch % 5 == 0 and batch < 10:
                        visualize_latent_space_individual(z)                           # z is of shape (B,C,H,W) = (16,4,32,32)'''

                    reconstructed = self.vae.decode(z).sample

                    target = target.squeeze(dim=1)

                    print(reconstructed.unique())
                    print(target.unique())

                    #recon_loss = mse_loss(reconstructed, target)
                    recon_loss = perceptual_loss(target,reconstructed)
                    #recon_loss = ce_loss(reconstructed,target)
                    kl_loss = posterior.kl().mean()
                    #sharpness_loss = (reconstructed * (1 - reconstructed)).mean()
                    loss = recon_loss + kl_weight * kl_loss


                    # Backpropogation based on every step in the epoch
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                    optimizer.step()

                    # Aggregating the loss values for each step in the epoch to output finally as a epoch loss values
                    train_vae_loss += loss.item()
                    train_mse_loss += recon_loss.item()
                    train_kl_loss += kl_loss.item()
                    #reconstructed = torch.argmax(reconstructed, dim=1)

                    if epoch%5 == 0 :
                        visulaize_images_individual(batch,epoch, reconstructed, images, None,f'./Individual Reconstruction Results/individual_images_Fold {fold}/train',False)
                        #latent_space_exploration(individual_mu, None, None, self.vae)
                        #visualize_feature_maps(images, individual_mu, False)

                    # Printing the step loss values in the progress bar
                    train_progress_bar.set_postfix({
                        'Total loss': loss.item(),
                        'reconstruction_loss': recon_loss.item(),
                        'KL_loss': kl_loss.item(),
                    })

            #scheduler.step()

            val_vae_loss = 0.0
            val_mse_loss = 0.0
            val_kl_loss = 0.0
            val_iou_dl = 0.0
            val_iou_img_prc = 0.0

            val_progress_bar = tqdm(val_loader,desc=f'Epoch {epoch}/{epochs}')
            if val_loader:
                print('The length of val loader', len(val_loader))
                self.vae.eval()
                with torch.no_grad():
                    if self.multi_image_mode:
                        for batch, input in enumerate(val_progress_bar):
                            images, labels, avg_result = input
                            target = labels
                            target = target.to(self.device)

                            batch_size, num_imgs = images.shape[0], images.shape[1]
                            combinedimages = images.reshape(-1, *images.shape[2:]).to(self.device)
                            #combinedimages = combinedimages.unsqueeze(dim=1)                    # Only for grayscale images

                            posterior = self.vae.encode(combinedimages).latent_dist
                            #target_posterior = self.vae.encode(target).latent_dist

                            individual_mu = posterior.mean  # Individual mu is mean represented using 4 latent variables (10,4,32,32)
                            log_var = posterior.logvar  # Log variance represent the uncertainty or spread of the distribution. (log for stability)

                            # Reshape it to (B,N,latent_dimension,H,W)
                            mu = individual_mu.reshape(batch_size, num_imgs, *individual_mu.shape[1:])  # (B,N,latent_dimension,H,W) = (2,5,4,32,32)
                            log_var = log_var.reshape(batch_size, num_imgs,*log_var.shape[1:])  # (B,N,latent_dimension,H,W) = (2,5,4,32,32)

                            #target_mu = target_posterior.mean  # (B,latent_dimension,H,W) = (2,4,32,32)
                            #target_logvar = target_posterior.logvar  # (B,latent_dimension,H,W) = (2,4,32,32)

                            if latent_technique == 'product_of_gaussians':
                                # Apply Product of Gaussians across the N images (dim=1)
                                # Calculate the precision (B,N,latent_dim, H,W)
                                precision = torch.exp(-log_var)
                                combined_precision = precision.sum(dim=1)  # Precision summed along the number of images (B,latent_dim,H,W)
                                combined_mu = (mu * precision).sum(dim=1) / combined_precision
                                combined_log_variance = -torch.log(combined_precision)

                                combined_std_dev = (0.5 * combined_log_variance).exp()  # Standard deviation
                                z_averaged = combined_mu + combined_std_dev * torch.randn_like(combined_mu)

                            elif latent_technique == 'averaging':
                                # Averaging the distributions of images in the latent space
                                averaged_mu = mu.mean(dim=1)  # (B,C,H,W) = (2,4,32,32)
                                averaged_logvar = log_var.mean(dim=1)  # (B,C,H,W) = (2,4,32,32)
                                averaged_std_dev = (0.5 * averaged_logvar).exp()  # Converting the log variance $square(\sigma)$ into standard deviation (just sigma) - (B,C,H,W)

                                # Sample a Z from the mean distribution using the REPARAMETERIZATION TRICK.  Z = \mu + \std * \epsilon (noise sampled from N(0,1)
                                z_averaged = averaged_mu + averaged_std_dev * torch.randn_like(averaged_mu)  # (B,C,H,W)

                            '''if epoch % 5 == 0 and batch < 10:
                                visualize_latent_space(individual_mu, combined_mu, target_mu,epoch,batch,)'''

                            reconstructed = self.vae.decode(z_averaged).sample

                            target = target.squeeze(dim=1)

                            print("reconstructed:", reconstructed.shape, reconstructed.dtype)
                            print("target:", target.shape, target.dtype)

                            #reconstruction_val_loss = ce_loss(reconstructed,target)

                            #latent_space_exploration(individual_mu, averaged_mu, target_mu, self.vae)

                            reconstruction_val_loss = mse_loss(reconstructed, target)
                            #reconstruction_val_loss = perceptual_loss(target,reconstructed)

                            # if latent_technique == 'product_of_gaussians':
                            #     kl_val_loss = -0.5 *(1+combined_log_variance - combined_mu.pow(2)- combined_log_variance.exp()).mean()
                            # elif latent_technique == 'averaging':
                            #     kl_val_loss = -0.5 * (1 + averaged_logvar - averaged_mu.pow(2) - averaged_logvar.exp()).mean()

                            #sharpness_val_loss  = self.sharpness_loss(reconstructed)

                            val_loss = reconstruction_val_loss
                            #val_loss = sharpness_val_loss

                            val_vae_loss += val_loss.item()
                            val_mse_loss += reconstruction_val_loss.item()
                            #val_kl_loss += kl_val_loss.item()
                            #val_sharpness_loss += sharpness_val_loss.item()

                            #reconstructed = torch.argmax(reconstructed,dim=1)

                            if epoch % 5 ==0:
                                if not os.path.exists(os.path.join(f'./output_images_{fold}/val/Epoch {epoch}')):
                                    os.mkdir(os.path.join(f'./output_images_{fold}/val/Epoch {epoch}'))
                                for i in range(len(reconstructed)):
                                    iou_dilated, iou_dilated_image_processing, dilated_overlap, img_proc_dilated_overlap = visulaize_iou_metrics(
                                        images[i], epoch, batch, reconstructed[i], target[i], avg_result[i], tolerance=0,
                                        save_path=f'./output_images_{fold}/val/Epoch {epoch}')
                                    visulaize_images(self.multi_image_mode, batch, epoch, reconstructed[i], images[i], target[i],
                                                     avg_result[i], iou_dilated, iou_dilated_image_processing, dilated_overlap, img_proc_dilated_overlap, f'./output_images_{fold}/val/Epoch {epoch}', index, True)
                                    val_iou_dl += iou_dilated
                                    val_iou_dl += iou_dilated_image_processing
                                    index +=1

                            # Printing the step loss values in the progress bar
                            val_progress_bar.set_postfix({
                                'Total Val loss': val_loss.item(),
                                'Val reconstruction_loss': reconstruction_val_loss.item(),
                                #'Val KL_loss': kl_val_loss.item(),
                                #'Val Sharpness Loss': sharpness_val_loss.item()
                                #'Val SSIM Loss': ssim_loss.item()
                            })

                    else:
                        for batch, (images,target) in enumerate(val_progress_bar):
                            images = images.to(self.device)
                            target = target.to(self.device)

                            posterior = self.vae.encode(images).latent_dist
                            individual_mu = posterior.mean
                            z = posterior.sample()
                            reconstructed = self.vae.decode(z).sample

                            target = target.squeeze(dim=1)
                            #recon_loss = mse_loss(reconstructed, target)
                            #recon_loss = ce_loss(reconstructed,target)
                            recon_loss = perceptual_loss(target, reconstructed)
                            kl_loss = posterior.kl().mean()
                            #ssim_loss = 1 - ssim(reconstructed, target)

                            val_loss = recon_loss + kl_weight * kl_loss

                            #reconstructed = torch.argmax(reconstructed,dim=1)
                            if epoch % 5 == 0:
                                visulaize_images_individual( batch, epoch, reconstructed, images, None,
                                                 f'./Individual Reconstruction Results/individual_images_Fold {fold}/val',
                                                 False)
                                # latent_space_exploration(individual_mu, None, None, self.vae)
                                # visualize_feature_maps(images, individual_mu, False)

                            val_vae_loss += val_loss.item()
                            val_mse_loss += recon_loss.item()
                            val_kl_loss += kl_loss.item()
                            #val_ssim_loss += ssim_loss.item()

                            # Printing the step loss values in the progress bar
                            val_progress_bar.set_postfix({
                                'Total Val loss': val_loss.item(),
                                'Val reconstruction_loss': recon_loss.item(),
                                'Val KL_loss': kl_loss.item(),
                                #'Val SSIM Loss': ssim_loss.item()
                            })


            test_vae_loss = 0.0
            test_mse_loss = 0.0
            test_kl_loss = 0.0
            test_ssim_loss = 0.0
            test_sharpness_loss =0.0
            test_iou_dl =0.0
            test_iou_img_prc = 0.0
            test_progress_bar = tqdm(test_loader, desc=f"Epoch {epoch}/{epochs}")
            if test_loader:
                print('The length of test loader',len(test_loader))
                self.vae.eval()
                index =0
                with torch.no_grad():
                    if self.multi_image_mode:
                        for batch, input in enumerate(test_progress_bar):
                            images, labels, avg_image = input
                            target = labels
                            target = target.to(self.device)

                            batch_size, num_imgs = images.shape[0], images.shape[1]
                            combinedimages = images.reshape(-1, *images.shape[2:]).to(self.device)

                            #combinedimages = combinedimages.unsqueeze(dim=1)

                            posterior = self.vae.encode(combinedimages).latent_dist
                            #target_posterior = self.vae.encode(target).latent_dist

                            individual_mu = posterior.mean  # Individual mu is mean represented using 4 latent variables (10,4,32,32)
                            log_var = posterior.logvar  # Log variance represent the uncertainty or spread of the distribution. (log for stability)

                            # Reshape it to (B,N,latent_dimension,H,W)
                            mu = individual_mu.reshape(batch_size, num_imgs, *individual_mu.shape[1:])  # (B,N,latent_dimension,H,W) = (2,5,4,32,32)
                            log_var = log_var.reshape(batch_size, num_imgs, *log_var.shape[1:])  # (B,N,latent_dimension,H,W) = (2,5,4,32,32)

                            '''z_individual = posterior.sample()
                            original_reconstruction = self.vae.decode(z_individual).sample

                            original_reconstruction = original_reconstruction.view(batch_size, num_imgs, *original_reconstruction.shape[1:])'''

                            #target_mu = target_posterior.mean  # (B,latent_dimension,H,W) = (2,4,32,32)
                            #target_logvar = target_posterior.logvar  # (B,latent_dimension,H,W) = (2,4,32,32)


                            if latent_technique == 'product_of_gaussians':
                                # Apply Product of Gaussians across the N images (dim=1)
                                # Calculate the precision (B,N,latent_dim, H,W)
                                precision = torch.exp(-log_var)
                                combined_precision = precision.sum(
                                    dim=1)  # Precision summed along the number of images (B,latent_dim,H,W)
                                combined_mu = (mu * precision).sum(dim=1) / combined_precision
                                combined_log_variance = -torch.log(combined_precision)

                                combined_std_dev = (0.5 * combined_log_variance).exp()  # Standard deviation
                                z_averaged = combined_mu + combined_std_dev * torch.randn_like(combined_mu)

                            elif latent_technique == 'averaging':
                                # Averaging the distributions of images in the latent space
                                averaged_mu = mu.mean(dim=1)  # (B,C,H,W) = (2,4,32,32)
                                averaged_logvar = log_var.mean(dim=1)  # (B,C,H,W) = (2,4,32,32)
                                averaged_std_dev = (0.5 * averaged_logvar).exp()  # Converting the log variance $square(\sigma)$ into standard deviation (just sigma) - (B,C,H,W)

                                # Sample a Z from the mean distribution using the REPARAMETERIZATION TRICK.  Z = \mu + \std * \epsilon (noise sampled from N(0,1)
                                z_averaged = averaged_mu + averaged_std_dev * torch.randn_like(averaged_mu)  # (B,C,H,W)
                            #if epoch % 5 == 0 and batch < 10:
                                #visualize_latent_space(individual_mu, combined_mu, target_mu)
                                #latent_space_exploration(individual_mu, averaged_mu, target_mu, self.vae)

                            reconstructed = self.vae.decode(z_averaged).sample

                            target = target.squeeze(dim=1)

                            print("reconstructed:", reconstructed.shape, reconstructed.dtype)
                            print("target:", target.shape, target.dtype)

                            #reconstruction_test_loss = ce_loss(reconstructed, target)
                            #reconstruction_test_loss = perceptual_loss(target,reconstructed)
                            reconstruction_test_loss = mse_loss(reconstructed, target)

                            # if latent_technique == 'product_of_gaussians':
                            #     kl_test_loss = -0.5 *(1+combined_log_variance - combined_mu.pow(2)- combined_log_variance.exp()).mean()
                            # elif latent_technique =='averaging':
                            #     kl_test_loss = -0.5 * (1 + averaged_logvar - averaged_mu.pow(2) - averaged_logvar.exp()).mean()

                            #sharpness_test_loss = self.sharpness_loss(reconstructed)

                            t_loss = reconstruction_test_loss
                            #t_loss = sharpness_test_loss

                            test_vae_loss += t_loss.item()
                            test_mse_loss += reconstruction_test_loss.item()
                            #test_kl_loss += kl_test_loss.item()
                            #test_sharpness_loss += sharpness_test_loss.item()

                            #reconstructed = torch.argmax(reconstructed, dim=1)

                            if epoch % 5 == 0:
                                if not os.path.exists(f'./output_images_{fold}/test/Epoch {epoch}'):
                                    os.mkdir(f'./output_images_{fold}/test/Epoch {epoch}')
                                for i in range(len(reconstructed)):
                                    iou_dilated, iou_dilated_image_processing, dilated_overlap, img_proc_dilated_overlap = visulaize_iou_metrics(
                                        images[i], epoch, batch, reconstructed[i], target[i], avg_image[i], tolerance=0,
                                        save_path=f'./output_images_{fold}/test/Epoch {epoch}')
                                    visulaize_images(self.multi_image_mode, batch, epoch, reconstructed[i], images[i], target[i],
                                                     avg_image[i], iou_dilated, iou_dilated_image_processing, dilated_overlap,img_proc_dilated_overlap, f'./output_images_{fold}/test/Epoch {epoch}', index, True)
                                    test_iou_dl += iou_dilated
                                    test_iou_img_prc += iou_dilated_image_processing
                                    index +=1

                            test_progress_bar.set_postfix({
                                'Test Total loss': t_loss.item(),
                                'Test Reconstruction_loss': reconstruction_test_loss.item(),
                                #'KL_loss': kl_test_loss.item(),
                            })


                    else:
                        for batch, (images,target) in enumerate(test_progress_bar):

                            images = images.to(self.device)
                            target = target.to(self.device)

                            posterior = self.vae.encode(images).latent_dist
                            individual_mu = posterior.mean
                            z = posterior.sample()
                            reconstructed = self.vae.decode(z).sample

                            '''if epoch > 10 and batch < 5:
                                visualize_feature_maps(images, individual_mu, False)
                                latent_space_exploration(individual_mu, None, None, self.vae)'''

                            target = target.squeeze(dim=1)

                            #recon_loss = mse_loss(reconstructed, target)
                            #recon_loss = ce_loss(reconstructed, target)
                            recon_loss = perceptual_loss(target,reconstructed)
                            kl_loss = posterior.kl().mean()
                            #ssim_loss = 1 - ssim(reconstructed, target)
                            t_loss = recon_loss + kl_weight * kl_loss

                            test_vae_loss += t_loss.item()
                            test_mse_loss += recon_loss.item()
                            test_kl_loss += kl_loss.item()
                            #test_ssim_loss += ssim_loss.item()

                            #reconstructed = torch.argmax(reconstructed, dim=1)

                            if epoch%5 == 0:
                                visulaize_images_individual(batch,epoch, reconstructed, images,None,f'./Individual Reconstruction Results/individual_images_Fold {fold}/test/', True)

                                # Printing the step loss values in the progress bar
                                test_progress_bar.set_postfix({
                                    'Test Total loss': t_loss.item(),
                                    'Test Reconstruction_loss': recon_loss.item(),
                                    'KL_loss': kl_loss.item(),
                                    #'SSIM Loss': ssim_loss.item()
                                })

            #print(f"Epoch {epoch + 1}:  \n Train Loss: {train_vae_loss/len(train_loader):.4f}, \n Val Loss: {val_vae_loss/len(val_loader)} \n Test Loss: {test_vae_loss/len(test_loader):.4f}")
            #print(f"Epoch {epoch}:  \n Train Loss: {train_vae_loss / len(train_loader):.4f} \n Val Loss: {val_vae_loss/len(val_loader)} \n Test Loss: {test_vae_loss / len(test_loader):.4f}")
            #print(f"Train MSE Loss: {train_mse_loss / len(train_loader):.4f} \n Val MSE Loss: {val_mse_loss/len(val_loader)} \n Test MSE Loss: {test_mse_loss / len(test_loader):.4f}")
            #print(f"Train KL Loss: {train_kl_loss / len(train_loader):.4f} \n Val KL Loss: {val_kl_loss/len(val_loader)} \n Test KL Loss: {test_kl_loss / len(test_loader):.4f}")
            #print(f"Train SSIM Loss: {train_ssim_loss / len(train_loader):.4f} \n Val SSIM Loss: {val_ssim_loss/len(val_loader)}  \n Test SSIM Loss: {test_ssim_loss / len(test_loader):.4f}")
            #print(f"Train Sharpness Loss: {train_sharpness_loss / len(train_loader):.4f} \n Val Sharpness Loss: {val_sharpness_loss / len(val_loader)}  \n Test Sharpness Loss: {test_sharpness_loss / len(test_loader):.4f}")

            if test_vae_loss < best_loss:
                best_loss = test_vae_loss
                if self.multi_image_mode:
                    torch.save(self.vae, './finetuned_vae/VAE_common_structures_checkpoint_averaging.pth')
                    self.vae.save_pretrained('finetuned_vae/common_structures_recon_averaging')

                else:
                    torch.save(self.vae,'./finetuned_vae/VAE_individual_checkpoint.pth')
                    self.vae.save_pretrained('finetuned_vae/individually_trained_models')

                print(f'Model Saved with test loss {test_vae_loss / len(test_loader):.4f}')

            #run.log({ 'Train_Loss': train_vae_loss/len(train_loader), 'Val_Loss': val_vae_loss/len(val_loader), 'Test_Loss': test_vae_loss/len(test_loader) }) ## Group multiple metrics as one in the UI
            print('Logging values to W&B...')
            run.log({'fold':fold,'epoch':epoch,
                     'Train_Total_Loss': train_vae_loss / len(train_loader), 'Test_Total_Loss': test_vae_loss / len(test_loader), 'Val Total Loss': val_vae_loss/len(val_loader),
                    'Train IoU Img_prc': train_iou_img_prc/76, 'Train IoU DL': train_iou_dl/76, 'Val IoU Img_prc': val_iou_img_prc/19, 'Val IoU DL': val_iou_dl/19, 'Test IoU Img_prc': test_iou_img_prc/10, 'Test IoU DL': test_iou_dl/10,

                    'Learning Rate':decoder_optimizer.param_groups[0]['lr'],
                    #'Train_MSE_Loss': train_mse_loss / len(train_loader), 'Test_MSE_Loss': test_mse_loss / len(test_loader),'Val MSE Loss': val_mse_loss/len(val_loader),
                     #'Train_SSIM_Loss': train_ssim_loss / len(train_loader), 'Test_SSIM_Loss': test_ssim_loss / len(test_loader), 'Val SSIM Loss':val_ssim_loss/len(val_loader),
                     #'Train_KL_Loss': train_kl_loss / len(train_loader), 'Test_KL_Loss': test_kl_loss / len(test_loader),'Val KL Loss':val_kl_loss/len(val_loader),
                   # 'Train_Sharpness_Loss': train_sharpness_loss / len(train_loader), 'Test_Sharpness_Loss': test_sharpness_loss / len(test_loader),'Val Sharpness Loss':val_sharpness_loss/len(val_loader)

                   })

        wandb.finish()













