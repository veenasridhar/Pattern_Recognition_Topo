import torch
import wandb
from diffusers import AutoencoderKL
import torch.nn as nn
from torchvision.transforms import v2
from torchinfo import summary
from input_pipeline import *
from tqdm import tqdm
from torchmetrics import StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from visualizations.visualizations import *

class VAEGANFineTuner:

    def __init__(self, model_name='stabilityai/sd-vae-ft-mse', multi_image_mode=True, device=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multi_image_mode = multi_image_mode
        self.vae = AutoencoderKL.from_pretrained(model_name)
        #self.vae = AutoencoderKL.from_pretrained('./finetuned_vae/individually_trained_models')
        #self.vae = torch.load('./finetuned_vae/individually_trained_models/vae_checkpoint.pth',weights_only=False)
        self.vae.to(self.device)
        #print(summary(self.vae,(8,3,256,256)))

        #self.discriminator = PatchDiscriminator(in_channels=3).to(self.device)
        #print(f"Model parameters: {sum(p.numel() for p in self.vae.parameters()):,}")
        #print('Running on ', self.device)
        #self.disc_warmup = 5


    def finetune(self, train_loader, val_loader, test_loader,fold, epochs=50, lr=0.00001, kl_weight=1e-6, latent_technique = 'product_of_gaussians', save_path="./finetuned_vae"):

        # Start a new wandb run to track this script.
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="veenasridhar96-university-of-stuttgart",
            # Set the wandb project where this run will be logged.
            project="Topology Optimization",
            # Track hyperparameters and run metadata.
            config={
                "learning_rate": lr,
                'kl_weight': kl_weight,
                'SSIM Weight': '0.0001',
                'latent_technique': 'individual_reconstruction',
                'loss': 'MSE + KL ',
                "architecture": "VAE",
                "dataset": "TO",
                "epochs": 50,

            }, )

        #torch.autograd.set_detect_anomaly(True)

        optimizer = torch.optim.AdamW(self.vae.parameters(), lr=lr, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        ssim = StructuralSimilarityIndexMeasure().to(self.device)
        mse_loss = torch.nn.MSELoss()
        best_loss = float('inf')

        for epoch in range(epochs):
            self.vae.train()
            train_vae_loss = 0.0
            train_mse_loss = 0.0
            train_kl_loss = 0.0
            train_ssim_loss = 0.0

            train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
            val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs}")
            print('The learning rate is ',lr)
            #kl_weight = min(1.0,epoch/40) *1e-5     #KL Annealing

            if self.multi_image_mode:

                for batch, (images, gt) in enumerate(train_progress_bar):
                    images = images.to(self.device)         # (B,N,C,H,W) = (2,5,3,256,256)
                    target = gt.to(self.device)             # (B,C,H,W) = (2,3,256,256)

                    batch_size, num_imgs = images.shape[0], images.shape[1]
                    combinedimages = images.reshape(-1, *images.shape[2:]).to(self.device)          #(B*N,C,H,W) = (10,3,256,256)
                    # The above line reshapes the input images from (B,N,C,H,W) to (B*N,C,H,W) where *images.shape[2:] (the * is called the argument unpacking unpacks (C,H,W) and the -1
                    # in the first indes allows python to infer the dimensions on its own while maintaining the data count so for the first indes B*N is done.)

                    # TODO: Check how this sampling happens --- > Sampling happens from Diffusers library
                    print('The shape of combined images',combinedimages.shape)
                    posterior = self.vae.encode(combinedimages).latent_dist        # Posterior - N(mu,std), latent_dist - Encoded outputs of Encoder represented as the mean and logvar of DiagonalGaussianDistribution. DiagonalGaussianDistribution allows for sampling latents from the distribution
                    target_posterior = self.vae.encode(target).latent_dist

                    individual_mu = posterior.mean       # Individual mu is mean represented using 4 latent variables (10,4,32,32)
                    log_var = posterior.logvar          # Log variance represent the uncertainty or spread of the distribution. (log for stability)

                    # Reshape it to (B,N,latent_dimension,H,W)
                    mu = individual_mu.reshape(batch_size, num_imgs, *individual_mu.shape[1:])  # (B,N,latent_dimension,H,W) = (2,5,4,32,32)
                    log_var = log_var.reshape(batch_size, num_imgs, *log_var.shape[1:])         # (B,N,latent_dimension,H,W) = (2,5,4,32,32)

                    target_mu = target_posterior.mean           # (B,latent_dimension,H,W) = (2,4,32,32)
                    target_logvar = target_posterior.logvar     # (B,latent_dimension,H,W) = (2,4,32,32)


                    if latent_technique == 'product_of_gaussians':
                        # Apply Product of Gaussians across the N images (dim=1)
                        # Calculate the precision (B,N,latent_dim, H,W)
                        precision = torch.exp(-log_var)
                        combined_precision = precision.sum(dim=1)  # Precision summed along the number of images (B,latent_dim,H,W)
                        combined_mu = (mu * precision).sum(dim=1) / combined_precision
                        log_combined_precision = -torch.log(combined_precision)

                        sampled_precision = (0.5 * log_combined_precision).exp()                              # Standard deviation
                        z_averaged = combined_mu + sampled_precision * torch.randn_like(combined_mu)

                    elif latent_technique == 'averaging':
                        # Averaging the distributions of images in the latent space
                        averaged_mu = mu.mean(dim=1)                #(B,C,H,W) = (2,4,32,32)
                        averaged_logvar = log_var.mean(dim=1)       #(B,C,H,W) = (2,4,32,32)
                        std_logvar = (0.5 * averaged_logvar).exp()  # Converting the log variance $square(\sigma)$ into standard deviation (just sigma) - (B,C,H,W)

                        # Sample a Z from the mean distribution using the REPARAMETERIZATION TRICK.  Z = \mu + \std * \epsilon (noise sampled from N(0,1)
                        z_averaged = averaged_mu + std_logvar * torch.randn_like(averaged_mu)           # (B,C,H,W)

                    # Check if there is any exploding values in the variance.
                    #print("individual_mu NaN:", torch.isnan(individual_mu).any().item())
                    #print("combined_mu NaN:", torch.isnan(combined_mu).any().item())
                    #print("target_mu NaN:", torch.isnan(target_mu).any().item())

                    '''if epoch%5 ==0 and batch < 10:
                        visualize_latent_space(individual_mu, combined_mu,target_mu,)'''

                    reconstructed = self.vae.decode(z_averaged).sample
                    if batch <5:
                        visulaize_images(self.multi_image_mode, batch, epoch, reconstructed, images, target, f'./output_images_{fold}',False)
                        latent_space_exploration(individual_mu, combined_mu, target_mu, self.vae)

                    reconstruction_loss = mse_loss(reconstructed,target)
                    kl_loss = -0.5 *(1+log_combined_precision - combined_mu.pow(2)- log_combined_precision.exp()).mean()  # KL Divergence is calculated using closed form equation insetead of posterior.kl().mean() which might work for individual image decoding
                    #kl_loss = -0.5 * (1 + averaged_logvar  - averaged_mu.pow(2) - averaged_logvar.exp()).mean()

                    loss = reconstruction_loss + kl_loss*kl_weight

                    #Backpropogation based on every step in the epoch
                    '''optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(),1.0)
                    optimizer.step()'''

                    #Aggregating the loss values for each step in the epoch to output finally as a epoch loss values
                    train_vae_loss +=loss.item()
                    train_mse_loss +=reconstruction_loss.item()
                    train_kl_loss +=kl_loss.item()

                    # Printing the step loss values in the progress bar
                    train_progress_bar.set_postfix({
                        'Total loss': loss.item(),
                        'reconstruction_loss': reconstruction_loss.item(),
                        'KL_loss': kl_loss.item(),
                    })

                with torch.no_grad():
                    self.vae.eval()
                    val_vae_loss = 0.0
                    val_mse_loss = 0.0
                    val_kl_loss = 0.0

                    for batch, (images, gt) in enumerate(val_progress_bar):
                        images = images.to(self.device)  # (B,N,C,H,W) = (2,5,3,256,256)
                        target = gt.to(self.device)  # (B,C,H,W) = (2,3,256,256)

                        batch_size, num_imgs = images.shape[0], images.shape[1]
                        combinedimages = images.reshape(-1, *images.shape[2:]).to(self.device)  # (B*N,C,H,W) = (10,3,256,256)
                        # The above line reshapes the input images from (B,N,C,H,W) to (B*N,C,H,W) where *images.shape[2:] (the * is called the argument unpacking unpacks (C,H,W) and the -1
                        # in the first indes allows python to infer the dimensions on its own while maintaining the data count so for the first indes B*N is done.)

                        # TODO: Check how this sampling happens
                        posterior = self.vae.encode(
                            combinedimages).latent_dist  # Posterior - N(mu,std), latent_dist - Encoded outputs of Encoder represented as the mean and logvar of DiagonalGaussianDistribution. DiagonalGaussianDistribution allows for sampling latents from the distribution
                        target_posterior = self.vae.encode(target).latent_dist

                        individual_mu = posterior.mean  # Individual mu is mean represented using 4 latent variables (10,4,32,32)
                        log_var = posterior.logvar  # Log variance represent the uncertainty or spread of the distribution. (log for stability)

                        # Reshape it to (B,N,latent_dimension,H,W)
                        mu = individual_mu.reshape(batch_size, num_imgs,
                                                   *individual_mu.shape[1:])  # (B,N,latent_dimension,H,W) = (2,5,4,32,32)
                        log_var = log_var.reshape(batch_size, num_imgs,
                                                  *log_var.shape[1:])  # (B,N,latent_dimension,H,W) = (2,5,4,32,32)

                        target_mu = target_posterior.mean  # (B,latent_dimension,H,W) = (2,4,32,32)
                        target_logvar = target_posterior.logvar  # (B,latent_dimension,H,W) = (2,4,32,32)

                        if latent_technique == 'product_of_gaussians':
                            # Apply Product of Gaussians across the N images (dim=1)
                            # Calculate the precision (B,N,latent_dim, H,W)
                            precision = torch.exp(-log_var)
                            combined_precision = precision.sum(
                                dim=1)  # Precision summed along the number of images (B,latent_dim,H,W)
                            combined_mu = (mu * precision).sum(dim=1) / combined_precision
                            log_combined_precision = -torch.log(combined_precision)

                            sampled_precision = (0.5 * log_combined_precision).exp()  # Standard deviation
                            z_averaged = combined_mu + sampled_precision * torch.randn_like(combined_mu)

                        elif latent_technique == 'averaging':
                            # Averaging the distributions of images in the latent space
                            averaged_mu = mu.mean(dim=1)  # (B,C,H,W) = (2,4,32,32)
                            averaged_logvar = log_var.mean(dim=1)  # (B,C,H,W) = (2,4,32,32)
                            std_logvar = (
                                        0.5 * averaged_logvar).exp()  # Converting the log variance $square(\sigma)$ into standard deviation (just sigma) - (B,C,H,W)

                            # Sample a Z from the mean distribution using the REPARAMETERIZATION TRICK.  Z = \mu + \std * \epsilon (noise sampled from N(0,1)
                            z_averaged = averaged_mu + std_logvar * torch.randn_like(averaged_mu)  # (B,C,H,W)

                        '''if epoch%5 ==0 and batch < 10:
                            visualize_latent_space(individual_mu, combined_mu,target_mu,)'''

                        reconstructed = self.vae.decode(z_averaged).sample
                        if batch < 5:
                            visulaize_images(self.multi_image_mode, batch, epoch, reconstructed, images, target,f'./output_images_{fold}/val_images', False)

                        reconstruction_loss = mse_loss(reconstructed, target)
                        kl_loss = -0.5 * (1 + log_combined_precision - combined_mu.pow(2) - log_combined_precision.exp()).mean()

                        loss= reconstruction_loss + kl_loss*kl_weight

                        val_vae_loss += loss.item()
                        val_mse_loss += reconstruction_loss.item()
                        val_kl_loss += kl_loss.item()

                        # Printing the step loss values in the progress bar
                        val_progress_bar.set_postfix({
                            'Total loss': loss.item(),
                            'reconstruction_loss': reconstruction_loss.item(),
                            'KL_loss': kl_loss.item(),
                        })



            ###################################
            ### Single Image Mode
            ###################################
            else:
                for batch, (images,target) in enumerate(train_progress_bar):

                    images = images.to(self.device)
                    target = target.to(self.device)

                    posterior = self.vae.encode(images).latent_dist      #  Posterior - N(mu,std), latent_dist - Encoded outputs of Encoder represented as the mean and logvar of DiagonalGaussianDistribution. DiagonalGaussianDistribution allows for sampling latents from the distribution
                    individual_mu = posterior.mean

                    '''if epoch > 10 and batch <3:
                        visualize_feature_maps(images,individual_mu,False)
                        latent_space_exploration(individual_mu,None,None,self.vae)'''

                    z = posterior.sample()                              # Reparametrization happens internally as it is a function from DiagonalGaussianDistribution class based on the Stable Diffusion VAE model used.
                    '''if epoch % 5 == 0 and batch < 10:
                        visualize_latent_space_individual(z)                           # z is of shape (B,C,H,W) = (16,4,32,32)'''

                    reconstructed = self.vae.decode(z).sample

                    if epoch%5 == 0 and batch <5:
                        visulaize_images(self.multi_image_mode, batch, epoch, reconstructed, images, None,'./output_images/individual_images/',False)
                        #latent_space_exploration(individual_mu, None, None, self.vae)
                        #visualize_feature_maps(images, individual_mu, False)

                    recon_loss = mse_loss(reconstructed, target)
                    kl_loss = posterior.kl().mean()
                    ssim_loss = 1 - ssim(reconstructed, target)
                    #sharpness_loss = (reconstructed * (1 - reconstructed)).mean()
                    loss = recon_loss + kl_weight * kl_loss + 0 * ssim_loss

                    # Backpropogation based on every step in the epoch
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                    optimizer.step()

                    # Aggregating the loss values for each step in the epoch to output finally as a epoch loss values
                    train_vae_loss += loss.item()
                    train_mse_loss += recon_loss.item()
                    train_kl_loss += kl_loss.item()
                    train_ssim_loss += ssim_loss.item()

                    # Printing the step loss values in the progress bar
                    train_progress_bar.set_postfix({
                        'Total loss': loss.item(),
                        'reconstruction_loss': recon_loss.item(),
                        'KL_loss': kl_loss.item(),
                        'SSIM Loss': ssim_loss.item()
                    })

            scheduler.step()

            val_vae_loss = 0.0
            val_mse_loss = 0.0
            val_kl_loss = 0.0
            val_ssim_loss = 0.0
            val_progress_bar = tqdm(val_loader,desc=f'Epoch {epoch}/{epochs}')
            if val_loader:
                print('The length of val loader', len(val_loader))
                self.vae.eval()
                with torch.no_grad():
                    if self.multi_image_mode:
                        for batch, (images, labels) in enumerate(test_progress_bar):
                            target = labels
                            target = target.to(self.device)

                            batch_size, num_imgs = images.shape[0], images.shape[1]
                            combinedimages = images.reshape(-1, *images.shape[2:]).to(self.device)

                            posterior = self.vae.encode(combinedimages).latent_dist
                            target_posterior = self.vae.encode(target).latent_dist

                            individual_mu = posterior.mean  # Individual mu is mean represented using 4 latent variables (10,4,32,32)
                            log_var = posterior.logvar  # Log variance represent the uncertainty or spread of the distribution. (log for stability)

                            # Reshape it to (B,N,latent_dimension,H,W)
                            mu = individual_mu.reshape(batch_size, num_imgs, *individual_mu.shape[1:])  # (B,N,latent_dimension,H,W) = (2,5,4,32,32)
                            log_var = log_var.reshape(batch_size, num_imgs,
                                                      *log_var.shape[1:])  # (B,N,latent_dimension,H,W) = (2,5,4,32,32)

                            target_mu = target_posterior.mean  # (B,latent_dimension,H,W) = (2,4,32,32)
                            target_logvar = target_posterior.logvar  # (B,latent_dimension,H,W) = (2,4,32,32)

                            latent_technique = 'product_of_gaussians'
                            if latent_technique == 'product_of_gaussians':
                                # Apply Product of Gaussians across the N images (dim=1)
                                # Calculate the precision (B,N,latent_dim, H,W)
                                precision = torch.exp(-log_var)
                                combined_precision = precision.sum(
                                    dim=1)  # Precision summed along the number of images (B,latent_dim,H,W)
                                combined_mu = (mu * precision).sum(dim=1) / combined_precision
                                log_combined_precision = -torch.log(combined_precision)

                                sampled_precision = (0.5 * log_combined_precision).exp()
                                z_averaged = combined_mu + sampled_precision * torch.randn_like(combined_mu)

                            elif latent_technique == 'averaging':
                                # Averaging the distributions of images in the latent space
                                averaged_mu = mu.mean(dim=1)  # (B,C,H,W) = (2,4,32,32)
                                averaged_logvar = log_var.mean(dim=1)  # (B,C,H,W) = (2,4,32,32)
                                std_logvar = (0.5 * averaged_logvar).exp()  # Converting the log variance $square(\sigma)$ into standard deviation (just sigma) - (B,C,H,W)

                                # Sample a Z from the mean distribution using the REPARAMETERIZATION TRICK.  Z = \mu + \std * \epsilon (noise sampled from N(0,1)
                                z_averaged = averaged_mu + std_logvar * torch.randn_like(averaged_mu)  # (B,C,H,W)
                            '''if epoch % 5 == 0 and batch < 10:
                                visualize_latent_space(individual_mu, combined_mu, target_mu)'''

                            reconstructed = self.vae.decode(z_averaged).sample


                            visulaize_images(self.multi_image_mode,batch, epoch, reconstructed, images, target, f'./output_images_{fold}/test_images/', True)

                            reconstruction_test_loss = mse_loss(reconstructed, target)

                            kl_test_loss = -0.5 *(1+log_combined_precision - combined_mu.pow(2)- log_combined_precision.exp()).mean()
                            #kl_test_loss = -0.5 * (1 + averaged_logvar - averaged_mu.pow(2) - averaged_logvar.exp()).mean()

                            t_loss = reconstruction_test_loss + kl_weight * kl_test_loss
                            test_vae_loss += t_loss.item()
                            test_mse_loss += reconstruction_test_loss.item()
                            test_kl_loss += kl_test_loss.item()

                    else:
                        for batch, (images,target) in enumerate(val_progress_bar):
                            images = images.to(self.device)
                            target = target.to(self.device)
                            posterior = self.vae.encode(images).latent_dist
                            individual_mu = posterior.mean
                            z = posterior.sample()
                            reconstructed = self.vae.decode(z).sample

                            if epoch%5 == 0 and batch < 5:
                                visulaize_images(self.multi_image_mode,batch, epoch, reconstructed, images, None,'./output_images/individual_images/val_images', False)
                                #latent_space_exploration(individual_mu, None, None, self.vae)
                                #visualize_feature_maps(images, individual_mu, False)

                            recon_loss = mse_loss(reconstructed, target)
                            kl_loss = posterior.kl().mean()
                            ssim_loss = 1 - ssim(reconstructed, target)

                            val_loss = recon_loss + kl_weight * kl_loss + 0 * ssim_loss

                            val_vae_loss += val_loss.item()
                            val_mse_loss += recon_loss.item()
                            val_kl_loss += kl_loss.item()
                            val_ssim_loss += ssim_loss.item()

                            # Printing the step loss values in the progress bar
                            val_progress_bar.set_postfix({
                                'Total Val loss': val_loss.item(),
                                'Val reconstruction_loss': recon_loss.item(),
                                'Val KL_loss': kl_loss.item(),
                                'Val SSIM Loss': ssim_loss.item()
                            })


            test_vae_loss = 0.0
            test_mse_loss = 0.0
            test_kl_loss = 0.0
            test_ssim_loss = 0.0

            test_progress_bar = tqdm(test_loader, desc=f"Epoch {epoch}/{epochs}")
            if test_loader:
                print('The length of test loader',len(test_loader))
                self.vae.eval()
                with torch.no_grad():
                    if self.multi_image_mode:
                        for batch, (images, labels) in enumerate(test_progress_bar):
                            target = labels
                            target = target.to(self.device)

                            batch_size, num_imgs = images.shape[0], images.shape[1]
                            combinedimages = images.reshape(-1, *images.shape[2:]).to(self.device)

                            posterior = self.vae.encode(combinedimages).latent_dist
                            target_posterior = self.vae.encode(target).latent_dist

                            individual_mu = posterior.mean  # Individual mu is mean represented using 4 latent variables (10,4,32,32)
                            log_var = posterior.logvar  # Log variance represent the uncertainty or spread of the distribution. (log for stability)

                            # Reshape it to (B,N,latent_dimension,H,W)
                            mu = individual_mu.reshape(batch_size, num_imgs, *individual_mu.shape[1:])  # (B,N,latent_dimension,H,W) = (2,5,4,32,32)
                            log_var = log_var.reshape(batch_size, num_imgs,
                                                      *log_var.shape[1:])  # (B,N,latent_dimension,H,W) = (2,5,4,32,32)

                            target_mu = target_posterior.mean  # (B,latent_dimension,H,W) = (2,4,32,32)
                            target_logvar = target_posterior.logvar  # (B,latent_dimension,H,W) = (2,4,32,32)

                            latent_technique = 'product_of_gaussians'
                            if latent_technique == 'product_of_gaussians':
                                # Apply Product of Gaussians across the N images (dim=1)
                                # Calculate the precision (B,N,latent_dim, H,W)
                                precision = torch.exp(-log_var)
                                combined_precision = precision.sum(
                                    dim=1)  # Precision summed along the number of images (B,latent_dim,H,W)
                                combined_mu = (mu * precision).sum(dim=1) / combined_precision
                                log_combined_precision = -torch.log(combined_precision)

                                sampled_precision = (0.5 * log_combined_precision).exp()
                                z_averaged = combined_mu + sampled_precision * torch.randn_like(combined_mu)

                            elif latent_technique == 'averaging':
                                # Averaging the distributions of images in the latent space
                                averaged_mu = mu.mean(dim=1)  # (B,C,H,W) = (2,4,32,32)
                                averaged_logvar = log_var.mean(dim=1)  # (B,C,H,W) = (2,4,32,32)
                                std_logvar = (0.5 * averaged_logvar).exp()  # Converting the log variance $square(\sigma)$ into standard deviation (just sigma) - (B,C,H,W)

                                # Sample a Z from the mean distribution using the REPARAMETERIZATION TRICK.  Z = \mu + \std * \epsilon (noise sampled from N(0,1)
                                z_averaged = averaged_mu + std_logvar * torch.randn_like(averaged_mu)  # (B,C,H,W)
                            '''if epoch % 5 == 0 and batch < 10:
                                visualize_latent_space(individual_mu, combined_mu, target_mu)'''

                            reconstructed = self.vae.decode(z_averaged).sample


                            visulaize_images(self.multi_image_mode,batch, epoch, reconstructed, images, target, f'./output_images_{fold}/test_images/', True)

                            reconstruction_test_loss = mse_loss(reconstructed, target)

                            kl_test_loss = -0.5 *(1+log_combined_precision - combined_mu.pow(2)- log_combined_precision.exp()).mean()
                            #kl_test_loss = -0.5 * (1 + averaged_logvar - averaged_mu.pow(2) - averaged_logvar.exp()).mean()

                            t_loss = reconstruction_test_loss + kl_weight * kl_test_loss
                            test_vae_loss += t_loss.item()
                            test_mse_loss += reconstruction_test_loss.item()
                            test_kl_loss += kl_test_loss.item()

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

                            recon_loss = mse_loss(reconstructed, target)
                            kl_loss = posterior.kl().mean()
                            ssim_loss = 1 - ssim(reconstructed, target)
                            t_loss = recon_loss + kl_weight * kl_loss + 0 * ssim_loss

                            test_vae_loss += t_loss.item()
                            test_mse_loss += recon_loss.item()
                            test_kl_loss += kl_loss.item()
                            test_ssim_loss += ssim_loss.item()

                            if epoch%5 == 0 and batch < 5:
                                visulaize_images(self.multi_image_mode,batch, epoch, reconstructed, images,None,'./output_images/individual_images/test_images/', True)

                                # Printing the step loss values in the progress bar
                                test_progress_bar.set_postfix({
                                    'Test Total loss': t_loss.item(),
                                    'Test Reconstruction_loss': recon_loss.item(),
                                    'KL_loss': kl_loss.item(),
                                    'SSIM Loss': ssim_loss.item()
                                })

                            '''plot_image = reconstructed[0].permute(1, 2, 0).detach().cpu().numpy()
                            plot_image = ((plot_image + 1)/2).clip(0,1)
                            #print(plot_image.min(),plot_image.max())
                            wandb_images = wandb.Image(plot_image)
                            run.log({'examples':wandb_images})'''


            #print(f"Epoch {epoch + 1}:  \n Train Loss: {train_vae_loss/len(train_loader):.4f}, \n Val Loss: {val_vae_loss/len(val_loader)} \n Test Loss: {test_vae_loss/len(test_loader):.4f}")
            print(f"Epoch {epoch + 1}:  \n Train Loss: {train_vae_loss / len(train_loader):.4f} \n Val Loss: {val_vae_loss/len(val_loader)} \n Test Loss: {test_vae_loss / len(test_loader):.4f}")
            print(f"Train MSE Loss: {train_mse_loss / len(train_loader):.4f} \n Val MSE Loss: {val_mse_loss/len(val_loader)} \n Test MSE Loss: {test_mse_loss / len(test_loader):.4f}")
            print(f"Train KL Loss: {train_kl_loss / len(train_loader):.4f} \n Val KL Loss: {val_kl_loss/len(val_loader)} \n Test KL Loss: {test_kl_loss / len(test_loader):.4f}")
            print(f"Train SSIM Loss: {train_ssim_loss / len(train_loader):.4f} \n Val SSIM Loss: {val_ssim_loss/len(val_loader)}  \n Test SSIM Loss: {test_ssim_loss / len(test_loader):.4f}")

            if test_vae_loss < best_loss:
                best_loss = test_vae_loss
                torch.save(self.vae,'./finetuned_vae/VAE_individual_checkpoint.pth')
                print(f'Model Saved with test loss {test_vae_loss/len(test_loader):.4f}')
                self.vae.save_pretrained('finetuned_vae/individually_trained_models')

            #run.log({ 'Train_Loss': train_vae_loss/len(train_loader), 'Val_Loss': val_vae_loss/len(val_loader), 'Test_Loss': test_vae_loss/len(test_loader) }) ## Group multiple metrics as one in the UI
            print('Logging values to W&B...')
            run.log({'Train_Total_Loss': train_vae_loss / len(train_loader), 'Test_Total_Loss': test_vae_loss / len(test_loader), 'Val Total Loss': val_vae_loss/len(val_loader),
                     'Train_MSE_Loss': train_mse_loss / len(train_loader), 'Test_MSE_Loss': test_mse_loss / len(test_loader),'Val MSE Loss': val_mse_loss/len(val_loader),
                     'Train_SSIM_Loss': train_ssim_loss / len(train_loader), 'Test_SSIM_Loss': test_ssim_loss / len(test_loader), 'Val SSIM Loss':val_ssim_loss/len(val_loader),
                     'Train_KL_Loss': train_kl_loss / len(train_loader), 'Test_KL_Loss': test_kl_loss / len(test_loader),'Val KL Loss':val_kl_loss/len(val_loader)
                     })













