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
from cldice_vis import *
#from input_pipeline_discrete import *
from tqdm import tqdm
from torchmetrics import StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt
import torchvision.models as models
from thesis_visualisation import visualize_comparison_with_overlay
from visualizations import *
from attSets import AttSets
from cldice_loss_vae import *

class VAEGANFineTuner:

    def __init__(self, model_name='stabilityai/sd-vae-ft-mse', multi_image_mode=True, device=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multi_image_mode = multi_image_mode
        self.vae = AutoencoderKL.from_pretrained(model_name)
        self.aggregation_module = AttSets(latent_channels=4)
        self.aggregation_module = self.aggregation_module.to(device)
        self.vae.to(self.device)
        #self.PROJECT_NAME = 'Individual Reconstruction Cross Validation'
        self.PROJECT_NAME = 'Topology Optimization_final'


    def finetune(self, train_loader, val_loader, test_loader,fold, epochs=101, lr=0.001, kl_weight=1e-6, latent_technique = 'learned_aggregation', save_path="./finetuned_vae"):

        # Start a new wandb run to track this script.
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="veenasridhar96-university-of-stuttgart",
            # Set the wandb project where this run will be logged.
            #project=self.PROJECT_NAME,
            name = f'Topology Optimization_{latent_technique}_{fold}',
            #job_type= f'run- {fold}',
            # Track hyperparameters and run metadata.
            config={
                "learning_rate": lr,
                #'kl_weight': kl_weight,
                'latent_technique': latent_technique,
                'loss': 'CE + clDice',
                "architecture": "VAE_Decoder training last two layers",
                "dataset": "OpenTO + TO",
                "epochs": epochs,
                "fold": fold,
            }, )

        #torch.autograd.set_detect_anomaly(True)
        save_path = f'VAE TO Dataset Results_{latent_technique}_{fold}_with decoder training_0.3ce+0.7cld'
        #save_path = 'Pretrained Individual Reconstruction Results'

        if self.multi_image_mode:

            for name, module in self.vae.named_modules():
            #print(name, "->", module.__class__.__name__)
                for p in module.parameters():
                        p.requires_grad = False

            trainable_modules = [
                #self.vae.decoder.up_blocks[2],
                self.vae.decoder.up_blocks[3],   # final upsampling block (highest resolution)
                self.vae.decoder.conv_norm_out,
                self.vae.decoder.conv_out,
            ]
            for module in trainable_modules:
                for p in module.parameters():
                    p.requires_grad = True

            decoder_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.vae.decoder.parameters()), lr=lr, weight_decay=1e-4)
            
            if latent_technique == 'learned_aggregation':
                aggregation_module_optimizer = torch.optim.AdamW(self.aggregation_module.parameters(), lr=0.001, weight_decay=1e-4)
                print('The training parameters for the aggregation module are',sum(p.numel() for p in self.aggregation_module.parameters() if p.requires_grad))

            #print('The training parameters for the decoder module are',sum(p.numel() for p in self.vae.decoder.parameters() if p.requires_grad))
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(decoder_optimizer, T_max=epochs)
        else:
            optimizer = torch.optim.AdamW(self.vae.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        
        cldice_loss = soft_dice_cldice()
        #ce_weights = torch.tensor([0.25, 0.35, 0.5],device=self.device)weight= ce_weights
        ce_loss = torch.nn.CrossEntropyLoss()
        best_loss = float('inf')
        best_val_loss = float('inf')

        for epoch in range(epochs):
            
            self.vae.train()
            train_vae_loss = 0.0
            train_recon_loss = 0.0
            train_kl_loss = 0.0
            train_iou_dl = 0.0
            train_iou_img_prc = 0.0
            train_cldice_dl =0.0
            train_cldice_img_prc = 0.0
            train_clprec_dl =0.0
            train_clprec_img_prc = 0.0
            train_clrec_dl =0.0
            train_clrec_img_prc = 0.0
            train_total_ce_loss = 0.0
            train_total_cldice_loss = 0.0

            train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
            index =0

            if self.multi_image_mode:
                accum_steps = 8
                #decoder_optimizer.zero_grad()
                
                if latent_technique == 'learned_aggregation':
                    self.aggregation_module.train()
                    aggregation_module_optimizer.zero_grad()

                for batch, input in enumerate(train_progress_bar):

                    images, gt, avg_result = input
                    print('The image shape is',images.shape)
                    images = images.to(self.device)         # (B,N,C,H,W) = (1,5,3,256,256)
                    target = gt.to(self.device)             # (B,C,H,W) = (1,3,256,256)

                    batch_size, num_imgs = images.shape[0], images.shape[1]
                    combinedimages = images.reshape(-1, *images.shape[2:]).to(self.device)          #(B*N,C,H,W) = (10,3,256,256)
                    # The above line reshapes the input images from (B,N,C,H,W) to (B*N,C,H,W) where *images.shape[2:] (the * is called the argument unpacking unpacks (C,H,W) and the -1
                    # in the first indes allows python to infer the dimensions on its own while maintaining the data count so for the first indes B*N is done.)

                    # Sampling happens from Diffusers library
                    #print('The shape of combined images',combinedimages.shape)

                    with torch.no_grad():                                               # Encoder is not trained during the multi iamge mode
                        print('The unique values of the input images are',combinedimages.unique())
                        posterior = self.vae.encode(combinedimages).latent_dist        # Posterior - N(mu,std), latent_dist - Encoded outputs of Encoder represented as the mean and logvar of DiagonalGaussianDistribution. DiagonalGaussianDistribution allows for sampling latents from the distribution
                        #target_posterior = self.vae.encode(target).latent_dist
                        individual_mu = posterior.mean       # Individual mu is mean represented using 4 latent variables (10,4,32,32)
                        log_var = posterior.logvar          # Log variance represent the uncertainty or spread of the distribution. (log for stability)
                        #target_mu = target_posterior.mean

                    # Reshape it to (B,N,latent_dimension,H,W)
                    mu = individual_mu.reshape(batch_size, num_imgs, *individual_mu.shape[1:])  # (B,N,latent_dimension,H,W) = (2,5,4,32,32)
                    log_var = log_var.reshape(batch_size, num_imgs, *log_var.shape[1:])         # (B,N,latent_dimension,H,W) = (2,5,4,32,32)

                    #visualize_interpolation(images, mu,self.vae)

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
                        z_averaged = averaged_mu
                        #z_averaged = averaged_mu + averaged_std_dev * torch.randn_like(averaged_mu)           # (B,C,H,W)

                    elif latent_technique == 'median':
                        median_mu = mu.median(dim=1).values
                        median_logvar = log_var.median(dim=1).values
                        median_std_dev = (0.5 * median_logvar).exp()
                        z_averaged = median_mu + median_std_dev * torch.randn_like(median_mu)

                    elif latent_technique == 'learned_aggregation':
                        z_averaged =  self.aggregation_module(individual_mu, sample = False,num_imgs=num_imgs, batch_size=batch_size)

                    '''if epoch%5 ==0 and batch < 10:
                        visualize_latent_space(individual_mu, combined_mu,target_mu,)'''

                    #with torch.no_grad():
                    reconstructed = self.vae.decode(z_averaged).sample

                    print('The unique elements in reconstructed image',reconstructed.unique())
                    print('The unique elements in target image',target.unique())


                    train_ce_loss = ce_loss(reconstructed,target)
                    probs       = torch.softmax(reconstructed, dim=1)
                    member_prob = probs[:, 2:3]
                    member_gt   = (target == 2).float().unsqueeze(1)
                    train_cldice_loss = cldice_loss(member_prob,member_gt)
                    reconstruction_loss =  0.3 * train_ce_loss + 0.7 * train_cldice_loss
                    #reconstruction_loss =  train_cldice_loss
                    #latent_space_exploration(individual_mu, z_averaged, target_mu, self.vae)

                    loss = reconstruction_loss

                    #Backpropogation based on every step in the epoch - Gradient Accumulation
                    (loss / accum_steps).backward()
                    if (batch + 1) % accum_steps == 0:
                        # torch.nn.utils.clip_grad_norm_(self.vae.decoder.parameters(), 1.0)
                        if latent_technique == 'learned_aggregation':
                            torch.nn.utils.clip_grad_norm_(self.aggregation_module.parameters(), 1.0)
                            aggregation_module_optimizer.step()
                            aggregation_module_optimizer.zero_grad()
                        decoder_optimizer.step()
                        decoder_optimizer.zero_grad()

                    #Aggregating the loss values for each step in the epoch to output finally as a epoch loss values
                    train_vae_loss +=loss.item()
                    train_recon_loss +=reconstruction_loss.item()
                    train_total_ce_loss += train_ce_loss.item()
                    train_total_cldice_loss += train_cldice_loss.item()

                    for i in range(len(reconstructed)):
                        
                        # print('The shape of the reconstructed image',reconstructed[i].shape)
                        # print('The shape of the target image',target[i].shape)
                        # print('The shape of avg_result image',avg_result[i].shape)

                        reconstructed = torch.argmax(reconstructed[i], dim=0)

                        predicted_binary = (reconstructed == 2).detach().cpu().numpy().astype(np.float32)
                        gt_binary = (target[i] == 2).detach().cpu().numpy().astype(np.float32)
                        img_prc_binary = (avg_result[i] ==2).detach().cpu().numpy().astype(np.float32)

                        predicted_skeleton = skeletonize(predicted_binary > 0).astype(np.float32)
                        gt_skeleton = skeletonize(gt_binary > 0).astype(np.float32)
                        img_prc_skeleton = skeletonize(img_prc_binary > 0).astype(np.float32)

                        cl_prec = np.sum(predicted_skeleton * gt_binary) / (np.sum(predicted_skeleton) + 1e-7)
                        cl_rec = np.sum(gt_skeleton * predicted_binary) / (np.sum(gt_skeleton) + 1e-7)
                        cl_dice_dl = 2 * cl_prec * cl_rec / (cl_prec + cl_rec)

                        cl_prec_i = np.sum(img_prc_skeleton * gt_binary) / (np.sum(img_prc_skeleton) + 1e-7)
                        cl_rec_i = np.sum(gt_skeleton * img_prc_binary) / (np.sum(gt_skeleton) + 1e-7)
                        cl_dice_img_prc = 2 * cl_prec_i * cl_rec_i / (cl_prec_i + cl_rec_i)

                        train_cldice_dl += cl_dice_dl
                        train_cldice_img_prc += cl_dice_img_prc
                        train_clprec_dl += cl_prec
                        train_clprec_img_prc += cl_prec_i
                        train_clrec_dl += cl_rec
                        train_clrec_img_prc += cl_rec_i
                        iou_dilated,iou_dilated_image_processing,dilated_overlap,img_proc_dilated_overlap = visulaize_iou_metrics(predicted_binary, gt_binary, img_prc_binary, tolerance=0)
                        train_iou_dl += iou_dilated
                        train_iou_img_prc += iou_dilated_image_processing

                        if epoch % 50 == 0 and index<500:
                            if not os.path.exists(os.path.join(f'./{save_path}_{latent_technique}_{fold}')):
                                os.mkdir(os.path.join(f'./{save_path}_{latent_technique}_{fold}'))
                            if not os.path.exists(os.path.join(f'./{save_path}_{latent_technique}_{fold}/train')):
                                os.mkdir(os.path.join(f'./{save_path}_{latent_technique}_{fold}/train'))
                            if not os.path.exists(os.path.join(f'./{save_path}_{latent_technique}_{fold}/train/Epoch {epoch}')):
                                os.mkdir(os.path.join(f'./{save_path}_{latent_technique}_{fold}/train/Epoch {epoch}'))
                            if not os.path.exists(os.path.join(f'./{save_path}_{latent_technique}_{fold}/train/Epoch {epoch}', 'Metrics')):
                                os.mkdir(os.path.join(f'./{save_path}_{latent_technique}_{fold}/train/Epoch {epoch}','Metrics'))

                            # plt.imsave(os.path.join(f'./{save_path}_{latent_technique}_{fold}/train/Epoch {epoch}', f'{save_path}_{index}.png'),
                            #        reconstructed.detach().cpu().numpy(), cmap='gray', vmin = reconstructed.min(), vmax = reconstructed.max())
                            
                            group = {
                                'inputs': images[i].detach().cpu().numpy(),
                                'gt': target[i].detach().cpu().numpy(),
                                'gt_foreground': gt_binary,
                                'pred': reconstructed.detach().cpu().numpy(),
                                'pred_foreground':predicted_binary,
                                'baseline': avg_result[i].detach().cpu().numpy(),
                                'baseline_foreground':img_prc_binary,
                                'label': f'Batch {index}',
                                # clDice already computed above — just pass them in
                                'cld_pred': cl_dice_dl,
                                'prec_pred': cl_prec,
                                'rec_pred': cl_rec,
                                'cld_base': cl_dice_img_prc,
                                'prec_base': cl_prec_i,
                                'rec_base': cl_rec_i,
                                'iou_base': iou_dilated_image_processing,
                                'iou_pred': iou_dilated
                            }
                            visualize_cldice_comparison_grid_discrete([group], epoch=epoch, batch=index, fold=fold,save_path=f'./{save_path}_{latent_technique}_{fold}/train/Epoch {epoch}/Metrics')
                        
                            # visulaize_images(self.multi_image_mode, batch, epoch, reconstructed[i], images[i], target[i], avg_result[i], iou_dilated,iou_dilated_image_processing,dilated_overlap,
                            #                  img_proc_dilated_overlap, f'./{save_path}_{latent_technique}_{fold}/train/Epoch {epoch}/Metrics', index, False)
                            index += 1

                    # Printing the step loss values in the progress bar
                    train_progress_bar.set_postfix({
                        'Total loss': loss.item(),
                        'reconstruction_loss': reconstruction_loss.item(),
                    })

                if (batch + 1) % accum_steps != 0:
                    torch.nn.utils.clip_grad_norm_(self.vae.decoder.parameters(), 1.0)
                    if latent_technique == 'learned_aggregation':
                        torch.nn.utils.clip_grad_norm_(self.aggregation_module.parameters(), 1.0)
                        aggregation_module_optimizer.step()
                        aggregation_module_optimizer.zero_grad()
                    decoder_optimizer.step()
                    decoder_optimizer.zero_grad()
                
                #scheduler.step()
        
            ###################################
            ### Single Image Mode
            ###################################
            else:
                accum_steps = 256
                optimizer.zero_grad()
                for batch, (images,target) in enumerate(train_progress_bar):

                    images = images.to(self.device)
                    target = target.to(self.device)

                    print('The shape of the input images are',images.shape)

                    posterior = self.vae.encode(images).latent_dist      #  Posterior - N(mu,std), latent_dist - Encoded outputs of Encoder represented as the mean and logvar of DiagonalGaussianDistribution. DiagonalGaussianDistribution allows for sampling latents from the distribution
                    individual_mu = posterior.mean

                    z = posterior.sample()                              # Reparametrization happens internally as it is a function from DiagonalGaussianDistribution class based on the Stable Diffusion VAE model used.
                    '''if epoch % 5 == 0 and batch < 10:
                        visualize_latent_space_individual(z)                           # z is of shape (B,C,H,W) = (16,4,32,32)'''

                    reconstructed = self.vae.decode(z).sample

                    print('Reconstructed Unique Values',reconstructed.unique())
                    print('Target Unique Values',target.unique())
                    print('Reconstructed shape', reconstructed.shape)
                    print('Target Unique Values', target.shape)

                    recon_loss = ce_loss(reconstructed,target)
                    probs       = torch.softmax(reconstructed, dim=1)
                    member_prob = probs[:, 2:3]
                    member_gt   = (target == 2).float().unsqueeze(1)
                    train_cldice_loss = cldice_loss(member_prob,member_gt)
                    kl_loss = posterior.kl().mean()
                    loss = recon_loss + kl_weight * kl_loss + train_cldice_loss

                    (loss / accum_steps).backward()
                    if (batch + 1) % accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()

                    # Aggregating the loss values for each step in the epoch to output finally as a epoch loss values
                    train_vae_loss += loss.item()
                    train_recon_loss += recon_loss.item()
                    train_kl_loss += kl_loss.item()
                    

                    if epoch%5 == 0 :
                        if not os.path.exists(os.path.join(save_path)):
                                os.mkdir(os.path.join(save_path))
                        if not os.path.exists(os.path.join(f'{save_path}/individual_images_Fold {fold}')):
                                    os.mkdir(os.path.join(f'{save_path}/individual_images_Fold {fold}'))
                        if not os.path.exists(os.path.join(f'{save_path}/individual_images_Fold {fold}/train')):
                            os.mkdir(os.path.join(f'{save_path}/individual_images_Fold {fold}/train'))
                        if not os.path.exists(os.path.join(f'{save_path}/individual_images_Fold {fold}/train/Epoch {epoch}')):
                            os.mkdir(os.path.join(f'{save_path}/individual_images_Fold {fold}/train/Epoch {epoch}'))
                        print('The shape of the reconstructed individual image',reconstructed.shape)
                        reconstructed = torch.argmax(reconstructed, dim=1)

                        if index < 50:
                            visulaize_images_individual(batch,epoch, reconstructed, target,None, f'{save_path}/individual_images_Fold {fold}/train/Epoch {epoch}',False)
                        index += 1
                        #latent_space_exploration(individual_mu, None, None, self.vae)
                        #visualize_feature_maps(images, individual_mu, False)

                    # Printing the step loss values in the progress bar
                    train_progress_bar.set_postfix({
                        'Total loss': loss.item(),
                        'reconstruction_loss': recon_loss.item(),
                        'KL_loss': kl_loss.item(),
                        'CLDice_loss': train_cldice_loss.item()
                    })

                if (batch + 1) % accum_steps != 0:
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            val_vae_loss = 0.0
            val_mse_loss = 0.0
            val_kl_loss = 0.0
            val_iou_dl = 0.0
            val_iou_img_prc = 0.0
            val_cldice_dl =0.0
            val_cldice_img_prc =0.0
            val_clprec_dl =0.0
            val_clprec_img_prc =0.0
            val_clrec_dl =0.0
            val_clrec_img_prc =0.0
            val_total_ce_loss = 0.0
            val_total_cldice_loss = 0.0

            val_progress_bar = tqdm(val_loader,desc=f'Epoch {epoch}/{epochs}')
            if val_loader:
                val_index = 0
                print('The length of val loader', len(val_loader))
                self.vae.eval()
                if latent_technique == 'learned_aggregation':
                    self.aggregation_module.eval()
                with torch.no_grad():
                    if self.multi_image_mode:
                        index = 0
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

                            elif latent_technique == 'median':
                                median_mu = mu.median(dim=1).values
                                median_logvar = log_var.median(dim=1).values
                                median_std_dev = (0.5 * median_logvar).exp()

                                z_averaged = median_mu + median_std_dev * torch.randn_like(median_mu)

                            elif latent_technique == 'learned_aggregation':
                                 z_averaged =  self.aggregation_module(individual_mu, sample = False, num_imgs=num_imgs, batch_size=batch_size)

                            '''if epoch % 5 == 0 and batch < 10:
                                visualize_latent_space(individual_mu, combined_mu, target_mu,epoch,batch,)'''

                            reconstructed = self.vae.decode(z_averaged).sample

                            #latent_space_exploration(individual_mu, averaged_mu, target_mu, self.vae)

                            val_ce_loss = ce_loss(reconstructed,target)
                            probs       = torch.softmax(reconstructed, dim=1)
                            member_prob = probs[:, 2:3]
                            member_gt   = (target == 2).float().unsqueeze(1)
                            val_cldice_loss = cldice_loss(member_prob,member_gt)
                            reconstruction_val_loss = 0.3 * val_ce_loss + 0.7 * val_cldice_loss
                            #reconstruction_val_loss = val_cldice_loss

                            val_loss = reconstruction_val_loss

                            val_vae_loss += val_loss.item()
                            val_mse_loss += reconstruction_val_loss.item()
                            val_total_ce_loss += val_ce_loss.item()
                            val_total_cldice_loss += val_cldice_loss.item()

                            for i in range(len(reconstructed)):
                        
                                # print('The shape of the reconstructed image',reconstructed[i].shape)
                                # print('The shape of the target image',target[i].shape)
                                # print('The shape of avg_result image',avg_result[i].shape)

                                reconstructed = torch.argmax(reconstructed[i], dim=0)
                                
                                predicted_binary = (reconstructed == 2).detach().cpu().numpy().astype(np.float32)
                                gt_binary = (target[i] == 2).detach().cpu().numpy().astype(np.float32)
                                img_prc_binary = (avg_result[i] ==2).detach().cpu().numpy().astype(np.float32)

                                predicted_skeleton = skeletonize(predicted_binary > 0).astype(np.float32)
                                gt_skeleton = skeletonize(gt_binary > 0).astype(np.float32)
                                img_prc_skeleton = skeletonize(img_prc_binary > 0).astype(np.float32)

                                cl_prec = np.sum(predicted_skeleton * gt_binary) / (np.sum(predicted_skeleton) + 1e-7)
                                cl_rec = np.sum(gt_skeleton * predicted_binary) / (np.sum(gt_skeleton) + 1e-7)
                                cl_dice_dl = 2 * cl_prec * cl_rec / (cl_prec + cl_rec)

                                cl_prec_i = np.sum(img_prc_skeleton * gt_binary) / (np.sum(img_prc_skeleton) + 1e-7)
                                cl_rec_i = np.sum(gt_skeleton * img_prc_binary) / (np.sum(gt_skeleton) + 1e-7)
                                cl_dice_img_prc = 2 * cl_prec_i * cl_rec_i / (cl_prec_i + cl_rec_i)

                                val_cldice_dl += cl_dice_dl
                                val_cldice_img_prc += cl_dice_img_prc
                                val_clprec_dl += cl_prec
                                val_clprec_img_prc += cl_prec_i
                                val_clrec_dl += cl_rec
                                val_clrec_img_prc += cl_rec_i
                                iou_dilated,iou_dilated_image_processing,dilated_overlap,img_proc_dilated_overlap = visulaize_iou_metrics(predicted_binary, gt_binary, img_prc_binary, tolerance=0)
                                val_iou_dl += iou_dilated
                                val_iou_img_prc += iou_dilated_image_processing

                                if val_vae_loss < best_val_loss and index<500:
                                    best_val_loss = val_vae_loss
                                    if not os.path.exists(os.path.join(f'./{save_path}_{latent_technique}_{fold}')):
                                        os.mkdir(os.path.join(f'./{save_path}_{latent_technique}_{fold}'))
                                    if not os.path.exists(os.path.join(f'./{save_path}_{latent_technique}_{fold}/val')):
                                        os.mkdir(os.path.join(f'./{save_path}_{latent_technique}_{fold}/val'))
                                    if not os.path.exists(os.path.join(f'./{save_path}_{latent_technique}_{fold}/val/Epoch {epoch}')):
                                        os.mkdir(os.path.join(f'./{save_path}_{latent_technique}_{fold}/val/Epoch {epoch}'))
                                    if not os.path.exists(os.path.join(f'./{save_path}_{latent_technique}_{fold}/val/Epoch {epoch}', 'Metrics')):
                                        os.mkdir(os.path.join(f'./{save_path}_{latent_technique}_{fold}/val/Epoch {epoch}','Metrics'))

                                    # plt.imsave(os.path.join(f'./{save_path}_{latent_technique}_{fold}/val/Epoch {epoch}', f'{save_path}_{index}.png'),
                                    #     reconstructed.detach().cpu().numpy(), cmap='gray', vmin = reconstructed.min(), vmax = reconstructed.max())


                                    group = {
                                        'inputs': images[i].detach().cpu().numpy(),
                                        'gt': target[i].detach().cpu().numpy(),
                                        'gt_foreground': gt_binary,
                                        'pred': reconstructed.detach().cpu().numpy(),
                                        'pred_foreground':predicted_binary,
                                        'baseline': avg_result[i].detach().cpu().numpy(),
                                        'baseline_foreground':img_prc_binary,
                                        'label': f'Batch {index}',
                                        # clDice already computed above — just pass them in
                                        'cld_pred': cl_dice_dl,
                                        'prec_pred': cl_prec,
                                        'rec_pred': cl_rec,
                                        'cld_base': cl_dice_img_prc,
                                        'prec_base': cl_prec_i,
                                        'rec_base': cl_rec_i,
                                        'iou_base': val_iou_img_prc,
                                        'iou_pred': val_iou_dl
                                    }
                                    visualize_cldice_comparison_grid_discrete([group], epoch=epoch, batch=index, fold=fold,save_path=f'./{save_path}_{latent_technique}_{fold}/val/Epoch {epoch}/Metrics')
                                
                                    # visulaize_images(self.multi_image_mode, batch, epoch, reconstructed[i], images[i], target[i], avg_result[i], iou_dilated,iou_dilated_image_processing,dilated_overlap,
                                    #                  img_proc_dilated_overlap, f'./{save_path}_{latent_technique}_{fold}/train/Epoch {epoch}/Metrics', index, False)
                                    index += 1


                            # Printing the step loss values in the progress bar
                            val_progress_bar.set_postfix({
                                'Total Val loss': val_loss.item(),
                                'Val reconstruction_loss': reconstruction_val_loss.item()
                            })

                    else:
                        for batch, (images,target) in enumerate(val_progress_bar):

                            images = images.to(self.device)
                            target = target.to(self.device)

                            posterior = self.vae.encode(images).latent_dist
                            individual_mu = posterior.mean
                            z = posterior.sample()
                            reconstructed = self.vae.decode(z).sample

                            
                            recon_loss = ce_loss(reconstructed,target)
                            probs       = torch.softmax(reconstructed, dim=1)
                            member_prob = probs[:, 2:3]
                            member_gt   = (target == 2).float().unsqueeze(1)
                            val_cldice_loss = cldice_loss(member_prob,member_gt)
                                
                            kl_loss = posterior.kl().mean()

                            val_loss = recon_loss + kl_weight * kl_loss + val_cldice_loss

                            if epoch%5 == 0 :
                                if not os.path.exists(os.path.join(f'{save_path}')):
                                        os.mkdir(os.path.join(f'{save_path}'))
                                if not os.path.exists(os.path.join(f'{save_path}/individual_images_Fold {fold}')):
                                    os.mkdir(os.path.join(f'{save_path}/individual_images_Fold {fold}'))
                                if not os.path.exists(os.path.join(f'{save_path}/individual_images_Fold {fold}/val')):
                                    os.mkdir(os.path.join(f'{save_path}/individual_images_Fold {fold}/val'))
                                if not os.path.exists(os.path.join(f'{save_path}/individual_images_Fold {fold}/val/Epoch {epoch}')):
                                    os.mkdir(os.path.join(f'{save_path}/individual_images_Fold {fold}/val/Epoch {epoch}'))
                                print('The shape of the reconstructed individual image',reconstructed.shape)
                                reconstructed = torch.argmax(reconstructed, dim=1)
                                if val_index < 50:
                                    visulaize_images_individual(batch,epoch, reconstructed, target,None, f'{save_path}/individual_images_Fold {fold}/val/Epoch {epoch}',False)
                                val_index += 1

                                # latent_space_exploration(individual_mu, None, None, self.vae)
                                # visualize_feature_maps(images, individual_mu, False)

                            val_vae_loss += val_loss.item()
                            val_mse_loss += recon_loss.item()
                            val_kl_loss += kl_loss.item()
                            val_cldice_loss += val_cldice_loss.item()
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
            test_iou_dl =0.0
            test_iou_img_prc = 0.0
            test_cldice_dl =0.0
            test_cldice_img_prc = 0.0
            test_clprec_dl =0.0
            test_clprec_img_prc =0.0
            test_clrec_dl =0.0
            test_clrec_img_prc =0.0
            test_total_ce_loss = 0.0
            test_total_cldice_loss = 0.0

            test_progress_bar = tqdm(test_loader, desc=f"Epoch {epoch}/{epochs}")
            if test_loader:
                print('The length of test loader',len(test_loader))
                self.vae.eval()
                index =0
                with torch.no_grad():
                    if self.multi_image_mode:
                        for batch, input in enumerate(test_progress_bar):
                            images, labels, avg_result = input
                            target = labels
                            target = target.to(self.device)

                            batch_size, num_imgs = images.shape[0], images.shape[1]
                            combinedimages = images.reshape(-1, *images.shape[2:]).to(self.device)

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
                                combined_precision = precision.sum(dim=1)  # Precision summed along the number of images (B,latent_dim,H,W)
                                combined_mu = (mu * precision).sum(dim=1) / combined_precision
                                combined_log_variance = -torch.log(combined_precision)
                                combined_std_dev = (0.5 * combined_log_variance).exp()  # Standard deviation
                                z_averaged = combined_mu + combined_std_dev * torch.randn_like(combined_mu)

                            elif latent_technique == 'median':
                                median_mu = mu.median(dim=1).values
                                median_logvar = log_var.median(dim=1).values
                                median_std_dev = (0.5 * median_logvar).exp()

                                z_averaged = median_mu + median_std_dev * torch.randn_like(median_mu)

                            elif latent_technique == 'averaging':
                                # Averaging the distributions of images in the latent space
                                averaged_mu = mu.mean(dim=1)  # (B,C,H,W) = (2,4,32,32)
                                averaged_logvar = log_var.mean(dim=1)  # (B,C,H,W) = (2,4,32,32)
                                averaged_std_dev = (0.5 * averaged_logvar).exp()  # Converting the log variance $square(\sigma)$ into standard deviation (just sigma) - (B,C,H,W)

                                # Sample a Z from the mean distribution using the REPARAMETERIZATION TRICK.  Z = \mu + \std * \epsilon (noise sampled from N(0,1)
                                z_averaged = averaged_mu + averaged_std_dev * torch.randn_like(averaged_mu)  # (B,C,H,W)

                            elif latent_technique == 'learned_aggregation':
                                 z_averaged =  self.aggregation_module(individual_mu, sample = False,num_imgs=num_imgs, batch_size=batch_size)

                            reconstructed = self.vae.decode(z_averaged).sample

                            test_ce_loss = ce_loss(reconstructed,target)
                            probs       = torch.softmax(reconstructed, dim=1)
                            member_prob = probs[:, 2:3]
                            member_gt   = (target == 2).float().unsqueeze(1)
                            test_cldice_loss = cldice_loss(member_prob,member_gt)
                            #reconstruction_loss =  test_ce_loss + test_cldice_loss
                            reconstruction_test_loss = 0.3 * test_ce_loss + 0.7 * test_cldice_loss
                            #reconstruction_test_loss = test_cldice_loss
                            #reconstruction_test_loss = mse_loss(reconstructed, target)

                            t_loss = reconstruction_test_loss

                            test_vae_loss += t_loss.item()
                            test_mse_loss += reconstruction_test_loss.item()
                            test_total_ce_loss += test_ce_loss.item()
                            test_total_cldice_loss += test_cldice_loss.item()

                            for i in range(len(reconstructed)):

                                reconstructed = torch.argmax(reconstructed[i], dim=0)
                                
                                predicted_binary = (reconstructed == 2).detach().cpu().numpy().astype(np.float32)
                                gt_binary = (target[i] == 2).detach().cpu().numpy().astype(np.float32)
                                img_prc_binary = (avg_result[i] ==2).detach().cpu().numpy().astype(np.float32)

                                predicted_skeleton = skeletonize(predicted_binary > 0).astype(np.float32)
                                gt_skeleton = skeletonize(gt_binary > 0).astype(np.float32)
                                img_prc_skeleton = skeletonize(img_prc_binary > 0).astype(np.float32)

                                cl_prec = np.sum(predicted_skeleton * gt_binary) / (np.sum(predicted_skeleton) + 1e-7)
                                cl_rec = np.sum(gt_skeleton * predicted_binary) / (np.sum(gt_skeleton) + 1e-7)
                                cl_dice_dl = 2 * cl_prec * cl_rec / (cl_prec + cl_rec)

                                cl_prec_i = np.sum(img_prc_skeleton * gt_binary) / (np.sum(img_prc_skeleton) + 1e-7)
                                cl_rec_i = np.sum(gt_skeleton * img_prc_binary) / (np.sum(gt_skeleton) + 1e-7)
                                cl_dice_img_prc = 2 * cl_prec_i * cl_rec_i / (cl_prec_i + cl_rec_i)

                                test_cldice_dl += cl_dice_dl
                                test_cldice_img_prc += cl_dice_img_prc
                                test_clprec_dl += cl_prec
                                test_clprec_img_prc += cl_prec_i
                                test_clrec_dl += cl_rec
                                test_clrec_img_prc += cl_rec_i
                                iou_dilated,iou_dilated_image_processing,dilated_overlap,img_proc_dilated_overlap = visulaize_iou_metrics(predicted_binary, gt_binary, img_prc_binary, tolerance=0)
                                test_iou_dl += iou_dilated
                                test_iou_img_prc += iou_dilated_image_processing

                                if test_vae_loss<best_loss and index<500:
                                
                                    if not os.path.exists(os.path.join(f'./{save_path}_{latent_technique}_{fold}')):
                                        os.mkdir(os.path.join(f'./{save_path}_{latent_technique}_{fold}'))
                                    if not os.path.exists(os.path.join(f'./{save_path}_{latent_technique}_{fold}/test')):
                                        os.mkdir(os.path.join(f'./{save_path}_{latent_technique}_{fold}/test'))
                                    if not os.path.exists(os.path.join(f'./{save_path}_{latent_technique}_{fold}/test/Epoch {epoch}')):
                                        os.mkdir(os.path.join(f'./{save_path}_{latent_technique}_{fold}/test/Epoch {epoch}'))
                                    if not os.path.exists(os.path.join(f'./{save_path}_{latent_technique}_{fold}/test/Epoch {epoch}', 'Metrics')):
                                        os.mkdir(os.path.join(f'./{save_path}_{latent_technique}_{fold}/test/Epoch {epoch}','Metrics'))

                                    # plt.imsave(os.path.join(f'./{save_path}_{latent_technique}_{fold}/test/Epoch {epoch}', f'{save_path}_{index}.png'),
                                    #         reconstructed.detach().cpu().numpy(), cmap='gray', vmin = reconstructed.min(), vmax = reconstructed.max())

                                    group = {
                                        'inputs': images[i].detach().cpu().numpy(),
                                        'gt': target[i].detach().cpu().numpy(),
                                        'gt_foreground': gt_binary,
                                        'pred': reconstructed.detach().cpu().numpy(),
                                        'pred_foreground':predicted_binary,
                                        'baseline': avg_result[i].detach().cpu().numpy(),
                                        'baseline_foreground':img_prc_binary,
                                        'label': f'Batch {index}',
                                        # clDice already computed above — just pass them in
                                        'cld_pred': cl_dice_dl,
                                        'prec_pred': cl_prec,
                                        'rec_pred': cl_rec,
                                        'cld_base': cl_dice_img_prc,
                                        'prec_base': cl_prec_i,
                                        'rec_base': cl_rec_i,
                                        'iou_base': test_iou_img_prc,
                                        'iou_pred': test_iou_dl
                                    }
                                    visualize_cldice_comparison_grid_discrete([group], epoch=epoch, batch=index, fold=fold,save_path=f'./{save_path}_{latent_technique}_{fold}/test/Epoch {epoch}/Metrics')
                                    
                                    # visulaize_images(self.multi_image_mode, batch, epoch, reconstructed, images[i], target[i], avg_result[i], iou_dilated,iou_dilated_image_processing,dilated_overlap,
                                    #              img_proc_dilated_overlap, f'./{save_path}_{latent_technique}_{fold}/test/Epoch {epoch}/Metrics', index, False)
                                    index += 1


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

                            #target = target.squeeze(dim=1)
                            # print('Reconstructed',reconstructed.shape)
                            # print('Target',target.shape)

                            recon_loss = ce_loss(reconstructed,target)
                            probs       = torch.softmax(reconstructed, dim=1)
                            member_prob = probs[:, 2:3]
                            member_gt   = (target == 2).float().unsqueeze(1)
                            test_cldice_loss = cldice_loss(member_prob,member_gt)
                                
                            kl_loss = posterior.kl().mean()

                            t_loss = recon_loss + kl_weight * kl_loss + test_cldice_loss

                            if epoch%5 == 0 :
                                if not os.path.exists(os.path.join(save_path)):
                                        os.mkdir(os.path.join(f'{save_path}'))
                                if not os.path.exists(os.path.join(f'{save_path}/individual_images_Fold {fold}')):
                                    os.mkdir(os.path.join(f'{save_path}/individual_images_Fold {fold}'))
                                if not os.path.exists(os.path.join(f'{save_path}/individual_images_Fold {fold}/test')):
                                    os.mkdir(os.path.join(f'{save_path}/individual_images_Fold {fold}/test'))
                                if not os.path.exists(os.path.join(f'{save_path}/individual_images_Fold {fold}/test/Epoch {epoch}')):
                                    os.mkdir(os.path.join(f'{save_path}/individual_images_Fold {fold}/test/Epoch {epoch}'))
                                print('The shape of the reconstructed individual image',reconstructed.shape)
                                reconstructed = torch.argmax(reconstructed, dim=1)
                                if index < 50:
                                    visulaize_images_individual(batch,epoch, reconstructed, target,None, f'{save_path}/individual_images_Fold {fold}/test/Epoch {epoch}',False)
                                index += 1

                            test_vae_loss += t_loss.item()
                            test_mse_loss += recon_loss.item()
                            test_kl_loss += kl_loss.item()
                            test_cldice_loss += test_cldice_loss.item()
                            #test_ssim_loss += ssim_loss.item()

                            #reconstructed = torch.argmax(reconstructed, dim=1)

                            # if epoch%50 == 0:
                            #     visulaize_images_individual(batch,epoch, reconstructed, images,None,f'./Individual Reconstruction Results/individual_images_Fold {fold}/test/', True)

                                # Printing the step loss values in the progress bar
                            test_progress_bar.set_postfix({
                                'Test Total loss': t_loss.item(),
                                'Test Reconstruction_loss': recon_loss.item(),
                                'KL_loss': kl_loss.item(),
                                #'SSIM Loss': ssim_loss.item()
                            })

            if test_vae_loss < best_loss:
                best_loss = test_vae_loss
                if self.multi_image_mode:
                    torch.save(self.vae, f'./finetuned_vae/VAE_common_structures_checkpoint_{latent_technique}.pth')
                    self.vae.save_pretrained(f'finetuned_vae/common_structures_recon_{latent_technique}')
                    if latent_technique == 'learned_aggregation':
                        torch.save({
                            'vae': self.vae,
                            'aggregation_module': self.aggregation_module,
                        }, f'./finetuned_vae/checkpoint_{latent_technique}.pth')

                else:
                    torch.save(self.vae,'./finetuned_vae/VAE_individual_checkpoint_ce_cldice.pth')
                    self.vae.save_pretrained('finetuned_vae/individually_trained_models_ce_cldice')

                print(f'Model Saved with test loss {test_vae_loss / len(test_loader):.4f}')

            #run.log({ 'Train_Loss': train_vae_loss/len(train_loader), 'Val_Loss': val_vae_loss/len(val_loader), 'Test_Loss': test_vae_loss/len(test_loader) }) ## Group multiple metrics as one in the UI
            print('Logging values to W&B...')
            run.log({'fold':fold,'epoch':epoch,
                    'Train_Total_Loss': train_vae_loss / len(train_loader), 'Test_Total_Loss': test_vae_loss / len(test_loader), 'Val Total Loss': val_vae_loss/len(val_loader),
                    'Train CE Loss':train_total_ce_loss / len(train_loader) ,'Test CE Loss':test_total_ce_loss / len(test_loader),'Val CE Loss':val_total_ce_loss/len(val_loader),
                    'Train Cldice Loss':train_total_cldice_loss / len(train_loader), 'Test Cldice Loss':test_total_cldice_loss / len(test_loader), 'Val Cldice Loss': val_total_cldice_loss/len(val_loader),
                    'Train cldice Img_prc': train_cldice_img_prc/len(train_loader.dataset), 'Train cldice DL': train_cldice_dl/len(train_loader.dataset),
                    'Val cldice Img_prc': val_cldice_img_prc/len(val_loader.dataset), 'Val cldice DL': val_cldice_dl /len(val_loader.dataset),
                    'Test cldice Img_prc': test_cldice_img_prc/len(test_loader.dataset), 'Test cldice DL': test_cldice_dl/len(test_loader.dataset),
                    'Train clrec Img_prc': train_clrec_img_prc/len(train_loader.dataset), 'Train clrec DL': train_clrec_dl/len(train_loader.dataset),
                    'Val clrec Img_prc': val_clrec_img_prc/len(val_loader.dataset), 'Val clrec DL': val_clrec_dl /len(val_loader.dataset),
                    'Test clrec Img_prc': test_clrec_img_prc/len(test_loader.dataset), 'Test clrec DL': test_clrec_dl/len(test_loader.dataset),
                    'Train cplrec Img_prc': train_clprec_img_prc/len(train_loader.dataset), 'Train clprec DL': train_clprec_dl/len(train_loader.dataset),
                    'Val cplrec Img_prc': val_clprec_img_prc/len(val_loader.dataset), 'Val clprec DL': val_clprec_dl /len(val_loader.dataset),
                    'Test cplrec Img_prc': test_clprec_img_prc/len(test_loader.dataset), 'Test clprec DL': test_clprec_dl/len(test_loader.dataset),
                    #'Learning Rate':decoder_optimizer.param_groups[0]['lr'],
                    'Train IoU Img_prc': train_iou_img_prc/len(train_loader.dataset), 'Train IoU DL': train_iou_dl/len(train_loader.dataset),
                    'Val IoU Img_prc': val_iou_img_prc/len(val_loader.dataset), 'Val IoU DL': val_iou_dl /len(val_loader.dataset),
                    'Test IoU Img_prc': test_iou_img_prc/len(test_loader.dataset), 'Test IoU DL': test_iou_dl/len(test_loader.dataset),
                    #'Train_MSE_Loss': train_mse_loss / len(train_loader), 'Test_MSE_Loss': test_mse_loss / len(test_loader),'Val MSE Loss': val_mse_loss/len(val_loader),
                     #'Train_SSIM_Loss': train_ssim_loss / len(train_loader), 'Test_SSIM_Loss': test_ssim_loss / len(test_loader), 'Val SSIM Loss':val_ssim_loss/len(val_loader),
                     #'Train_KL_Loss': train_kl_loss / len(train_loader), 'Test_KL_Loss': test_kl_loss / len(test_loader),'Val KL Loss':val_kl_loss/len(val_loader),
                   })

        wandb.finish()













