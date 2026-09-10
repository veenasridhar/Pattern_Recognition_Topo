import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from logger import setup_logger
import wandb
from cldice_loss_vae import *
from visualizations.visualizations import *
from visualizations.thesis_visualisation import *
from models import *
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Canonical label-map conventions (matches your to_label_map bucketing)
# ----------------------------------------------------------------------------
CLS_VOID, CLS_MEMBER = 0, 1
NUM_CLASSES = 2

decoder_lr = 0.0001

class CustomVAE:

    def __init__(self, model, multi_image_mode=True,device=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multi_image_mode = multi_image_mode
        self.vae = model
        #checkpoint = torch.load('finetuned_vae/Custom_VAE_OpenTO_aggmodule_checkpoint_learned_aggregation_0.pth',weights_only=False)
        self.aggregation_module = AttSets(latent_channels=4)
        #self.aggregation_module = checkpoint['aggregation_module']
        self.aggregation_module = self.aggregation_module.to(device)
        self.vae.to(self.device)

    def finetune(self, train_loader, val_loader, test_loader,fold, epochs=50, lr=0.001, kl_weight=1e-3, latent_technique = 'learned_aggregation', save_path="./finetuned_vae"):

        training_params = 0

        if self.multi_image_mode:

            logger = setup_logger(name="finetune", log_file="logs/VAE_finetune.log", level=logging.INFO)
            save_path = f'Custom VAE RailwayTO Consensus Reconstruction_{latent_technique}_{fold}_final'
            optimizer = torch.optim.AdamW(self.vae.decoder.parameters(), lr= decoder_lr) # weight_decay=1e-2
            if latent_technique == 'learned_aggregation':
                aggregation_module_optimizer = torch.optim.AdamW(self.aggregation_module.parameters(), lr=lr)
                logger.info('The training parameters for the aggregation module are',sum(p.numel() for p in self.aggregation_module.parameters() if p.requires_grad))
                training_params = sum(p.numel() for p in self.aggregation_module.parameters() if p.requires_grad)
            
            #To freeze the decoder. 
            # for name, module in self.vae.named_modules():
            #     #print(name, "->", module.__class__.__name__)
            #     for p in module.parameters():
            #         p.requires_grad = False

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            training_params +=sum(p.numel() for p in self.vae.decoder.parameters() if p.requires_grad)
            logger.info('The training parameters for the decoder module are %d',sum(p.numel() for p in self.vae.decoder.parameters() if p.requires_grad))
            logger.info('The training parameters for the entire module are %d',training_params)
        
        else:
            optimizer = torch.optim.AdamW(self.vae.parameters(), lr=lr) # weight_decay=1e-2
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            training_params = sum(p.numel() for p in self.vae.parameters() if p.requires_grad)
            logger.info('The training parameters for the individual Training are %d',training_params)

        # Start a new wandb run to track this script.
        run = wandb.init(
            entity="veenasridhar96-university-of-stuttgart",
            # Set the wandb project where this run will be logged.
            project='Masterthesis Curvess',
            name = f'Custom VAE RailwayTO {latent_technique} - Fold {fold}',
            # Hyperparameters and Run metadta
            config={
                "learning_rate": lr,
                'training_parameters': training_params,
                #'kl_weight': kl_weight,
                'latent_technique': latent_technique,
                'loss': '0.3*CE +0.7* Cldice',
                "architecture": " Custom VAE ",
                "dataset": "Railway TO (Augmented)",
                "epochs": epochs,
                "fold": fold,
            }, )
        
        cldice_loss = soft_dice_cldice()
        # ce_weights = torch.tensor([0.25, 0.35, 0.5],device=self.device)
        # ce_loss = torch.nn.CrossEntropyLoss(weight= ce_weights)
        ce_loss = torch.nn.CrossEntropyLoss()
        best_loss = float('inf')

        for epoch in range(epochs):
            
            self.vae.train()
            train_vae_loss = 0.0
            train_total_ce_loss = 0.0
            train_total_cld_loss = 0.0
            train_recon_loss = 0.0
            train_kl_loss = 0.0
            train_cldice_dl =0.0
            train_cldice_img_prc = 0.0
            train_clprec_dl =0.0
            train_clprec_img_prc = 0.0
            train_clrec_dl =0.0
            train_clrec_img_prc = 0.0

            alpha = get_alpha(epoch)

            train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
            index =0

            if self.multi_image_mode:
                accum_steps = 4
                train_batch = 0
                optimizer.zero_grad()
                
                if latent_technique == 'learned_aggregation':
                    self.aggregation_module.train()
                    aggregation_module_optimizer.zero_grad()

                for batch, input in enumerate(train_progress_bar):

                    images, gt, avg_result = input
                    images = images.to(self.device)         # (B,N,C,H,W) = (1,5,3,256,256)
                    target = gt.to(self.device)             # (B,C,H,W) = (1,3,256,256)

                    batch_size, num_imgs = images.shape[0], images.shape[1]
                    combinedimages = images.reshape(-1, *images.shape[2:]).to(self.device)          #(B*N,C,H,W) = (10,3,256,256)
                    # The above line reshapes the input images from (B,N,C,H,W) to (B*N,C,H,W) where *images.shape[2:] (the * is called the argument unpacking unpacks (C,H,W) and the -1
                    # in the first indes allows python to infer the dimensions on its own while maintaining the data count so for the first indes B*N is done.)

                    with torch.no_grad():                                               # Encoder is not trained during the multi iamge mode
                        # print('The unique values of the input images are',combinedimages.unique())
                        individual_mu, individual_log_var = self.vae.encode(combinedimages)      #(N,C,H,W) - # Posterior - N(mu,std), latent_dist - Encoded outputs of Encoder represented as the mean and logvar of DiagonalGaussianDistribution. DiagonalGaussianDistribution allows for sampling latents from the distribution
                        
                    # Reshape it to (B,N,latent_dimension,H,W)
                    mu = individual_mu.reshape(batch_size, num_imgs, *individual_mu.shape[1:])                        # (B,N,latent_dimension,H,W) = (2,5,4,32,32)
                    log_var = individual_log_var.reshape(batch_size, num_imgs, *individual_log_var.shape[1:])         # (B,N,latent_dimension,H,W) = (2,5,4,32,32)

                    #visualize_interpolation(images, mu,self.vae)

                    if latent_technique == 'product_of_gaussians':
                        # Apply Product of Gaussians across the N images (dim=1)
                        # Calculate the precision (B,N,latent_dim, H,W)
                        precision = torch.exp(-log_var)
                        combined_precision = precision.sum(dim=1)                                           # Precision summed along the number of images (B,latent_dim,H,W)
                        combined_mu = (mu * precision).sum(dim=1) / combined_precision
                        combined_log_variance = -torch.log(combined_precision)
                        combined_std_dev = (0.5 * combined_log_variance).exp()                              # Standard deviation
                        z_averaged = combined_mu + combined_std_dev * torch.randn_like(combined_mu)

                    elif latent_technique == 'averaging':
                        # Averaging the distributions of images in the latent space
                        averaged_mu = mu.mean(dim=1)                    #(B,C,H,W) = (2,4,32,32)
                        averaged_var = log_var.exp().mean(dim=1)        #(B,C,H,W) = (2,4,32,32)
                        averaged_std_dev = averaged_var.sqrt()          # Converting the variance $square(\sigma)$ into standard deviation (just sigma) - (B,C,H,W)

                        # Sample a Z from the mean distribution using the REPARAMETERIZATION TRICK.  Z = \mu + \std * \epsilon (noise sampled from N(0,1)
                        #z_averaged = averaged_mu
                        z_averaged = averaged_mu + averaged_std_dev * torch.randn_like(averaged_mu)           # (B,C,H,W)

                    elif latent_technique == 'median':
                        median_mu = mu.median(dim=1).values
                        median_var = log_var.exp().median(dim=1).values
                        median_std_dev = median_var.sqrt()
                        z_averaged = median_mu + median_std_dev * torch.randn_like(median_mu)

                    elif latent_technique == 'learned_aggregation':
                        individual_latent = individual_mu + torch.randn_like(individual_mu) * torch.exp(0.5 * individual_log_var)
                        individual_latent = individual_latent.squeeze(0)
                        # print('Shape of reparametrized individual latents',individual_latent.shape)  
                        z_averaged,weights =  self.aggregation_module(individual_latent, sample = False,num_imgs=num_imgs, batch_size=batch_size)

                    # To freeze decoder
                    #with torch.no_grad():
                    reconstructed = self.vae.decode(z_averaged)

                    train_ce_loss = ce_loss(reconstructed,target)
                    probs       = torch.softmax(reconstructed, dim=1)
                    member_prob = probs[:, 1:2]
                    member_gt   = (target == 1).float().unsqueeze(1)
                    train_cldice_loss = cldice_loss(member_prob,member_gt)
                    reconstruction_loss =   train_ce_loss + train_cldice_loss
                    #latent_space_exploration(individual_mu, z_averaged, target_mu, self.vae)
                    #latent_grid_from_views(individual_mu, self.vae,batch = train_batch)

                    loss = reconstruction_loss

                    #Backpropogation based on every step in the epoch - Gradient Accumulation
                    (loss / accum_steps).backward()
                    if (batch + 1) % accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.vae.decoder.parameters(), 1.0)
                        if latent_technique == 'learned_aggregation':
                            torch.nn.utils.clip_grad_norm_(self.aggregation_module.parameters(), 1.0)
                            aggregation_module_optimizer.step()
                            aggregation_module_optimizer.zero_grad()
                        optimizer.step()
                        optimizer.zero_grad()

                    #Aggregating the loss values for each step in the epoch to output finally as a epoch loss values
                    train_vae_loss +=loss.item()
                    train_total_ce_loss += train_ce_loss
                    train_total_cld_loss += train_total_cld_loss
                    
                    for i in range(len(reconstructed)):

                        reconstructed_image = torch.argmax(reconstructed[i], dim=0)

                        predicted_binary = (reconstructed_image == 1).detach().cpu().numpy().astype(np.float32)
                        gt_binary = (target[i] == 1).detach().cpu().numpy().astype(np.float32)
                        img_prc_binary = (avg_result[i] ==1).detach().cpu().numpy().astype(np.float32)

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

                        if epoch % 5 == 0 and index < 500:
                            if not os.path.exists(save_path):
                                os.mkdir(save_path)
                            if not os.path.exists(os.path.join(f'{save_path}/train')):
                                os.mkdir(os.path.join(f'{save_path}/train'))
                            if not os.path.exists(os.path.join(f'{save_path}/train/Epoch {epoch}')):
                                os.mkdir(os.path.join(f'{save_path}/train/Epoch {epoch}'))
                            if not os.path.exists(os.path.join(f'{save_path}/train/Epoch {epoch}', 'Metrics')):
                                os.mkdir(os.path.join(f'{save_path}/train/Epoch {epoch}','Metrics'))
                            if not os.path.exists(os.path.join(f'{save_path}/train/Epoch {epoch}', 'Weights')):
                                 os.mkdir(os.path.join(f'{save_path}/train/Epoch {epoch}','Weights'))
                            
                            # iou,img_prc_iou =  visulaize_iou_metrics(raw_images=images[i], epoch=epoch, batch=train_batch, predicted_structures = reconstructed_image,
                            #                                     gt = target[i],image_processing_result=avg_result[i],betti_error_pred=cl_dice_dl,betti_error_img_prc=cl_dice_img_prc,save_path= f'{save_path}/train')
                            
                            #plt.imsave(os.path.join(f'{save_path}/train/Epoch {epoch}', f'output_images_{train_batch}.png'),
                            #       reconstructed_image.detach().cpu().numpy(), cmap='gray', vmin = reconstructed_image.min(), vmax = reconstructed_image.max())
                            
                            group = {
                            'inputs': images[i],
                            'gt': target[i].detach().cpu().numpy(),
                            'pred': reconstructed_image.detach().cpu().numpy(),
                            'baseline': avg_result[i].squeeze(0).detach().cpu().numpy(),
                            'label': f'Batch {train_batch}',
                            # clDice already computed above — just pass them in
                            'cld_pred': cl_dice_dl,
                            'prec_pred': cl_prec,
                            'rec_pred': cl_rec,
                            'cld_base': cl_dice_img_prc,
                            'prec_base': cl_prec_i,
                            'rec_base': cl_rec_i,
                            #'iou_dl':iou,
                            #'iou_img_prc': img_prc_iou
                            }
                            visualize_comparison_with_overlay(
                                groups=[group],
                                structure_val=1,  # label index for structure class
                                save_path=f'{save_path}/train/Epoch {epoch}/Metrics',
                                batch=train_batch
                            )
                            visualize_attsets_weights(weights,save_path = f'{save_path}/train/Epoch {epoch}/Weights',batch_index = train_batch)
                            index += 1
                            train_batch += 1

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
                    optimizer.step()
                    optimizer.zero_grad()
                
                scheduler.step()
                logger.info(f'Epoch %d | Train Loss: %.4f', epoch, train_vae_loss / len(train_loader))
                
        
            ###################################
            ### Single Image Mode
            ###################################
            else:
                accum_steps = 64
                optimizer.zero_grad()
                for batch, (images,target) in enumerate(train_progress_bar):

                    images = images.to(self.device)
                    target = target.to(self.device)

                    mu, log_var = self.vae.encode(images)

                    z =  self.vae.reparameterize(mu, log_var)              
                    reconstructed = self.vae.decode(z)

                    recon_loss = ce_loss(reconstructed,target)
                    probs       = torch.softmax(reconstructed, dim=1)
                    member_prob = probs[:, 1:2]
                    member_gt   = (target == 1).float().unsqueeze(1)
                    train_cldice_loss = cldice_loss(member_prob,member_gt)
                    kl_loss = self.vae.kl_loss(mu, log_var)
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
                    

                    if epoch%10 == 0 :
                        if not os.path.exists(os.path.join(save_path)):
                            os.mkdir(os.path.join(save_path))
                        if not os.path.exists(os.path.join(save_path,f'individual_images_Fold {fold}')):
                            os.mkdir(os.path.join(save_path,f'individual_images_Fold {fold}'))
                        if not os.path.exists(os.path.join(save_path,f'individual_images_Fold {fold}/train')):
                            os.mkdir(os.path.join(save_path,f'individual_images_Fold {fold}/train'))
                        if not os.path.exists(os.path.join(save_path,f'individual_images_Fold {fold}/train/Epoch {epoch}')):
                            os.mkdir(os.path.join(save_path,f'individual_images_Fold {fold}/train/Epoch {epoch}'))
                        reconstructed_image = torch.argmax(reconstructed, dim=1)
                        
                        if index <50:
                            visulaize_images_individual(batch,epoch, reconstructed_image, target,None, f'{save_path}/individual_images_Fold {fold}/train/Epoch {epoch}',False)
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
            val_total_ce_loss = 0.0
            val_total_cld_loss = 0.0
            val_mse_loss = 0.0
            val_kl_loss = 0.0
            # val_iou_dl = 0.0
            # val_iou_img_prc = 0.0
            val_cldice_dl =0.0
            val_cldice_img_prc =0.0
            val_clprec_dl =0.0
            val_clprec_img_prc =0.0
            val_clrec_dl =0.0
            val_clrec_img_prc =0.0
            epoch_saves = []

            val_progress_bar = tqdm(val_loader,desc=f'Epoch {epoch}/{epochs}')
            if val_loader:
                val_index = 0
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
                           
                            individual_mu, individual_log_var = self.vae.encode(combinedimages)
                            # Reshape it to (B,N,latent_dimension,H,W)
                            mu = individual_mu.reshape(batch_size, num_imgs, *individual_mu.shape[1:])  # (B,N,latent_dimension,H,W) = (2,5,4,32,32)
                            log_var = individual_log_var.reshape(batch_size, num_imgs,*individual_log_var.shape[1:])  # (B,N,latent_dimension,H,W) = (2,5,4,32,32)

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
                                averaged_var = log_var.exp().mean(dim=1)  # (B,C,H,W) = (2,4,32,32)
                                averaged_std_dev = averaged_var.sqrt()  # Converting the log variance $square(\sigma)$ into standard deviation (just sigma) - (B,C,H,W)

                                # Sample a Z from the mean distribution using the REPARAMETERIZATION TRICK.  Z = \mu + \std * \epsilon (noise sampled from N(0,1)
                                z_averaged = averaged_mu + averaged_std_dev * torch.randn_like(averaged_mu)  # (B,C,H,W)

                            elif latent_technique == 'median':
                                median_mu = mu.median(dim=1).values
                                median_var = log_var.exp().median(dim=1).values
                                median_std_dev = median_var.sqrt()

                                z_averaged = median_mu + median_std_dev * torch.randn_like(median_mu)

                            elif latent_technique == 'learned_aggregation':
                                individual_latent = individual_mu + torch.randn_like(individual_mu) * torch.exp(0.5 * individual_log_var)
                                individual_latent = individual_latent.squeeze(0)
                                z_averaged,weights =  self.aggregation_module(individual_latent, sample = False, num_imgs=num_imgs, batch_size=batch_size)
                                
                            '''if epoch % 5 == 0 and batch < 10:
                                visualize_latent_space(individual_mu, combined_mu, target_mu,epoch,batch,)'''

                            reconstructed = self.vae.decode(z_averaged)

                            #latent_space_exploration(individual_mu, averaged_mu, target_mu, self.vae)

                            val_ce_loss = ce_loss(reconstructed,target)
                            probs       = torch.softmax(reconstructed, dim=1)
                            member_prob = probs[:, 1:2]
                            member_gt   = (target == 1).float().unsqueeze(1)
                            val_cldice_loss = cldice_loss(member_prob,member_gt)
                            reconstruction_val_loss =  val_ce_loss +  val_cldice_loss

                            val_loss = reconstruction_val_loss

                            val_vae_loss += val_loss.item()
                            val_total_ce_loss += val_ce_loss
                            val_total_cld_loss += val_cldice_loss
                           
                            for i in range(len(reconstructed)):
                        
                                reconstructed_image = torch.argmax(reconstructed[i], dim=0)
                                
                                predicted_binary = (reconstructed_image == 1).detach().cpu().numpy().astype(np.float32)
                                gt_binary = (target[i] == 1).detach().cpu().numpy().astype(np.float32)
                                img_prc_binary = (avg_result[i] ==1).detach().cpu().numpy().astype(np.float32)

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
                                #index += 1
                               
                                epoch_saves.append(
                                    {
                                        'inputs': images[i],
                                        'gt': target[i],
                                        'pred': reconstructed_image,
                                        'baseline': avg_result[i],
                                        #'baseline': avg_result[i].squeeze(0).detach().cpu().numpy(),
                                        'label': f'Batch {val_index}',
                                        'weights': weights if weights is not None else None, 
                                        # clDice already computed above — just pass them in
                                        'cld_pred': cl_dice_dl,
                                        'prec_pred': cl_prec,
                                        'rec_pred': cl_rec,
                                        'cld_base': cl_dice_img_prc,
                                        'prec_base': cl_prec_i,
                                        'rec_base': cl_rec_i,
                                        # 'iou_dl':iou,
                                        # 'iou_img_prc': img_prc_iou
                                    }
                                )
                                sample = epoch_saves[val_index]
                                val_index += 1

                            # Printing the step loss values in the progress bar
                            val_progress_bar.set_postfix({
                                'Total Val loss': val_loss.item(),
                                'Val reconstruction_loss': reconstruction_val_loss.item()
                            })


                        logger.info(f'Epoch %d | Val Loss: %.4f', epoch, val_vae_loss / len(val_loader))    

                        epoch_val_loss = val_total_cld_loss /len(val_loader)
                        if epoch_val_loss <= best_loss:
                            best_loss = epoch_val_loss
                            if not os.path.exists(save_path):
                                os.mkdir(save_path)
                            if not os.path.exists(os.path.join(f'{save_path}/val')):
                                os.mkdir(os.path.join(f'{save_path}/val'))
                            if not os.path.exists(os.path.join(f'{save_path}/val/Epoch {epoch}')):
                                os.mkdir(os.path.join(f'{save_path}/val/Epoch {epoch}'))
                            if not os.path.exists(os.path.join(f'{save_path}/val/Epoch {epoch}', 'Metrics')):
                                os.mkdir(os.path.join(f'{save_path}/val/Epoch {epoch}','Metrics'))
                            if not os.path.exists(os.path.join(f'{save_path}/val/Epoch {epoch}', 'Weights')):
                                os.mkdir(os.path.join(f'{save_path}/val/Epoch {epoch}','Weights'))

                            for sample_index, sample in enumerate(epoch_saves):

                                iou,img_prc_iou =  visulaize_iou_metrics(raw_images=sample['inputs'], epoch=epoch, batch=val_index, predicted_structures = sample['pred'],
                                                        gt = sample['gt'],image_processing_result=sample['baseline'],betti_error_pred=sample['cld_pred'],betti_error_img_prc=sample['cld_base'],save_path= f'{save_path}/val')
                    
                                plt.imsave(os.path.join(f'{save_path}/val/Epoch {epoch}', f'output_images_{sample_index}.png'),
                                sample['pred'].detach().cpu().numpy(), cmap='gray', vmin = sample['pred'].min(), vmax = sample['pred'].max())
                                
                                group = {
                                'inputs': sample['inputs'],
                                'gt': sample['gt'].detach().cpu().numpy(),
                                'pred': sample['pred'].detach().cpu().numpy(),
                                'baseline': sample['baseline'].squeeze(0).detach().cpu().numpy(),
                                'label': f'Batch {sample_index}',
                                # clDice already computed above
                                'cld_pred': sample['cld_pred'],
                                'prec_pred': sample['prec_pred'],
                                'rec_pred': sample['rec_pred'],
                                'cld_base': sample['cld_base'],
                                'prec_base': sample['prec_base'],
                                'rec_base': sample['rec_base'],
                                #'iou_dl':iou,
                                #'iou_img_prc': img_prc_iou
                                }
                                visualize_comparison_with_overlay(
                                    groups=[group],
                                    structure_val=1,  # label index for structure class
                                    save_path=f'{save_path}/val/Epoch {epoch}/Metrics',
                                    batch=sample_index
                                )
                                visualize_attsets_weights(sample['weights'],save_path = f'{save_path}/val/Epoch {epoch}/Weights',batch_index = sample_index)
                            
                        

                    else:
                        for batch, (images,target) in enumerate(val_progress_bar):

                            images = images.to(self.device)
                            target = target.to(self.device)

                            mu, log_var = self.vae.encode(images)

                            z =  self.vae.reparameterize(mu, log_var)
                            reconstructed = self.vae.decode(z)

                            recon_loss = ce_loss(reconstructed,target)
                            probs       = torch.softmax(reconstructed, dim=1)
                            member_prob = probs[:, 1:2]
                            member_gt   = (target == 1).float().unsqueeze(1)
                            val_cldice_loss = cldice_loss(member_prob,member_gt)
                            kl_loss = self.vae.kl_loss(mu, log_var)

                            val_loss = recon_loss + kl_weight * kl_loss + val_cldice_loss

                            if epoch%5 == 0 :
                                if not os.path.exists(os.path.join(save_path)):
                                        os.mkdir(os.path.join(save_path))
                                if not os.path.exists(os.path.join(save_path,f'individual_images_Fold {fold}')):
                                    os.mkdir(os.path.join(save_path,f'individual_images_Fold {fold}'))
                                if not os.path.exists(os.path.join(save_path,f'individual_images_Fold {fold}/val')):
                                    os.mkdir(os.path.join(save_path,f'individual_images_Fold {fold}/val'))
                                if not os.path.exists(os.path.join(save_path,f'individual_images_Fold {fold}/val/Epoch {epoch}')):
                                    os.mkdir(os.path.join(save_path,f'individual_images_Fold {fold}/val/Epoch {epoch}'))
                                #print('The shape of the reconstructed individual image',reconstructed.shape)
                                reconstructed_image = torch.argmax(reconstructed, dim=1)
                                
                                if val_index <50:
                                    visulaize_images_individual(batch,epoch, reconstructed_image, target,None, f'{save_path}/individual_images_Fold {fold}/val/Epoch {epoch}',False)

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

            if epoch_val_loss <= best_loss:
                test_vae_loss = 0.0
                test_total_cld_loss = 0.0
                test_total_ce_loss = 0.0
                test_kl_loss = 0.0
                # test_iou_dl =0.0
                # test_iou_img_prc = 0.0
                test_cldice_dl =0.0
                test_cldice_img_prc = 0.0
                test_clprec_dl =0.0
                test_clprec_img_prc =0.0
                test_clrec_dl =0.0
                test_clrec_img_prc =0.0
                test_progress_bar = tqdm(test_loader, desc=f"Epoch {epoch}/{epochs}")
                if test_loader:
                    #print('The length of test loader',len(test_loader))
                    self.vae.eval()
                    test_index =0
                    with torch.no_grad():
                        if self.multi_image_mode:
                            for batch, input in enumerate(test_progress_bar):
                                images, labels, avg_result = input
                                target = labels
                                target = target.to(self.device)

                                batch_size, num_imgs = images.shape[0], images.shape[1]
                                combinedimages = images.reshape(-1, *images.shape[2:]).to(self.device)

                                individual_mu, individual_log_var = self.vae.encode(combinedimages)

                                # Individual mu is mean represented using 4 latent variables (10,4,32,32)
                                # Log variance represent the uncertainty or spread of the distribution. (log for stability)

                                # Reshape it to (B,N,latent_dimension,H,W)
                                mu = individual_mu.reshape(batch_size, num_imgs, *individual_mu.shape[1:])  # (B,N,latent_dimension,H,W) = (2,5,4,32,32)
                                log_var = individual_log_var.reshape(batch_size, num_imgs, *individual_log_var.shape[1:])  # (B,N,latent_dimension,H,W) = (2,5,4,32,32)

                                '''z_individual = posterior.sample()
                                original_reconstruction = self.vae.decode(z_individual).sample
                                original_reconstruction = original_reconstruction.view(batch_size, num_imgs, *original_reconstruction.shape[1:])'''

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
                                    median_mu = mu.median(dim=1).values         #Since torch.median returns a tuple of (values, indices), we need to access the values using .values
                                    median_var = log_var.exp().median(dim=1).values
                                    median_std_dev = median_var.sqrt()

                                    z_averaged = median_mu + median_std_dev * torch.randn_like(median_mu)

                                elif latent_technique == 'averaging':
                                    # Averaging the distributions of images in the latent space
                                    averaged_mu = mu.mean(dim=1)  # (B,C,H,W) = (2,4,32,32)
                                    averaged_var = log_var.exp().mean(dim=1)  # (B,C,H,W) = (2,4,32,32)
                                    averaged_std_dev = averaged_var.sqrt()  # Converting the log variance $square(\sigma)$ into standard deviation (just sigma) - (B,C,H,W)

                                    # Sample a Z from the mean distribution using the REPARAMETERIZATION TRICK.  Z = \mu + \std * \epsilon (noise sampled from N(0,1)
                                    z_averaged = averaged_mu + averaged_std_dev * torch.randn_like(averaged_mu)  # (B,C,H,W)

                                elif latent_technique == 'learned_aggregation':
                                    individual_latent = individual_mu + torch.randn_like(individual_mu) * torch.exp(0.5 * log_var)
                                    individual_latent = individual_latent.squeeze(0)
                                    z_averaged,weights =  self.aggregation_module(individual_latent, sample = False,num_imgs=num_imgs, batch_size=batch_size)

                                reconstructed = self.vae.decode(z_averaged)

                                test_ce_loss = ce_loss(reconstructed,target)
                                probs       = torch.softmax(reconstructed, dim=1)
                                member_prob = probs[:, 1:2]
                                member_gt   = (target == 1).float().unsqueeze(1)
                                test_cldice_loss = cldice_loss(member_prob,member_gt)
                                reconstruction_loss =  test_ce_loss + test_cldice_loss
                                reconstruction_test_loss = alpha*  test_ce_loss + (1-alpha)*test_cldice_loss
                                #reconstruction_test_loss = mse_loss(reconstructed, target)

                                t_loss = reconstruction_test_loss

                                test_vae_loss += t_loss.item()
                                test_total_ce_loss += test_ce_loss
                                test_total_cld_loss += test_cldice_loss
                                #test_mse_loss += reconstruction_test_loss.item()

                                for i in range(len(reconstructed)):

                                    reconstructed_image = torch.argmax(reconstructed[i], dim=0)
                                    # print('The shape of the reconstructed image',reconstructed.shape)
                                    # print('The unique values of the reconstructed image',reconstructed.unique())
                                    
                                    predicted_binary = (reconstructed_image == 1).detach().cpu().numpy().astype(np.float32)
                                    gt_binary = (target[i] == 1).detach().cpu().numpy().astype(np.float32)
                                    img_prc_binary = (avg_result[i] ==1).detach().cpu().numpy().astype(np.float32)

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
                                    # iou_dilated,iou_dilated_image_processing,dilated_overlap,img_proc_dilated_overlap = visulaize_iou_metrics(predicted_binary, gt_binary, img_prc_binary, tolerance=0)
                                    # test_iou_dl += iou_dilated
                                    # test_iou_img_prc += iou_dilated_image_processing

                                    if not os.path.exists(save_path):
                                        os.mkdir(save_path)
                                    if not os.path.exists(os.path.join(f'{save_path}/test')):
                                        os.mkdir(os.path.join(f'{save_path}/test'))
                                    if not os.path.exists(os.path.join(f'{save_path}/test/Epoch {epoch}')):
                                        os.mkdir(os.path.join(f'{save_path}/test/Epoch {epoch}'))
                                    if not os.path.exists(os.path.join(f'{save_path}/test/Epoch {epoch}', 'Metrics')):
                                        os.mkdir(os.path.join(f'{save_path}/test/Epoch {epoch}','Metrics'))
                                    if not os.path.exists(os.path.join(f'{save_path}/test/Epoch {epoch}', 'Weights')):
                                        os.mkdir(os.path.join(f'{save_path}/test/Epoch {epoch}','Weights'))
                                    
                                    plt.imsave(os.path.join(f'{save_path}/test/Epoch {epoch}', f'output_images_{test_index}.png'),
                                            reconstructed_image.detach().cpu().numpy(), cmap='gray', vmin = reconstructed_image.min(), vmax = reconstructed_image.max())

                                    #iou,img_prc_iou =  visulaize_iou_metrics(raw_images=images[i], epoch=epoch, batch=test_index, predicted_structures = reconstructed_image,
                                    #                        gt = target[i],image_processing_result=avg_result[i],betti_error_pred=cl_dice_dl,betti_error_img_prc=cl_dice_img_prc,save_path= f'{save_path}/test')
                                    
                                    group = {
                                    'inputs': images[i],
                                    'gt': target[i].detach().cpu().numpy(),
                                    'pred': reconstructed_image.detach().cpu().numpy(),
                                    'baseline': avg_result[i].squeeze(0).detach().cpu().numpy(),
                                    'label': f'Batch {test_index}',
                                    # clDice already computed above
                                    'cld_pred': cl_dice_dl,
                                    'prec_pred': cl_prec,
                                    'rec_pred': cl_rec,
                                    'cld_base': cl_dice_img_prc,
                                    'prec_base': cl_prec_i,
                                    'rec_base': cl_rec_i,
                                    'iou_dl':iou,
                                    'iou_img_prc': img_prc_iou
                                    }
                                    visualize_comparison_with_overlay(
                                        groups=[group],
                                        structure_val=1,  # label index for structure class
                                        save_path=f'{save_path}/test/Epoch {epoch}/Metrics',
                                        batch=test_index
                                    )
                                    visualize_attsets_weights(weights,save_path = f'{save_path}/test/Epoch {epoch}/Weights',batch_index = test_index)
                                    
                                    # visulaize_images(self.multi_image_mode, batch, epoch, reconstructed[i], images[i], target[i], avg_result[i], iou_dilated,iou_dilated_image_processing,dilated_overlap,
                                    #                  img_proc_dilated_overlap, f'./output_images_{latent_technique}_{fold}/train/Epoch {epoch}/Metrics', index, False)
                                test_index += 1

                                test_progress_bar.set_postfix({
                                    'Test Total loss': t_loss.item(),
                                    'Test Reconstruction_loss': reconstruction_test_loss.item(),
                                    #'KL_loss': kl_test_loss.item(),
                                })

                            logger.info(f'Epoch %d | Test Loss: %.4f', epoch, test_vae_loss / len(test_loader))    


                        else:
                            for batch, (images,target) in enumerate(test_progress_bar):

                                images = images.to(self.device)
                                target = target.to(self.device)

                                mu, log_var = self.vae.encode(images)
                            
                                z =  self.vae.reparameterize(mu, log_var)         
                                
                                reconstructed = self.vae.decode(z)

                                recon_loss = ce_loss(reconstructed,target)
                                probs       = torch.softmax(reconstructed, dim=1)
                                member_prob = probs[:, 1:2]
                                member_gt   = (target == 1).float().unsqueeze(1)
                                test_cldice_loss = cldice_loss(member_prob,member_gt)
                                kl_loss = self.vae.kl_loss(mu, log_var)

                                t_loss = recon_loss + kl_weight * kl_loss + test_cldice_loss

                                if epoch%10 == 0 :
                                    if not os.path.exists(os.path.join(save_path)):
                                            os.mkdir(os.path.join(save_path))
                                    if not os.path.exists(os.path.join(save_path,f'individual_images_Fold {fold}')):
                                        os.mkdir(os.path.join(save_path,f'individual_images_Fold {fold}'))
                                    if not os.path.exists(os.path.join(save_path,f'individual_images_Fold {fold}/test')):
                                        os.mkdir(os.path.join(save_path,f'individual_images_Fold {fold}/test'))
                                    if not os.path.exists(os.path.join(save_path,f'individual_images_Fold {fold}/test/Epoch {epoch}')):
                                        os.mkdir(os.path.join(save_path,f'individual_images_Fold {fold}/test/Epoch {epoch}'))
                                    reconstructed_image = torch.argmax(reconstructed, dim=1)
                                    if test_index <50:
                                        visulaize_images_individual(batch,epoch, reconstructed_image, target,None, f'{save_path}/individual_images_Fold {fold}/test/Epoch {epoch}',False)

                                    test_index += 1

                                test_vae_loss += t_loss.item()
                                test_mse_loss += recon_loss.item()
                                test_kl_loss += kl_loss.item()
                                test_cldice_loss += test_cldice_loss.item()

                                # Printing the step loss values in the progress bar
                                test_progress_bar.set_postfix({
                                    'Test Total loss': t_loss.item(),
                                    'Test Reconstruction_loss': recon_loss.item(),
                                    'KL_loss': kl_loss.item(),
                                })

            
                best_loss = epoch_cldice_dl
                if self.multi_image_mode:
                    torch.save(self.vae, f'Custom_VAE_RailwayTO_common_structures_checkpoint_{latent_technique}_{fold}_final.pth')
                    #self.vae.save_pretrained(f'finetuned_vae/common_structures_recon_{latent_technique}')
                    if latent_technique == 'learned_aggregation':
                        torch.save({
                            'vae': self.vae,
                            'aggregation_module': self.aggregation_module,
                        }, f'Custom_VAE_RailwayTO_aggmodule_checkpoint_{latent_technique}_{fold}_final.pth')
                else:
                    torch.save(self.vae,f'Custom_VAE_RailwayTO dataset_individual_checkpoint_ce_cldice_{fold}.pth')
                    #self.vae.save_pretrained('finetuned_vae/individually_trained_models_ce_cldice')

                logger.info(f'Model Saved with best_loss: %.4f',best_loss)

            run.log({'fold':fold,'epoch':epoch,
                    'Train_Total_Loss': train_vae_loss / len(train_loader), 'Test_Total_Loss': test_vae_loss / len(test_loader), 'Val_Total_Loss': val_vae_loss/len(val_loader),
                    'Train_CE_Loss':train_total_ce_loss/len(train_loader), 'Test_CE_Loss': test_total_ce_loss / len(test_loader), 'Val_CE_Loss': val_total_ce_loss/len(val_loader),
                    'Train_cldice_Loss':train_total_cld_loss/len(train_loader), 'Test_cldice_Loss': test_total_cld_loss / len(test_loader), 'Val_cldice_Loss': val_total_cld_loss/len(val_loader),
                    'Train cldice Img_prc': train_cldice_img_prc/len(train_loader.dataset), 'Train cldice DL': train_cldice_dl/len(train_loader.dataset),
                    'Val cldice Img_prc': val_cldice_img_prc/len(val_loader.dataset), 'Val cldice DL': val_cldice_dl /len(val_loader.dataset),
                    'Test cldice Img_prc': test_cldice_img_prc/len(test_loader.dataset), 'Test cldice DL': test_cldice_dl/len(test_loader.dataset),
                    'Train clrec Img_prc': train_clrec_img_prc/len(train_loader.dataset), 'Train clrec DL': train_clrec_dl/len(train_loader.dataset),
                    'Val clrec Img_prc': val_clrec_img_prc/len(val_loader.dataset), 'Val clrec DL': val_clrec_dl /len(val_loader.dataset),
                    'Test clrec Img_prc': test_clrec_img_prc/len(test_loader.dataset), 'Test clrec DL': test_clrec_dl/len(test_loader.dataset),
                    'Train cplrec Img_prc': train_clprec_img_prc/len(train_loader.dataset), 'Train clprec DL': train_clprec_dl/len(train_loader.dataset),
                    'Val cplrec Img_prc': val_clprec_img_prc/len(val_loader.dataset), 'Val clprec DL': val_clprec_dl /len(val_loader.dataset),
                    'Test cplrec Img_prc': test_clprec_img_prc/len(test_loader.dataset), 'Test clprec DL': test_clprec_dl/len(test_loader.dataset),
                    #'alpha':alpha,
                    #'Learning Rate':decoder_optimizer.param_groups[0]['lr'],
                   })

        wandb.finish()

def extract_red_mask(image):
    #print(image.min(),image.max())
    R = image
    mask = (R ==1)
    return mask.astype(float)

def visulaize_iou_metrics(raw_images, epoch,batch,predicted_structures, gt,image_processing_result,betti_error_pred,betti_error_img_prc,tolerance=0,save_path='Simple_Conv_Grayscale_output'):

    red_extracted_structures = extract_red_mask(predicted_structures.detach().cpu().numpy())
    red_gt_structures = extract_red_mask(gt.detach().cpu().numpy())
    red_image_processing_structures = extract_red_mask(image_processing_result.detach().cpu().numpy())

    if tolerance>0:
        r = tolerance
        y, x = np.ogrid[-r:r + 1, -r:r + 1]
        circle = (x ** 2 + y ** 2 <= r ** 2)
        predicted_dilated_structure = binary_dilation(red_extracted_structures.astype(bool), structure=circle)
        target_dilated_structure = binary_dilation(red_gt_structures.astype(bool), structure=circle)
        image_processing_structures = binary_dilation(red_image_processing_structures.astype(bool), structure=circle)
    else:
        predicted_dilated_structure = red_extracted_structures
        target_dilated_structure = red_gt_structures
        image_processing_structures = red_image_processing_structures

    TP = (predicted_dilated_structure * target_dilated_structure).sum()
    FP = (predicted_dilated_structure * (1 - target_dilated_structure)).sum()
    FN = ((1 - predicted_dilated_structure) * target_dilated_structure).sum()

    dilated_overlap = np.zeros((*red_extracted_structures.shape, 3))
    dilated_overlap[..., 0] = (predicted_dilated_structure > 0) & (target_dilated_structure == 0)  # FP = red
    dilated_overlap[..., 1] = (predicted_dilated_structure > 0) & (target_dilated_structure > 0)  # TP = green
    dilated_overlap[..., 2] = (predicted_dilated_structure == 0) & (target_dilated_structure > 0)  # FN = blue

    iou_dilated = TP / (TP + FP + FN + 1e-8)


    TP = (image_processing_structures * target_dilated_structure).sum()
    FP = (image_processing_structures * (1 - target_dilated_structure)).sum()
    FN = ((1 - image_processing_structures) * target_dilated_structure).sum()

    img_proc_dilated_overlap = np.zeros((*red_extracted_structures.shape, 3))
    img_proc_dilated_overlap[..., 0] = (image_processing_structures > 0) & (target_dilated_structure == 0)  # FP = red
    img_proc_dilated_overlap[..., 1] = (image_processing_structures > 0) & (target_dilated_structure > 0)  # TP = green
    img_proc_dilated_overlap[..., 2] = (image_processing_structures == 0) & (target_dilated_structure > 0)  # FN = blue

    iou_dilated_image_processing = TP / (TP + FP + FN + 1e-8)

    print('Predicted IoU ', iou_dilated)
    print('Image Processing IoU ',iou_dilated_image_processing)

    #visulaize_iou_results(raw_images,epoch,batch,predicted_structures,gt,dilated_overlap,red_image_processing_structures,img_proc_dilated_overlap,iou_dilated,iou_dilated_image_processing,betti_error_pred,betti_error_img_prc,save_path)

    return iou_dilated,iou_dilated_image_processing