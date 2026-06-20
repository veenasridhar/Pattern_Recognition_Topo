import os

import torch
import torchvision
from vae_latent_visualisation import *
from image_processing_approach import *
import cv2
import numpy as np
from siamese_network import *
#from attSets import *

def eval(model):
    #images_path = 'data/batched_data_resized_red/train'
    image_path = 'Batch 0'
    #gt_path = './data/eval_images/gt/gt 2.png'

    image_stack = []
    aggregation_module = AttSets(latent_channels=4)
    mse_loss = torch.nn.MSELoss()
    latent_technique = 'averaging'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_transformations = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((256, 256)),
            v2.ToTensor(),
            # v2.Normalize(mean=(0.5,), std= (1.0,)),
        ])
    aggregation_module = aggregation_module.to(device)
    higher_thresh = []
    # for image_batch in os.listdir(images_path):
    #     input_images = []
    #     image_path = os.path.join(images_path, image_batch)
    #     batch = re.split(r'(\d+)', image_batch)[-2]
        #print(batch)
    input_images = []
    for images in os.listdir(image_path):
        image = torchvision.io.decode_image(os.path.join(image_path,images),torchvision.io.ImageReadMode.RGB)
        image = test_transformations(image)
        # print(image.shape)
        input_images.append(image)
        # input_images.append(image.permute(2,0,1))
        #print(image.shape)
    # gt_path = f'data/Ground Truth Images Resized_red/train/gt {batch}.png'
    # gt_image = torchvision.io.decode_image(gt_path, torchvision.io.ImageReadMode.RGB)
    # gt_image = test_transformations(gt_image)


    images = torch.stack(input_images)
    perm = torch.randperm(images.size(0))
    images_permuted = images[perm]
    images = torch.unsqueeze(images,0)
    images_permuted = torch.unsqueeze(images_permuted,0)
    reconstructed = model_evaluation(model,images_permuted,gt=None,batch=0,latent_technique=latent_technique,device=device,permutated=False,aggregation_module = aggregation_module)
        #reconstructed_permutated = model_evaluation(model, images_permuted, gt_image,batch, latent_technique, device, permutated=True,aggregation_module = aggregation_module)
        # print(reconstructed_permutated.shape,reconstructed.shape)
        # diff = mse_loss(reconstructed_permutated,reconstructed)
        # print(diff)
    #     diff = (reconstructed - reconstructed_permutated).abs().max().item()
    #     max_diff = max(0, diff)
    #     print('Max diff',max_diff)
    #     if max_diff > 0.001:
    #         higher_thresh.append(batch)
    # print(higher_thresh)

def model_evaluation(model,images,gt,batch,latent_technique,device,permutated=False,aggregation_module=None):
    vae_eval_model = model
    vae_eval_model.eval()
    images = images.to(device)
    model = model.to(device)
    with torch.no_grad():

        batch_size, num_imgs = images.shape[0], images.shape[1]
        combinedimages = images.reshape(-1, *images.shape[2:]).to(device)

        posterior = vae_eval_model.encode(combinedimages).latent_dist

        individual_mu = posterior.mean  # Individual mu is mean represented using 4 latent variables (10,4,32,32)
        log_var = posterior.logvar  # Log variance represent the uncertainty or spread of the distribution. (log for stability)

        # Reshape it to (B,N,latent_dimension,H,W)
        mu = individual_mu.reshape(batch_size, num_imgs,
                                   *individual_mu.shape[1:])  # (B,N,latent_dimension,H,W) = (2,5,4,32,32)
        log_var = log_var.reshape(batch_size, num_imgs,
                                  *log_var.shape[1:])  # (B,N,latent_dimension,H,W) = (2,5,4,32,32)

        if latent_technique == 'product_of_gaussians':
            # Apply Product of Gaussians across the N images (dim=1)
            # Calculate the precision (B,N,latent_dim, H,W)
            precision = torch.exp(-log_var)
            combined_precision = precision.sum(
                dim=1)  # Precision summed along the number of images (B,latent_dim,H,W)
            combined_mu = (mu * precision).sum(dim=1) / combined_precision
            combined_log_variance = -torch.log(combined_precision)

            combined_std_dev = (0.5 * combined_log_variance).exp()
            z_averaged = combined_mu + combined_std_dev * torch.randn_like(combined_mu)

        elif latent_technique == 'averaging':
            # Averaging the distributions of images in the latent space
            averaged_mu = mu.mean(dim=1)  # (B,C,H,W) = (2,4,32,32)
            averaged_logvar = log_var.mean(dim=1)  # (B,C,H,W) = (2,4,32,32)
            averaged_std_dev = (
                        0.5 * averaged_logvar).exp()  # Converting the log variance $square(\sigma)$ into standard deviation (just sigma) - (B,C,H,W)

            # Sample a Z from the mean distribution using the REPARAMETERIZATION TRICK.  Z = \mu + \std * \epsilon (noise sampled from N(0,1)
            z_averaged = averaged_mu + averaged_std_dev * torch.randn_like(averaged_mu)  # (B,C,H,W)

        elif latent_technique == 'spatial_aggregation':
            posterior_mean_detached = posterior.mean.detach()

            z_averaged = aggregation_module(posterior_mean_detached, num_imgs, batch_size)

        '''if epoch % 5 == 0 and batch < 10:
            visualize_latent_space(individual_mu, combined_mu, target_mu,epoch, batch,test=True,val=False)'''

        reconstructed = vae_eval_model.decode(z_averaged).sample

        reconstructed_image = torch.clamp(reconstructed, 0, 1)
        print(reconstructed_image.shape)
        if permutated==True:
            plt.imsave(os.path.join('./Compiled Results/permuattion_results/', f'eval_{batch}_permuted.png'),reconstructed_image[0].permute(1, 2, 0).detach().cpu().numpy())
        else:
            plt.imsave(os.path.join('./Compiled Results/permuattion_results/', f'eval_{batch}.png'), reconstructed_image[0].permute(1, 2,0).detach().cpu().numpy())



        fig = plt.figure(figsize=(16, 8))
        grid_spec = fig.add_gridspec(2, 5, hspace=0.1, wspace=0.1)

        for j in range(5):
            ax = fig.add_subplot(grid_spec[0, j])
            orig_img = images[0][j].permute(1, 2, 0).detach().cpu().numpy()
            ax.imshow(orig_img)
            ax.axis('off')
        #print(gt.shape)
        dreamsim_metric_latent = dreamsim_metric(gt.permute(1, 2, 0).detach().cpu().numpy(), reconstructed_image[0].permute(1, 2, 0).detach().cpu().numpy()).item()
        #dreamsim_metric_self = dreamsim_metric(gt.permute(1, 2, 0).detach().cpu().numpy(), gt.permute(1, 2, 0).detach().cpu().numpy())
        #print('Self Comparison',dreamsim_metric_self)
        #ssim_loss = ssim(reconstructed_image, gt_image)
        ax_center = fig.add_subplot(grid_spec[1, 2])
        recon_images = reconstructed_image[0].permute(1, 2, 0).detach().cpu().numpy()
        ax_center.imshow(recon_images)
        ax_center.axis('off')
        ax_center.set_title(f'Spatial Aggregation \n Dreamsim: {dreamsim_metric_latent:.2f}')

        ax_left = fig.add_subplot(grid_spec[1, 1])
        # print(type(gt_image), gt_image.shape)
        gt_image_visual = gt.permute(1, 2, 0).detach().cpu().numpy()
        ax_left.imshow(gt_image_visual)
        ax_left.axis('off')
        ax_left.set_title(f'Ground Truth')

        # ax_right = fig.add_subplot(grid_spec[1, 3])
        # # print(type(gt_image), gt_image.shape)
        # image_processing_result = cv2.imread(f'Image Processing Results/train/result_{batch}.png')
        # image_processing_result = image_processing_result[:,:,::-1]
        # ax_right.imshow(image_processing_result)
        # ax_right.axis('off')
        #dreamsim_metric_pixel_avg = dreamsim_metric(gt.permute(1, 2, 0).detach().cpu().numpy(),image_processing_result).item()
        #ax_right.set_title(f'Pixelwise Averaging Results \n Dreamsim: {dreamsim_metric_pixel_avg:.2f}')

        if permutated==True:
            plt.savefig(os.path.join('./Compiled Results/permuattion_results/train', f'reconstructed_eval_{batch}_permuted.png'), )
        else:
            plt.savefig(os.path.join('./Compiled Results/permuattion_results/train', f'reconstructed_eval_{batch}.png'), )
        return reconstructed_image


    # visulaize_images(self.multi_image_mode,batch, epoch, reconstructed, images, target, f'./output_images/test_images/', True)
    #visualize_latent_and_images(self.multi_image_mode, batch, epoch, reconstructed, images, target, individual_mu,
    #                            combined_mu, target_mu, './output_images/test_images', True)

    #reconstruction_test_loss = mse_loss(reconstructed, target)

    #kl_test_loss = -0.5 * (1 + combined_log_variance - combined_mu.pow(2) - combined_log_variance.exp()).mean()