import os
import torchvision
from vae_latent_visualisation import *
from image_processing_approach import *
import cv2
import numpy as np

def eval(model):
    images_path = './eval_images/batch 3'
    input_images = []
    image_stack = []

    for images in os.listdir(images_path):
        image_stack.append(cv2.imread(os.path.join(images_path, images)))
    find_common_structures(image_stack, 'frequency')


    latent_technique = 'product_of_gaussians'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_transformations = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((256, 256)),
            v2.ToTensor(),
            # v2.Normalize(mean=(0.5,), std= (1.0,)),
        ])
    for images in os.listdir(images_path):
        image = torchvision.io.decode_image(os.path.join(images_path,images),torchvision.io.ImageReadMode.RGB)
        image = test_transformations(image)
        # print(image.shape)
        input_images.append(image)
        # input_images.append(image.permute(2,0,1))
        print(image.shape)


    images = torch.stack(input_images)
    images = torch.unsqueeze(images,0)


    vae_eval_model = model
    vae_eval_model.eval()
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

        '''if epoch % 5 == 0 and batch < 10:
            visualize_latent_space(individual_mu, combined_mu, target_mu,epoch, batch,test=True,val=False)'''

        reconstructed = vae_eval_model.decode(z_averaged).sample

        reconstructed_image = torch.clamp(reconstructed, 0, 1)
        plt.imsave(os.path.join('./', f'eval.png'), reconstructed_image[0].permute(1, 2,0).detach().cpu().numpy())

        fig = plt.figure(figsize=(16, 8))
        grid_spec = fig.add_gridspec(2, 5, hspace=0.1, wspace=0.1)

        for j in range(5):
            ax = fig.add_subplot(grid_spec[0, j])
            orig_img = images[0][j].permute(1, 2, 0).detach().cpu().numpy()
            ax.imshow(orig_img)
            ax.axis('off')
        #ssim_loss = ssim(reconstructed_image, gt_image)
        ax_center = fig.add_subplot(grid_spec[1, 2])
        recon_images = reconstructed_image[0].permute(1, 2, 0).detach().cpu().numpy()
        ax_center.imshow(recon_images)
        ax_center.axis('off')
        #ax_center.set_title(f'SSIM: {ssim_loss:.2f}')

        '''ax_left = fig.add_subplot(grid_spec[1, 1])
        # print(type(gt_image), gt_image.shape)
        gt_image_visual = gt_image[i].permute(1, 2, 0).detach().cpu().numpy()
        ax_left.imshow(gt_image_visual)
        ax_left.axis('off')'''
        plt.savefig(os.path.join('./', f'reconstructed_eval.png'),)

    # visulaize_images(self.multi_image_mode,batch, epoch, reconstructed, images, target, f'./output_images/test_images/', True)
    #visualize_latent_and_images(self.multi_image_mode, batch, epoch, reconstructed, images, target, individual_mu,
    #                            combined_mu, target_mu, './output_images/test_images', True)

    #reconstruction_test_loss = mse_loss(reconstructed, target)

    #kl_test_loss = -0.5 * (1 + combined_log_variance - combined_mu.pow(2) - combined_log_variance.exp()).mean()