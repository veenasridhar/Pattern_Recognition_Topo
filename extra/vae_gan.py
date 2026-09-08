import cv2
import torch
from diffusers import AutoencoderKL
import torch.nn as nn
from torchvision.transforms import v2
from input_pipeline import *
from tqdm import tqdm
from torchmetrics import StructuralSimilarityIndexMeasure
from visualizations import *
from torch.utils.tensorboard import SummaryWriter


# Diffusers Library Implementation based on: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/vae.py#L687


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator.
    Judges NxN local patches as real/fake rather than the whole image.
    More stable than a global discriminator, especially with ~100 samples.
    """

    def __init__(self, in_channels=3, base_filters=64):
        super().__init__()

        def block(in_c, out_c, normalize=True, stride=2):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4,
                                stride=stride, padding=1, bias=not normalize)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c, affine=True))
            # ✅ inplace=False to avoid autograd inplace errors
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, base_filters, normalize=False),
            *block(base_filters, base_filters * 2),
            *block(base_filters * 2, base_filters * 4),
            *block(base_filters * 4, base_filters * 8, stride=1),
            nn.Conv2d(base_filters * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)


class VAEGANFineTuner:

    def __init__(self, model_name='stabilityai/sd-vae-ft-mse', multi_image_mode=True, train_loader=None,
                 test_loader=None, device=None):
        """
        Initialize VAE model

        Args:
            model_name: HuggingFace model identifier
                - "stabilityai/sd-vae-ft-mse" (recommended, MSE finetuned)
                - "stabilityai/sd-vae-ft-ema" (EMA finetuned)
            device: torch device
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Running on... ', self.device)
        self.multi_image_mode = multi_image_mode
        self.vae = AutoencoderKL.from_pretrained(model_name)
        self.vae.to(self.device)

        self.discriminator = PatchDiscriminator(in_channels=3).to(self.device)
        #print(f"Model parameters: {sum(p.numel() for p in self.vae.parameters()):,}")
        #self.disc_warmup = 3

    def _discriminator_loss(self, real, fake,fake2):
        """Hinge loss — more stable than BCE for small datasets."""
        real_pred = self.discriminator(real)
        fake_pred = self.discriminator(fake)  # fake already detached at call site
        fake_pred2 = self.discriminator(fake2)
        print('The mean real predictions ',real_pred.mean(),'and mean fake predictions ',fake_pred.mean())
        d_loss = (
                torch.mean(torch.relu(1.0 - real_pred)) +
                torch.mean(torch.relu(1.0 + fake_pred)) +
                torch.mean(torch.relu(1.0 + fake_pred2))
        )
        return d_loss

    def finetune(self, train_loader, val_loader, test_loader, fold, epochs=30, lr=0.0001, kl_weight=1e-6, save_path="./finetuned_vae"):

        torch.autograd.set_detect_anomaly(True)
        encoder_optimizer = torch.optim.AdamW(self.vae.encoder.parameters(), lr=lr, weight_decay=1e-4)
        decoder_optimizer = torch.optim.AdamW(self.vae.decoder.parameters(), lr=lr, weight_decay=1e-4)
        disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(encoder_optimizer, T_max=epochs)
        disc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(disc_optimizer, T_max=epochs)

        mse_loss = torch.nn.MSELoss()
        ssim = StructuralSimilarityIndexMeasure().to(self.device)
        best_loss = float('inf')
        use_adv = True

        print("\n[VAEFineTuner] Starting VAE-GAN fine-tuning ...")

        for epoch in range(epochs):
            self.vae.train()
            self.discriminator.train()

            train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
            if self.multi_image_mode:

                for batch, (images, gt) in enumerate(progress_bar):
                    images = images.to(self.device)
                    gt = gt.to(self.device)

                    # ══════════════════════════════════════════════════════════
                    # DISCRIMINATOR STEP
                    # Use detached reconstruction — no gradients flow into VAE
                    # ══════════════════════════════════════════════════════════
                    with torch.no_grad():
                        batch_size, num_imgs = images.shape[0], images.shape[1]
                        # print(f'The batch_size is {batch_size} and number of images per sample {num_imgs}')

                        print('The shape of images', images.shape)

                        combinedimages = images.reshape(-1, *images.shape[2:]).to(self.device)
                        # The above line reshapes the input images from (B,N,C,H,W) to (B*N,C,H,W) where *images.shape[2:] (the * is called the argument unpacking unpacks (C,H,W) and the -1
                        # in the first indes allows python to infer the dimensions on its own while maintaining the data count so for the first indes B*N is done.)
                        posterior = self.vae.encode(combinedimages.clone()).latent_dist
                        z_all = posterior.sample()  # (B*N,latent_dimensions, h,w)

                        print('The shape of latent dimensions', z_all.shape)
                        z_all = z_all.reshape(batch_size, num_imgs, *z_all.shape[1:])  # Reshaping back into (B,N,C,H,W)
                        z_averaged = z_all.mean(
                            dim=1)  # dim=1 specifies that the averaging should be done across the first dimension
                        # print('The shape of averaged latent code',z_averaged.shape)

                        # reconstructed = self.vae.decode(z_averaged).sample
                        reconstructed = self.vae.decode(z_averaged.clone()).sample

                    disc_optimizer.zero_grad(set_to_none=True)
                    d_loss = self._discriminator_loss(gt, reconstructed.detach())
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.discriminator.parameters(), max_norm=1.0
                    )
                    disc_optimizer.step()

                    # ══════════════════════════════════════════════════════════
                    # GENERATOR (VAE DECODER) STEP
                    # Fresh forward pass — completely new computation graph
                    # ══════════════════════════════════════════════════════════
                    optimizer.zero_grad(set_to_none=True)

                    batch_size, num_imgs = images.shape[0], images.shape[1]
                    # print(f'The batch_size is {batch_size} and number of images per sample {num_imgs}')

                    print('The shape of images', images.shape)

                    combinedimages = images.reshape(-1, *images.shape[2:]).to(self.device)
                    # The above line reshapes the input images from (B,N,C,H,W) to (B*N,C,H,W) where *images.shape[2:] (the * is called the argument unpacking unpacks (C,H,W) and the -1
                    # in the first indes allows python to infer the dimensions on its own while maintaining the data count so for the first indes B*N is done.)
                    posterior = self.vae.encode(combinedimages.clone()).latent_dist
                    mu = posterior.mean
                    log_var = posterior.logvar
                    # Reshape it to (B,N,latent_dimension,H,W)
                    mu = mu.reshape(batch_size, num_imgs, *mu.shape[1:])
                    log_var = log_var.reshape(batch_size, num_imgs, *log_var.shape[1:])
                    latent_technique = 'averaging'
                    if latent_technique == 'product_of_gaussians':
                        # Apply Product of Gaussians across the N images (dim=1)
                        # Calculate the precision (B,N,latent_dim, H,W)
                        precision = torch.exp(-log_var)
                        combined_precision = precision.sum(dim=1)  # Precision summed along the number of images (B,latent_dim,H,W)
                        combined_mu = (mu * precision).sum(dim=1) / combined_precision
                        log_combined_precision = -torch.log(combined_precision)

                        sampled_precision = (0.5 * log_combined_precision).exp()
                        z_averaged = combined_mu + sampled_precision * torch.randn_like(combined_mu)

                    elif latent_technique == 'averaging':
                        # Averaging the distributions of images in the latent space
                        averaged_mu = mu.mean(dim=1)
                        averaged_logvar = log_var.mean(dim=1)
                        std_logvar = (0.5 * averaged_logvar).exp()
                        '''print('The standard deviation is ',std_logvar, 'and the shape is ',std_logvar.shape)
                        print('The shape of the averaged_mu',averaged_mu.shape)'''
                        # Sample a Z from the mean distribution
                        z_averaged = averaged_mu + std_logvar * torch.randn_like(averaged_mu)

                    #z_all = posterior.sample()  # (B*N,latent_dimensions, h,w)

                    #print('The shape of latent dimensions', z_all.shape)
                    #z_all = z_all.reshape(batch_size, num_imgs, *z_all.shape[1:])  # Reshaping back into (B,N,C,H,W)
                    #z_averaged = z_all.mean(dim=1)  # dim=1 specifies that the averaging should be done across the first dimension
                    # print('The shape of averaged latent code',z_averaged.shape)

                    # reconstructed = self.vae.decode(z_averaged).sample
                    reconstructed = self.vae.decode(z_averaged.clone()).sample

                    reconstruction_loss = mse_loss(reconstructed, gt)
                    kl_loss = -0.5 *(1+averaged_logvar - averaged_mu.pow(2)- averaged_logvar.exp()).mean()

                    #kl_loss = posterior.kl().reshape(batch_size, num_imgs, -1).mean()

                    if use_adv:
                        fake_pred = self.discriminator(reconstructed)
                        adv_loss = -torch.mean(fake_pred)
                        print('The prediction of discriminator on reconstructed is ',fake_pred.mean())
                    else:
                        adv_loss = torch.tensor(0.0, device=self.device)


                    g_loss = reconstruction_loss + kl_weight * kl_loss + adv_loss
                    g_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                    optimizer.step()
                    #if batch <= 3:
                    self.visulaize_images(batch, epoch, reconstructed, images, gt, './output_images/', False)

            else:

                for batch,(images,target) in enumerate(train_progress_bar):

                    images = images.to(self.device)


                    # ══════════════════════════════════════════════════════════
                    # DISCRIMINATOR STEP
                    # Use detached reconstruction — no gradients flow into VAE
                    # ══════════════════════════════════════════════════════════
                    with torch.no_grad():

                        posterior = self.vae.encode(images).latent_dist
                        z = posterior.sample() # Sampling function based on Diagonal Gaussian Distribution class from Diffusers library.
                        z_prior = torch.randn_like(z)
                        reconstructed = self.vae.decode(z).sample
                        reconstructed_prior = self.vae.decode(z_prior).sample

                        cv2.imshow('Reconstructed 1', reconstructed[0].permute(1,2,0).cpu().numpy())
                        cv2.waitKey()
                        cv2.destroyAllWindows()
                        cv2.imshow('Reconstructed 2', reconstructed2[0].permute(1, 2, 0).cpu().numpy())
                        cv2.waitKey()
                        cv2.destroyAllWindows()

                    disc_optimizer.zero_grad()
                    d_loss = self._discriminator_loss(images, reconstructed, reconstructed_prior)
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                    disc_optimizer.step()

                    # ══════════════════════════════════════════════════════════
                    # GENERATOR (VAE DECODER) STEP
                    # Fresh forward pass — completely new computation graph
                    # ══════════════════════════════════════════════════════════

                    posterior = self.vae.encode(images).latent_dist
                    z = posterior.sample()
                    reconstructed = self.vae.decode(z.clone()).sample
                    reconstruction_loss = mse_loss(reconstructed, images)
                    kl_loss = posterior.kl().mean()     #kl() is also implemented on the Diagonal Gaussian Distribution class from Diffusers library.

                    if use_adv:
                        fake_pred = self.discriminator(reconstructed)
                        print('The predictions for reconstructed images',fake_pred.mean())
                        adv_loss = -torch.mean(fake_pred)
                    else:
                        adv_loss = torch.tensor(0.0, device=self.device)

                    optimizer.zero_grad()
                    g_loss = reconstruction_loss + kl_weight * kl_loss + adv_loss *0.01
                    g_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                    optimizer.step()
                    if batch <= 5:
                        self.visulaize_images(batch, epoch, reconstructed, images, None, './output_images/', False)


            scheduler.step()
            disc_scheduler.step()

            progress_bar.set_postfix({
                'loss': g_loss.item(),
                'recon': reconstruction_loss.item(),
                'kl': kl_loss.item()
            })

            test_loss = 0
            if test_loader:
                self.vae.eval()
                with torch.no_grad():
                    if self.multi_image_mode:
                        for batch, (images, labels) in enumerate(test_loader):
                            batch_size, num_imgs = images.shape[0], images.shape[1]
                            combinedimages = images.reshape(-1, *images.shape[2:]).to(self.device)
                            posterior = self.vae.encode(combinedimages).latent_dist
                            mu = posterior.mean
                            log_var = posterior.logvar
                            # Reshape it to (B,N,latent_dimension,H,W)
                            mu = mu.reshape(batch_size, num_imgs, *mu.shape[1:])
                            log_var = log_var.reshape(batch_size, num_imgs, *log_var.shape[1:])
                            latent_technique = 'averaging'
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
                                averaged_mu = mu.mean(dim=1)
                                averaged_logvar = log_var.mean(dim=1)
                                std_logvar = (0.5 * averaged_logvar).exp()
                                '''print('The standard deviation is ',std_logvar, 'and the shape is ',std_logvar.shape)
                                print('The shape of the averaged_mu',averaged_mu.shape)'''
                                # Sample a Z from the mean distribution
                                z_averaged = averaged_mu + std_logvar * torch.randn_like(averaged_mu)
                            #z_all = posterior.sample()
                            #z_all = z_all.reshape(batch_size, num_imgs,*z_all.shape[1:])  # Reshaping back into (B,N,C,H,W)
                            #z_averaged = z_all.mean(dim=1)
                            reconstructed = self.vae.decode(z_averaged).sample
                            target = labels
                            target = target.to(self.device)
                            reconstruction_loss = mse_loss(reconstructed, target)
                            kl_loss = -0.5 * (1 + averaged_logvar - averaged_mu.pow(2) - averaged_logvar.exp()).mean()
                            test_loss += reconstruction_loss.item() +kl_loss.item() * kl_weight
                            # target = images.mean(dim=1).to(self.device)
                            self.visulaize_images(batch, epoch, reconstructed, images, target,
                                                  './output_images/test_images', True)
                            print('The shape of target', target.shape)

                    else:
                        for batch,images in enumerate(test_loader):
                            images = images.to(self.device)
                            posterior = self.vae.encode(images).latent_dist
                            z = posterior.sample()
                            reconstructed = self.vae.decode(z).sample
                            recon_loss = mse_loss(reconstructed.clone(), images)
                            kl_loss = posterior.kl().mean()
                            if use_adv:
                                fake_test_pred = self.discriminator(reconstructed)
                                print('The fake test discriminator prediction ',fake_test_pred.mean())
                                adv_test_loss = -torch.mean(fake_test_pred)
                            else:
                                adv_test_loss = torch.tensor(0.0,device=self.device)
                            loss = recon_loss + kl_weight * kl_loss + adv_test_loss *0.01
                            test_loss += loss.item()
                            if batch <= 3:
                                self.visulaize_images(batch, epoch,reconstructed,images,None,'./output_images/test_images/',True)

                test_loss /= len(test_loader)
                print(f"Epoch {epoch + 1}: Train Loss: {g_loss / len(train_loader):.4f}, Val Loss: {test_loss:.4f}")

                if test_loss < best_loss:
                    best_loss = test_loss
                    self.vae.save_pretrained(save_path)
                    print(f"Saved best model with val loss: {test_loss:.4f}")

    def visulaize_images(self, batch, epoch, reconstructed_image, original, gt_image, save_path,
                         test=False):

        reconstructed_image = torch.clamp(reconstructed_image, 0, 1)
        original = torch.clamp(original, 0, 1)

        ssim = StructuralSimilarityIndexMeasure().to(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        if self.multi_image_mode:
            gt_image = torch.clamp(gt_image, 0, 1)
            for i in range(len(reconstructed_image)):
                if test:
                    plt.imsave(os.path.join(save_path, f'output_test_images_{epoch}_{batch}_{i}.png'),
                               reconstructed_image[i].permute(1, 2, 0).detach().cpu().numpy())
                    # individual_reconstruction = random_reconstruction[i].squeeze(0)
                    # individual_reconstruction= torch.clamp(individual_reconstruction, 0.0,1.0)
                    # plt.imsave(os.path.join(save_path, f'random_recon/random_reconstruction_{epoch}_{batch}_{i}.png'),individual_reconstruction.permute(1, 2, 0).detach().cpu().numpy())
                else:
                    plt.imsave(os.path.join(save_path, f'output_images_{epoch}_{batch}_{i}.png'),
                               reconstructed_image[i].permute(1, 2, 0).detach().cpu().numpy())

            for i in range(len(reconstructed_image)):
                fig = plt.figure(figsize=(16, 8))
                grid_spec = fig.add_gridspec(2, 5, hspace=0.1, wspace=0.1)

                for j in range(5):
                    ax = fig.add_subplot(grid_spec[0, j])
                    orig_img = original[i][j].permute(1, 2, 0).detach().cpu().numpy()
                    ax.imshow(orig_img)
                    ax.axis('off')
                ssim_loss = ssim(reconstructed_image, gt_image)
                ax_center = fig.add_subplot(grid_spec[1, 2])
                recon_images = reconstructed_image[i].permute(1, 2, 0).detach().cpu().numpy()
                ax_center.imshow(recon_images)
                ax_center.axis('off')
                ax_center.set_title(f'SSIM: {ssim_loss:.2f}')

                ax_left = fig.add_subplot(grid_spec[1, 1])
                # print(type(gt_image), gt_image.shape)
                gt_image_visual = gt_image[i].permute(1, 2, 0).detach().cpu().numpy()
                ax_left.imshow(gt_image_visual)
                ax_left.axis('off')

                '''ax_right = fig.add_subplot(grid_spec[1, 3])
                individual_reconstruction = random_reconstruction[i].squeeze(0)
                individual_reconstruction = torch.clamp(individual_reconstruction, 0.0, 1.0)
                ax_right.imshow(individual_reconstruction.permute(1, 2, 0).detach().cpu().numpy())
                ax_right.axis('off')'''

                for l in [0, 3, 4]:
                    ax = fig.add_subplot(grid_spec[1, l])
                    ax.axis("off")

                plt.tight_layout()
                plt.savefig(os.path.join(save_path, f'reconstructed_images_{epoch}_{batch}_{i}.png'), dpi=150)
                plt.close()
        else:

            num_images = reconstructed_image.shape[0]
            fig, axes = plt.subplots(2, num_images, figsize=(16, 8))
            # Original Image
            for j in range(num_images):
                orig_img = original[j].permute(1, 2, 0).detach().cpu().numpy()
                axes[0, j].imshow(orig_img)
                axes[0, j].set_title(f'Original {j + 1}')
                axes[0, j].axis('off')

                # Reconstruction Images
                recon_images = reconstructed_image[j].permute(1, 2, 0).detach().cpu().numpy()
                axes[1, j].imshow(recon_images)
                axes[1, j].set_title(f'Recon {j + 1}')
                axes[1, j].axis('off')
            if test:
                plt.tight_layout()
                plt.savefig(f'./output_images/individual_images/test_images/reconstructed_test_{epoch}_{batch}.png',
                            dpi=150)
                plt.close()
            else:
                plt.tight_layout()
                plt.savefig(f'./output_images/individual_images/reconstructed_images_{epoch}_{batch}.png', dpi=150)
                plt.close()

            for i in range(len(reconstructed_image)):
                if test:
                    plt.imsave(os.path.join('./output_images/individual_images/test_images/',
                                            f'output_test_images_{epoch}_{batch}_{i}.png'),
                               reconstructed_image[i].permute(1, 2, 0).detach().cpu().numpy())
                else:
                    plt.imsave(
                        os.path.join('./output_images/individual_images/', f'output_images_{epoch}_{batch}_{i}.png'),
                        reconstructed_image[i].permute(1, 2, 0).detach().cpu().numpy())


