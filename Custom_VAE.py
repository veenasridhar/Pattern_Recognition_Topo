"""
Custom VAE for topology-optimization label maps with 2x spatial downsampling.

Design goals (vs. frozen SD-VAE, 8x):
  * 2x downsampling only -> 1px-wide members survive encoding (a 1px member
    still occupies ~0.5 latent cells instead of ~0.125, and the stem sees it
    at full resolution before the single stride-2 step).
  * Input/output are 3-class label maps (CLS_VOID=0, CLS_NONDESIGN=1,
    CLS_MEMBER=2) -- the latent space is organized around the classes you
    care about, not natural-image statistics.
  * Spatial Gaussian latent with (mu, logvar) heads -> drop-in compatible
    with PrecisionAwareAggregation (concat mu+logvar, softmax over N views,
    law-of-total-variance for aggregated uncertainty).
  * GroupNorm everywhere (no BatchNorm running-stats issues), no output
    activation on the decoder (softmax happens exactly once, in the trainer,
    per your established convention).

Shapes (example, 512x512 input):
    x        : [B, 3, 512, 512]   one-hot label map (float)
    mu,logvar: [B, Z, 256, 256]   Z = latent_channels (default 4)
    logits   : [B, 3, 512, 512]   raw logits, no activation

Memory note: at 2x downsampling the latent is large. Keep latent_channels
small (4-8) and keep the aggregation score network lightweight (1x1 convs),
otherwise N=5 views at 256x256xZ will hurt on the cluster.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from attSets import AttSets
import wandb
from cldice_loss_vae import *
from cldice_vis import *
from thesis_visualisation import visualize_comparison_with_overlay
#from input_pipeline_discrete import *
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Canonical label-map conventions (matches your to_label_map bucketing)
# ----------------------------------------------------------------------------
CLS_VOID, CLS_NONDESIGN, CLS_MEMBER = 0, 1, 2
NUM_CLASSES = 3


def to_label_map(gray: torch.Tensor) -> torch.Tensor:
    """Grayscale {0,128,255} (any float/int tensor, [B,1,H,W] or [B,H,W])
    -> integer label map [B,H,W] via thresholds at 64 and 192."""
    if gray.dim() == 4:
        gray = gray.squeeze(1)
    labels = torch.full_like(gray, CLS_VOID, dtype=torch.long)
    labels[gray > 64] = CLS_NONDESIGN
    labels[gray > 192] = CLS_MEMBER
    return labels


def labels_to_onehot(labels: torch.Tensor) -> torch.Tensor:
    """[B,H,W] long -> [B,3,H,W] float one-hot (VAE input)."""
    return F.one_hot(labels, NUM_CLASSES).permute(0, 3, 1, 2).float()


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------
def gn(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm1 = gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = gn(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


# ----------------------------------------------------------------------------
# Encoder: full-res stem -> single stride-2 downsample -> (mu, logvar)
# ----------------------------------------------------------------------------
class Encoder2x(nn.Module):
    def __init__(
        self,
        in_channels: int = NUM_CLASSES,
        base_channels: int = 64,
        latent_channels: int = 4,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        C = base_channels

        # Full-resolution stem: lets the network see 1px members BEFORE any
        # downsampling -- this is the whole point of the 2x design.
        self.stem = nn.Conv2d(in_channels, C, 3, padding=1)
        self.res_full = nn.Sequential(
            *[ResBlock(C, C) for _ in range(num_res_blocks)]
        )

        # The one and only downsample.
        self.down = nn.Conv2d(C, 2 * C, 3, stride=2, padding=1)
        self.res_half = nn.Sequential(
            *[ResBlock(2 * C, 2 * C) for _ in range(num_res_blocks)]
        )

        self.norm_out = gn(2 * C)
        # Separate heads keep mu and logvar statistics decoupled.
        self.to_mu = nn.Conv2d(2 * C, latent_channels, 1)
        self.to_logvar = nn.Conv2d(2 * C, latent_channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.res_full(self.stem(x))
        h = self.res_half(self.down(h))
        h = F.silu(self.norm_out(h))
        mu = self.to_mu(h)
        # Clamp for numerical stability in exp() / PoG downstream.
        logvar = self.to_logvar(h).clamp(-10.0, 10.0)
        return mu, logvar


# ----------------------------------------------------------------------------
# Decoder: latent -> single 2x upsample -> full-res refinement -> logits
# ----------------------------------------------------------------------------
class Decoder2x(nn.Module):
    def __init__(
        self,
        out_channels: int = NUM_CLASSES,
        base_channels: int = 64,
        latent_channels: int = 4,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        C = base_channels

        self.from_latent = nn.Conv2d(latent_channels, 2 * C, 3, padding=1)
        self.res_half = nn.Sequential(
            *[ResBlock(2 * C, 2 * C) for _ in range(num_res_blocks)]
        )

        # Nearest-neighbor upsample + conv (no ConvTranspose -> no
        # checkerboard artifacts on thin members).
        self.up_conv = nn.Conv2d(2 * C, C, 3, padding=1)
        self.res_full = nn.Sequential(
            *[ResBlock(C, C) for _ in range(num_res_blocks)]
        )

        self.norm_out = gn(C)
        # Raw logits. NO output activation -- softmax happens once, in the
        # trainer, feeding SoftDiceClDiceLoss + CE.
        self.conv_out = nn.Conv2d(C, out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.res_half(self.from_latent(z))
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.res_full(self.up_conv(h))
        return self.conv_out(F.silu(self.norm_out(h)))


# ----------------------------------------------------------------------------
# Full VAE
# ----------------------------------------------------------------------------
class TOVae2x(nn.Module):
    """VAE over 3-class TO label maps, 2x spatial compression.

    forward(x) with x = one-hot [B,3,H,W] returns a dict:
        logits [B,3,H,W], mu/logvar/z [B,Z,H/2,W/2], kl (scalar, per-element mean)

    For your aggregation pipeline:
        mu_i, logvar_i = model.encode(x_i)        # per view i = 1..N
        agg_mu, agg_logvar = your PrecisionAwareAggregation(...)
        logits = model.decode(agg_mu)             # deterministic path
    """

    def __init__(self, base_channels: int = 64, latent_channels: int = 4, num_res_blocks: int = 2, free_bits: float = 0.05):
        super().__init__()
        self.encoder = Encoder2x(
            NUM_CLASSES, base_channels, latent_channels, num_res_blocks
        )
        self.decoder = Decoder2x(
            NUM_CLASSES, base_channels, latent_channels, num_res_blocks
        )
        self.latent_channels = latent_channels
        self.free_bits = free_bits

    # -- pieces usable independently by the aggregation pipeline ------------
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    # -- KL with free bits (guards against posterior collapse; with a
    #    powerful decoder + small beta this matters) -------------------------
    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl_map = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)  # [B,Z,h,w]
        kl_per_ch = kl_map.mean(dim=(0, 2, 3))                    # [Z]
        if self.free_bits > 0:
            kl_per_ch = torch.clamp(kl_per_ch, min=self.free_bits)
        return kl_per_ch.sum()

    def forward(self, x: torch.Tensor, sample: bool = True) -> dict:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        logits = self.decode(z)
        return {
            "logits": logits,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "kl": self.kl_loss(mu, logvar),
        }


# ----------------------------------------------------------------------------
# Latent aggregation over N views (dim=1). [FIX-3] deterministic everywhere.
# ----------------------------------------------------------------------------
def aggregate_latents(
    mu: torch.Tensor,                # [B,N,Z,h,w]
    logvar: torch.Tensor,            # [B,N,Z,h,w]
    technique: str,
    aggregation_module=None,
) -> torch.Tensor:
    if technique == "averaging":
        return mu.mean(dim=1)
 
    if technique == "product_of_gaussians":
        precision = torch.exp(-logvar)                           # [B,N,Z,h,w]
        combined_precision = precision.sum(dim=1)
        return (mu * precision).sum(dim=1) / combined_precision  # PoG mean
 
    if technique == "median":
        return mu.median(dim=1).values
 
    if technique == "learned_aggregation":
        B, N = mu.shape[0], mu.shape[1]
        flat_mu = mu.reshape(B * N, *mu.shape[2:])
        return aggregation_module(flat_mu, sample=False, num_imgs=N, batch_size=B)
 
    raise ValueError(f"unknown latent technique: {technique}")

# ----------------------------------------------------------------------------
# Minimal training-step skeleton (plug in your SoftDiceClDiceLoss)
# ----------------------------------------------------------------------------
class CustomVAE:

    def __init__(self, model, multi_image_mode=True, device=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multi_image_mode = multi_image_mode
        self.vae = model
        self.aggregation_module = AttSets(latent_channels=4)
        self.aggregation_module = self.aggregation_module.to(device)
        self.vae.to(self.device)
        #self.PROJECT_NAME = 'Individual Reconstruction Cross Validation'
        self.PROJECT_NAME = 'Topology Optimization'


    def finetune(self, train_loader, val_loader, test_loader,fold, epochs=101, lr=0.001, kl_weight=1e-6, latent_technique = 'learned_aggregation', save_path="./finetuned_vae"):

        # Start a new wandb run to track this script.
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="veenasridhar96-university-of-stuttgart",
            # Set the wandb project where this run will be logged.
            #project=self.PROJECT_NAME,
            name = 'Topology Optimization',
            #job_type= f'run- {fold}',
            # Track hyperparameters and run metadata.
            config={
                "learning_rate": lr,
                #'kl_weight': kl_weight,
                'latent_technique': latent_technique,
                'loss': 'Perceptual Loss ',
                "architecture": "VAE",
                "dataset": "TO",
                "epochs": epochs,
                "fold": fold,
            }, )

        #torch.autograd.set_detect_anomaly(True)

        # for name, module in self.vae.named_modules():
        #     #print(name, "->", module.__class__.__name__)
        #     for p in module.parameters():
        #             p.requires_grad = False



        if self.multi_image_mode:

        #     trainable_modules = [
        #         self.vae.decoder.up_blocks[2],
        #         self.vae.decoder.up_blocks[3],   # final upsampling block (highest resolution)
        #         self.vae.decoder.conv_norm_out,
        #         self.vae.decoder.conv_out,
        #     ]
        #     for module in trainable_modules:
        #         for p in module.parameters():
        #             p.requires_grad = True

            optimizer = torch.optim.AdamW(self.vae.parameters(), lr=lr, weight_decay=1e-2)
            if latent_technique == 'learned_aggregation':
                aggregation_module_optimizer = torch.optim.AdamW(self.aggregation_module.parameters(), lr=0.001, weight_decay=1e-4)
                print('The training parameters for the aggregation module are',sum(p.numel() for p in self.aggregation_module.parameters() if p.requires_grad))

            print('The training parameters for the aggregation module are',sum(p.numel() for p in self.vae.decoder.parameters() if p.requires_grad))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        else:
            optimizer = torch.optim.AdamW(self.vae.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        
        cldice_loss = soft_dice_cldice()
        ce_weights = torch.tensor([0.25, 0.35, 0.5],device=self.device)
        ce_loss = torch.nn.CrossEntropyLoss(weight= ce_weights)
        best_loss = float('inf')

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

            train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
            index =0

            if self.multi_image_mode:
                accum_steps = 4
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

                    # Sampling happens from Diffusers library
                    print('The shape of combined images',combinedimages.shape)

                    with torch.no_grad():                                               # Encoder is not trained during the multi iamge mode
                        print('The unique values of the input images are',combinedimages.unique())
                        individual_mu, log_var = self.vae.encode(combinedimages).latent_dist        # Posterior - N(mu,std), latent_dist - Encoded outputs of Encoder represented as the mean and logvar of DiagonalGaussianDistribution. DiagonalGaussianDistribution allows for sampling latents from the distribution
                        #target_posterior = self.vae.encode(target).latent_dist
                        # individual_mu = posterior.mean       # Individual mu is mean represented using 4 latent variables (10,4,32,32)
                        # log_var = posterior.logvar          # Log variance represent the uncertainty or spread of the distribution. (log for stability)
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

                    train_ce_loss = ce_loss(reconstructed,target)
                    probs       = torch.softmax(reconstructed, dim=1)
                    member_prob = probs[:, 2:3]
                    member_gt   = (target == 2).float().unsqueeze(1)
                    train_cldice_loss = cldice_loss(member_prob,member_gt)
                    reconstruction_loss =  train_ce_loss + train_cldice_loss
                    #latent_space_exploration(individual_mu, z_averaged, target_mu, self.vae)

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
                    train_recon_loss +=reconstruction_loss.item()

                    #reconstructed = torch.argmax(reconstructed, dim=1)

                    for i in range(len(reconstructed)):
                        
                        print('The shape of the reconstructed image',reconstructed[i].shape)
                        print('The shape of the target image',target[i].shape)
                        print('The shape of avg_result image',avg_result[i].shape)

                        reconstructed = torch.argmax(reconstructed[i], dim=0)
                        print('The shape of the reconstructed image',reconstructed.shape)
                        print('The unique values of the reconstructed image',reconstructed.unique())

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
                        # iou_dilated,iou_dilated_image_processing,dilated_overlap,img_proc_dilated_overlap = visulaize_iou_metrics(predicted_binary, gt_binary, img_prc_binary, tolerance=0)
                        # train_iou_dl += iou_dilated
                        # train_iou_img_prc += iou_dilated_image_processing

                        if epoch % 5 == 0:
                            if not os.path.exists(os.path.join(f'./output_images_{latent_technique}_{fold}')):
                                os.mkdir(os.path.join(f'./output_images_{latent_technique}_{fold}'))
                            if not os.path.exists(os.path.join(f'./output_images_{latent_technique}_{fold}/train')):
                                os.mkdir(os.path.join(f'./output_images_{latent_technique}_{fold}/train'))
                            if not os.path.exists(os.path.join(f'./output_images_{latent_technique}_{fold}/train/Epoch {epoch}')):
                                os.mkdir(os.path.join(f'./output_images_{latent_technique}_{fold}/train/Epoch {epoch}'))
                            if not os.path.exists(os.path.join(f'./output_images_{latent_technique}_{fold}/train/Epoch {epoch}', 'Metrics')):
                                os.mkdir(os.path.join(f'./output_images_{latent_technique}_{fold}/train/Epoch {epoch}','Metrics'))

                            plt.imsave(os.path.join(f'./output_images_{latent_technique}_{fold}/train/Epoch {epoch}', f'output_images_{index}.png'),
                                   reconstructed.detach().cpu().numpy(), cmap='gray', vmin = reconstructed.min(), vmax = reconstructed.max())
                            
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
                            }
                            visualize_cldice_comparison_grid_discrete([group], epoch=epoch, batch=index, fold=fold,save_path=f'./output_images_{latent_technique}_{fold}/train/Epoch {epoch}/Metrics')
                        
                            # visulaize_images(self.multi_image_mode, batch, epoch, reconstructed[i], images[i], target[i], avg_result[i], iou_dilated,iou_dilated_image_processing,dilated_overlap,
                            #                  img_proc_dilated_overlap, f'./output_images_{latent_technique}_{fold}/train/Epoch {epoch}/Metrics', index, False)
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
                    optimizer.step()
                    optimizer.zero_grad()
                
                scheduler.step()
        
            ###################################
            ### Single Image Mode
            ###################################
            else:
                accum_steps = 256
                optimizer.zero_grad()
                for batch, (images,target) in enumerate(train_progress_bar):

                    images = images.to(self.device)
                    target = target.to(self.device)

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
                        if not os.path.exists(os.path.join(f'./Individual Reconstruction Results')):
                                os.mkdir(os.path.join(f'./Individual Reconstruction Results'))
                        if not os.path.exists(os.path.join(f'./Individual Reconstruction Results/individual_images_Fold {fold}')):
                                    os.mkdir(os.path.join(f'./Individual Reconstruction Results/individual_images_Fold {fold}'))
                        if not os.path.exists(os.path.join(f'./Individual Reconstruction Results/individual_images_Fold {fold}/train')):
                            os.mkdir(os.path.join(f'./Individual Reconstruction Results/individual_images_Fold {fold}/train'))
                        if not os.path.exists(os.path.join(f'./Individual Reconstruction Results/individual_images_Fold {fold}/train/Epoch {epoch}')):
                            os.mkdir(os.path.join(f'./Individual Reconstruction Results/individual_images_Fold {fold}/train/Epoch {epoch}'))
                        print('The shape of the reconstructed individual image',reconstructed.shape)
                        reconstructed = torch.argmax(reconstructed, dim=1)
                        visulaize_images_individual(batch,epoch, reconstructed, target,None, f'./Individual Reconstruction Results/individual_images_Fold {fold}/train/Epoch {epoch}',False)
                        
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

            val_progress_bar = tqdm(val_loader,desc=f'Epoch {epoch}/{epochs}')
            if val_loader:
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
                            reconstruction_val_loss =  val_ce_loss + val_cldice_loss

                            val_loss = reconstruction_val_loss

                            val_vae_loss += val_loss.item()
                            val_mse_loss += reconstruction_val_loss.item()
                            #val_kl_loss += kl_val_loss.item()

                            for i in range(len(reconstructed)):
                        
                                # print('The shape of the reconstructed image',reconstructed[i].shape)
                                # print('The shape of the target image',target[i].shape)
                                # print('The shape of avg_result image',avg_result[i].shape)

                                reconstructed = torch.argmax(reconstructed[i], dim=0)
                                # print('The shape of the reconstructed image',reconstructed.shape)
                                # print('The unique values of the reconstructed image',reconstructed.unique())
                                
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
                                # iou_dilated,iou_dilated_image_processing,dilated_overlap,img_proc_dilated_overlap = visulaize_iou_metrics(predicted_binary, gt_binary, img_prc_binary, tolerance=0)
                                # train_iou_dl += iou_dilated
                                # train_iou_img_prc += iou_dilated_image_processing

                                if epoch % 5 == 0:
                                    if not os.path.exists(os.path.join(f'./output_images_{latent_technique}_{fold}')):
                                        os.mkdir(os.path.join(f'./output_images_{latent_technique}_{fold}'))
                                    if not os.path.exists(os.path.join(f'./output_images_{latent_technique}_{fold}/val')):
                                        os.mkdir(os.path.join(f'./output_images_{latent_technique}_{fold}/val'))
                                    if not os.path.exists(os.path.join(f'./output_images_{latent_technique}_{fold}/val/Epoch {epoch}')):
                                        os.mkdir(os.path.join(f'./output_images_{latent_technique}_{fold}/val/Epoch {epoch}'))
                                    if not os.path.exists(os.path.join(f'./output_images_{latent_technique}_{fold}/val/Epoch {epoch}', 'Metrics')):
                                        os.mkdir(os.path.join(f'./output_images_{latent_technique}_{fold}/val/Epoch {epoch}','Metrics'))

                                    plt.imsave(os.path.join(f'./output_images_{latent_technique}_{fold}/val/Epoch {epoch}', f'output_images_{index}.png'),
                                        reconstructed.detach().cpu().numpy(), cmap='gray', vmin = reconstructed.min(), vmax = reconstructed.max())


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
                                    }
                                    visualize_cldice_comparison_grid_discrete([group], epoch=epoch, batch=index, fold=fold,save_path=f'./output_images_{latent_technique}_{fold}/val/Epoch {epoch}/Metrics')
                                
                                    # visulaize_images(self.multi_image_mode, batch, epoch, reconstructed[i], images[i], target[i], avg_result[i], iou_dilated,iou_dilated_image_processing,dilated_overlap,
                                    #                  img_proc_dilated_overlap, f'./output_images_{latent_technique}_{fold}/train/Epoch {epoch}/Metrics', index, False)
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
                                if not os.path.exists(os.path.join(f'./Individual Reconstruction Results')):
                                        os.mkdir(os.path.join(f'./Individual Reconstruction Results'))
                                if not os.path.exists(os.path.join(f'./Individual Reconstruction Results/individual_images_Fold {fold}')):
                                    os.mkdir(os.path.join(f'./Individual Reconstruction Results/individual_images_Fold {fold}'))
                                if not os.path.exists(os.path.join(f'./Individual Reconstruction Results/individual_images_Fold {fold}/val')):
                                    os.mkdir(os.path.join(f'./Individual Reconstruction Results/individual_images_Fold {fold}/val'))
                                if not os.path.exists(os.path.join(f'./Individual Reconstruction Results/individual_images_Fold {fold}/val/Epoch {epoch}')):
                                    os.mkdir(os.path.join(f'./Individual Reconstruction Results/individual_images_Fold {fold}/val/Epoch {epoch}'))
                                print('The shape of the reconstructed individual image',reconstructed.shape)
                                reconstructed = torch.argmax(reconstructed, dim=1)
                                visulaize_images_individual(batch,epoch, reconstructed, target,None, f'./Individual Reconstruction Results/individual_images_Fold {fold}/val/Epoch {epoch}',False)
                        
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
                            reconstruction_loss =  test_ce_loss + test_cldice_loss
                            reconstruction_test_loss =  test_ce_loss + test_cldice_loss
                            #reconstruction_test_loss = mse_loss(reconstructed, target)

                            t_loss = reconstruction_test_loss

                            test_vae_loss += t_loss.item()
                            test_mse_loss += reconstruction_test_loss.item()

                            for i in range(len(reconstructed)):

                                reconstructed = torch.argmax(reconstructed[i], dim=0)
                                print('The shape of the reconstructed image',reconstructed.shape)
                                print('The unique values of the reconstructed image',reconstructed.unique())
                                
                                
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
                                # iou_dilated,iou_dilated_image_processing,dilated_overlap,img_proc_dilated_overlap = visulaize_iou_metrics(predicted_binary, gt_binary, img_prc_binary, tolerance=0)
                                # test_iou_dl += iou_dilated
                                # test_iou_img_prc += iou_dilated_image_processing

                                if epoch % 5 == 0:
                                
                                    if not os.path.exists(os.path.join(f'./output_images_{latent_technique}_{fold}')):
                                        os.mkdir(os.path.join(f'./output_images_{latent_technique}_{fold}'))
                                    if not os.path.exists(os.path.join(f'./output_images_{latent_technique}_{fold}/test')):
                                        os.mkdir(os.path.join(f'./output_images_{latent_technique}_{fold}/test'))
                                    if not os.path.exists(os.path.join(f'./output_images_{latent_technique}_{fold}/test/Epoch {epoch}')):
                                        os.mkdir(os.path.join(f'./output_images_{latent_technique}_{fold}/test/Epoch {epoch}'))
                                    if not os.path.exists(os.path.join(f'./output_images_{latent_technique}_{fold}/test/Epoch {epoch}', 'Metrics')):
                                        os.mkdir(os.path.join(f'./output_images_{latent_technique}_{fold}/test/Epoch {epoch}','Metrics'))

                                    plt.imsave(os.path.join(f'./output_images_{latent_technique}_{fold}/test/Epoch {epoch}', f'output_images_{index}.png'),
                                            reconstructed.detach().cpu().numpy(), cmap='gray', vmin = reconstructed.min(), vmax = reconstructed.max())

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
                                    }
                                    visualize_cldice_comparison_grid_discrete([group], epoch=epoch, batch=index, fold=fold,save_path=f'./output_images_{latent_technique}_{fold}/test/Epoch {epoch}/Metrics')
                                    
                                    # visulaize_images(self.multi_image_mode, batch, epoch, reconstructed, images[i], target[i], avg_result[i], iou_dilated,iou_dilated_image_processing,dilated_overlap,
                                    #              img_proc_dilated_overlap, f'./output_images_{latent_technique}_{fold}/test/Epoch {epoch}/Metrics', index, False)
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
                                if not os.path.exists(os.path.join(f'./Individual Reconstruction Results')):
                                        os.mkdir(os.path.join(f'./Individual Reconstruction Results'))
                                if not os.path.exists(os.path.join(f'./Individual Reconstruction Results/individual_images_Fold {fold}')):
                                    os.mkdir(os.path.join(f'./Individual Reconstruction Results/individual_images_Fold {fold}'))
                                if not os.path.exists(os.path.join(f'./Individual Reconstruction Results/individual_images_Fold {fold}/test')):
                                    os.mkdir(os.path.join(f'./Individual Reconstruction Results/individual_images_Fold {fold}/test'))
                                if not os.path.exists(os.path.join(f'./Individual Reconstruction Results/individual_images_Fold {fold}/test/Epoch {epoch}')):
                                    os.mkdir(os.path.join(f'./Individual Reconstruction Results/individual_images_Fold {fold}/test/Epoch {epoch}'))
                                print('The shape of the reconstructed individual image',reconstructed.shape)
                                reconstructed = torch.argmax(reconstructed, dim=1)
                                visulaize_images_individual(batch,epoch, reconstructed, target,None, f'./Individual Reconstruction Results/individual_images_Fold {fold}/test/Epoch {epoch}',False)
                        

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
                    # 'Train IoU Img_prc': train_iou_img_prc/len(train_loader.dataset), 'Train IoU DL': train_iou_dl/len(train_loader.dataset),
                    # 'Val IoU Img_prc': val_iou_img_prc/len(val_loader.dataset), 'Val IoU DL': val_iou_dl /len(val_loader.dataset),
                    # 'Test IoU Img_prc': test_iou_img_prc/len(test_loader.dataset), 'Test IoU DL': test_iou_dl/len(test_loader.dataset),
                    #'Train_MSE_Loss': train_mse_loss / len(train_loader), 'Test_MSE_Loss': test_mse_loss / len(test_loader),'Val MSE Loss': val_mse_loss/len(val_loader),
                     #'Train_SSIM_Loss': train_ssim_loss / len(train_loader), 'Test_SSIM_Loss': test_ssim_loss / len(test_loader), 'Val SSIM Loss':val_ssim_loss/len(val_loader),
                     #'Train_KL_Loss': train_kl_loss / len(train_loader), 'Test_KL_Loss': test_kl_loss / len(test_loader),'Val KL Loss':val_kl_loss/len(val_loader),
                   })

        wandb.finish()