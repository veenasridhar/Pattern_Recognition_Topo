
import torch
import torch.nn as nn

class simple_Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(simple_Conv, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)  # 1st layer
        self.conv_layer2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=1, padding=1)
        self.conv_layer3 = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=1)
        self.conv_layer4 = nn.Conv2d(64, 128, kernel_size=kernel_size, stride=1, padding=1)
        self.conv_layer5 = nn.Conv2d(128, 256, kernel_size=kernel_size, stride=1, padding=1)
        self.conv_layer7 = nn.Conv2d(256, 64, kernel_size=kernel_size, stride=1, padding=1)
        self.conv_layer8 = nn.Conv2d(64, 16, kernel_size=kernel_size, stride=1, padding=1)
        self.output_layer = nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout2d(p=0.2)
        self.relu_layer = nn.ReLU()

    def forward(self, x):
        x1 = self.relu_layer(self.conv_layer1(x))
        x2 = self.relu_layer(self.conv_layer2(x1))
        x3 = self.relu_layer(self.conv_layer3(x2))
        x4 = self.relu_layer(self.conv_layer4(x3))
        x5 = self.relu_layer(self.conv_layer5(x4))
        #x6 = self.relu_layer(self.conv_layer6(x5))
        x7 = self.relu_layer(self.conv_layer7(x5))
        x8 = self.relu_layer(self.conv_layer8(x7))
        conv_output = self.output_layer(x8)
        output = conv_output
        #output = alpha * conv_output + avg_output              #Previously tried Residual Approach
        return output


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
# Encoder
# ----------------------------------------------------------------------------
class Encoder2x(nn.Module):
    def __init__(self, in_channels: int = NUM_CLASSES, base_channels: int = 64, latent_channels: int = 4, num_res_blocks: int = 2,):
        super().__init__()
        C = base_channels

        self.stem = nn.Conv2d(in_channels, C, 3, padding=1)
        self.res_full = nn.Sequential(
            *[ResBlock(C, C) for _ in range(num_res_blocks)]
        )

        # Downsample
        self.down = nn.Conv2d(C, 2 * C, 3, stride=2, padding=1)
        self.res_half = nn.Sequential(
            *[ResBlock(2 * C, 2 * C) for _ in range(num_res_blocks)]
        )

        self.norm_out = gn(2 * C)
        # Separate heads  mu and logvar
        self.to_mu = nn.Conv2d(2 * C, latent_channels, 1)
        self.to_logvar = nn.Conv2d(2 * C, latent_channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.res_full(self.stem(x))
        h = self.res_half(self.down(h))
        h = F.silu(self.norm_out(h))
        mu = self.to_mu(h)
        # Clamp for numerical stability in exp() / PoG downstream.
        logvar = self.to_logvar(h)                      #.clamp(-10.0, 10.0)
        return mu, logvar


# ----------------------------------------------------------------------------
# Decoder
# ----------------------------------------------------------------------------
class Decoder2x(nn.Module):
    def __init__(self, out_channels: int = NUM_CLASSES, base_channels: int = 64, latent_channels: int = 4, num_res_blocks: int = 2):
        super().__init__()
        C = base_channels

        self.from_latent = nn.Conv2d(latent_channels, 2 * C, 3, padding=1)
        self.res_half = nn.Sequential(*[ResBlock(2 * C, 2 * C) for _ in range(num_res_blocks)])

        # Nearest-neighbor upsample + conv (no ConvTranspose -> nocheckerboard artifacts on thin members).
        #self.up_conv = nn.Conv2d(2 * C, C, 3, padding=1)
        self.up_conv = nn.ConvTranspose2d(2 * C, C, kernel_size=4, stride=2, padding=1)
        self.res_full = nn.Sequential(*[ResBlock(C, C) for _ in range(num_res_blocks)] )

        self.norm_out = gn(C)
        self.conv_out = nn.Conv2d(C, out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.res_half(self.from_latent(z))
        #h = F.interpolate(h, scale_factor=2, mode="nearest")       #Alternate to the transpose Convolution
        h = self.res_full(self.up_conv(h))
        return self.conv_out(F.silu(self.norm_out(h)))


# ----------------------------------------------------------------------------
# Full VAE
# ----------------------------------------------------------------------------
class TOVae2x(nn.Module):
    """VAE over 2-class TO label maps, 2x spatial compression.
    """

    def __init__(self, base_channels: int = 64, latent_channels: int = 4, num_res_blocks: int = 2, free_bits: float = 0.05):
        super().__init__()
        self.encoder = Encoder2x(NUM_CLASSES, base_channels, latent_channels, num_res_blocks)
        
        self.decoder = Decoder2x(NUM_CLASSES, base_channels, latent_channels, num_res_blocks)

        self.latent_channels = latent_channels


    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    # -- KL with free bits (guards against posterior collapse; with a
    #    powerful decoder + small beta this matters) -------------------------
    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor, free_bits) -> torch.Tensor:
        kl_map = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)  # [B,Z,h,w]
        kl_per_ch = kl_map.mean(dim=(0, 2, 3))                    # [Z]
        # if free_bits > 0:
        #     kl_per_ch = torch.clamp(kl_per_ch, min=free_bits)
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
            "kl": self.kl_loss(mu, logvar, self.free_bits),
        }


# ----------------------------------------------------------------------------
# Latent aggregation over N views (dim=1).
# ----------------------------------------------------------------------------
def aggregate_latents( mu: torch.Tensor, logvar: torch.Tensor, technique: str, aggregation_module=None) -> torch.Tensor:
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

def get_alpha(epoch, anneal_epochs=50, start=0.7, end=0.3):
    frac = min(epoch / anneal_epochs, 1.0)
    return start + (end - start) * frac



class AttSets(nn.Module):
    """
       From AttSets paper — attentive aggregation of feature set        - https://arxiv.org/pdf/1808.00758
       Works with arbitrary N input features
       Permutation invariant (order of images doesn't matter)
       """
    def __init__(self,latent_channels = 4):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Conv2d(in_channels=latent_channels, out_channels=16, kernel_size=3,padding=1),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.LeakyReLU(),      
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=3,padding=1),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=3,padding=1),
            nn.GroupNorm(num_groups=4, num_channels=64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=3,padding=1),
            nn.GroupNorm(num_groups=4, num_channels=128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size=3,padding=1),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 8, kernel_size=3,padding=1),
            nn.GroupNorm(num_groups=4, num_channels=8),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = 8, out_channels = 4, kernel_size=3,padding=1),
        )
        self.score_net = self.score_net .to(torch.device("cuda"))

    def forward(self,input_latents, sample = None, permute = None, num_imgs=5,batch_size=4):

        '''Method-1
        This method weights individual images separately (sigmoid activations) and multiplies with its own input latent representations before going through the aggregation
         Finally for aggregation it sums it along the first dimension in (B,N,C,H,W) i.e along the number of images in each batch.'''

        scores = self.score_net(input_latents)  # Size = (B*N,C,H,W)

        batch_latents = input_latents.view(batch_size,num_imgs,*(input_latents.shape[1:]))
        
        batch_scores = scores.view(batch_size,num_imgs, *(scores.shape[1:]))
        
        weights = torch.softmax(batch_scores,dim=1)
        
        output = (weights *batch_latents).sum(dim=1)
       
        return output,weights
