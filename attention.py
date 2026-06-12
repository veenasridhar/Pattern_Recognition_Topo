import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, channels=4, spatial=64,num_heads=2):
        super().__init__()
        self.channels = channels

        #Q,K,V Projections - 1x1 convs (per pixel linear)
        self.to_q = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.to_k = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.to_v = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.out = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)

    def forward(self,z_list,z_target,batch_size,num_imgs):

        # Query is what to find? or what is common? - comes from target
        q = self.to_q(z_target)


class ChannelWiseSetTransformer(nn.Module):
    """
    For each channel independently:
    Apply a full Set Transformer (attention across the 5 images)

    This is the most expressive version —
    learns complex cross-image relationships per channel
    """

    def __init__(self, n_channels=4, n_images=5, d_model=16):
        super().__init__()
        self.n_channels = n_channels

        # Shared transformer across channels (weight sharing)
        # OR separate transformer per channel (more expressive)
        self.transformers = nn.ModuleList([
            ChannelTransformer(n_images=n_images, d_model=d_model)
            for _ in range(n_channels)
        ])

    def forward(self, z_stack):
        # z_stack: [B, 5, 4, H, W]
        B, N, C, H, W = z_stack.shape

        output_channels = []

        for c in range(self.n_channels):
            channel_stack = z_stack[:, :, c, :, :]  # [B, 5, H, W]
            aggregated = self.transformers[c](channel_stack)  # [B, H, W]
            output_channels.append(aggregated)

        return torch.stack(output_channels, dim=1)  # [B, 4, H, W]


class ChannelTransformer(nn.Module):
    """
    Treats the 5 images as a SET at each spatial location
    Applies self-attention across the set dimension
    """

    def __init__(self, n_images=5, d_model=16):
        super().__init__()

        # Project each image's scalar value to d_model dimensions
        self.input_proj = nn.Conv2d(n_images, d_model, 1)

        # Self attention across the set
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            batch_first=True
        )

        # Project back to single aggregated channel
        self.output_proj = nn.Conv2d(d_model, 1, 1)

        # Learned query — "what does common structure look like?"
        self.query = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, channel_stack):
        # channel_stack: [B, 5, H, W]
        B, N, H, W = channel_stack.shape

        # Project 5 images to d_model
        x = self.input_proj(channel_stack)  # [B, d_model, H, W]

        # Reshape for attention: treat spatial locations as sequence
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, d_model]

        # Cross-attention: query is learned, keys/values from input
        q = self.query.expand(B, -1, -1)  # [B, 1, d_model]
        out, weights = self.attn(q, x, x)  # [B, 1, d_model]

        # Reshape back to spatial
        out = out.transpose(1, 2).view(B, -1, H, W)  # [B, d_model, H, W]

        return self.output_proj(out).squeeze(1)  # [B, H, W]

