# import numpy as np
# import torch
# from torch import nn
# from diffusers import AutoencoderKL
# from torch.cuda import device
# from tqdm import tqdm
# from torchmetrics.image import StructuralSimilarityIndexMeasure
# #from VAE_Perceptual_Loss import VGGPerceptualLoss
# import cv2
# from visualizations import *
# import wandb





# # class AttSets(nn.Module):
# #     """
# #        From AttSets paper — attentive aggregation of feature set        - https://arxiv.org/pdf/1808.00758
# #        Works with arbitrary N input features
# #        Permutation invariant (order of images doesn't matter)
# #        """
# #     def __init__(self,latent_channels = 4):
# #         super().__init__()
# #         self.score_net = nn.Sequential(
# #             nn.Conv2d(in_channels=latent_channels, out_channels=8, kernel_size=3,padding=1),
# #             nn.LeakyReLU(),
# #             #nn.GroupNorm(num_groups=4, num_channels=8),
# #             nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size=3,padding=1),
# #             nn.LeakyReLU(),
# #             nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=3,padding=1),
# #             nn.LeakyReLU(),
# #             nn.Conv2d(in_channels = 32, out_channels = 8, kernel_size=3,padding=1),
# #             nn.LeakyReLU(),
# #             nn.Conv2d(in_channels = 8, out_channels = 4, kernel_size=3,padding=1),
# #         )
# #         self.score_net = self.score_net .to(torch.device("cuda"))

# #     def forward(self,input_latents, sample = None, permute = None, num_imgs=5,batch_size=4):

# #         '''Method-1
# #         This method weights individual images separately (sigmoid activations) and multiplies with its own input latent representations before going through the aggregation
# #          Finally for aggregation it sums it along the first dimension in (B,N,C,H,W) i.e along the number of images in each batch.'''

# #         scores = self.score_net(input_latents)  # Size = (B*N,C,H,W)

# #         # print('The shape of each score',scores.shape)

# #         batch_latents = input_latents.view(batch_size,num_imgs,*(input_latents.shape[1:]))
# #         # print('The shape of the batch latents',batch_latents.shape)
        
# #         batch_scores = scores.view(batch_size,num_imgs, *(scores.shape[1:]))
# #         # print('The shape of batch scores', batch_scores.shape)
# #         weights = torch.softmax(batch_scores,dim=1)
# #         # print('The shape of the weights ', weights.shape)
# #         # print('Requires grad status of weights', weights.requires_grad)

# #         output = (weights *batch_latents).sum(dim=1)
# #         # print('The shape of output', output.shape)
# #         # print('Requires grad status of output',output.requires_grad)

# #         return output,weights

# class ChannelWiseAggregator(nn.Module):

#     def __init__(self,num_channels,num_images):

#         super().__init__()
#         self.n_channels = num_channels
#         self.n_images = num_images

#         self.channel_agg_net = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(in_channels=self.n_images,out_channels=16,kernel_size=3,padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(16, 32, kernel_size=1),
#                 nn.ReLU(),
#                 # nn.Conv2d(32,num_images, kernel_size=1),
#                 nn.Softmax(dim=1)
#             )
#             for _ in range(self.n_channels)
#         ])

#     def forward(self,input_latents,num_imgs,batch_size):

#         output_channels = []
#         batch_latents = input_latents.view(batch_size, num_imgs, *(input_latents.shape[1:]))
#         for c in range(self.n_channels):

#             channel_stack = batch_latents[:,:,c,:,:] # Selecting only one channel at a time.
#             print('The shape of channel', channel_stack.shape)

#             attn = self.channel_agg_net[c](channel_stack)
#             print('The shape of the output of the model (attention scores)', attn.shape)

#             aggregated = (channel_stack * attn)
#             print('Multiply it with the original input ', aggregated.shape)
#             aggregated = aggregated.sum(dim=1)
#             print('The shape of the final aggregated summing across all channels', aggregated.shape)

#             output_channels.append(aggregated)

#         z_common = torch.stack(output_channels, dim=1)
#         print('The final z_common', z_common.shape)
#         return z_common
