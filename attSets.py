import numpy as np
import torch
from torch import nn
from diffusers import AutoencoderKL
from torch.cuda import device
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure
#from VAE_Perceptual_Loss import VGGPerceptualLoss
import cv2
from visualizations import *
import wandb

class AttSets(nn.Module):
    """
       From AttSets paper — attentive aggregation of feature set        - https://arxiv.org/pdf/1808.00758
       Works with arbitrary N input features
       Permutation invariant (order of images doesn't matter)
       """
    def __init__(self,latent_channels =4):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Conv2d(in_channels=latent_channels, out_channels=16, kernel_size=3,padding=1),
            nn.ReLU(),
            # nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,padding=1),
            # nn.ReLU(),
            nn.Conv2d(in_channels = 16, out_channels = 4, kernel_size=3,padding=1),
            #nn.Sigmoid()
        )
        self.score_net = self.score_net .to(torch.device("cuda"))

    def forward(self,input_latents,num_imgs=5,batch_size=4):

        '''Method-1
        This method weights individual images separately (sigmoid activations) and multiplies with its own input latent representations before going through the aggregation
         Finally for aggregation it sums it along the first dimension in (B,N,C,H,W) i.e along the number of images in each batch.'''

        print('The shape of the input latents',input_latents.dtype)

        scores = self.score_net(input_latents)  # Size = (B*N,C,H,W)

        print('The shape of each score',scores.shape)

        batch_latents = input_latents.view(batch_size,num_imgs,*(input_latents.shape[1:]))
        batch_scores = scores.view(batch_size,num_imgs, *(scores.shape[1:]))
        print('The shape of batch scores', batch_scores.shape)

        weights = torch.softmax(batch_scores,dim=1)
        print('The shape of the weights ', weights.shape)
        print('Requires grad status of weights', weights.requires_grad)

        output = (weights *batch_latents).sum(dim=1)
        print('The shape of output', output.shape)
        print('Requires grad status of output',output.requires_grad)

        return output

class ChannelWiseAggregator(nn.Module):

    def __init__(self,num_channels,num_images):

        super().__init__()
        self.n_channels = num_channels
        self.n_images = num_images

        self.channel_agg_net = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=self.n_images,out_channels=16,kernel_size=3,padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(32,num_images, kernel_size=1),
                nn.Softmax(dim=1)
            )
            for _ in range(self.n_channels)
        ])

    def forward(self,input_latents,num_imgs,batch_size):

        output_channels = []
        batch_latents = input_latents.view(batch_size, num_imgs, *(input_latents.shape[1:]))
        for c in range(self.n_channels):

            channel_stack = batch_latents[:,:,c,:,:] # Selecting only one channel at a time.
            print('The shape of channel', channel_stack.shape)

            attn = self.channel_agg_net[c](channel_stack)
            print('The shape of the output of the model (attention scores)', attn.shape)

            aggregated = (channel_stack * attn)
            print('Multiply it with the original input ', aggregated.shape)
            aggregated = aggregated.sum(dim=1)
            print('The shape of the final aggregated summing across all channels', aggregated.shape)

            output_channels.append(aggregated)

        z_common = torch.stack(output_channels, dim=1)
        print('The final z_common', z_common.shape)
        return z_common

class SetAggregation(nn.Module):
    def __init__(self,model_name=None):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')
        self.vae = AutoencoderKL.from_pretrained('./finetuned_vae/individually_trained_models',use_safetensors=True)
        self.vae.to(self.device)
        self.aggregation_module = AttSets(latent_channels=4)
        #self.aggregation_module = ChannelWiseAggregator(num_channels=4, num_images=5 )
        self.aggregation_module.to(self.device)

        # Keeping the VAE parameters frozen and training only the aggregation module
        for p in self.vae.encoder.parameters():
            p.requires_grad = False

        trainable_decoder_layers = [
            self.vae.decoder.up_blocks[-1],
            self.vae.decoder.up_blocks[-2],
            self.vae.decoder.conv_out,
        ]
        for layer in trainable_decoder_layers:
            for param in layer.parameters():
                param.requires_grad = True

        # ── Step 4: Verify what is trainable ────────────────────────
        decoder_params = [
            p for p in self.vae.decoder.parameters()
            if p.requires_grad
        ]

        print('The number of aggregation parameters trained are', sum(p.numel() for p in self.aggregation_module.parameters()))
        self.agg_optimizer = torch.optim.Adam(self.aggregation_module.parameters(), lr=3e-3)
        #self.decoder_optimizer = torch.optim.Adam(self.vae.decoder.parameters(), lr=3e-3)
        '''self.optimizer = torch.optim.AdamW(
            [
                {
                    'params': self.aggregation_module.parameters(),
                    'lr': 1e-3,  # aggregator: learn fast
                    'weight_decay': 1e-4,
                    'name': 'aggregator'
                },
                {
                    'params': decoder_params,
                    'lr': 1e-5,  # decoder: fine-tune slow
                    'weight_decay': 1e-6,  # less decay — pretrained
                    'name': 'decoder'
                },
            ]
        )'''

    def sharpness_loss(self,reconstructed):

        t = [(1.0, 0.0, 0.0),  # full red
             (0.5, 0.5, 0.5),  # full grey
             (1.0, 1.0, 1.0), ]  # full white
        t = torch.tensor(t,dtype= torch.float,device=self.device)  # The shape of the tensor is (T,C)
        #print(t.shape)
        reconstructed = reconstructed.permute(0,2,3,1).unsqueeze(-2)            # This reshapes the tensor to be (B,H,W,1,C)
        print(reconstructed.shape)

        dist = ((reconstructed - t) ** 2).sum(dim=-1)  # This first subtracts (H,W,1,C) - (T,C) and the result is summed along dim(-1) thus resulting in (H,W,T)
        '''
        [[[0.9,0,0],[0,0,0],[0,0,0],[0,0,0]],
        [[[0.9,0,0],[0,0,0],[0,0,0],[0,0,0]],
        [[0.5,0,0],[0.5,0,0],[0,0,0],[0,0,0]],
        [[0,0,0],[0,0,0],[0.9,0.9,0.9],[0,0,0]],
        [[0,0,0],[0,0,0],[0.9,0.9,0.9],[0,0,0]]]

        This is a 4x4 image with 3 channels. Calculating distance would mean subtracting each pixel [0.9,0,0] with the target values (red,grey and white). This why unsqueeze(-2) is important to match dimensions for subtraction.
        Sum along dimension -1  will ensure that the red, grey and white each have 1 value instream of 3 channels. The shape of dist is [B,H,W,T] - T is the distance to the each of the target color '''
        minimum_dist = dist.min(-1).values.mean()
        ''' min(-1) selects the  minimum distance to the target value and is averaged across all pixels of all images. This is equal to calculating for each images individually and then averaging.
        '''
        return minimum_dist

    def forward(self,train_loader,val_loader,test_loader, fold, epochs=1,lr=3e-3 ):

        # Start a new wandb run to track this script.
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="veenasridhar96-university-of-stuttgart",
            # Set the wandb project where this run will be logged.
            project='Topology Optimization',
            # group=self.GROUP_NAME,
            name='Sharpness Loss Tracking',
            # job_type= f'run- {fold}',
            # Track hyperparameters and run metadata.
            config={
                "learning_rate": lr,
                #'kl_weight': kl_weight,
                'SSIM Weight': '0.00001',
                'latent_technique': 'Individual Reconstruction',
                'loss': ' KL + Perceptual Loss ',
                "architecture": "VAE",
                "dataset": "TO",
                "epochs": epochs,
                "fold": fold,
            }, )

        mse_loss = torch.nn.MSELoss()
        bce_loss = torch.nn.BCEWithLogitsLoss()
        perceptual_loss = VGGPerceptualLoss().to(self.device)
        ssim = StructuralSimilarityIndexMeasure().to(self.device)

        for epoch in range(epochs):

            train_progress_bar = tqdm(train_loader,desc=f'Epochs = {epoch}/{epochs}')
            train_total_loss = 0.0
            train_mse_loss= 0.0
            train_precep_loss = 0.0
            index =0

            for batch, (images,gt) in enumerate(train_progress_bar):
                images = images.to(self.device)
                target = gt.to(self.device)
                binary_mask = (target !=0.5).float()

                batch_size, num_imgs = images.shape[0], images.shape[1]
                combinedimages = images.reshape(-1, *images.shape[2:]).to(self.device)

                with torch.no_grad():
                    posterior = self.vae.encode(combinedimages).latent_dist
                    z_individual = posterior.sample()
                    posterior_mean_detached = posterior.mean.detach()

                    target_posterior = self.vae.encode(target).latent_dist
                    z_target = target_posterior.sample()

                z_common = self.aggregation_module(posterior_mean_detached, num_imgs,batch_size)
                #print('Requires grad status ',z_common.requires_grad)
                #print('The shape of the common aggregator output',z_common.shape)

                with torch.no_grad():
                    original_reconstruction = self.vae.decode(z_individual).sample
                    original_reconstruction = original_reconstruction.view(batch_size, num_imgs, *original_reconstruction.shape[1:])
                    target_reconstruction = self.vae.decode(z_target).sample

                    print('The shape of z_individual',z_individual.shape)
                    channelwise_recon = {}
                    common_recon_dict = {}

                    ## Channel wise Reconstruction
                    '''for channel in range(4): # For now num_channels =4
                        z_isolated = torch.zeros_like(z_individual)
                        z_isolated[:,channel] = z_individual[:,channel,:,:]
                        print('The shape of z isolated',z_isolated.shape)
                        channelwise_reconstruction = self.vae.decode(z_isolated).sample
                        channelwise_reconstruction = channelwise_reconstruction.reshape(batch_size, num_imgs, *channelwise_reconstruction.shape[1:])
                        print('Input channel wise reconstrcution',channelwise_reconstruction.shape)
                        channelwise_recon[channel] = channelwise_reconstruction

                        common_channels = torch.zeros_like(z_common)
                        common_channels[:,channel] = z_common[:,channel,:,:]
                        common_reconstruction = self.vae.decode(common_channels).sample
                        print('Common Reconstruction',common_reconstruction.shape)
                        common_recon_dict[channel] = common_reconstruction'''



                reconstructed = self.vae.decode(z_common).sample            # / 0.18215 -- check if it makes any difference
                #visualize_agg_feature_map(original_reconstruction, z_common, reconstructed)
                if epoch % 5 == 0:
                    visulaize_images(True, batch, epoch, reconstructed, images, target_reconstruction,
                                     'output_files/output_images_1', index, False)
                    index+=2
                #if epoch % 10 ==0 and batch < 5:
                    #z_individual = z_individual.reshape(batch_size,num_imgs,*z_individual.shape[1:])
                    #visualize_temp(batch,epoch,original_reconstruction,target_reconstruction,reconstructed,channelwise_recon,z_common,'./output_images_1')
                    #channelwise_visualisation(batch,epoch, original_reconstruction,target_reconstruction,reconstructed,channelwise_recon,common_recon_dict,self.vae,'./output_images_1')


                pixelwise_recon_loss = mse_loss(reconstructed, target)
                #binary_loss = torch.tensor(0.0)
                #perceptual_recon_loss = torch.tensor(0.0)
                #perceptual_recon_loss = self.sharpness_loss(reconstructed)
                '''if binary_mask.sum() >0:
                    pred_binary = reconstructed[binary_mask.bool()]
                    target_binary = target[binary_mask.bool()]
                    binary_loss = bce_loss(pred_binary, target_binary)'''

                perceptual_recon_loss = perceptual_loss(reconstructed,target)
                #loss =  pixelwise_recon_loss

                #+perceptual_recon_loss +binary_loss)

                #self.optimizer.zero_grad()
                #loss.backward()
                #self.optimizer.step()

                train_mse_loss = pixelwise_recon_loss
                self.agg_optimizer.zero_grad()
                train_mse_loss.backward()
                self.agg_optimizer.step()

                train_total_loss += train_mse_loss.item()
                train_mse_loss += pixelwise_recon_loss.item()
                train_precep_loss += perceptual_recon_loss.item()


                train_progress_bar.set_postfix({'loss': train_mse_loss.item(), 'MSE': pixelwise_recon_loss.item(), 'Perceptual Loss': perceptual_recon_loss.item()})

            # val_progress_bar = tqdm(val_loader,desc=f'Epochs = {epoch}/{epochs}')
            #
            # total_val_loss = 0.0
            # total_val_mse_loss = 0.0
            # total_val_precep_loss = 0.0
            #
            # for batch, (images,gt) in enumerate(val_progress_bar):
            #     self.vae.eval()
            #     self.aggregation_module.eval()
            #     with torch.no_grad():
            #         images = images.to(self.device)
            #         target = gt.to(self.device)
            #
            #         batch_size, num_imgs = images.shape[0], images.shape[1]
            #         combinedimages = images.reshape(-1, *images.shape[2:]).to(self.device)
            #
            #         posterior = self.vae.encode(combinedimages).latent_dist
            #         posterior_mean = posterior.mean
            #
            #         target_posterior = self.vae.encode(target).latent_dist
            #         z_target = target_posterior.sample()
            #
            #         z_individual = posterior.sample()
            #         z_common = self.aggregation_module(posterior_mean,num_imgs,batch_size)
            #
            #         original_reconstruction = self.vae.decode(z_individual).sample
            #         target_reconstruction = self.vae.decode(z_target).sample
            #
            #         original_reconstruction = original_reconstruction.view(batch_size, num_imgs, *original_reconstruction.shape[1:])
            #         reconstructed = self.vae.decode(z_common).sample  # / 0.18215 -- check if it makes any difference
            #         #visualize_agg_feature_map(original_reconstruction, z_common,reconstructed)
            #         if epoch % 5 == 0 :
            #             visulaize_images(True, batch, epoch, reconstructed, images, target_reconstruction, './output_images_1/val_images', False)
            #
            #         '''if epoch % 10 == 0:
            #             z_individual = z_individual.reshape(batch_size, num_imgs, *z_individual.shape[1:])
            #             visualize_temp(batch, epoch, original_reconstruction, target_reconstruction, reconstructed, z_individual, z_common, './output_images_1/val_images')'''
            #
            #         pixelwise_recon_loss = mse_loss(z_common, z_target)
            #         perceptual_recon_loss = self.sharpness_loss(reconstructed)
            #         #perceptual_recon_loss = perceptual_loss(reconstructed, target)
            #         val_loss = pixelwise_recon_loss + perceptual_recon_loss
            #
            #         total_val_loss +=val_loss.item()
            #         total_val_mse_loss += pixelwise_recon_loss.item()
            #         total_val_precep_loss += perceptual_recon_loss.item()
            #
            #         val_progress_bar.set_postfix({'Val Loss': val_loss.item(), 'Val MSE Loss':pixelwise_recon_loss.item(), 'Val Perceptual Loss':perceptual_recon_loss.item()})

            test_progress_bar = tqdm(test_loader, desc=f'Epochs = {epoch}/{epochs}')

            total_test_loss = 0.0
            total_test_mse_loss = 0.0
            total_test_precep_loss = 0.0

            for batch, (images, gt) in enumerate(test_progress_bar):
                self.vae.eval()
                self.aggregation_module.eval()
                with torch.no_grad():
                    images = images.to(self.device)
                    target = gt.to(self.device)

                    batch_size, num_imgs = images.shape[0], images.shape[1]
                    combinedimages = images.reshape(-1, *images.shape[2:]).to(self.device)

                    posterior = self.vae.encode(combinedimages).latent_dist
                    posterior_mean = posterior.mean
                    target_posterior = self.vae.encode(target).latent_dist

                    z_individual = posterior.sample()
                    z_common = self.aggregation_module(posterior_mean,num_imgs,batch_size)
                    z_target = target_posterior.sample()


                    original_reconstruction = self.vae.decode(z_individual).sample
                    original_reconstruction = original_reconstruction.view(batch_size, num_imgs,*original_reconstruction.shape[1:])
                    reconstructed = self.vae.decode(z_common).sample  # / 0.18215 -- check if it makes any difference
                    target_reconstruction = self.vae.decode(z_target).sample

                    if epoch % 5 == 0 :
                        visulaize_images(True, batch, epoch, reconstructed, images, target_reconstruction,
                                         'output_files/output_images_1/test_images', index, True)
                        index +=2

                    '''if epoch % 10 == 0:
                        z_individual = z_individual.reshape(batch_size, num_imgs, *z_individual.shape[1:])
                        visualize_temp(batch, epoch, original_reconstruction, target_reconstruction, reconstructed, z_individual, z_common, './output_images_1/test_images')'''


                    pixelwise_recon_loss = mse_loss(z_common, z_target)
                    perceptual_recon_loss = self.sharpness_loss(reconstructed)
                    #perceptual_recon_loss = perceptual_loss(reconstructed, target)
                    test_loss = pixelwise_recon_loss + perceptual_recon_loss

                    total_test_loss += test_loss.item()
                    total_test_mse_loss += pixelwise_recon_loss.item()
                    total_test_precep_loss += perceptual_recon_loss.item()

                    test_progress_bar.set_postfix(
                        {'Test Loss': test_loss.item(), 'Test MSE Loss': pixelwise_recon_loss.item(),
                         'Test Perceptual Loss': perceptual_recon_loss.item()})

            #print(f'Epoch: {epoch} \n Train Total Loss: {train_total_loss/len(train_loader)}, Val Total Loss : {total_val_loss/len(val_loader)}, Test Total Loss : {total_test_loss/len(test_loader)}')
            #run.log({'Train Total Loss': train_total_loss/len(train_loader), 'Val Total Loss': total_val_loss/len(val_loader), 'Test Total Loss': total_test_loss/len(test_loader)})






