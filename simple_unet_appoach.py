from diffusers import UNet2DModel
import torch.nn as nn

class ResnetUNetAutoencoder(nn.Module):

    def __init__(self,latent_dimension=1024,pretrained=True):
        super().__init__()
        self.unet_model = UNet2DModel(latent_dimension=latent_dimension,pretrained=pretrained)



    def make_decoder_block(self, in_channels, out_channels):
        """
        Decoder block: Conv → BN → ReLU → Conv → BN → ReLU
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def encode(self,x):
        x0 = self.encoder_conv1(x)
        skip0 = x0

        x1 = self.encoder_pool(x0)

        skip1 = self.encoder_layer1(x1)

        skip2 = self.encoder_layer2(skip1)

        skip3 = self.encoder_layer3(skip2)

        x = self.encoder_layer4(skip3)

        z=self.bottleneck_flatten(x)
        z = self.bottleneck_fc(z)

        return z, [skip0,skip1, skip2, skip3]

    def decode(self,z, skips):

        x = self.decoder_fc(z)
        x = x.view(-1,512,8,8)

        x = self.upsample_layer1(x)
        if skips is not None and len(skips) >= 4:
            x = torch.cat((x, skips[3]), dim=1)
        x= self.dec_conv1(x)

        x = self.upsample_layer2(x)
        if skips is not None and len(skips) >= 3:
            x = torch.cat((x, skips[2]), dim=1)
        x = self.dec_conv2(x)

        x = self.upsample_layer3(x)
        if skips is not None and len(skips) >= 2:
            x = torch.cat((x, skips[1]), dim=1)
        x = self.dec_conv3(x)

        x = self.upsample_layer4(x)
        if skips is not None and len(skips) >= 1:
            x = torch.cat((x, skips[0]), dim=1)
        x= self.dec_conv4(x)

        x= self.upsample_layer5(x)
        x =self.dec_conv5(x)

        x = self.final_conv(x)
        return x

    def forward(self, x):
        z, skips = self.encode(x)
        z = z.reshape(2,5,*z.shape[1:])                     #Averaging in the latent space
        z_averaged = z.mean(dim=1)
        #x_recon = self.decode(z, skips)                    #Applicable only for single images
        x_recon = self.decode(z_averaged, None)
        return x_recon, z

    def aggregate_and_reconstruct(self,images):

        latents = []

        for img in images:
            z,_ = self.encode(img.unsqueeze(0))
            latents.append(z)

        z_avg = torch.stack(latents).mean(dim=0)

        common = self.decode(z_avg, skips=None)
        return common