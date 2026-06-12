from diffusers import UNet2DConditionModel
import torchvision.models as models
import torch.nn as nn
import os
import torch
import torchvision
from torchvision.transforms import v2
from diffusers import AutoencoderKL, VQModel
from torch.utils.data import DataLoader,Dataset
from torchinfo import summary
from tqdm import tqdm
import matplotlib.pyplot as plt
from input_pipeline import *
import cv2
from torchmetrics import StructuralSimilarityIndexMeasure
import wandb

class UNetDecoderBlock(nn.Module):

    def __init__(self,in_channels, skip_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Convolutions after concatenation
        self.conv =  self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        # Upsample
        x = self.upsample(x)

        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)

        # Convolutions
        x = self.conv(x)

        return x


class ResnetUNetAutoencoder(nn.Module):

    def __init__(self,latent_dimension=1024,pretrained=True):
        super().__init__()

        self.latent_dim = latent_dimension
        resnet = models.resnet18(pretrained=pretrained)

        self.encoder_conv1 = nn.Sequential(resnet.conv1,resnet.bn1,resnet.relu) # [3, 256, 256] → [64, 128, 128]
        self.encoder_pool = resnet.maxpool   # [64, 128, 128] → [64, 64, 64]

        self.encoder_layer1 = resnet.layer1
        self.encoder_layer2 = resnet.layer2  # [64, 64, 64] → [128, 32, 32]
        self.encoder_layer3 = resnet.layer3  # [128, 32, 32] → [256, 16, 16]
        #self.encoder_layer4 = resnet.layer4  # [256, 16, 16] → [512, 8, 8]

        #Latent Vector
        #self.bottleneck_flatten = nn.Flatten()
        #self.bottleneck_fc = nn.Linear(512*8*8, latent_dimension)

        #self.decoder_fc = nn.Linear(latent_dimension, 512 * 8 * 8)

        #self.upsample_layer1 = nn.ConvTranspose2d(512,256,kernel_size=4, stride=2, padding=1)
        #self.dec_conv1 = self.make_decoder_block(256+256, 256)                     # With skips
        self.dec_conv1 = self.make_decoder_block(256, 256)        # Without skips

        self.upsample_layer2 = nn.ConvTranspose2d(256,128,kernel_size=4, stride=2, padding=1)
        #self.dec_conv2 = self.make_decoder_block(128+128,128)                      # With skips
        self.dec_conv2 = self.make_decoder_block(128 , 128)    # Without skips


        self.upsample_layer3 = nn.ConvTranspose2d(128,64,kernel_size=4, stride=2, padding=1)
        #self.dec_conv3 = self.make_decoder_block(64+64,64)                         # With skips
        self.dec_conv3 = self.make_decoder_block(64 , 64)      # Without skips

        self.upsample_layer4 = nn.ConvTranspose2d(64,64,kernel_size=4, stride=2, padding=1)
        #self.dec_conv4 = self.make_decoder_block(64+64,64)                         # With skips
        self.dec_conv4 = self.make_decoder_block(64 , 64)       # Without skips

        self.upsample_layer5 = nn.ConvTranspose2d(64,32,kernel_size=4, stride=2, padding=1)
        self.dec_conv5 = self.make_decoder_block(32,32)            #Same for both with and without skips

        self.final_conv = nn.Sequential(
            #nn.Conv2d(32, 16, kernel_size=3, padding=1),
            #nn.BatchNorm2d(16),
            #nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
        )

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


class FineTuner(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_data(self, image_directory, gt_directory=None, test=False):

        # Image Transformations
        train_transformations = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),  # Scale = true already scales the images between 0,1.
            v2.Resize((256, 256)),
            v2.ToTensor(),
            # v2.Normalize(mean=(0.5,), std= (1.0,)),
        ])

        test_transformations = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((256, 256)),
            v2.ToTensor(),
            # v2.Normalize(mean=(0.5,), std= (1.0,)),
        ])

        # image_groups = create_image_groups(image_directory)

        # train_dataset = SimpleCustomBatch(image_groups=image_groups,gt_directory = gt_directory,transform=train_transformations)
        # print("The length of train dataset",len(train_dataset))
        train_dataset, test_dataset = input_pipeline_residual_grayscale(single_image_reconstruction=False)

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=False, num_workers=2,
                                      pin_memory=True)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, num_workers=2,
                                     pin_memory=True)

        return train_dataloader,test_dataloader

    def finetune(self, model, train_dataloader, test_dataloader, lr=1e-4, epochs=150, single_image_recon=False):

        print('Running on ', self.device)
        best_loss = float('inf')
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        model.to(self.device)
        run = wandb.init(
            entity='veenasridhar96-university-of-stuttgart',
            project='Topology Optimization',
            config={
                "learning_rate": 0.1e-4,
                "architecture": "Resnet Encoder & Decoder",
                "dataset": "TO",
                "epochs": 100,
            },
        )

        #criterion = nn.BCEWithLogitsLoss()
        #criterion = nn.MSELoss()
        criterion = nn.CrossEntropyLoss()

        if single_image_recon:
            for epoch in range(epochs):
                model.train()
                train_loss = 0.0

                progress_bar = tqdm(train_dataloader, desc=f'Epochs {epoch + 1}/{epochs}')
                for batch_idx, images in enumerate(progress_bar):

                    images = images.to(self.device)
                    recon_logits, z = model(images)
                    loss = criterion(recon_logits, images)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                    progress_bar.set_postfix({'loss': loss.item(), })
                    if batch_idx <= 5:
                        self.visualize_images(batch_idx, epoch, recon_logits, images, test=False,
                                              save_path='./output_images/')

                scheduler.step()

                test_loss = 0.0
                if test_dataloader:
                    model.eval()
                    with torch.no_grad():
                        for batch_idx, images in enumerate(test_dataloader):
                            images = images.to(self.device)
                            recon_logits, z = model(images)
                            loss = criterion(recon_logits, images)
                            self.visualize_images(batch_idx, epoch, recon_logits, images, True,
                                                  './output_images/test_images')
                            test_loss += loss.item()

                print(
                    f"Epoch {epoch + 1}: Train Loss: {train_loss / len(train_dataloader):.4f}, Val Loss: {test_loss / len(test_dataloader):.4f}")

                if test_loss < best_loss:
                    best_loss = test_loss
                    # model.save_pretrained('./runs/')
                    print(f"Saved best model with val loss: {test_loss:.4f}")

                else:
                    print(f"Epoch {epoch + 1}: Train Loss: {train_loss / len(train_dataloader):.4f}")
        else:
            for epoch in range(epochs):
                model.train()
                train_loss = 0.0

                progress_bar = tqdm(train_dataloader, desc=f'Epochs {epoch + 1}/{epochs}')
                for batch_idx, (images, labels) in enumerate(progress_bar):

                    batch_size, num_images = images.shape[0], images.shape[1]
                    print('The shape of images', images.shape)
                    combinedimages = images.reshape(-1, *images.shape[2:]).to(self.device)

                    print('The shape of combined_images', combinedimages.shape)

                    recon_logits, z = model(combinedimages)
                    print('The shape of reconstruction logits ', recon_logits.shape)

                    labels = labels.to(self.device)
                    loss = criterion(recon_logits, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                    progress_bar.set_postfix({'loss': loss.item(), })
                    if batch_idx <= 5:
                        self.visualize_images(batch_idx, epoch, recon_logits, images, labels, test=False,
                                              save_path='./output_images/')

                scheduler.step()

                test_loss = 0.0
                if test_dataloader:
                    model.eval()
                    with torch.no_grad():
                        for batch_idx, (images, labels) in enumerate(test_dataloader):
                            batch_size, num_imgs = images.shape[0], images.shape[1]

                            combinedimages = images.reshape(-1, *images.shape[2:]).to(self.device)

                            recon_logits, z = model(combinedimages)
                            labels = labels.to(self.device)
                            loss = criterion(recon_logits, labels)
                            self.visualize_images(batch_idx, epoch, recon_logits, images, labels, test=True,
                                                  save_path='./output_images/test_images')
                            test_loss += loss.item()

                print(
                    f"Epoch {epoch + 1}: Train Loss: {train_loss / len(train_dataloader):.4f}, Val Loss: {test_loss / len(test_dataloader):.4f}")
                run.log(
                    {'train loss': train_loss / len(train_dataloader), 'test loss': test_loss / len(test_dataloader)})
                if test_loss < best_loss:
                    best_loss = test_loss
                    # model.save_pretrained('./runs/')
                    print(f"Saved best model with val loss: {test_loss:.4f}")

                else:
                    print(f"Epoch {epoch + 1}: Train Loss: {train_loss / len(train_dataloader):.4f}")

    def visualize_images(self, batch, epoch, reconstructed_image, original, gt_image, save_path, test=False):

        reconstructed_image = torch.sigmoid(reconstructed_image)
        reconstructed_image = torch.clamp(reconstructed_image, 0, 1)
        ssim = StructuralSimilarityIndexMeasure().to(self.device)
        # print(original.shape)
        for i in range(len(reconstructed_image)):
            if test:
                plt.imsave(os.path.join(save_path, f'output_test_images_{epoch}_{batch}_{i}.png'),
                           reconstructed_image[i].permute(1, 2, 0).detach().cpu().numpy())
            else:
                plt.imsave(os.path.join(save_path, f'output_images_{epoch}_{batch}_{i}.png'),
                           reconstructed_image[i].permute(1, 2, 0).detach().cpu().numpy())

        # Create figure
        for i in range(2):
            fig = plt.figure(figsize=(16, 8))
            grid_spec = fig.add_gridspec(2, 5, hspace=0.1, wspace=0.1)

            for j in range(5):
                ax = fig.add_subplot(grid_spec[0, j])
                orig_img = original[i][j].permute(1, 2, 0).detach().cpu().numpy()
                ax.imshow(orig_img)
                ax.axis('off')
            ssim_value = ssim(reconstructed_image, gt_image)
            ax_center = fig.add_subplot(grid_spec[1, 2])
            recon_images = reconstructed_image[i].permute(1, 2, 0).detach().cpu().numpy()
            ax_center.imshow(recon_images)
            ax_center.axis('off')
            ax_center.set_title(f'SSIM: {ssim_value:.2f}')

            ax_left = fig.add_subplot(grid_spec[1, 1])
            print(type(gt_image), gt_image.shape)
            gt_image_visual = gt_image[i].permute(1, 2, 0).detach().cpu().numpy()
            ax_left.imshow(gt_image_visual)
            ax_left.axis('off')

            for l in [0, 3, 4]:
                ax = fig.add_subplot(grid_spec[1, l])
                ax.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'reconstructed_test_images_{epoch}_{batch}_{i}.png'), dpi=150)
            plt.close()

            if test and epoch == 99:
                resized_image = cv2.resize(reconstructed_image[i].permute(1, 2, 0).detach().cpu().numpy(), (1024, 512),
                                           interpolation=cv2.INTER_NEAREST)
                plt.imshow(resized_image)

    '''def visualize_images(self,batch_index,epoch,reconstructed_image, original,test=False,save_path='./output_images/'):
        reconstructed_image = torch.sigmoid(reconstructed_image)
        #reconstructed_image = torch.clamp(reconstructed_image,0,1)
        #ssim = StructuralSimilarityIndexMeasure().to(device)
        for i in range(len(reconstructed_image)):
            if test:
                plt.imsave(os.path.join(save_path,f'output_test_images_{epoch}_{batch_index}_{i}.png'),reconstructed_image[i].permute(1,2,0).detach().cpu().numpy())
            else:
                plt.imsave(os.path.join(save_path,f'output_images_{epoch}_{batch_index}_{i}.png'),reconstructed_image[i].permute(1,2,0).detach().cpu().numpy())

        fig = plt.figure(figsize=(16,8))
        grid_spec = fig.add_gridspec(2,5,hspace=0.1,wspace=0.1)
        for i in range(2):
            ax = fig.add_subplot(grid_spec[0,i])
            orig_img = original[i].permute(1, 2, 0).detach().cpu().numpy()
            ax.imshow(orig_img)
            ax.axis('off')

            ax = fig.add_subplot(grid_spec[1,i])
            recon_img = reconstructed_image[i].permute(1, 2, 0).detach().cpu().numpy()
            ax.imshow(recon_img)
            ax.axis('off')

            for l in [0, 3, 4]:
                ax = fig.add_subplot(grid_spec[1, l])
                ax.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_path,f'reconstructed_test_images_{epoch}_{batch_index}.png'), dpi=150)
        plt.close()'''













