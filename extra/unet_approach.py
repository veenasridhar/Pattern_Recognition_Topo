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
from convert_sdf_structures import *
from image_processing_approach import visualize_images
from input_pipeline import *
import cv2
import wandb

class UNetDecoderBlock(nn.Module):

    def __init__(self,in_channels, skip_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Convolutions after concatenation
        self.conv =  nn.Sequential(
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

class UNetEncoderBlock(nn.Module):

    def __init__(self,in_channels, skip_channels, out_channels):
        super().__init__()

        # Convolutions after concatenation
        self.conv =  nn.Sequential(
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

        self.encoder_conv1 = nn.Conv2d(5,64, kernel_size=7, stride=2, padding=3,bias=False) #Early fusion step with 5 channels
        #self.encoder_conv1 = nn.Sequential(resnet.conv1,resnet.bn1,resnet.relu) # [3, 256, 256] → [64, 128, 128]
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
            nn.Conv2d(32, 1, kernel_size=1),
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

        z = skip3
        #x = self.encoder_layer4(skip3)

        #z=self.bottleneck_flatten(x)
        #z = self.bottleneck_fc(z)

        return z, [None, None, None, None]
        #return z, [skip0,skip1, skip2, skip3]

    def decode(self,z, skips):

        x = z
        #x = self.decoder_fc(z)
        #x = x.view(-1,512,8,8)

        # x = self.upsample_layer1(x)
        # if skips is not None and len(skips) >= 4:
        #     x = torch.cat((x, skips[3]), dim=1)
        # x= self.dec_conv1(x)

        x = self.upsample_layer2(x)
        print('The shape of x is ', x.shape)
        # if skips is not None and len(skips) >= 3:
        #     x = torch.cat((x, skips[2]), dim=1)
        x = self.dec_conv2(x)

        x = self.upsample_layer3(x)
        # if skips is not None and len(skips) >= 2:
        #     x = torch.cat((x, skips[1]), dim=1)
        x = self.dec_conv3(x)

        x = self.upsample_layer4(x)
        # if skips is not None and len(skips) >= 1:
        #     x = torch.cat((x, skips[0]), dim=1)
        x= self.dec_conv4(x)

        x= self.upsample_layer5(x)
        x =self.dec_conv5(x)

        x = self.final_conv(x)
        return x

    def forward(self, x, batch_size,num_imgs):
        z, skips = self.encode(x)
        #print('The latent code shape ', z.shape)
        #z = z.reshape(batch_size,num_imgs,*z.shape[1:])                     #Averaging in the latent space
        #z_averaged = z.mean(dim=1)
        #print('The averaged latent code shape ', z_averaged.shape)
        x_recon = self.decode(z, skips)                    #Applicable only for single images
        #x_recon = self.decode(z_averaged, None)
        return x_recon, z

    def aggregate_and_reconstruct(self,images):

        latents = []

        for img in images:
            z,_ = self.encode(img.unsqueeze(0))
            latents.append(z)

        z_avg = torch.stack(latents).mean(dim=0)

        common = self.decode(z_avg, skips=None)
        return common


def train_unet(single_image_recon=False,input = 'sdf'):

    input = input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k_fold = KFold(n_splits=5, shuffle=True)
    batch_size = 4
    best_loss = float('inf')
    #train_dataset, test_dataset = input_pipeline_grayscale(single_image_reconstruction=False)
    train_dataset, test_dataset = input_pipeline_sdf(single_image_reconstruction=False)
    train_indices = np.arange(len(train_dataset))

    for fold, (train_index, val_index) in enumerate(k_fold.split(train_indices)):
        if fold > 0:
            break

        model = ResnetUNetAutoencoder()
        model = model.to(device)
        print(summary(model))
        print('Running on', device)

        train_subset = torch.utils.data.Subset(train_dataset, train_index)
        val_subset = torch.utils.data.Subset(train_dataset, val_index)

        # Transform the Dataset into DataLoader
        train_dataloader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=False, num_workers=2,
                                      pin_memory=True)
        val_dataloader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False, num_workers=2,
                                    pin_memory=True)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                                     pin_memory=True)

        lr = 0.0003
        epochs = 500

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer1,gamma= 0.95)

        weights = torch.tensor([0.25, 0.35, 0.5])
        crossentropy_loss = torch.nn.CrossEntropyLoss(weight=weights).to(device)
        #cl_dice = soft_dice_cldice()
        save_parent_dir = f'UNet Results'

        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="veenasridhar96-university-of-stuttgart",
            # Set the wandb project where this run will be logged.
            # project=self.PROJECT_NAME,
            # group=self.GROUP_NAME,
            name='Topology Optimization',

            # job_type= f'run- {fold}',
            # Track hyperparameters and run metadata.
            config={
                "architecture": "Resnet Encoder & Decoder",
                "learning_rate": lr,
                "epochs": epochs,
            }, )

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
                train_sdf_loss = 0.0
                train_batch = 0
                train_iou_dl = 0.0
                train_iou_img_prc = 0.0

                progress_bar = tqdm(train_dataloader, desc=f'Epochs {epoch + 1}/{epochs}')
                for batch_idx, (images, labels,img_prc_result) in enumerate(progress_bar):

                    batch_size, num_images = images.shape[0], images.shape[1]
                    images = images.to(device)

                    #print('The shape of images', images.shape)
                    #combinedimages = images.reshape(-1, *images.shape[2:]).to(device) # B*N,C,H,W
                    #print('The shape of combined_images', combinedimages.shape)

                    recon_logits, z = model(images,batch_size,num_images)
                    print('The shape of reconstruction logits ', recon_logits.shape)
                    print('The labels unique vales', labels.unique(), labels.shape)
                    labels = labels.to(device)
                    if input == 'images':
                        ce_loss = crossentropy_loss(recon_logits, labels.squeeze(1))
                        sdf_loss = torch.norm((recon_logits.squeeze(1) - labels) / labels)
                        loss = ce_loss + sdf_loss
                        #sdf_loss = F.l1_loss(sdf_pred, gt_sdf)
                    elif input == 'sdf':
                        sdf_loss = torch.norm((recon_logits.squeeze(1) - labels) / labels)
                        loss = sdf_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                    progress_bar.set_postfix({'loss': loss.item(), })

                    # if epoch>100:
                    #     for i in range(reconstructed_labels.shape[0]):
                    #         plt.imshow(reconstructed_labels[i].detach().cpu().numpy(), cmap='gray',vmin=0,vmax=2)
                    #         plt.show()

                    if not os.path.exists(os.path.join(save_parent_dir, 'train', 'Epoch ' + str(epoch))):
                          os.mkdir(os.path.join(save_parent_dir, 'train', 'Epoch ' + str(epoch)))
                    if not os.path.exists(os.path.join(save_parent_dir, 'train', 'Epoch ' + str(epoch), 'sdf_map')):
                        os.mkdir(os.path.join(save_parent_dir, 'train', 'Epoch ' + str(epoch), 'sdf_map'))
                    for i in range(len(recon_logits)):
                        if input == 'images':
                            reconstructed_labels = torch.argmax(recon_logits, dim=1)
                            print('The reconstructed labels', reconstructed_labels.shape)
                            visualize_images(batch_idx, epoch, recon_logits, images, labels, test=False, save_path= os.path.join(save_parent_dir,'train', 'Epoch ' + str(epoch)))
                        elif input == 'sdf':
                            #np.save(os.path.join(save_parent_dir, 'train', 'Epoch ' + str(epoch), 'numpy',f'result_{train_batch}.npy'),recon_logits[i].permute(1, 2, 0).detach().cpu().numpy())
                            structure = convert_sdf_to_structures(recon_logits[i].permute(1, 2, 0).detach().cpu().numpy())
                            plt.imsave(os.path.join(os.path.join(save_parent_dir, 'train', 'Epoch ' + str(epoch),
                                                                 f'result_{train_batch}.png')), structure.squeeze(2), cmap='gray', vmin=0, vmax=1)
                            iou, img_prc_iou,_,_ = iou_sdf(recon_logits[i],labels[i],img_prc_result[i],os.path.join(save_parent_dir,'train', 'Epoch '+str(epoch),'iou_metrics'),train_batch)
                            train_iou_dl += iou
                            train_iou_img_prc += img_prc_iou
                            sdf_vis = recon_logits[i].permute(1, 2, 0).detach().cpu().numpy()
                            # vmax = max(abs(sdf_vis.min()), abs(sdf_vis.max()))
                            fig, ax = plt.subplots(figsize=(6, 5))
                            im = ax.imshow(sdf_vis, cmap='seismic', vmin=-1, vmax=1)
                            print('Exact Zeros: ', int(((sdf_vis > -0.05) & (sdf_vis < 0.05)).sum()))
                            ax.contour(sdf_vis.squeeze(-1), levels=[-1, 0.001], colors=['k'], linewidths=1.0)
                            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                            plt.savefig(os.path.join(save_parent_dir, 'train', 'Epoch ' + str(epoch), 'sdf_map',
                                                     f'sdf_{train_batch}'))
                            plt.close(fig)
                            train_batch += 1
                #scheduler.step()

                if epoch % 10 ==0:
                    val_loss = 0.0
                    val_batch = 0
                    if val_dataloader:
                        model.eval()
                        with torch.no_grad():
                            for batch_idx, (images, labels,img_prc_result) in enumerate(val_dataloader):

                                batch_size, num_imgs = images.shape[0], images.shape[1]

                                #combinedimages = images.reshape(-1, *images.shape[2:]).to(device)
                                images = images.to(device)
                                recon_logits, z = model(images,batch_size, num_imgs)
                                labels = labels.to(device)
                                if input == 'images':
                                    ce_loss = crossentropy_loss(recon_logits, labels.squeeze(1))
                                    sdf_loss = torch.norm((recon_logits.squeeze(1) - labels) / labels)
                                    loss = ce_loss + sdf_loss
                                    # sdf_loss = F.l1_loss(sdf_pred, gt_sdf)
                                elif input == 'sdf':
                                    sdf_loss = torch.norm((recon_logits.squeeze(1) - labels) / labels)
                                    loss = sdf_loss

                                val_loss += loss.item()

                                #progress_bar.set_postfix({'loss': loss.item(), })
                                if not os.path.exists(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch))):
                                    os.mkdir(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch)))
                                if not os.path.exists(
                                        os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch), 'sdf_map')):
                                    os.mkdir(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch), 'sdf_map'))
                                for i in range(len(recon_logits)):
                                    if input == 'images':
                                        reconstructed_labels = torch.argmax(recon_logits, dim=1)
                                        print('The reconstructed labels', reconstructed_labels.shape)
                                        visualize_images(batch_idx, epoch, recon_logits, images, labels, test=False,
                                                         save_path=os.path.join(save_parent_dir, 'val','Epoch ' + str(epoch)))
                                    elif input == 'sdf':
                                        #np.save(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch), 'numpy',
                                        #                     f'result_{val_batch}.npy'),
                                        #        recon_logits[i].permute(1, 2, 0).detach().cpu().numpy())
                                        structure = convert_sdf_to_structures(
                                            recon_logits[i].permute(1, 2, 0).detach().cpu().numpy())
                                        plt.imsave(
                                            os.path.join(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch),
                                                                      f'result_{val_batch}.png')), structure.squeeze(2),
                                            cmap='gray', vmin=0, vmax=1)
                                        iou, img_prc_iou,_,_ = iou_sdf(recon_logits[i], labels[i], img_prc_result[i],
                                                                   os.path.join(save_parent_dir, 'val',
                                                                                'Epoch ' + str(epoch), 'iou_metrics'),
                                                                   val_batch)
                                        train_iou_dl += iou
                                        train_iou_img_prc += img_prc_iou
                                        sdf_vis = recon_logits[i].permute(1, 2, 0).detach().cpu().numpy()
                                        # vmax = max(abs(sdf_vis.min()), abs(sdf_vis.max()))
                                        fig, ax = plt.subplots(figsize=(6, 5))
                                        im = ax.imshow(sdf_vis, cmap='seismic', vmin=-1, vmax=1)
                                        print('Exact Zeros: ', int(((sdf_vis > -0.05) & (sdf_vis < 0.05)).sum()))
                                        ax.contour(sdf_vis.squeeze(-1), levels=[-1, 0.001], colors=['k'], linewidths=1.0)
                                        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                                        plt.savefig(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch), 'sdf_map',
                                                                 f'sdf_{val_batch}'))
                                        plt.close(fig)
                                        val_batch += 1


                    test_loss = 0.0
                    test_batch = 0
                    test_iou_dl =0.0
                    test_iou_img_prc = 0.0
                    if test_dataloader:
                        model.eval()
                        with torch.no_grad():

                            for batch_idx, (images, labels, img_prc_result) in enumerate(test_dataloader):

                                batch_size, num_imgs = images.shape[0], images.shape[1]

                                #combinedimages = images.reshape(-1, *images.shape[2:]).to(device)
                                images = images.to(device)

                                recon_logits, z = model(images, batch_size, num_imgs)
                                labels = labels.to(device)

                                if input == 'images':
                                    ce_loss = crossentropy_loss(recon_logits, labels.squeeze(1))
                                    sdf_loss = torch.norm((recon_logits.squeeze(1) - labels) / labels)
                                    loss = ce_loss + sdf_loss
                                    # sdf_loss = F.l1_loss(sdf_pred, gt_sdf)
                                elif input == 'sdf':
                                    sdf_loss = torch.norm((recon_logits.squeeze(1) - labels) / labels)
                                    loss = sdf_loss

                                test_loss += loss.item()

                                #progress_bar.set_postfix({'loss': loss.item(), })
                                if not os.path.exists(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch))):
                                    os.mkdir(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch)))
                                if not os.path.exists(
                                        os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch), 'sdf_map')):
                                    os.mkdir(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch), 'sdf_map'))
                                for i in range(len(recon_logits)):
                                    if input == 'images':
                                        reconstructed_labels = torch.argmax(recon_logits, dim=1)
                                        print('The reconstructed labels', reconstructed_labels.shape)
                                        visualize_images(batch_idx, epoch, recon_logits, images, labels, test=False,
                                                         save_path=os.path.join(save_parent_dir, 'test','Epoch ' + str(epoch)))
                                    elif input == 'sdf':
                                        #np.save(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch), 'numpy',
                                        #                     f'result_{test_batch}.npy'),
                                        #        recon_logits[i].permute(1, 2, 0).detach().cpu().numpy())
                                        structure = convert_sdf_to_structures(
                                            recon_logits[i].permute(1, 2, 0).detach().cpu().numpy())
                                        plt.imsave(
                                            os.path.join(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch),
                                                                      f'result_{test_batch}.png')),
                                            structure.squeeze(2),
                                            cmap='gray', vmin=0, vmax=1)
                                        iou, img_prc_iou,_,_ = iou_sdf(recon_logits[i], labels[i], img_prc_result[i],
                                                                   os.path.join(save_parent_dir, 'test',
                                                                                'Epoch ' + str(epoch), 'iou_metrics'),
                                                                   test_batch)
                                        test_iou_dl += iou
                                        test_iou_img_prc += img_prc_iou
                                        sdf_vis = recon_logits[i].permute(1, 2, 0).detach().cpu().numpy()
                                        # vmax = max(abs(sdf_vis.min()), abs(sdf_vis.max()))
                                        fig, ax = plt.subplots(figsize=(6, 5))
                                        im = ax.imshow(sdf_vis, cmap='seismic', vmin=-1, vmax=1)
                                        print('Exact Zeros: ', int(((sdf_vis > -0.05) & (sdf_vis < 0.05)).sum()))
                                        ax.contour(sdf_vis.squeeze(-1), levels=[-1, 0.001], colors=['k'],
                                                   linewidths=1.0)
                                        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                                        plt.savefig(
                                            os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch), 'sdf_map',
                                                         f'sdf_{test_batch}'))
                                        plt.close(fig)
                                        test_batch += 1

                    print(f"Epoch {epoch + 1}: Train Loss: {train_loss / len(train_dataloader):.4f}, Val Loss: {val_loss / len(val_dataloader):.4f}, Test Loss: {test_loss / len(test_dataloader):.4f}")

                    run.log({'train loss': train_loss / len(train_dataloader),'val loss': val_loss / len(val_dataloader), 'test loss': test_loss / len(test_dataloader),
                             })
                # if test_loss < best_loss:
                #     best_loss = test_loss
                #     # model.save_pretrained('./runs/')
                #     print(f"Saved best model with val loss: {test_loss:.4f}")
                #
                # else:
                #     print(f"Epoch {epoch + 1}: Train Loss: {train_loss / len(train_dataloader):.4f}")



if __name__ == "__main__":
    train_unet()








