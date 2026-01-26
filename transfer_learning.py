import torch
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import v2
from torchvision import transforms
from diffusers import AutoencoderKL, VQModel
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision

def create_image_groups(image_directory):
    image_groups = []
    if image_directory:
        for sub_dir in os.listdir(image_directory):
            image_group = []
            for images in os.listdir(os.path.join(image_directory,sub_dir)):
                if str(images).endswith('.png'):
                    image_group.append(os.path.join(image_directory,sub_dir,images))
                    #print(image_group)
            image_groups.append(image_group)
        #print(image_groups)
    return image_groups

class PreBatchedImageDataset(Dataset):

    def __init__(self,image_groups,transform):
        self.image_groups = image_groups
        self.transform = transform

    def __getitem__(self,index):

        image_paths = self.image_groups[index]
        
        #Array of images instead of a single image
        images = []

        for imagepath in image_paths:
            image = torchvision.io.decode_image(imagepath)
            if self.transform:
                image = self.transform(image)
            images.append(image)

        return torch.stack(images)          #This created a array of images (num_images,C,H,W)

    def __len__(self):
        return len(self.image_groups)

class VAEFineTuner:

    def __init__(self, model_name='stabilityai/sd-vae-ft-mse',multi_image_mode=True,train_loader=None,test_loader=None,device=None):
        """
        Initialize VAE model
        
        Args:
            model_name: HuggingFace model identifier
                - "stabilityai/sd-vae-ft-mse" (recommended, MSE finetuned)
                - "stabilityai/sd-vae-ft-ema" (EMA finetuned)
            device: torch device
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multi_image_mode = multi_image_mode
        self.vae = AutoencoderKL.from_pretrained(model_name)
        self.vae.to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.vae.parameters()):,}")
        print('Running on ',self.device)

    def prepare_data(self,image_directory):
        #Image Transformations
        train_transformations = v2.Compose([
        v2.ToDtype(torch.float32, scale = True), # Scale = true already scales the iamges between 0,1.
        v2.Resize((256,256)),
        v2.ToTensor(),
        #v2.Normalize(mean=(0.5,), std= (1.0,)),
        ])

        test_transformations = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((256,256)),
            v2.ToTensor(),
            #v2.Normalize(mean=(0.5,), std= (1.0,)),
        ])

        image_groups = create_image_groups(image_directory)

        train_dataset = PreBatchedImageDataset(image_groups=image_groups,transform=train_transformations)

        train_dataloader = DataLoader(train_dataset,batch_size=2,shuffle=True,num_workers=2,pin_memory=True)

        print('The length of train data loader', len(train_dataloader))

        return train_dataloader


    
    def finetune(self,train_loader,test_loader,epochs=50, lr=0.0001,kl_weight = 1e-6,save_path = "./finetuned_vae"):
        
        optimizer = torch.optim.AdamW(self.vae.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        mse_loss = torch.nn.MSELoss()
        best_loss = float('inf')

        for epoch in range(epochs):

            self.vae.train()
            train_loss =0
            train_recon_loss =0
            train_kl_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epochs {epoch+1}/{epochs}')
            for batch,images in enumerate(progress_bar):

                if self.multi_image_mode:
                    
                    # Convert the (Batch, Num_of_images, Channels, Height, Weight) where Num_of_images is the images to be averaged in that group
                    batch_size, num_imgs = images.shape[0], images.shape[1]
                    #print(f'The batch_size is {batch_size} and number of images per sample {num_imgs}')

                    combinedimages = images.reshape(-1, *images.shape[2:]).to(self.device)
                    # The above line reshapes the input images from (B,N,C,H,W) to (B*N,C,H,W) where *images.shape[2:] (the * is called the argument unpacking unpacks (C,H,W) and the -1
                    #in the first indes allows python to infer the dimensions on its own while maintaining the data count so for the first indes B*N is done.)

                    posterior = self.vae.encode(combinedimages).latent_dist
                    z_all = posterior.sample()            #(B*N,latent_dimensions, h,w)

                    print('The shape of latent dimensions',z_all.shape)
                    z_all = z_all.reshape(batch_size,num_imgs,*z_all.shape[1:]) # Reshaping back into (B,N,C,H,W)
                    z_averaged = z_all.mean(dim=1)               # dim=1 specifies that the averaging should be done across the first dimension
                    #print('The shape of averaged latent code',z_averaged.shape)

                    reconstructed = self.vae.decode(z_averaged).sample
                    #print('The shape of reconstructed',reconstructed.shape)
                    if batch <= 3:
                        self.visulaize_images(batch, epoch,reconstructed,images,'.\\output_images\\',False)

                    # Target: also average the input images (common structures)
                    target = images.mean(dim=1).to(self.device)
                    #print('The shape of target', target.shape)

                    reconstruction_loss = mse_loss(reconstructed,target)

                    kl_loss = posterior.kl().reshape(batch_size,num_imgs,-1).mean()
                
                else:
                    images = images.to(self.device)
                    posterior = self.vae.encode(images).latent_dist
                    z = posterior.sample()
                    reconstructed = self.vae.decode(z).sample
                    if batch == 0:
                        self.visulaize_images(batch, reconstructed,images,'.\\output_images\\',False)

                    reconstruction_loss = mse_loss(reconstructed,images)
                    kl_loss = posterior.kl().mean()
                    
                loss = reconstruction_loss + kl_weight * kl_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                train_recon_loss += reconstruction_loss.item()
                train_kl_loss += kl_loss.item()

                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'recon': reconstruction_loss.item(),
                    'kl': kl_loss.item()
                })
            scheduler.step()

            test_loss =0
            if test_loader:
                self.vae.eval()
                with torch.no_grad():
                    for batch,images in enumerate(test_loader):
                        if self.multi_image_mode:
                            batch_size, num_imgs = images.shape[0], images.shape[1]
                            combinedimages = images.reshape(-1, *images.shape[2:]).to(self.device)
                            posterior = self.vae.encode(combinedimages).latent_dist
                            z_all = posterior.sample()
                            z_all = z_all.reshape(batch_size,num_imgs,*z_all.shape[1:]) # Reshaping back into (B,N,C,H,W)
                            z_averaged = z_all.mean(dim=1)
                            reconstructed = self.vae.decode(z_averaged).sample
                            self.visulaize_images(batch, epoch,reconstructed,images,'.\\output_images\\test_images\\',True)
                            target = images.mean(dim=1).to(self.device)
                            print('The shape of target', target.shape)

                            reconstruction_test_loss = mse_loss(reconstructed,target)

                            kl_test_loss = posterior.kl().reshape(batch_size,num_imgs,-1).mean()

                            t_loss = reconstruction_test_loss + kl_weight * kl_test_loss
                            test_loss += t_loss.item()

                        else:
                            images = images.to(self.device)
                            posterior = self.vae.encode(images).latent_dist
                            z=posterior.sample()
                            reconstructed = self.vae.decode(z).sample
                            recon_loss = mse_loss(reconstructed,images)
                            kl_loss = posterior.kl().mean()
                            loss = recon_loss + kl_weight*kl_loss
                            test_loss += loss.item()
                        '''if batch <= 3:
                            self.visulaize_images(batch, epoch,reconstructed,images,'.\\output_images\\test_images\\',True)'''
                
                test_loss /= len(test_loader)
                print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {test_loss:.4f}")

                if test_loss < best_loss:
                    best_loss = test_loss
                    self.vae.save_pretrained(save_path)
                    print(f"Saved best model with val loss: {test_loss:.4f}")
            
                else:
                    print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}")

    def visulaize_images(self,batch,epoch,reconstructed_image,original, save_path,test =False):
        reconstructed_image = torch.clamp(reconstructed_image,0,1)
        #print(original.shape)
        for i in range(len(reconstructed_image)):
            if test:
                plt.imsave(os.path.join(save_path,f'output_test_images_{epoch}_{batch}_{i}.png'),reconstructed_image[i].permute(1,2,0).detach().cpu().numpy())
            else:
                plt.imsave(os.path.join(save_path,f'output_images_{epoch}_{batch}_{i}.png'),reconstructed_image[i].permute(1,2,0).detach().cpu().numpy())

       
        # Create figure
        for i in range(2):
            fig = plt.figure(figsize=(16,8))
            grid_spec = fig.add_gridspec(2,5,hspace=0.1,wspace=0.1)

            for j in range(5):
                ax = fig.add_subplot(grid_spec[0,j])
                orig_img = original[i][j].permute(1, 2, 0).detach().cpu().numpy()
                ax.imshow(orig_img)
                ax.axis('off')
            
            ax_center = fig.add_subplot(grid_spec[1,2])
            recon_images = reconstructed_image[i].permute(1,2,0).detach().cpu().numpy()
            ax_center.imshow(recon_images)
            ax_center.axis('off')

            for l in [0, 1, 3, 4]:
                ax = fig.add_subplot(grid_spec[1, l])
                ax.axis("off")
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path,f'reconstructed_test_images_{epoch}_{batch}_{i}.png'), dpi=150)
            plt.close()

            '''fig, axes = plt.subplots(2, 5, figsize=(16, 8))
            #Original Image
            for j in range(5):
                orig_img = original[i][j].permute(1, 2, 0).detach().cpu().numpy()
                axes[0, j].imshow(orig_img)
                axes[0, j].set_title(f'Original {j+1}')
                axes[0, j].axis('off')

            #Reconstruction Images
            recon_images = reconstructed_image[i].permute(1,2,0).detach().cpu().numpy()
            axes[1,0].imshow(recon_images)
            axes[1,0].set_title(f'Recon {i+1}')
            axes[1,0].axis('off')

            plt.tight_layout()
            plt.savefig(f'./output_images/output_images_{epoch}_{i}.png', dpi=150)
            plt.close()'''
    






