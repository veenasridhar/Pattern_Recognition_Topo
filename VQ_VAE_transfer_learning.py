import os
import torch
import torchvision
from torchvision.transforms import v2 
from diffusers import AutoencoderKL, VQModel
from torch.utils.data import DataLoader,Dataset
from torchinfo import summary
from tqdm import tqdm
import matplotlib.pyplot as plt


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

class ImageDataset:

    def __init__(self, transforms,image_directory=None,prebatched_dataset=False):
        
        self.transforms = transforms
        self.image_directory = image_directory
        self.prebatched_dataset = prebatched_dataset

    def __getitem__(self,index):

        
        if self.prebatched_dataset:
            image_paths = self.image_directory[index]
            images = []
            for image_path in image_paths:
                decoded_image = torchvision.io.decode_image(image_path)
                if self.transforms:
                    image = self.transforms(decoded_image)
                images.append(decoded_image)
            
            return torch.stack(images)               #This created a array of images (num_images,C,H,W)

        else:
            image_path = os.path.join(self.image_directory,f'image{index}'+'.png')
            decoded_image = torchvision.io.decode_image(image_path)
            if self.transforms:
                    image = self.transforms(decoded_image)
            return image
    
    def __len__(self):
        return len(os.listdir(self.image_directory))


class VQVAEFineTuner:

    def __init__(self, model_name="stabilityai/sd-vae-ft-mse",prebatched_images=False,device=None):
        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vqvae =  AutoencoderKL.from_pretrained(model_name)
        self.vqvae.to(self.device)
        summary(self.vqvae, input_size = (4,3,256,256))
        self.prebatched_images = prebatched_images
    
    def prepare_data(self,train_image_dir,test_image_dir,batch_size=2,img_size=256):

        train_transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((img_size,img_size)),
            v2.ToTensor()
        ])

        test_transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((img_size,img_size)),
            v2.ToTensor()
        ])
        
         ##Select wether batched images or single images.
        if self.prebatched_images:
            train_image_groups = create_image_groups(train_image_dir)
            train_dataset = ImageDataset(transforms = train_transforms,image_directory=train_image_groups,prebatched_dataset=self.prebatched_images)
            test_image_groups = create_image_groups(test_image_dir)
            test_dataset = ImageDataset(transforms=test_transforms,image_directory=test_image_groups,prebatched_dataset=self.prebatched_images)
        else:
            train_dataset = ImageDataset(transforms = train_transforms,image_directory=train_image_dir)
            test_dataset = ImageDataset(test_transforms,test_image_dir)

        train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=2,pin_memory=True)
        test_dataloader = DataLoader(test_dataset,batch_size=batch_size)

        return train_dataloader,test_dataloader
    
    def finetune(self, train_loader,test_loader, epochs=50,lr=0.0001,save_path = './finetuned_vae'):

        optimizer = torch.optim.AdamW(self.vqvae.parameters(), lr=lr, weight_decay = 1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)

        mse_loss = torch.nn.MSELoss()
        best_loss = float('inf')

        for epoch in range(epochs):

            self.vqvae.train()
            train_loss=0

            progress_bar = tqdm(train_loader,desc=f'Epoch {epoch+1}/{epochs}')

            for batch,images in enumerate(progress_bar):

                images  = images.to(self.device)

                outputs = self.vqvae(images)
                reconstructed = outputs.sample

                loss = mse_loss(reconstructed,images)

                #Backward Pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vqvae.parameters(),1.0)
                optimizer.step()
                if batch >4:
                    self.visulaize_images(batch,epoch,reconstructed,images,'./output_images/',False)

                train_loss+=loss
                progress_bar.set_postfix({'loss':loss.item()})

            scheduler.step()

            if test_loader:
                self.vqvae.eval()
                test_loss = 0
                
                with torch.no_grad():
                    for batch,images in enumerate(test_loader):
                        images = images.to(self.device)
                        outputs = self.vqvae(images)
                        test_reconstructed = outputs.sample
                        loss = mse_loss(test_reconstructed,images)
                        test_loss += loss.item()
                        self.visulaize_images(batch,epoch,test_reconstructed,images,'./output_images/test_images/',True)
                
                test_loss /= len(test_loader)
                print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {test_loss:.4f}")

                '''if test_loss < best_loss:
                    best_loss = test_loss
                    self.vqvae.save_pretrained(save_path)
                    print(f"Saved best model with val loss: {test_loss:.4f}")
            
                else:
                    print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}")'''

    def visulaize_images(self,batch,epoch,reconstructed_image,original, save_path,test =False):
        reconstructed_image = torch.clamp(reconstructed_image,0,1)
        #print(original.shape)
        for i in range(len(reconstructed_image)):
            if test:
                plt.imsave(os.path.join(save_path,f'output_test_images_{epoch}_{batch}_{i}.png'),reconstructed_image[i].permute(1,2,0).detach().cpu().numpy())
            else:
                plt.imsave(os.path.join(save_path,f'output_images_{epoch}_{batch}_{i}.png'),reconstructed_image[i].permute(1,2,0).detach().cpu().numpy())

        if True:
            fig, ax = plt.subplots(2,4,figsize=(16,8))
            print('The shape of reconstructed_image',reconstructed_image.shape,original.shape)
            for i in range(len(reconstructed_image)):
                ax[0,i].imshow(original[i].permute(1,2,0).detach().cpu().numpy())
                ax[0,i].set_title(f'Original {i+1}')
                ax[0,i].axis('off')

                ax[1,i].imshow(reconstructed_image[i].permute(1,2,0).detach().cpu().numpy())
                ax[1,i].set_title(f'Reconstructed {i+1}')
                ax[1,i].axis('off')

            plt.tight_layout()
            if test:
                plt.savefig(f'./output_images/test_images/reconstructed_{epoch}_{batch}.png', dpi=180)
            else:
                plt.savefig(f'./output_images/reconstructed_images_{epoch}_{batch}.png', dpi=180)
            plt.close()

        else:
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



