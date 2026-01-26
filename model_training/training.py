import torch
from torch import nn
from model_training.model import AutoEncoder
from torchinfo import summary
import torch.optim as optim
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import random

def training(dataloader,test_dataloader,use_saved_model=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on', device)

    model = AutoEncoder()
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    if use_saved_model:
        model_checkpoint = torch.load('./runs/best_model.pt')
        model.load_state_dict(model_checkpoint['model_state_dict'],weights_only=True)
        optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
        
    model = model.to(device)
    summary(model, input_size = (16,3,256,256))
    best_loss = float("inf")

    #Training Loop
    for epoch in range(501):
        running_loss =0.0
        avg_loss =0.0
        model.train()
        for batch, images in enumerate(dataloader):
            optimizer.zero_grad()
            images = images.to(device)
            recon = model(images)
            #print('The reeconstructed iamge shape',recon.shape,'and the target image size',images.shape)
            #print(recon.min().item(), recon.max().item())
            #print(images.min().item(), images.max().item())
            loss = loss_function(recon,images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss/len(dataloader)
        print(f'The loss in {epoch} is', avg_loss )

        #Saving the model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = {'epoch':epoch,
                                'model_state_dict':model.state_dict(),
                                'optimizer_state_dict':optimizer.state_dict(),
                                'loss':avg_loss}
            torch.save(best_model_state,'./runs/best_model.pt')

        #Visulaise the Recosntructed Outputs
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                visualize_images = next(iter(dataloader)).to(device)
                image_reconstruction_logits = model(visualize_images)
                image_reconstruction = torch.sigmoid(image_reconstruction_logits)
                for i in range(4):
                    plt.imsave(f'.\\output_images\\output_images_{epoch}_{i}.png',image_reconstruction[i].permute(1,2,0).detach().cpu().numpy())

                # Create figure
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                
                for i in range(4):
                    #Original Image
                    orig_img = visualize_images[i].permute(1, 2, 0).detach().cpu().numpy()
                    axes[0, i].imshow(orig_img)
                    axes[0, i].set_title(f'Original {i+1}')
                    axes[0, i].axis('off')

                    #Reconstruction Images
                    recon_images = image_reconstruction[i].permute(1,2,0).detach().cpu().numpy()
                    axes[1,i].imshow(recon_images)
                    axes[1,i].set_title(f'Recon {i+1}')
                    axes[1,i].axis('off')

                plt.suptitle(f'Epoch {epoch} - Avg Loss: {avg_loss:.6f}', fontsize=16)
                plt.tight_layout()
                plt.savefig(f'./output_images/reconstruction_epoch_{epoch:03d}.png', dpi=150)
                plt.close()

            #Test the model
            #model.eval()
            running_test_loss = 0.0
            with torch.no_grad():
                for batch,test_image in enumerate(test_dataloader):
                    test_image = test_image.to(device)
                    output = model(test_image)
                    loss = loss_function(output,test_image)
                    print('The loss for batch', batch,'is', loss.item())
                    running_test_loss += loss.item()
                    reconstructed_image = torch.sigmoid(output)
                    
                    for i in range(len(output)):
                        plt.imsave(f'.\\output_images\\test_images\\output_test_images_{batch}_{i}.png',reconstructed_image[i].permute(1,2,0).detach().cpu().numpy())

                    if batch == 0:
                        # Create figure
                        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                        for i in range(4):
                            #Original Image
                            orig_img = test_image[i].permute(1, 2, 0).detach().cpu().numpy()
                            axes[0, i].imshow(orig_img)
                            axes[0, i].set_title(f'Original {i+1}')
                            axes[0, i].axis('off')

                            #Reconstruction Images
                            recon_images = reconstructed_image[i].permute(1,2,0).detach().cpu().numpy()
                            axes[1,i].imshow(recon_images)
                            axes[1,i].set_title(f'Recon {i+1}')
                            axes[1,i].axis('off')

                        plt.tight_layout()
                        plt.savefig(f'./output_images/test_images/reconstruction_test_{epoch}.png', dpi=150)
                        plt.close()
            
                print('The test loss is', (running_test_loss/len(test_dataloader)))
                    






