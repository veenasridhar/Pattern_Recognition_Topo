import os
import numpy as np
import torchvision
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torchinfo import summary
import cv2
from input_pipeline import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import distance_transform_edt
import gudhi
from cldice_loss import *
from metrics import compute_betti_numbers
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from convert_sdf_structures import *
import glob
import collections
from util import convert_labels


class simple_Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(simple_Conv, self).__init__()

        # self.conv_layer1 = nn.Conv2d(in_channels, 16, kernel_size, stride, padding)               #1st layer
        # self.conv_layer2  = nn.Conv2d(16, 64, kernel_size=5, stride=1, padding=2)
        # self.conv_layer3 = nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3)
        # self.conv_layer4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        # self.conv_layer5 = nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1)
        # self.output_layer = nn.Conv2d(16, 3, kernel_size=1)            #Output Channels is 3 for 3 classes in the labels.

        self.conv_layer1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=2)  # 1st layer
        self.conv_layer2 = nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=3)
        self.conv_layer3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv_layer4 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer5 = nn.Conv2d(128, 32, kernel_size=5, stride=1, padding=2)
        self.conv_layer6 = nn.Conv2d( 32, 16, kernel_size=5, stride=1, padding=2)
        self.output_layer = nn.Conv2d(16, 3, kernel_size=1)

        self.relu_layer = nn.ReLU()
        self.sdf_head = nn.Sequential(
                nn.Conv2d(16, 1, kernel_size=1),
                nn.Tanh())

    def forward(self,x,avg_output,alpha):

        x1 = self.relu_layer(self.conv_layer1(x))
        x2 = self.relu_layer(self.conv_layer2(x1))
        x3 = self.relu_layer(self.conv_layer3(x2))
        x4 = self.relu_layer(self.conv_layer4(x3))
        x5 = self.relu_layer(self.conv_layer5(x4))
        x6 = self.relu_layer(self.conv_layer6(x5))
        conv_output = self.output_layer(x6)
        #output = alpha * conv_output +  avg_output

        sdf_output = self.sdf_head(x6)
        avg_output = avg_output.unsqueeze(1)
        print('Avg Output shape for addition',avg_output.shape)
        print('Model Output shape', sdf_output.shape)
        return (alpha * sdf_output + avg_output)/2
        #return sdf_output



def sdf_loss(prediction_softmax, gt_sdf):

    #prediction_softmax = torch.softmax(prediction_logits, dim=1)
    print('Softmax shape',prediction_softmax.shape)
    print('gt ssdf',gt_sdf.shape)
    loss = (prediction_softmax * gt_sdf).mean()

    return loss

def get_persistance_diagram(prob_map):

    '''prob_map: numpy (H, W) - Softmax probabilities of the for class 2 (white strutures)'''

    cc = gudhi.CubicalComplex(
        dimensions=prob_map.shap,
        top_dimensional_cells= prob_map.flatten()
    )
    cc.compute_persistance()

    # Get birth-death pairs for dimension 0 (components) and 1 (holes)
    dgm_0 = cc.persistence_intervals_in_dimension(0)  # components
    dgm_1 = cc.persistence_intervals_in_dimension(1)  # holes

    return dgm_0, dgm_1


def topo_loss(pred_diagram, gt_diagram):
    # Wasserstein distance — optimal transport between diagrams
    # Penalizes unmatched births/deaths
    return gudhi.wasserstein.wasserstein_distance(
        pred_diagram, gt_diagram, order=2
    )


def train_grayscale():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k_fold = KFold(n_splits=5, shuffle=True)
    batch_size = 4

    subbatch_dirs, group_ids = [], []
    for batch_id, batch_dir in enumerate(sorted(glob.glob('data/Final Data/Bigger Batches of Data SDF'))):  # 20 parent batches
        for sub in sorted(glob.glob(os.path.join(batch_dir, '*'))):  # its sub-batches
            subbatch_dirs.append(sub)
            group_ids.append(batch_id)

    full_dataset = TopoDataset(subbatch_dirs, group_ids)
    print('Total Sub Batches ', len(full_dataset.items))
    print("number of views:", len(set(full_dataset.group_of_item)))
    print("sub-batches per view:", collections.Counter(full_dataset.group_of_item.tolist()))

    #train_dataset,test_dataset = input_pipeline_sdf(single_image_reconstruction=False)
    #train_indices = np.arange(len(train_dataset))

    for fold in range (5):


        train_idx, val_idx, test_idx = grouped_split(full_dataset.group_of_item, fold)

        print('Train Index',train_idx)
        print('Test Index',test_idx)
        print('Val Index',val_idx)

        model = simple_Conv(5, 3, kernel_size=3, stride=1, padding=1)
        model = model.to(device)
        print(summary(model))
        print('Running on', device)

        if fold > 0:
            break

        train_dataloader = DataLoader(Subset(full_dataset, train_idx), batch_size=batch_size, shuffle=False)
        val_dataloader = DataLoader(Subset(full_dataset, val_idx), batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(Subset(full_dataset, test_idx), batch_size = batch_size, shuffle=False)


        lr = 0.0003
        epochs = 1000

        optimizer1 = torch.optim.Adam(model.parameters(), lr= lr)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=1000, eta_min=1e-5)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode = 'min', factor=0.5, patience=10, min_lr = 1e-5)

        weights = torch.tensor([0.25, 0.35, 0.5])
        crossentropy_loss = torch.nn.CrossEntropyLoss(weight=weights).to(device)
        cl_dice = soft_dice_cldice()
        save_parent_dir = f'Residual Convolutional Network Results_sdf'


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
                'architecture': 'Simple Convolution (Residual)',
                "learning_rate": lr,
                "epochs": epochs,
            }, )

        for epoch in range(epochs):

            model.train()
            print('Epoch: ',epoch, ' learning rate: ', lr)
            train_loss = 0.0
            train_image_proc_loss = 0.0
            train_cross_entropy_loss = 0.0
            train_sdf_loss =0.0
            train_batch = 0
            train_iou_dl = 0.0
            train_iou_img_prc = 0.0

            if epoch < 250:             # For weighing ce_loss and sdf loss
                alpha = epoch / 250
            else:
                alpha =  1

            for images, labels, avg_output in train_dataloader:
                #print('Image min and max',images[0][0].min(),images[0][0].max())
                #print('Avg Output Range',avg_output.min(),avg_output.max())

                images = images.to(device)
                print(images.shape)
                labels = labels.to(device)
                print('Labels shape',labels.shape)
                print('Labels Min and Max', labels[0].min(),labels[0].max())
                avg_output = avg_output.to(device)



                output = model(images, avg_output, alpha)
                print('Output shape for loss', output.shape)
                # print('Avg Output shape for loss',avg_output.shape)

                #sdf_loss = F.mse_loss(output.squeeze(1),labels) / (torch.norm(labels) + 1e-8)
                sdf_loss = torch.norm((output.squeeze(1) - labels)/ labels)
                img_processing_sdf_train_loss =torch.norm((avg_output- labels)/ labels)

                loss = sdf_loss
                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()

                #train_cross_entropy_loss += cross_entropy_loss.item()
                train_sdf_loss +=  sdf_loss.item()
                train_loss += loss.item()
                train_image_proc_loss+= img_processing_sdf_train_loss.item()


                if epoch % 10 ==0:
                    if not os.path.exists(os.path.join(save_parent_dir,'train', 'Epoch '+str(epoch))):
                        os.mkdir(os.path.join(save_parent_dir,'train', 'Epoch '+str(epoch)))
                    if not os.path.exists(os.path.join(save_parent_dir,'train', 'Epoch '+str(epoch),'iou_metrics')):
                        os.mkdir(os.path.join(save_parent_dir,'train', 'Epoch '+str(epoch),'iou_metrics'))
                    if not os.path.exists(os.path.join(save_parent_dir, 'train', 'Epoch ' + str(epoch),'numpy')):
                        os.mkdir(os.path.join(save_parent_dir, 'train', 'Epoch ' + str(epoch),'numpy'))
                    if not os.path.exists(os.path.join(save_parent_dir, 'train', 'Epoch ' + str(epoch), 'sdf_map')):
                        os.mkdir(os.path.join(save_parent_dir, 'train', 'Epoch ' + str(epoch), 'sdf_map'))

                    for i in range(len(output)):
                        # Output as Numpy
                        np.save(os.path.join(save_parent_dir,'train', 'Epoch '+str(epoch),'numpy',f'result_{train_batch}.npy'),output[i].permute(1, 2, 0).detach().cpu().numpy())

                        # Convert it to binary structures
                        structure = convert_sdf_to_structures(output[i].permute(1, 2, 0).detach().cpu().numpy())
                        #print('Converted Structures shape',structure.shape)
                        plt.imsave(os.path.join(os.path.join(save_parent_dir,'train', 'Epoch '+str(epoch), f'result_{train_batch}.png')), structure.squeeze(2), cmap='gray',vmin=0, vmax=1)
                        plt.close()

                        # IoU Metrics
                        iou, img_prc_iou = iou_sdf(output[i],labels[i],avg_output[i],os.path.join(save_parent_dir,'train', 'Epoch '+str(epoch),'iou_metrics'),train_batch)
                        train_iou_dl += iou
                        train_iou_img_prc += img_prc_iou

                        # SDF Visualisation
                        sdf_vis = output[i].permute(1,2,0).detach().cpu().numpy()
                        #vmax = max(abs(sdf_vis.min()), abs(sdf_vis.max()))
                        fig, ax = plt.subplots(figsize=(6,5))
                        im = ax.imshow(sdf_vis,cmap='seismic',vmin=-1,vmax=1)
                        ax.contour(sdf_vis.squeeze(-1), levels=[-1,0.001], colors=['k'], linewidths=1.0)  #Contour on all negative distances
                        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        plt.savefig(os.path.join(save_parent_dir, 'train', 'Epoch ' + str(epoch),'sdf_map',f'sdf_{train_batch}'))
                        plt.close(fig)
                        train_batch += 1


            print(f'Train Loss for Epoch {epoch}', train_loss / len(train_dataloader))
            print('Train IoU for deep learning', train_iou_dl / len(train_idx))
            print('Train IoU for image processing ', train_iou_img_prc / len(train_idx))

            #scheduler.step()

            # Starting a val loop
            #if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_img_proc_loss = 0.0
                val_cross_entropy_loss = 0.0
                val_dice_loss = 0.0
                val_sdf_loss = 0.0
                val_batch = 0
                val_iou_dl = 0.0
                val_iou_img_prc = 0.0

                for images, labels, avg_output in val_dataloader:
                    images = images.to(device)
                    labels = labels.to(device)
                    avg_output = avg_output.to(device)
                    # print('Val Avg OUtput Shape', avg_output.shape)
                    # print('Val Labels shape',labels.shape)

                    output= model(images, avg_output, alpha)
                    # print('Val Output shape',output.shape)

                    #sdf_loss = F.mse_loss(output.squeeze(1), labels)
                    #sdf_loss =  torch.norm(output.squeeze(1) - labels)/torch.norm(labels)
                    #val_image_processing_loss = F.mse_loss(avg_output,labels)
                    sdf_loss = torch.norm((output.squeeze(1) - labels) / labels)
                    val_image_processing_loss = torch.norm((avg_output - labels) / labels)

                    loss = sdf_loss

                    #val_cross_entropy_loss += cross_entropy_loss.item()
                    val_sdf_loss += sdf_loss.item()
                    val_loss += loss.item()
                    val_img_proc_loss+= val_image_processing_loss.item()


                    if epoch % 10 == 0:
                        if not os.path.exists(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch))):
                            os.mkdir(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch)))
                        if not os.path.exists(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch),'iou_metrics')):
                            os.mkdir(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch),'iou_metrics'))
                        if not os.path.exists(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch),'numpy')):
                            os.mkdir(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch), 'numpy'))
                        if not os.path.exists(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch), 'sdf_map')):
                            os.mkdir(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch), 'sdf_map'))

                        for i in range(len(output)):
                            np.save(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch), 'numpy',f'result_{val_batch}.npy'), output[i].permute(1, 2, 0).detach().cpu().numpy())

                            structure = convert_sdf_to_structures(output[i].permute(1, 2, 0).detach().cpu().numpy())
                            #print('Converted SDF structure shape',structure.shape)
                            plt.imsave(os.path.join(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch),f'result_{val_batch}.png')), structure.squeeze(2),cmap='gray',vmin=0, vmax=1)
                            plt.close()

                            iou, img_prc_iou = iou_sdf(output[i], labels[i], avg_output[i],os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch),'iou_metrics'), val_batch)
                            val_iou_dl += iou
                            val_iou_img_prc += img_prc_iou

                            sdf_vis = output[i].permute(1, 2, 0).detach().cpu().numpy()
                            # vmax = max(abs(sdf_vis.min()), abs(sdf_vis.max()))
                            fig, ax = plt.subplots(figsize=(6, 5))
                            im = ax.imshow(sdf_vis, cmap='seismic', vmin=-1, vmax=1)
                            ax.contour(sdf_vis.squeeze(-1), levels=[-1, 0.001], colors=['k'], linewidths=1.0)
                            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                            plt.savefig(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch), 'sdf_map',
                                                     f'sdf_{val_batch}'))
                            plt.close(fig)
                            val_batch += 1

                if epoch % 10 == 0:
                    print(f'Val Loss for Epoch {epoch}', val_loss / len(val_dataloader))
                    print('Val IoU for deep learning', val_iou_dl / len(val_idx))
                    print('Val IoU for image processing ', val_iou_img_prc / len(val_idx))



            # Starting a test loop
            if epoch % 10 ==0:
                model.eval()
                with torch.no_grad():
                    test_loss =0.0
                    test_image_processing_loss = 0.0
                    test_cross_entropy_loss = 0.0
                    test_dice_loss = 0.0
                    test_sdf_loss = 0.0
                    test_batch = 0
                    test_iou_dl = 0.0
                    test_iou_img_prc = 0.0

                    for images, labels,avg_output in test_dataloader:
                        images = images.to(device)
                        labels = labels.to(device)
                        avg_output = avg_output.to(device)
                        # print('Test Avg Output Shape', avg_output.shape)
                        # print('Test Labels shape', labels.shape)


                        output = model(images,avg_output,alpha)
                        # print('Test output shape for loss ',output.shape)
                        # print('Test Avg Output Shape for loss', avg_output.shape)
                        # print('Test Labels shape for loss', labels.shape)

                        #sdf_loss = F.mse_loss(output.squeeze(1), labels)
                        # sdf_loss = torch.norm(output.squeeze(1) - labels) / torch.norm(labels)
                        # image_processing_mse_loss = F.mse_loss(avg_output,labels)
                        sdf_loss = torch.norm((output.squeeze(1) - labels) / labels)
                        image_processing_mse_loss = torch.norm((avg_output - labels) / labels)
                        loss = sdf_loss

                        #test_cross_entropy_loss += cross_entropy_loss.item()
                        test_sdf_loss += sdf_loss.item()
                        #test_dice_loss +=  dice_loss
                        test_loss += loss.item()
                        test_image_processing_loss += image_processing_mse_loss.item()

                        #if epoch %10 == 0:
                        if not os.path.exists(os.path.join(save_parent_dir,'test', 'Epoch '+str(epoch))):
                            os.mkdir(os.path.join(save_parent_dir,'test', 'Epoch '+str(epoch)))
                        if not os.path.exists(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch),'iou_metrics')):
                            os.mkdir(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch),'iou_metrics'))
                        if not os.path.exists(os.path.join(save_parent_dir,'test', 'Epoch '+str(epoch),'numpy')):
                            os.mkdir(os.path.join(save_parent_dir,'test', 'Epoch '+str(epoch),'numpy'))
                        if not os.path.exists(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch), 'sdf_map')):
                            os.mkdir(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch), 'sdf_map'))
                        for i in range(len(output)):
                            np.save(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch),'numpy'
                                                 f'result_{test_batch}.npy'), output[i].permute(1, 2, 0).detach().cpu().numpy())
                            structure = convert_sdf_to_structures(output[i].permute(1, 2, 0).detach().cpu().numpy())
                            #print('The shape of structures ',structure.shape)
                            plt.imsave(os.path.join(
                                os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch), f'result_{test_batch}.png')),
                                       structure.squeeze(2),
                                       cmap='gray',
                                       vmin=0, vmax=1)
                            plt.close()
                            iou, img_prc_iou = iou_sdf(output[i], labels[i], avg_output[i],
                                                       os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch),
                                                                    'iou_metrics'), test_batch)
                            test_iou_dl += iou
                            test_iou_img_prc += img_prc_iou
                            sdf_vis = output[i].permute(1, 2, 0).detach().cpu().numpy()
                            # vmax = max(abs(sdf_vis.min()), abs(sdf_vis.max()))
                            fig, ax = plt.subplots(figsize=(6, 5))
                            im = ax.imshow(sdf_vis, cmap='seismic', vmin=-1, vmax=1)
                            ax.contour(sdf_vis.squeeze(-1), levels=[-1, 0.001], colors=['k'], linewidths=1.0)
                            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                            plt.savefig(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch), 'sdf_map',
                                                     f'sdf_{test_batch}'))
                            plt.close(fig)
                            test_batch += 1


                        print(f'Test Loss for Epoch {epoch}',test_loss/len(test_dataloader))
                        print('Test IoU for deep learning', test_iou_dl / len(test_idx))
                        print('Test IoU for image processing ', test_iou_img_prc / len(test_idx))



                    run.log({ 'epoch': epoch,
                              'alpha':alpha,
                             ##'learning_rate': scheduler.get_last_lr()[0],
                             'Train_Total_Loss': train_loss / len(train_dataloader),'Test_Total_Loss': test_loss / len(test_dataloader), 'Val Loss': val_loss/len(val_dataloader),
                             'Train Image Processing Loss': train_image_proc_loss/len(train_dataloader),'Val Image Processing Loss': val_image_processing_loss/len(val_dataloader),
                            'Test Image Processing Loss': test_image_processing_loss/len(test_dataloader),
                            'Train IoU DL':train_iou_dl / len(train_idx), 'Train IoU Image Processing':train_iou_img_prc/ len(train_idx),
                            'Val IoU DL': val_iou_dl / len(val_idx),'Val IoU Image Processing': val_iou_img_prc / len(val_idx),
                            'Test IoU DL':test_iou_dl /len(test_idx),'Test Image Processing':test_iou_img_prc /len(test_idx),
                        })

        wandb.finish()


if __name__ == "__main__":
    train_grayscale()
