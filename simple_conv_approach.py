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
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import distance_transform_edt
import gudhi
from resnet_network import *
from metrics import compute_betti_numbers
from sklearn.model_selection import KFold
from convert_sdf_structures import *

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
        self.conv_layer3 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3)
        self.conv_layer4 = nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2)
        self.conv_layer5 = nn.Conv2d( 32, 16, kernel_size=5, stride=1, padding=2)
        self.output_layer = nn.Conv2d(16, 3, kernel_size=1)

        self.relu_layer = nn.ReLU()
        self.sdf_head = nn.Sequential(
                nn.Conv2d(16, 1, kernel_size=1),
                nn.Tanh()
        )

    def forward(self,x,avg_output,alpha):

        x1 = self.relu_layer(self.conv_layer1(x))
        x2 = self.relu_layer(self.conv_layer2(x1))
        x3 = self.relu_layer(self.conv_layer3(x2))
        x4 = self.relu_layer(self.conv_layer4(x3))
        x5 = self.relu_layer(self.conv_layer5(x4))
        conv_output = self.output_layer(x5)
        #output = alpha * conv_output +  avg_output

        sdf_output = self.sdf_head(x5)
        avg_output = avg_output.unsqueeze(1)
        #return output,sdf_output
        # print('Output Range',sdf_output.min(),sdf_output.max())
        # print('Output Shape',sdf_output.shape)
        # print('Avg Output', avg_output.shape)
        # print('Avg Output Range',avg_output.min(),avg_output.max())
        # print('Final Output',((alpha * sdf_output +  avg_output)/2).min(), ((alpha * sdf_output +  avg_output)/2).max())
        # print('Final Output Shape', ((alpha * sdf_output + avg_output) / 2).shape)
        return (alpha * sdf_output +  avg_output)/2
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
    model = simple_Conv(5,3,kernel_size=3, stride=1, padding=1)
    model = model.to(device)
    print(summary(model))
    print('Running on',device)

    print('going into input piptliune')
    train_dataset,test_dataset = input_pipeline_sdf(single_image_reconstruction=False)
    train_indices = np.arange(len(train_dataset))

    for fold, (train_index, val_index) in enumerate(k_fold.split(train_indices)):

        if fold > 0:
            break
        train_subset = torch.utils.data.Subset(train_dataset, train_index)
        val_subset = torch.utils.data.Subset(train_dataset, val_index)

        # Transform the Dataset into DataLoader
        train_dataloader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        val_dataloader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2,pin_memory=True)


        lr = 0.0003
        epochs = 500

        optimizer1 = torch.optim.Adam(model.parameters(), lr= lr)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer1,gamma= 0.95)

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
            for images, labels,avg_output in train_dataloader:
                print('Image min and max',images[0][0].min(),images[0][0].max())
                print('Avg Output Range',avg_output.min(),avg_output.max())
                images = images.to(device)
                print(images.shape)
                labels = labels.to(device)
                print('Labels shape',labels.shape)
                avg_output = avg_output.to(device)

                output = model(images, avg_output, alpha)
                print('Output shape', output.shape)
                print('Avg Output shape',avg_output.shape)

                #sdf_loss = F.mse_loss(output.squeeze(1),labels) / (torch.norm(labels) + 1e-8)
                #label_norm =
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
                        np.save(os.path.join(save_parent_dir,'train', 'Epoch '+str(epoch),'numpy',f'result_{train_batch}.npy'),output[i].permute(1, 2, 0).detach().cpu().numpy())
                        structure = convert_sdf_to_structures(output[i].permute(1, 2, 0).detach().cpu().numpy())
                        plt.imsave(os.path.join(os.path.join(save_parent_dir,'train', 'Epoch '+str(epoch), f'result_{train_batch}.png')), structure.squeeze(2), cmap='gray',
                                   vmin=0, vmax=1)
                        # iou, img_prc_iou = iou_sdf(output[i],labels[i],avg_output[i],os.path.join(save_parent_dir,'train', 'Epoch '+str(epoch),'iou_metrics'),train_batch)
                        # train_iou_dl += iou
                        # train_iou_img_prc += img_prc_iou
                        sdf_vis = output[i].permute(1,2,0).detach().cpu().numpy()
                        #vmax = max(abs(sdf_vis.min()), abs(sdf_vis.max()))
                        fig, ax = plt.subplots(figsize=(6,5))
                        im = ax.imshow(sdf_vis,cmap='RdBu',vmin=-1,vmax=1)
                        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        plt.savefig(os.path.join(save_parent_dir, 'train', 'Epoch ' + str(epoch),'sdf_map',f'sdf_{train_batch}'))
                        plt.close(fig)
                        train_batch += 1


            print(f'Train Loss for Epoch {epoch}', train_loss / len(train_dataloader))
            print('Train IoU for deep learning', train_iou_dl / 95)
            print('Train IoU for image processing ', train_iou_img_prc / 95)

            # scheduler.step()

            # Starting a val loop
            if epoch % 10 == 0:
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
                    if epoch < 250:  # For weighing ce_loss and sdf loss
                        alpha = epoch / 250
                    else:
                        alpha = 1

                    for images, labels, avg_output in val_dataloader:
                        images = images.to(device)
                        labels = labels.to(device)
                        avg_output = avg_output.to(device)
                        print('Val Avg OUtput Shape', avg_output.shape)
                        print('Val Labels shape',labels.shape)

                        output= model(images, avg_output, alpha)
                        print('Val Output shape',output.shape)

                        #sdf_loss = F.mse_loss(output.squeeze(1), labels)
                        # sdf_loss =  torch.norm(output.squeeze(1) - labels)/torch.norm(labels)
                        # val_image_processing_loss = F.mse_loss(avg_output,labels)
                        sdf_loss = torch.norm((output.squeeze(1) - labels) / labels)
                        val_image_processing_loss = torch.norm((avg_output - labels) / labels)

                        loss = sdf_loss

                        #val_cross_entropy_loss += cross_entropy_loss.item()
                        val_sdf_loss = sdf_loss.item()
                        # test_dice_loss +=  dice_loss
                        val_loss += loss.item()
                        val_img_proc_loss+= val_image_processing_loss.item()

                        #if epoch % 10 == 0:
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
                            plt.imsave(os.path.join(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch),f'result_{val_batch}.png')), structure.squeeze(2),
                                       cmap='gray',
                                       vmin=0, vmax=1)
                            # iou, img_prc_iou = iou_sdf(output[i], labels[i], avg_output[i],
                            #                            os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch),
                            #                                         'iou_metrics'), val_batch)
                            # val_iou_dl += iou
                            # val_iou_img_prc += img_prc_iou
                            sdf_vis = output[i].permute(1, 2, 0).detach().cpu().numpy()
                            # vmax = max(abs(sdf_vis.min()), abs(sdf_vis.max()))
                            fig, ax = plt.subplots(figsize=(6, 5))
                            im = ax.imshow(sdf_vis, cmap='RdBu', vmin=-1, vmax=1)
                            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                            plt.savefig(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch), 'sdf_map',
                                                     f'sdf_{val_batch}'))
                            plt.close(fig)
                            val_batch += 1

                    print(f'Val Loss for Epoch {epoch}', val_loss / len(val_dataloader))
                    print('Val IoU for deep learning', val_iou_dl / 19)
                    print('Val IoU for image processing ', val_iou_img_prc / 19)

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
                    if epoch < 250:  # For weighing ce_loss and sdf loss
                        alpha = epoch / 250
                    else:
                        alpha = 1
                    for images, labels,avg_output in test_dataloader:
                        images = images.to(device)
                        labels = labels.to(device)
                        avg_output = avg_output.to(device)
                        print('Test Avg Output Shape', avg_output.shape)
                        print('Test Labels shape', labels.shape)


                        output = model(images,avg_output,alpha)
                        print('Test output shape',output.shape)

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
                        test_image_processing_loss+=image_processing_mse_loss.item()

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
                            plt.imsave(os.path.join(
                                os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch), f'result_{test_batch}.png')),
                                       structure.squeeze(2),
                                       cmap='gray',
                                       vmin=0, vmax=1)
                            # iou, img_prc_iou = iou_sdf(output[i], labels[i], avg_output[i],
                            #                            os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch),
                            #                                         'iou_metrics'), val_batch)
                            # test_iou_dl += iou
                            # test_iou_img_prc += img_prc_iou
                            sdf_vis = output[i].permute(1, 2, 0).detach().cpu().numpy()
                            # vmax = max(abs(sdf_vis.min()), abs(sdf_vis.max()))
                            fig, ax = plt.subplots(figsize=(6, 5))
                            im = ax.imshow(sdf_vis, cmap='RdBu', vmin=-1, vmax=1)
                            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                            plt.savefig(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch), 'sdf_map',
                                                     f'sdf_{test_batch}'))
                            plt.close(fig)
                            test_batch += 1


                        print(f'Test Loss for Epoch {epoch}',test_loss/len(test_dataloader))
                        print('Test IoU for deep learning', test_iou_dl / 10)
                        print('Test IoU for image processing ', test_iou_img_prc / 10)


                    run.log({ 'epoch': epoch,
                             'alpha':alpha,
                            #'learning_rate': scheduler.get_last_lr()[0],
                             'Train_Total_Loss': train_loss / len(train_dataloader),'Test_Total_Loss': test_loss / len(test_dataloader), 'Val Loss': val_loss/len(val_dataloader),
                            'Train_image_processing_mse_loss': train_image_proc_loss/len(train_dataloader),'Val Image Processing mse loss': val_image_processing_loss/len(val_dataloader),
                              'Test IMage Processing Loss': test_image_processing_loss/len(test_dataloader),
                            # 'Train IoU DL':train_iou_dl, 'Train Image Processing':train_iou_img_prc, 'Val IoU DL': val_iou_dl,'Val IoU Image Processing': val_iou_img_prc,
                            #   'Test IoU DL':test_iou_dl,'Test Image Processing':test_iou_img_prc

                        })

        wandb.finish()

def extract_red_mask(image):
    print(image.min(),image.max())
    R = image[:,:,0]
    # G = image[:,:,1]
    # B = image[:,:,2]
    mask = (R ==2)          # Output Labels of 2 is white structures.
    return mask.astype(float)

def visulaize_iou_metrics(raw_images, epoch,batch,predicted_structures, gt,image_processing_result,betti_error_pred,betti_error_img_prc,tolerance=0,save_path='Simple_Conv_Grayscale_output'):

    red_extracted_structures = extract_red_mask(predicted_structures.unsqueeze(dim = -1).detach().cpu().numpy())
    red_gt_structures = extract_red_mask(gt.unsqueeze(dim = -1).detach().cpu().numpy())
    red_image_processing_structures = extract_red_mask(image_processing_result.permute(1,2,0).detach().cpu().numpy())

    if tolerance>0:
        r = tolerance
        y, x = np.ogrid[-r:r + 1, -r:r + 1]
        circle = (x ** 2 + y ** 2 <= r ** 2)
        predicted_dilated_structure = binary_dilation(red_extracted_structures.astype(bool), structure=circle)
        target_dilated_structure = binary_dilation(red_gt_structures.astype(bool), structure=circle)
        image_processing_structures = binary_dilation(red_image_processing_structures.astype(bool), structure=circle)
    else:
        predicted_dilated_structure = red_extracted_structures
        target_dilated_structure = red_gt_structures
        image_processing_structures = red_image_processing_structures

    TP = (predicted_dilated_structure * target_dilated_structure).sum()
    FP = (predicted_dilated_structure * (1 - target_dilated_structure)).sum()
    FN = ((1 - predicted_dilated_structure) * target_dilated_structure).sum()

    dilated_overlap = np.zeros((*red_extracted_structures.shape, 3))
    dilated_overlap[..., 0] = (predicted_dilated_structure > 0) & (target_dilated_structure == 0)  # FP = red
    dilated_overlap[..., 1] = (predicted_dilated_structure > 0) & (target_dilated_structure > 0)  # TP = green
    dilated_overlap[..., 2] = (predicted_dilated_structure == 0) & (target_dilated_structure > 0)  # FN = blue

    iou_dilated = TP / (TP + FP + FN + 1e-8)


    TP = (image_processing_structures * target_dilated_structure).sum()
    FP = (image_processing_structures * (1 - target_dilated_structure)).sum()
    FN = ((1 - image_processing_structures) * target_dilated_structure).sum()

    img_proc_dilated_overlap = np.zeros((*red_extracted_structures.shape, 3))
    img_proc_dilated_overlap[..., 0] = (image_processing_structures > 0) & (target_dilated_structure == 0)  # FP = red
    img_proc_dilated_overlap[..., 1] = (image_processing_structures > 0) & (target_dilated_structure > 0)  # TP = green
    img_proc_dilated_overlap[..., 2] = (image_processing_structures == 0) & (target_dilated_structure > 0)  # FN = blue

    iou_dilated_image_processing = TP / (TP + FP + FN + 1e-8)

    print('Predicted IoU ', iou_dilated)
    print('Image Processing IoU ',iou_dilated_image_processing)

    visulaize_iou_results(raw_images,epoch,batch,predicted_structures,gt,dilated_overlap,red_image_processing_structures,img_proc_dilated_overlap,iou_dilated,iou_dilated_image_processing,betti_error_pred,betti_error_img_prc,save_path)

    return iou_dilated,iou_dilated_image_processing

def visulaize_iou_results(raw_images,epoch,batch,predicted_structures,gt,overlap, image_processing_result,img_proc_dilated_overlap,iou_dilated,iou_dilated_image_processing,betti_error_pred,betti_error_img_prc,save_path='Simple_Conv_Grayscale_output'):

    fig = plt.figure(figsize=(16, 8))
    grid_spec = fig.add_gridspec(2, 5, hspace=0.2, wspace=0.2)

    for j in range(5):
        ax = fig.add_subplot(grid_spec[0, j])
        ax.imshow(raw_images[j].detach().cpu().numpy(),cmap='gray')
        if j == 2:
            ax.set_title('Input Images')
        ax.axis('off')

    ax = fig.add_subplot(grid_spec[1, 0])
    ax.imshow(gt.detach().cpu().numpy(),cmap='gray')
    ax.set_title('Ground Truth')
    ax.axis('off')

    ax = fig.add_subplot(grid_spec[1, 1])
    ax.imshow(predicted_structures.detach().cpu().numpy(),cmap='gray',vmin=0,vmax=2)
    ax.set_title('Predicted Dilated Structure')
    ax.axis('off')

    ax = fig.add_subplot(grid_spec[1, 2])
    ax.imshow(overlap)
    ax.set_title(f'IOU: {iou_dilated:0.2f} \n Betti Number: {betti_error_pred}')
    ax.axis('off')

    ax = fig.add_subplot(grid_spec[1, 3])
    #print('Image Processing Shape',image_processing_result.shape)
    ax.imshow(image_processing_result,cmap='gray')
    ax.set_title(f'Image Processing Result')
    ax.axis('off')

    ax = fig.add_subplot(grid_spec[1, 4])
    ax.imshow(img_proc_dilated_overlap)
    ax.set_title(f'IOU: {iou_dilated_image_processing:0.2f} \n Betti Number: {betti_error_pred}')
    ax.axis('off')

    plt.tight_layout()
    #plt.show()
    if not os.path.exists(os.path.join(save_path, 'Epoch ' + str(epoch),'IoU Metric')):
        os.mkdir(os.path.join(save_path, 'Epoch ' + str(epoch),'IoU Metric'))

    plt.savefig(f'{save_path}/Epoch {epoch}/IoU Metric/result_{batch}.png')
    #plt.savefig(f'Latent_averaging_Results/IoU Metric/spatial_agg_result{batch}.png')
    plt.close()

def visualize_sdf_training(model, diff_model, dataloader, device,
                            epoch, save_path='.'):
    """
    Run this every N epochs to see SDF loss effect on your predictions.
    Shows: inputs | GT | GT SDF | model pred | SDF error | improvement
    """
    model.eval()
    diff_model.eval()

    with torch.no_grad():
        for images, labels, avg_output, sdf_white in dataloader:
            images     = images.to(device)
            labels     = labels.squeeze(1).long().to(device)
            avg_output = avg_output.to(device)
            sdf_white  = sdf_white.to(device)

            output      = model(images)
            diff_output = diff_model(output, avg_output)

            # Convert to numpy for plotting
            B = min(images.shape[0], 3)  # show 3 samples

            fig = plt.figure(figsize=(20, B * 4))
            fig.suptitle(f'Epoch {epoch} — SDF loss visualisation',
                        fontsize=13)

            for b in range(B):
                # ── Compute values ─────────────────────────────────
                gt_np   = labels[b].cpu().numpy()          # (H, W) 0,1,2
                sdf_np  = sdf_white[b].cpu().numpy()       # (H, W) [-1,1]

                pred_soft_m1   = torch.softmax(output,      dim=1)[b, 2].cpu().numpy()
                pred_soft_diff = torch.softmax(diff_output, dim=1)[b, 2].cpu().numpy()

                gt_white  = (gt_np == 2).astype(float)

                # SDF error — how far pred is from GT SDF
                # Kervadec convention: sdf negative inside → target for pred is HIGH
                # sdf_white is negative inside (outside structure is positive)
                # so -sdf_white gives us what pred_soft SHOULD look like
                sdf_target  = np.clip(-sdf_np, 0, 1)      # ideal pred_soft
                sdf_error_m1   = np.abs(pred_soft_m1   - sdf_target)
                sdf_error_diff = np.abs(pred_soft_diff - sdf_target)

                # Loss value per pixel
                loss_map_m1   = pred_soft_m1   * sdf_np   # positive=bad, negative=good
                loss_map_diff = pred_soft_diff * sdf_np

                # ── Plot row ────────────────────────────────────────
                cols = 8
                row  = b * cols

                gs = gridspec.GridSpec(B, cols, figure=fig,
                                      hspace=0.35, wspace=0.15)

                def ax(c):
                    a = fig.add_subplot(gs[b, c])
                    a.axis('off')
                    return a

                # Col 0 — input image (first channel)
                a = ax(0)
                a.imshow(images[b, 0].cpu().numpy(), cmap='gray')
                a.set_title('Input', fontsize=9)

                # Col 1 — ground truth
                a = ax(1)
                a.imshow(gt_white, cmap='gray', vmin=0, vmax=1)
                a.set_title('GT (white)', fontsize=9)

                # Col 2 — GT SDF
                a = ax(2)
                vmax = max(abs(sdf_np.min()), abs(sdf_np.max()))
                im = a.imshow(sdf_np, cmap='RdBu', vmin=-vmax, vmax=vmax)
                a.set_title('GT SDF\nred=+, blue=-', fontsize=9)
                plt.colorbar(im, ax=a, fraction=0.046, pad=0.04)

                # Col 3 — Model 1 predicted prob
                a = ax(3)
                a.imshow(pred_soft_m1, cmap='hot', vmin=0, vmax=1)
                a.set_title(f'Model1 pred\nmean={pred_soft_m1.mean():.2f}',
                           fontsize=9)

                # Col 4 — Model 1 loss map
                a = ax(4)
                lv = max(abs(loss_map_m1.min()), abs(loss_map_m1.max()))
                a.imshow(loss_map_m1, cmap='RdBu', vmin=-lv, vmax=lv)
                loss_val_m1 = loss_map_m1.mean()
                a.set_title(f'Loss map M1\n{loss_val_m1:.4f}', fontsize=9)

                # Col 5 — Refiner predicted prob
                a = ax(5)
                a.imshow(pred_soft_diff, cmap='hot', vmin=0, vmax=1)
                a.set_title(f'Refiner pred\nmean={pred_soft_diff.mean():.2f}',
                           fontsize=9)

                # Col 6 — Refiner loss map
                a = ax(6)
                lv = max(abs(loss_map_diff.min()), abs(loss_map_diff.max()))
                a.imshow(loss_map_diff, cmap='RdBu', vmin=-lv, vmax=lv)
                loss_val_diff = loss_map_diff.mean()
                a.set_title(f'Loss map Ref\n{loss_val_diff:.4f}', fontsize=9)

                # Col 7 — improvement map
                a = ax(7)
                improvement = sdf_error_m1 - sdf_error_diff
                a.imshow(improvement, cmap='RdYlGn',
                        vmin=-0.5, vmax=0.5)
                mean_imp = improvement.mean()
                a.set_title(f'Improvement\n{mean_imp:+.4f}', fontsize=9)

            plt.savefig(f'{save_path}/sdf_epoch_{epoch:04d}.png',
                       dpi=120, bbox_inches='tight')
            plt.close()
            print(f'Saved sdf_epoch_{epoch:04d}.png')
            break   # only first batch





def visualise_sdf(label_map,mask,dist_in,dist_out,raw_sdf,normalized):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('SDF Step by Step', fontsize=14)

    # 1. Label map
    axes[0, 0].imshow(label_map, cmap='gray', vmin=0, vmax=2)
    axes[0, 0].set_title('Label map\n(0=black, 1=gray, 2=white)')
    axes[0, 0].axis('off')

    # 2. Binary mask
    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title('Binary mask\n(white class only)')
    axes[0, 1].axis('off')

    # 3. dist_inside
    im = axes[0, 2].imshow(dist_in, cmap='hot')
    axes[0, 2].set_title('dist_inside\n(0 outside, +d inside)')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    # 4. dist_outside
    im = axes[1, 0].imshow(dist_out, cmap='hot')
    axes[1, 0].set_title('dist_outside\n(0 inside, +d outside)')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    # 5. Raw SDF
    vmax = max(abs(raw_sdf.min()), abs(raw_sdf.max()))
    im = axes[1, 1].imshow(raw_sdf, cmap='RdBu', vmin=-vmax, vmax=vmax)
    axes[1, 1].set_title('Raw SDF\n(red=inside +, blue=outside -)')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    # 6. Normalized SDF
    im = axes[1, 2].imshow(normalized, cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 2].set_title('Normalized SDF [-1, +1]\n(red=inside, blue=outside, white=boundary)')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    plt.savefig('sdf_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved to sdf_visualization.png")

    # Print stats
    print(f"\nSDF stats:")
    print(f"  Raw     — min: {raw_sdf.min():.2f}  max: {raw_sdf.max():.2f}")
    print(f"  Norm    — min: {normalized.min():.2f}  max: {normalized.max():.2f}")
    print(f"  Inside  pixels: {mask.sum()}")
    print(f"  Outside pixels: {(~mask).sum()}")
    print(f"  Boundary pixels (SDF=0): {(raw_sdf == 0).sum()}")



if __name__ == "__main__":
    train_grayscale()
