import os
import numpy as np
import torchvision
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader,Subset
from torchinfo import summary
import cv2
from input_pipeline import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_dilation
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
#import gudhi
from cldice_loss import *
from metrics import compute_betti_numbers
import collections
from thesis_visualisation import *


class simple_Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(simple_Conv, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=2)  # 1st layer
        self.conv_layer2 = nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=3)
        self.conv_layer3 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3)
        self.conv_layer4 = nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3)
        self.conv_layer5 = nn.Conv2d(128, 256, kernel_size=7, stride=1, padding=3)
        #self.conv_layer6 = nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer7 = nn.Conv2d(256, 64, kernel_size=5, stride=1, padding=2)
        self.conv_layer8 = nn.Conv2d(64, 16, kernel_size=5, stride=1, padding=2)
        self.output_layer = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

        self.dropout = nn.Dropout2d(p=0.2)
        self.relu_layer = nn.ReLU()
        self.sdf_head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x, avg_output, alpha):
        x1 = self.relu_layer(self.conv_layer1(x))
        x2 = self.dropout(self.relu_layer(self.conv_layer2(x1)))
        x3 = self.dropout(self.relu_layer(self.conv_layer3(x2)))
        x4 = self.dropout(self.relu_layer(self.conv_layer4(x3)))
        x5 = self.relu_layer(self.conv_layer5(x4))
        #x6 = self.relu_layer(self.conv_layer6(x5))
        x7 = self.relu_layer(self.conv_layer7(x5))
        x8 = self.relu_layer(self.conv_layer8(x7))
        conv_output = self.output_layer(x8)
        #output = conv_output
        output = alpha * conv_output + avg_output

        #sdf_output = self.sdf_head(x5)
        return output


def build_dataset_items(data_directory):
    batches_list , items = [], []
    for dirpath, dirnames, filenames in os.walk(data_directory):
        if any(f.endswith(".png") for f in filenames):                              # a leaf that holds data = a sub-batch
            rel = os.path.relpath(dirpath, data_directory).split(os.sep)
            view = os.path.join(*rel[:1])                                           # the main-batch
            items.append(dirpath)                                                   # Items have the full address of all batches from data directory to final leaf batch
            batches_list.append(view)
    uniq = {v: k for k, v in enumerate(sorted(set(batches_list)))}                  #Dictionary of Main batches as keys and a number as the value

    batches_list = np.array([uniq[v] for v in batches_list])                        # Convert the list of batch names (string) into number.
    assert len(items) > 0, f"no .png-containing folders under {data_directory}"
    return batches_list, items




def train_grayscale():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size=1

    data_dir = 'data/Bigger Batches of Data_original_size'
    ground_truth_dir = 'data/Edited Ground Truth_ Encoded'
    image_processing_dir = 'data/Image Processing Original Size Grayscale'
    batches_list, items = build_dataset_items(data_dir)                   # batches_list is the list of batches as main batch indices [7,7,11,11,11,11,....]
                                                                         # items is the list of  full address of each batch
    print('Total Sub Batches ', len(items))
    print("Number of Main Batches:", len(set(batches_list)))
    print("Sub-Batches per Main Batch:", collections.Counter(batches_list.tolist()))

    for fold in range(5):

        train_idx, val_idx, test_idx = grouped_split(batches_list, fold=fold, ratios=[0.7, 0.2, 0.1])

        print('Train Index', train_idx)
        print('Test Index', test_idx)
        print('Val Index', val_idx)

        model = simple_Conv(5, 3, kernel_size=3, stride=1, padding=1)
        model = model.to(device)
        # print(summary(model))

        train_dataset = TopoDatasetImg(items,train_idx,ground_truth_dir= ground_truth_dir, img_proc_dir = image_processing_dir, augment= True)
        val_dataset = TopoDatasetImg(items,val_idx,ground_truth_dir= ground_truth_dir, img_proc_dir = image_processing_dir, augment=False)
        test_dataset = TopoDatasetImg(items,test_idx,ground_truth_dir= ground_truth_dir, img_proc_dir = image_processing_dir,augment=False)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=4,pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=4,pin_memory=True)

        lr = 0.001
        epochs = 300

        optimizer1 = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay = 1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=300)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer1,gamma= 0.95)

        weights = torch.tensor([0.25, 0.35, 0.5])
        crossentropy_loss = torch.nn.CrossEntropyLoss(weight=weights).to(device)
        cl_dice = soft_dice_cldice()

        save_parent_dir = f'Residual Convolutional Network Results_original size_{fold}'
        best_loss = float('inf')

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
        #wandb.define_metric("*", step_metric="epoch")

        for epoch in range(epochs):
            
            model.train()
            print('Epoch: ',epoch, ' learning rate: ', lr)
            
            train_loss = 0.0
            train_cross_entropy_loss = 0.0
            train_dice_loss =0.0
            train_img_proc_dice_loss = 0.0
            train_batch = 0
            train_iou_dl = 0.0
            train_iou_img_prc = 0.0
            train_cl_dice_image_prc = 0.0
            train_cl_dice_dl = 0.0    
            
            if epoch < 150:             # For weighing ce_loss and sdf loss --- 150,250
                alpha = epoch / 150
            else:
                alpha =  1

            #alpha = epoch/epochs
            accum_steps = 4
            optimizer1.zero_grad()

            for i,( images, labels,avg_output) in enumerate(train_dataloader):

                images = images.to(device)
                labels = labels.to(device)
                avg_output = avg_output.to(device)
                avg_output_classes = avg_output.squeeze(1)
                labels = labels.squeeze(1)
                #print('Avg shape',avg_output.shape,'Labels', labels.shape)
                
                avg_output_one_hot = F.one_hot(avg_output_classes, num_classes=3).permute(0, 3, 1, 2).to(device)

                output_logits  = model(images,avg_output_one_hot,alpha)
                #output_logits = output_logits.unsqueeze(0)

                output_softmax = torch.softmax(output_logits, dim=1)
                output = torch.argmax(output_logits,dim=1)                          # Using argmax since the output is only raw logits and argmax makes you choose the maximum of the labels. So the 3 channels (3 classes) is converted to 1 channel the max value.
                targets_onehot = F.one_hot(labels, num_classes=3) # (B, H, W, C)
                targets_onehot = targets_onehot.permute(0, 3, 1, 2)  # (B, C, H, W)
                targets_onehot = targets_onehot.float()  # scatter_ gives long by default


                cross_entropy_loss = crossentropy_loss(output_logits, labels)
                #sdf_loss = F.l1_loss(sdf_pred, gt_sdf)
                dice_loss = cl_dice(targets_onehot,output_softmax,train_batch)
                img_processing_dice_loss = cl_dice(targets_onehot,avg_output_one_hot.float(),train_batch)
                
                loss = cross_entropy_loss +dice_loss
                train_img_proc_dice_loss += img_processing_dice_loss.item()
                train_dice_loss += dice_loss.item()

                (loss/accum_steps).backward()
                if (i + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer1.step()
                    optimizer1.zero_grad()

                # optimizer1.zero_grad()
                # loss.backward()
                # optimizer1.step()

                train_loss += loss.item()

                if epoch % 10 ==0:
                    if not os.path.exists(os.path.join(save_parent_dir,'train', 'Epoch '+str(epoch))):
                        os.mkdir(os.path.join(save_parent_dir,'train', 'Epoch '+str(epoch)))
                    if not os.path.exists(os.path.join(save_parent_dir, 'train', 'Epoch ' + str(epoch), 'IoU Metric')):
                        os.mkdir(os.path.join(save_parent_dir, 'train', 'Epoch ' + str(epoch), 'IoU Metric'))
                    if not os.path.exists(os.path.join(save_parent_dir, 'train', 'Epoch ' + str(epoch), 'sdf_map')):
                        os.mkdir(os.path.join(save_parent_dir, 'train', 'Epoch ' + str(epoch), 'sdf_map'))
                    
                    for i in range(len(output)):
                        #print('Unique Outputs',output.unique(return_counts=True), 'Unique Labels',labels.unique(return_counts=True))
                        # plt.imsave(f'{save_parent_dir}/train/Epoch {epoch}/label_{train_batch}.png', labels[i].cpu().detach().numpy(),cmap='gray',vmin=0,vmax=2)
                        # plt.close()
                        plt.imsave(f'{save_parent_dir}/train/Epoch {epoch}/result_{train_batch}.png',output[i].detach().cpu().numpy(),cmap='gray',vmin=0,vmax=2)
                        plt.close()
                        
                        predicted_binary = (output[i].detach().cpu().numpy() == 2).astype(np.float32)
                        gt_binary = (labels[i].detach().cpu().numpy() == 2).astype(np.float32)
                        img_prc_binary = (avg_output[i].squeeze(0).detach().cpu().numpy() == 2).astype(np.float32)
                        predicted_skeleton = skeletonize(predicted_binary > 0).astype(np.float32)
                        gt_skeleton = skeletonize(gt_binary > 0).astype(np.float32)
                        img_prc_skeleton = skeletonize(img_prc_binary > 0).astype(np.float32)

                        cl_prec = np.sum(predicted_skeleton * gt_binary) / (np.sum(predicted_skeleton) + 1e-7)
                        cl_rec = np.sum(gt_skeleton * predicted_binary) / (np.sum(gt_skeleton) + 1e-7)
                        cl_dice_dl = 2 * cl_prec * cl_rec / (cl_prec + cl_rec)

                        cl_prec_i = np.sum(img_prc_skeleton * gt_binary) / (np.sum(img_prc_skeleton) + 1e-7)
                        cl_rec_i = np.sum(gt_skeleton * img_prc_binary) / (np.sum(gt_skeleton) + 1e-7)
                        cl_dice_img_prc = 2 * cl_prec_i * cl_rec_i / (cl_prec_i + cl_rec_i)
                        
                        # b0_pred, b1_pred = compute_betti_numbers(output[i])
                        # b0_img_prc, b1_img_prc = compute_betti_numbers(avg_output[i].squeeze(0))
                        # b0_gt, b1_gt = compute_betti_numbers(labels[i])
                        # betti_error_pred = abs(b0_pred - b0_gt) + abs(b1_pred - b1_gt)
                        # betti_error_img_prc = abs(b0_img_prc - b0_gt) + abs(b1_img_prc - b1_gt)
                        # iou, img_prc_iou = visulaize_iou_metrics(raw_images=images[i], epoch=epoch, batch=train_batch,
                        #                                          predicted_structures=output[i],gt = labels[i],image_processing_result=avg_output[i],betti_error_pred=cl_dice_dl,betti_error_img_prc=cl_dice_img_prc,save_path= f'{save_parent_dir}/train')

                        # dl_sdf, normalised_dl_sdf= sdf_distance_function(output[i].detach().cpu().numpy())
                        # img_prc_sdf,normalised_img_prc_sdf = sdf_distance_function(avg_output[i].squeeze(0).detach().cpu().numpy())
                        # gt_sdf, normlaised_gt_sdf = sdf_distance_function(labels[i].detach().cpu().numpy())
                        # dl_sdf_mse = np.mean((dl_sdf - gt_sdf) ** 2)
                        # img_prc_sdf_mse = np.mean((img_prc_sdf - gt_sdf) ** 2)
                        # print(dl_sdf_mse.shape)
                        
                        iou,img_prc_iou =  visulaize_iou_metrics(raw_images=images[i], epoch=epoch, batch=train_batch, predicted_structures = output[i],
                                                                 gt = labels[i],image_processing_result=avg_output[i],betti_error_pred=cl_dice_dl,betti_error_img_prc=cl_dice_img_prc,save_path= f'{save_parent_dir}/train')
                        group = {
                            'inputs': images[i],
                            'gt': labels[i].detach().cpu().numpy(),
                            'pred': output[i].detach().cpu().numpy(),
                            'baseline': avg_output[i].squeeze(0).detach().cpu().numpy(),
                            'label': f'Batch {train_batch}',
                            # clDice already computed above — just pass them in
                            'cld_pred': cl_dice_dl,
                            'prec_pred': cl_prec,
                            'rec_pred': cl_rec,
                            'cld_base': cl_dice_img_prc,
                            'prec_base': cl_prec_i,
                            'rec_base': cl_rec_i,
                        }
                        visualize_comparison_with_overlay(
                            groups=[group],
                            structure_val=2,  # your label index for structure class
                            save_path=f'{save_parent_dir}/train/Epoch {epoch}/IoU Metric',
                            batch=train_batch
                        )
                        plot2_overlay(labels[i].detach().cpu().numpy(),output[i].detach().cpu().numpy(),avg_output[i].squeeze(0).detach().cpu().numpy(),(cl_dice_dl,cl_prec,cl_rec),
                                      (cl_dice_img_prc,cl_prec_i,cl_rec_i),save_path=f'{save_parent_dir}/train/Epoch {epoch}/IoU Metric',batch = train_batch)
                        
                        train_iou_dl += iou
                        train_iou_img_prc += img_prc_iou
                        train_cl_dice_image_prc += cl_dice_img_prc
                        train_cl_dice_dl += cl_dice_dl
                        # sdf_vis = sdf_pred[i].permute(1, 2, 0).detach().cpu().numpy()
                        # # vmax = max(abs(sdf_vis.min()), abs(sdf_vis.max()))
                        # fig, ax = plt.subplots(figsize=(6, 5))
                        # im = ax.imshow(sdf_vis, cmap='RdBu', vmin=-1, vmax=1)
                        # cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        # plt.savefig(os.path.join(save_parent_dir, 'train', 'Epoch ' + str(epoch), 'sdf_map',
                        #                          f'sdf_{train_batch}'))
                        # plt.close(fig)
                        train_batch += 1

            scheduler.step()
            print(f'Train Loss for Epoch {epoch}', train_loss / len(train_dataloader))
            print('Train IoU for deep learning', train_iou_dl / len(train_idx))
            print('Train IoU for image processing ', train_iou_img_prc / len(train_idx))

            # Starting a val loop
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    val_cross_entropy_loss = 0.0
                    val_dice_loss = 0.0
                    val_sdf_loss = 0.0
                    val_batch = 0
                    val_iou_dl = 0.0
                    val_iou_img_prc = 0.0
                    val_img_proc_dice_loss = 0.0
                    val_cl_dice_image_prc = 0.0
                    val_cl_dice_dl = 0.0

                    for images, labels, avg_output in val_dataloader:
                        images = images.to(device)
                        labels = labels.to(device)
                        avg_output = avg_output.to(device)
                        labels = labels.squeeze(1)
                        avg_output_classes = avg_output.squeeze(1)
                        

                        avg_output_one_hot = F.one_hot(avg_output_classes, num_classes=3).permute(0, 3, 1, 2).to(device)

                        output_logits = model(images, avg_output_one_hot, alpha)
                        output_softmax = torch.softmax(output_logits.detach(), dim=1)
                        output = torch.argmax(output_logits,
                                              dim=1)  # Using argmax since the output is only raw logits and argmax makes you choose the maximum of the labels. So the 3 channels (3 classes) is converted to 1 channel the max value.
                        
                        targets_onehot = F.one_hot(labels, num_classes=3)  # (B, H, W, C)
                        targets_onehot = targets_onehot.permute(0, 3, 1, 2)  # (B, C, H, W)
                        targets_onehot = targets_onehot.float()
                        # print('One Hot Targets', targets_onehot.shape, 'Unique Values',
                        #      targets_onehot.unique(return_counts=True))

                        
                        cross_entropy_loss = crossentropy_loss(output_logits, labels)

                        #sdf_loss = F.l1_loss(sdf_pred, gt_sdf)
                        dice_loss = cl_dice(targets_onehot,output_softmax,val_batch)
                        img_processing_dice_loss = cl_dice(targets_onehot,avg_output_one_hot.float(),val_batch)
                        
                        loss = cross_entropy_loss + dice_loss

                        val_img_proc_dice_loss += img_processing_dice_loss
                        # val_cross_entropy_loss += cross_entropy_loss.item()
                        #val_sdf_loss = sdf_loss.item()
                        val_dice_loss +=  dice_loss
                        val_loss += loss.item()

                        # if epoch % 10 == 0:
                        if not os.path.exists(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch))):
                            os.mkdir(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch)))
                        if not os.path.exists(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch), 'sdf_map')):
                            os.mkdir(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch), 'sdf_map'))
                        if not os.path.exists(
                                os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch), 'IoU Metric')):
                            os.mkdir(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch), 'IoU Metric'))
                        for i in range(len(output)):
                            plt.axis('off')
                            plt.imsave(f'{save_parent_dir}/val/Epoch {epoch}/result_{val_batch}.png',
                                       output[i].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=2)
                            plt.close()
                            
                            predicted_binary = (output[i].detach().cpu().numpy() == 2).astype(np.float32)
                            gt_binary = (labels[i].detach().cpu().numpy() == 2).astype(np.float32)
                            img_prc_binary = (avg_output[i].squeeze(0).detach().cpu().numpy()==2).astype(np.float32)
                            predicted_skeleton = skeletonize(predicted_binary > 0).astype(np.float32)
                            gt_skeleton = skeletonize(gt_binary > 0).astype(np.float32)
                            img_prc_skeleton = skeletonize(img_prc_binary > 0).astype(np.float32)

                            cl_prec = np.sum(predicted_skeleton * gt_binary) / (np.sum(predicted_skeleton) + 1e-7)
                            cl_rec = np.sum(gt_skeleton * predicted_binary) / (np.sum(gt_skeleton) + 1e-7)
                            cl_dice_dl = 2 * cl_prec * cl_rec / (cl_prec + cl_rec)

                            cl_prec_i = np.sum(img_prc_skeleton * gt_binary) / (np.sum(img_prc_skeleton) + 1e-7)
                            cl_rec_i = np.sum(gt_skeleton * img_prc_binary) / (np.sum(gt_skeleton) + 1e-7)
                            cl_dice_img_prc = 2 * cl_prec_i * cl_rec_i / (cl_prec_i + cl_rec_i)
                            # b0_pred, b1_pred = compute_betti_numbers(output[i])
                            # b0_img_prc, b1_img_prc = compute_betti_numbers(avg_output[i])
                            # b0_gt, b1_gt = compute_betti_numbers(labels[i])
                            # betti_error_pred = abs(b0_pred - b0_gt) + abs(b1_pred - b1_gt)
                            # betti_error_img_prc = abs(b0_img_prc - b1_gt) + abs(b1_img_prc - b1_gt)
                            # dl_sdf = sdf_distance_function(output[i].detach().cpu().numpy())
                            # img_prc_sdf = sdf_distance_function(avg_output[i].squeeze(0).detach().cpu().numpy())
                            # gt_sdf = sdf_distance_function(labels[i].detach().cpu().numpy())
                            # dl_sdf_mse = np.mean((dl_sdf - gt_sdf) ** 2)
                            # img_prc_sdf_mse = np.mean((img_prc_sdf - gt_sdf) ** 2)
                            iou_dl, iou_img_prc = visulaize_iou_metrics(raw_images=images[i], epoch=epoch,
                                                                         batch=val_batch,
                                                                         predicted_structures=output[i],
                                                                         gt=labels[i],
                                                                         image_processing_result=avg_output[i],
                                                                         betti_error_pred=cl_dice_dl,
                                                                         betti_error_img_prc=cl_dice_img_prc,
                                                                         save_path=f'{save_parent_dir}/val')
                            group = {
                                'inputs': images[i],
                                'gt': labels[i].detach().cpu().numpy(),
                                'pred': output[i].detach().cpu().numpy(),
                                'baseline': avg_output[i].squeeze(0).detach().cpu().numpy(),
                                'label': f'Batch {train_batch}',
                                # clDice already computed above — just pass them in
                                'cld_pred': cl_dice_dl,
                                'prec_pred': cl_prec,
                                'rec_pred': cl_rec,
                                'cld_base': cl_dice_img_prc,
                                'prec_base': cl_prec_i,
                                'rec_base': cl_rec_i,
                            }
                            visualize_comparison_with_overlay(
                                groups=[group],
                                structure_val=2,  # your label index for structure class
                                save_path=f'{save_parent_dir}/val/Epoch {epoch}/IoU Metric',
                                batch=val_batch
                            )
                            plot2_overlay(labels[i].detach().cpu().numpy(), output[i].detach().cpu().numpy(),
                                          avg_output[i].squeeze(0).detach().cpu().numpy(),
                                          (cl_dice_dl, cl_prec, cl_rec),
                                          (cl_dice_img_prc, cl_prec_i, cl_rec_i),
                                          save_path=f'{save_parent_dir}/val/Epoch {epoch}/IoU Metric',
                                          batch=val_batch)
                            val_iou_dl += iou_dl
                            val_iou_img_prc += iou_img_prc
                            val_cl_dice_image_prc +=cl_dice_img_prc
                            val_cl_dice_dl += cl_dice_dl

                            # sdf_vis = sdf_pred[i].permute(1, 2, 0).detach().cpu().numpy()
                            # # vmax = max(abs(sdf_vis.min()), abs(sdf_vis.max()))
                            # fig, ax = plt.subplots(figsize=(6, 5))
                            # im = ax.imshow(sdf_vis, cmap='RdBu', vmin=-1, vmax=1)
                            # cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                            # plt.savefig(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch), 'sdf_map',
                            #                          f'sdf_{val_batch}'))
                            # plt.close(fig)
                            val_batch += 1

            print(f'Val Loss for Epoch {epoch}', val_loss / len(val_dataloader))
            print('Val IoU for deep learning', val_iou_dl / len(val_idx))
            print('Val IoU for image processing ', val_iou_img_prc / len(val_idx))

            # Starting a test loop
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    test_loss = 0.0
                    test_cross_entropy_loss = 0.0
                    test_dice_loss = 0.0
                    test_sdf_loss = 0.0
                    test_batch = 0
                    test_iou_dl = 0.0
                    test_iou_img_prc = 0.0
                    test_cl_dice_img_proc = 0.0
                    test_cl_dice_dl = 0.0
                    test_image_processing_dice_loss = 0.0
                
                    
                    for images, labels, avg_output in test_dataloader:
                        images = images.to(device)
                        labels = labels.to(device)
                        avg_output = avg_output.to(device)
                        labels = labels.squeeze(1)
                        avg_output_classes = avg_output.squeeze(1)
                        #gt_sdf = sdf.to(device)
                        # avg_output_classes = (avg_output *2).round().long().squeeze(1)

                        avg_output_one_hot = F.one_hot(avg_output_classes, num_classes=3).permute(0, 3, 1, 2).to(device)

                        output_logits = model(images, avg_output_one_hot.float(), alpha)
                        #output_logits = output_logits.unsqueeze(0)
                        # output_softmax = torch.softmax(output_logits.detach(), dim=1)[:,2]
                        output_softmax = torch.softmax(output_logits.detach(), dim=1)
                        output = torch.argmax(output_logits,
                                              dim=1)  # Using argmax since the output is only raw logits and argmax makes you choose the maximum of the labels. So the 3 channels (3 classes) is converted to 1 channel the max value.
                        targets_onehot = F.one_hot(labels, num_classes=3)  # (B, H, W, C)
                        targets_onehot = targets_onehot.permute(0, 3, 1, 2)  # (B, C, H, W)
                        targets_onehot = targets_onehot.float()
                        # print('One Hot Targets', targets_onehot.shape, 'Unique Values',
                        #       targets_onehot.unique(return_counts=True))

                        # print('Labels dtype',labels.dtype)
                        cross_entropy_loss = crossentropy_loss(output_logits, labels)

                        #sdf_loss = F.l1_loss(sdf_pred, gt_sdf)
                        dice_loss = cl_dice(targets_onehot,output_softmax,test_batch)
                        image_processing_dice_loss = cl_dice(targets_onehot,avg_output_one_hot.float(),test_batch)
                        # boundary_loss = sdf_loss(output_softmax, gt_sdf_white)
                        # loss = cross_entropy_loss + sdf_loss
                        loss = cross_entropy_loss + dice_loss
                        test_image_processing_dice_loss += image_processing_dice_loss
                        # test_cross_entropy_loss += cross_entropy_loss.item()
                        #test_sdf_loss += sdf_loss.item()
                        test_dice_loss +=  dice_loss
                        test_loss += loss.item()

                        # if epoch %10 == 0:
                        if not os.path.exists(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch))):
                            os.mkdir(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch)))
                        if not os.path.exists(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch), 'sdf_map')):
                            os.mkdir(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch), 'sdf_map'))
                        if not os.path.exists(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch), 'IoU Metric')):
                            os.mkdir(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch), 'IoU Metric'))
                        for i in range(len(output)):
                            plt.axis('off')
                            plt.imsave(f'{save_parent_dir}/test/Epoch {epoch}/result_{test_batch}.png',
                                       output[i].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=2)
                            plt.close()
                            predicted_binary = (output[i].detach().cpu().numpy() == 2).astype(np.float32)
                            gt_binary = (labels[i].detach().cpu().numpy() == 2).astype(np.float32)
                            img_prc_binary = (avg_output[i].squeeze(0).detach().cpu().numpy()==2).astype(np.float32)
                            predicted_skeleton = skeletonize(predicted_binary > 0).astype(np.float32)
                            gt_skeleton = skeletonize(gt_binary > 0).astype(np.float32)
                            img_prc_skeleton = skeletonize(img_prc_binary > 0).astype(np.float32)



                            cl_prec = np.sum(predicted_skeleton * gt_binary) / (np.sum(predicted_skeleton) + 1e-7)
                            cl_rec = np.sum(gt_skeleton * predicted_binary) / (np.sum(gt_skeleton) + 1e-7)
                            cl_dice_dl = 2 * cl_prec * cl_rec / (cl_prec + cl_rec)

                            cl_prec_i = np.sum(img_prc_skeleton * gt_binary) / (np.sum(img_prc_skeleton) + 1e-7)
                            cl_rec_i = np.sum(gt_skeleton * img_prc_binary) / (np.sum(gt_skeleton) + 1e-7)
                            cl_dice_img_prc = 2 * cl_prec_i * cl_rec_i / (cl_prec_i + cl_rec_i)

                            # b0_pred, b1_pred = compute_betti_numbers(output[i])
                            # b0_img_prc, b1_img_prc = compute_betti_numbers(avg_output[i])
                            # b0_gt, b1_gt = compute_betti_numbers(labels[i])
                            # betti_error_pred = abs(b0_pred - b0_gt) + abs(b1_pred - b1_gt)
                            # betti_error_img_prc = abs(b0_img_prc - b1_gt) + abs(b1_img_prc - b1_gt)
                            # dl_sdf = sdf_distance_function(output[i].detach().cpu().numpy())
                            # img_prc_sdf = sdf_distance_function(avg_output[i].squeeze(0).detach().cpu().numpy())
                            # gt_sdf = sdf_distance_function(labels[i].detach().cpu().numpy())
                            # dl_sdf_mse = np.mean((dl_sdf - gt_sdf) ** 2)
                            # img_prc_sdf_mse = np.mean((img_prc_sdf - gt_sdf) ** 2)
                            group = {
                                'inputs': images[i],
                                'gt': labels[i].detach().cpu().numpy(),
                                'pred': output[i].detach().cpu().numpy(),
                                'baseline': avg_output[i].squeeze(0).detach().cpu().numpy(),
                                'label': f'Batch {train_batch}',
                                # clDice already computed above — just pass them in
                                'cld_pred': cl_dice_dl,
                                'prec_pred': cl_prec,
                                'rec_pred': cl_rec,
                                'cld_base': cl_dice_img_prc,
                                'prec_base': cl_prec_i,
                                'rec_base': cl_rec_i,
                            }
                            visualize_comparison_with_overlay(
                                groups=[group],
                                structure_val=2,  # your label index for structure class
                                save_path=f'{save_parent_dir}/test/Epoch {epoch}/IoU Metric',
                                batch=test_batch
                            )
                            plot2_overlay(labels[i].detach().cpu().numpy(), output[i].detach().cpu().numpy(),
                                          avg_output[i].squeeze(0).detach().cpu().numpy(),
                                          (cl_dice_dl, cl_prec, cl_rec),
                                          (cl_dice_img_prc, cl_prec_i, cl_rec_i),
                                          save_path=f'{save_parent_dir}/test/Epoch {epoch}/IoU Metric',
                                          batch=test_batch)
                            iou_dl, iou_img_prc = visulaize_iou_metrics(raw_images=images[i], epoch=epoch,
                                                                        batch=test_batch,
                                                                        predicted_structures=output[i], gt=labels[i],
                                                                        image_processing_result=avg_output[i],
                                                                        betti_error_pred=cl_dice_dl,
                                                                        betti_error_img_prc=cl_dice_img_prc,
                                                                        save_path=f'{save_parent_dir}/test')
                            test_iou_dl += iou_dl
                            test_iou_img_prc += iou_img_prc
                            test_cl_dice_dl += cl_dice_dl
                            test_cl_dice_img_proc += cl_dice_img_prc
                            # sdf_vis = sdf_pred[i].permute(1, 2, 0).detach().cpu().numpy()
                            # # vmax = max(abs(sdf_vis.min()), abs(sdf_vis.max()))
                            # fig, ax = plt.subplots(figsize=(6, 5))
                            # im = ax.imshow(sdf_vis, cmap='RdBu', vmin=-1, vmax=1)
                            # cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                            # plt.savefig(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch), 'sdf_map',
                            #                          f'sdf_{test_batch}'))
                            # plt.close(fig)
                            test_batch += 1

                print(f'Test Loss for Epoch {epoch}', test_loss / len(test_dataloader))
                print('Test IoU for deep learning', test_iou_dl / len(test_idx))
                print('Test IoU for image processing ', test_iou_img_prc / len(test_idx))

                if test_loss / len(test_dataloader) < best_loss:
                    best_loss = test_loss / len(test_dataloader)
                    torch.save(model,'./convolutional_network_chkpoint/convolutional_chkpoint.pth')
                    print(f'Model Saved with test loss {test_loss / len(test_dataloader):.4f}')

                run.log({'epoch': epoch,
                        'alpha': alpha,
                        # 'learning_rate': scheduler.get_last_lr()[0],
                        'Train_Total_Loss': train_loss / len(train_dataloader),
                        'Test_Total_Loss': test_loss / len(test_dataloader),
                        'Val Loss': val_loss / len(val_dataloader),
                        # 'Train_CE_Loss': train_cross_entropy_loss / len(train_dataloader),'Test_CE_Loss': test_cross_entropy_loss / len(test_dataloader),
                        'Train IoU': train_iou_dl / len(train_idx),
                        'Train IoU_img': train_iou_img_prc / len(train_idx),
                        'Test IoU': test_iou_dl / len(test_idx),
                        'Test IoU_img': test_iou_img_prc / len(test_idx),
                        'Val IoU': val_iou_dl / len(val_idx), 'Val IoU img': val_iou_img_prc / len(val_idx),
                        'Train_Img_Processing_Dice Loss': train_img_proc_dice_loss/len(train_dataloader),
                        'Val_Img_Proccessing_Dice Loss': val_img_proc_dice_loss/len(val_dataloader),
                        'Test_Img_Processing_Dice Loss': test_image_processing_dice_loss/len(test_dataloader),
                        'Train Dice Loss': train_dice_loss / len(train_dataloader), 'Test_Dice_Loss': test_dice_loss / len(test_dataloader),
                        'Val Dice Loss': val_dice_loss/len(val_dataloader),

                        'Train_Img_Processing_Cldice': train_cl_dice_image_prc / len(train_idx),
                        'Val_Img_Proccessing_Cldice': val_cl_dice_image_prc / len(val_idx),
                        'Test_Img_Processing_Cldice': test_cl_dice_img_proc / len(test_idx),
                        'Train Cldice DL': train_cl_dice_dl / len(train_idx),
                        'Test Cldice DL': test_cl_dice_dl / len(test_idx),
                        'Val Cldice DL': val_cl_dice_dl / len(val_idx),
                        })

        wandb.finish()

def extract_red_mask(image):
    #print(image.min(),image.max())
    R = image[:,:,0]
    # G = image[:,:,1]
    # B = image[:,:,2]
    mask = (R ==2)
    return mask.astype(float)

def convert_labels(input):

        input_label = input
        #print('Before decoding',torch.unique(input,return_counts = True))
        input[input_label == 0.0] = 0
        input[input_label == 1.0] = 2
        input[(input_label >= 0.25) & (input < 0.75)] =1  #since floating point precision might vary
        #print('After decoding',torch.unique(input, return_counts=True))
        return input

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

    H, W = raw_images.shape[-2], raw_images.shape[-1]
    ar = H / W
    fig_w = 16
    fig_h = fig_w * (2 / 5) * ar + 0.9  # +0.9" headroom for the row-1 titles
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    grid_spec = fig.add_gridspec(2, 5, hspace=0.05, wspace=0.2)

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
    ax.set_title('Predicted Structure')
    ax.axis('off')

    ax = fig.add_subplot(grid_spec[1, 2])
    ax.imshow(overlap)
    ax.set_title(f'IOU: {iou_dilated:0.2f}  \n SDF MSE: {betti_error_pred}')
    ax.axis('off')

    ax = fig.add_subplot(grid_spec[1, 3])
    #print('Image Processing Shape',image_processing_result.shape)
    ax.imshow(image_processing_result,cmap='gray')
    ax.set_title(f'Image Processing Result')
    ax.axis('off')

    ax = fig.add_subplot(grid_spec[1, 4])
    ax.imshow(img_proc_dilated_overlap)
    ax.set_title(f'IOU: {iou_dilated_image_processing:0.2f} \n Img Proc SDF MSE: {betti_error_img_prc} ')
    ax.axis('off')

    # Single shared legend for the entire bottom row
    legend_patches = [
        mpatches.Patch(color='green', label='True Positive'),
        mpatches.Patch(color='blue', label='False Negative'),
        mpatches.Patch(color='red', label='False Positive'),
    ]
    fig.legend(
        handles=legend_patches,
        loc='lower center',
        ncol=3,  # all three side by side in one row
        fontsize=10,
        framealpha=0.6,
        bbox_to_anchor=(0.5, 0.01),  # centred horizontally, just below the figure
    )
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
            #print(f'Saved sdf_epoch_{epoch:04d}.png')
            break   # only first batch





def visulaise_sdf_images(normalised_dl_sdf, normalised_img_prc_sdf, normlaised_gt_sdf):
    fig, axes = plt.subplots(1, 3, figsize=(14, 8))
    #fig.suptitle('SDF Step by Step', fontsize=14)

    # 1. Label map
    # axes[0, 0].imshow(label_map, cmap='gray', vmin=0, vmax=2)
    # axes[0, 0].set_title('Label map\n(0=black, 1=gray, 2=white)')
    # axes[0, 0].axis('off')
    #
    # # 2. Binary mask
    # axes[0, 1].imshow(mask, cmap='gray')
    # axes[0, 1].set_title('Binary mask\n(white class only)')
    # axes[0, 1].axis('off')
    #
    # # 3. dist_inside
    # im = axes[0, 2].imshow(dist_in, cmap='hot')
    # axes[0, 2].set_title('dist_inside\n(0 outside, +d inside)')
    # axes[0, 2].axis('off')
    # plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
    #
    # # 4. dist_outside
    # im = axes[1, 0].imshow(dist_out, cmap='hot')
    # axes[1, 0].set_title('dist_outside\n(0 inside, +d outside)')
    # axes[1, 0].axis('off')
    # plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    # 5. Raw SDF
    # vmax = max(abs(raw_sdf.min()), abs(raw_sdf.max()))
    # im = axes[1, 1].imshow(raw_sdf, cmap='seismic', vmin=-vmax, vmax=vmax)
    # axes[1, 1].set_title('Raw SDF\n(red=inside +, blue=outside -)')
    # axes[1, 1].axis('off')
    # plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    # 6. Normalized SDF
    im = axes[0, 1].imshow(normalised_dl_sdf, cmap='seismic', vmin=-1, vmax=1)
    axes[0, 1].set_title('Normalized Predicted SDF [-1, +1]\n')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    im = axes[0, 2].imshow(normalised_img_prc_sdf, cmap='seismic', vmin=-1, vmax=1)
    axes[0, 2].set_title('Normalized Image Processing SDF [-1, +1]\n')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0,2], fraction=0.046)

    im = axes[0, 0].imshow(normlaised_gt_sdf, cmap='seismic', vmin=-1, vmax=1)
    axes[0, 0].set_title('Normalized gt SDF [-1, +1]\n')
    axes[0, 0].axis('off')
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)

    # im = axes[0, 1].imshow(normalised_dl_sdf, cmap='seismic', vmin=-1, vmax=1)
    # axes[0, 1].set_title('Normalized Predicted SDF [-1, +1]\n')
    # axes[0, 1].axis('off')
    # plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    plt.tight_layout()
    plt.savefig('sdf_visualization2.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved to sdf_visualization.png")

    # Print stats
    # print(f"\nSDF stats:")
    # print(f"  Raw     — min: {raw_sdf.min():.2f}  max: {raw_sdf.max():.2f}")
    # print(f"  Norm    — min: {normalized.min():.2f}  max: {normalized.max():.2f}")
    # print(f"  Inside  pixels: {mask.sum()}")
    # print(f"  Outside pixels: {(~mask).sum()}")
    # print(f"  Boundary pixels (SDF=0): {(raw_sdf == 0).sum()}")


def visulaise_softmax(argmax,probs_np):
    class_names = {0: "black", 1: "gray", 2: "white"}  # adjust if needed
    fig, axes = plt.subplots(1, 3 + 1, figsize=(4 * (3 + 1), 4))
    im0 = axes[0].imshow(argmax, cmap="tab10", vmin=0, vmax=3 - 1)
    axes[0].set_title("Argmax (predicted class)")
    plt.colorbar(im0, ax=axes[0])

    # --- Per-class softmax probability maps ---
    for c in range(3):
        im = axes[c + 1].imshow(probs_np[c], cmap="viridis", vmin=0, vmax=1)
        axes[c + 1].set_title(f"Class {c} ({class_names.get(c, c)}) prob")
        plt.colorbar(im, ax=axes[c + 1])

    plt.tight_layout()
    plt.show()

    # --- Also print numeric summary per class ---
    print("Softmax probability stats per class:")
    for c in range(3):
        p = probs_np[c]
        print(f"  Class {c} ({class_names.get(c, c)}): "
              f"mean={p.mean():.4f}, max={p.max():.4f}, "
              f"min={p.min():.4f}, pixels>{0.5:.0f}: {(p > 0.5).sum()}")

    print(f"\nArgmax class distribution:")
    for c in range(3):
        count = (argmax == c).sum()
        print(f"  Class {c} ({class_names.get(c, c)}): {count} pixels "
              f"({100 * count / argmax.size:.1f}%)")


if __name__ == "__main__":
    train_grayscale()
