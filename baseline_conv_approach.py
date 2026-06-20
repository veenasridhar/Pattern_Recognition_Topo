import os
import numpy as np
import torchvision
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Subset
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
import collections


class simple_Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(simple_Conv, self).__init__()

        # self.conv_layer1 = nn.Conv2d(in_channels, 16, kernel_size, stride, padding)  # 1st layer
        # self.conv_layer2 = nn.Conv2d(16, 64, kernel_size=5, stride=1, padding=2)
        # self.conv_layer3 = nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3)
        # self.conv_layer4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        # self.conv_layer5 = nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1)
        # self.output_layer = nn.Conv2d(16, 3, kernel_size=1)  # Output Channels is 3 for 3 classes in the labels.

        self.conv_layer1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)  # 1st layer
        self.conv_layer2 = nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=3)
        self.conv_layer3 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3)
        self.conv_layer4 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer5 = nn.Conv2d(128, 32, kernel_size=5, stride=1, padding=2)
        self.conv_layer6 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.output_layer = nn.Conv2d(16, 3, kernel_size=1)

        self.relu_layer = nn.ReLU()
        self.sdf_head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x, avg_output, alpha):
        x1 = self.relu_layer(self.conv_layer1(x))
        x2 = self.relu_layer(self.conv_layer2(x1))
        x3 = self.relu_layer(self.conv_layer3(x2))
        x4 = self.relu_layer(self.conv_layer4(x3))
        x5 = self.relu_layer(self.conv_layer5(x4))
        x6 = self.relu_layer(self.conv_layer6(x5))
        conv_output = self.output_layer(x6)
        output = alpha * conv_output + avg_output
        #output = conv_output

        sdf_output = self.sdf_head(x6)
        #sdf_final_output = (alpha * sdf_output + avg_sdf)/2
        return output, sdf_output

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

    batch_size=4

    data_dir = 'data/Final Data/Bigger Batches of Data'
    ground_truth_dir = 'data/Final Data/gt_grayscale_resized'
    image_processing_dir = 'data/Final Data/image processing resized'
    batches_list, items = build_dataset_items(data_dir)  # batches_list is the list of batches as main batch indices [7,7,11,11,11,11,....]
    # items is the list of  full address of each batch
    print('Total Sub Batches ', len(items))
    print("number of views:", len(set(batches_list)))
    print("sub-batches per view:", collections.Counter(batches_list.tolist()))


    for fold in range(5):

        train_idx, val_idx, test_idx = grouped_split(batches_list, fold, ratios=[0.7, 0.2, 0.1])

        print('Train Index', train_idx)
        print('Test Index', test_idx)
        print('Val Index', val_idx)

        model = simple_Conv(5, 3, kernel_size=3, stride=1, padding=1)
        model = model.to(device)
        # print(summary(model))

        train_dataset = TopoDatasetImg(items, train_idx, ground_truth_dir,image_processing_dir, augment=True)
        val_dataset = TopoDatasetImg(items, val_idx, ground_truth_dir,image_processing_dir, augment=False)
        test_dataset = TopoDatasetImg(items, test_idx, ground_truth_dir,image_processing_dir, augment=False)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        lr = 0.0003
        epochs = 1000

        optimizer1 = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1,T_max=1000, eta_min=1e-5)

        weights = torch.tensor([0.25, 0.35, 0.5])
        crossentropy_loss = torch.nn.CrossEntropyLoss().to(device)
        cl_dice = soft_dice_cldice()
        save_parent_dir = 'Residual Convolutional Network Results'

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
                'architecture': 'Simple Convolution (Residual) CE+SDF',
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
            train_sdf_loss = 0.0
            train_batch = 0
            train_iou_dl = 0.0
            train_iou_img_prc = 0.0
            train_img_proc_dice_loss = 0.0
            if epoch < 200:
                alpha = epoch / 200
            else:
                alpha =  1
            accum_steps = 4

            for i,(images, labels,avg_output) in enumerate(train_dataloader):

                images = images.to(device)
                labels = labels.to(device)
                avg_output = avg_output.to(device)

                avg_output_classes = avg_output.squeeze(1)                              # [B, H, W] - Labels - 0,1,2
                #gt_sdf = sdf.to(device)
                #avg_output_sdf = avg_output_sdf.to(device)
                labels = labels.squeeze(1)                                              # [B, H,W] - Labels - 0,1,2
                avg_output_one_hot = F.one_hot(avg_output_classes, num_classes=3).permute(0, 3, 1, 2).to(device)

                #output_logits, sdf_pred  = model(images,avg_output_one_hot,avg_output_sdf, alpha)
                output_logits, sdf_pred = model(images, avg_output_one_hot, alpha)
                output_softmax = torch.softmax(output_logits, dim=1)
                output = torch.argmax(output_logits,dim=1)                              # Using argmax since the output is only raw logits and argmax makes you choose the maximum of the labels. So the 3 channels (3 classes) is converted to 1 channel the max value.

                targets_onehot = F.one_hot(labels, num_classes=3)  # (B, H, W, C)
                targets_onehot = targets_onehot.permute(0, 3, 1, 2)  # (B, C, H, W)
                targets_onehot = targets_onehot.float()  # scatter_ gives long by default

                cross_entropy_loss = crossentropy_loss(output_logits, labels)
                #sdf_loss = F.l1_loss(sdf_pred, gt_sdf)
                dice_loss = cl_dice(targets_onehot,output_softmax,train_batch)
                image_processing_dice_loss = cl_dice(targets_onehot,avg_output_one_hot.float(),train_batch)
                #loss = (1-alpha) * cross_entropy_loss + alpha * sdf_loss
                #loss = cross_entropy_loss + sdf_loss +dice_loss
                loss = cross_entropy_loss + dice_loss

                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()

                train_cross_entropy_loss += cross_entropy_loss.item()
                #train_sdf_loss += sdf_loss.item()
                train_dice_loss += dice_loss.item()
                train_loss += loss.item()
                train_img_proc_dice_loss += image_processing_dice_loss.item()


                if epoch % 10 ==0:
                    if not os.path.exists(os.path.join(save_parent_dir,'train', 'Epoch '+str(epoch))):
                        os.mkdir(os.path.join(save_parent_dir,'train', 'Epoch '+str(epoch)))
                    if not os.path.exists(os.path.join(save_parent_dir, 'train', 'Epoch ' + str(epoch), 'sdf_map')):
                        os.mkdir(os.path.join(save_parent_dir, 'train', 'Epoch ' + str(epoch), 'sdf_map'))

                    for i in range(len(output)):
                        plt.imsave(f'{save_parent_dir}/train/Epoch {epoch}/result_{train_batch}.png',output[i].detach().cpu().numpy(),cmap='gray',vmin=0,vmax=2)
                        plt.close()

                        b0_pred, b1_pred = compute_betti_numbers(output[i])
                        b0_img_prc, b1_img_prc = compute_betti_numbers(avg_output[i].squeeze(0))
                        b0_gt, b1_gt = compute_betti_numbers(labels[i])
                        betti_error_pred = abs(b0_pred - b0_gt) + abs(b1_pred - b1_gt)
                        betti_error_img_prc = abs(b0_img_prc - b0_gt) + abs(b1_img_prc - b1_gt)

                        iou,img_prc_iou =  visulaize_iou_metrics(raw_images=images[i], epoch=epoch, batch=train_batch, predicted_structures = output[i], gt = labels[i],image_processing_result=avg_output[i],betti_error_pred=betti_error_pred,betti_error_img_prc=betti_error_img_prc,save_path= f'{save_parent_dir}/train')
                        train_iou_dl += iou
                        train_iou_img_prc += img_prc_iou

                        # sdf_vis = sdf_pred[i].permute(1, 2, 0).detach().cpu().numpy()
                        # # vmax = max(abs(sdf_vis.min()), abs(sdf_vis.max()))
                        # fig, ax = plt.subplots(figsize=(6, 5))
                        # im = ax.imshow(sdf_vis, cmap='seismic', vmin=-1, vmax=1)
                        # cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        # plt.savefig(os.path.join(save_parent_dir, 'train', 'Epoch ' + str(epoch), 'sdf_map',
                        #                          f'sdf_{train_batch}'))
                        # plt.close(fig)
                        train_batch += 1


            print(f'Train Loss for Epoch {epoch}', train_loss / len(train_dataloader))
            print('Train IoU for deep learning', train_iou_dl / len(train_idx))
            print('Train IoU for image processing ', train_iou_img_prc / len(train_idx))

            scheduler.step()

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
                    val_image_processing_dice_loss = 0.0

                    for images, labels, avg_output in val_dataloader:
                        images = images.to(device)
                        labels = labels.to(device)
                        avg_output = avg_output.to(device)
                        #avg_output_sdf = avg_output_sdf.to(device)

                        labels = labels.squeeze(1)
                        avg_output_classes = avg_output.squeeze(1)
                        #gt_sdf = sdf.to(device)
                        avg_output_one_hot = F.one_hot(avg_output_classes, num_classes=3).permute(0, 3, 1, 2).to(device)

                        output_logits, sdf_pred = model(images, avg_output_one_hot, alpha)
                        output_softmax = torch.softmax(output_logits.detach(), dim=1)
                        output = torch.argmax(output_logits,dim=1)  # Using argmax since the output is only raw logits and argmax makes you choose the maximum of the labels. So the 3 channels (3 classes) is converted to 1 channel the max value.
                        targets_onehot = F.one_hot(labels, num_classes=3)  # (B, H, W, C)
                        targets_onehot = targets_onehot.permute(0, 3, 1, 2)  # (B, C, H, W)
                        targets_onehot = targets_onehot.float()



                        cross_entropy_loss = crossentropy_loss(output_logits, labels)
                        #sdf_loss = F.l1_loss(sdf_pred, gt_sdf)
                        dice_loss = cl_dice(targets_onehot,output_softmax,val_batch)
                        image_processing_dice_loss = cl_dice(targets_onehot,avg_output_one_hot.float(),val_batch)
                        #loss = cross_entropy_loss + sdf_loss + dice_loss
                        loss = cross_entropy_loss + dice_loss

                        val_cross_entropy_loss += cross_entropy_loss.item()
                        #val_sdf_loss += sdf_loss.item()
                        val_dice_loss +=  dice_loss.item()
                        val_loss += loss.item()
                        val_image_processing_dice_loss += image_processing_dice_loss.item()

                        # if epoch % 10 == 0:
                        if not os.path.exists(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch))):
                            os.mkdir(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch)))
                        if not os.path.exists(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch), 'sdf_map')):
                            os.mkdir(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch), 'sdf_map'))
                        for i in range(len(output)):
                            plt.axis('off')
                            plt.imsave(f'{save_parent_dir}/val/Epoch {epoch}/result_{val_batch}.png',
                                       output[i].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=2)
                            plt.close()
                            b0_pred, b1_pred = compute_betti_numbers(output[i])
                            b0_img_prc, b1_img_prc = compute_betti_numbers(avg_output[i])
                            b0_gt, b1_gt = compute_betti_numbers(labels[i])
                            betti_error_pred = abs(b0_pred - b0_gt) + abs(b1_pred - b1_gt)
                            betti_error_img_prc = abs(b0_img_prc - b0_gt) + abs(b1_img_prc - b1_gt)
                            iou_dl, iou_img_prc = visulaize_iou_metrics(raw_images=images[i], epoch=epoch,
                                                                        batch=val_batch,
                                                                        predicted_structures=output[i],
                                                                        gt=labels[i],
                                                                        image_processing_result=avg_output[i],
                                                                        betti_error_pred=betti_error_pred,
                                                                        betti_error_img_prc=betti_error_img_prc,
                                                                        save_path=f'{save_parent_dir}/val')
                            val_iou_dl += iou_dl
                            val_iou_img_prc += iou_img_prc

                            # sdf_vis = sdf_pred[i].permute(1, 2, 0).detach().cpu().numpy()
                            # # vmax = max(abs(sdf_vis.min()), abs(sdf_vis.max()))
                            # fig, ax = plt.subplots(figsize=(6, 5))
                            # im = ax.imshow(sdf_vis, cmap='seismic', vmin=-1, vmax=1)
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
                    test_image_processing_dice_loss = 0.0
                    for images, labels, avg_output in test_dataloader:
                        images = images.to(device)
                        labels = labels.to(device)
                        avg_output = avg_output.to(device)
                        labels = labels.squeeze(1)
                        avg_output_classes = avg_output.squeeze(1)
                        #gt_sdf = sdf.to(device)
                        #avg_output_sdf = avg_output_sdf.to(device)
                        # avg_output_classes = (avg_output *2).round().long().squeeze(1)

                        avg_output_one_hot = F.one_hot(avg_output_classes, num_classes=3).permute(0, 3, 1, 2).to(device)

                        output_logits, sdf_pred = model(images, avg_output_one_hot, alpha)
                        output_softmax = torch.softmax(output_logits.detach(), dim=1)
                        output = torch.argmax(output_logits,
                                              dim=1)  # Using argmax since the output is only raw logits and argmax makes you choose the maximum of the labels. So the 3 channels (3 classes) is converted to 1 channel the max value.
                        targets_onehot = F.one_hot(labels, num_classes=3)  # (B, H, W, C)
                        targets_onehot = targets_onehot.permute(0, 3, 1, 2)  # (B, C, H, W)
                        targets_onehot = targets_onehot.float()
                        print('One Hot Targets', targets_onehot.shape, 'Unique Values',
                              targets_onehot.unique(return_counts=True))

                        cross_entropy_loss = crossentropy_loss(output_logits, labels)
                        #sdf_loss = F.l1_loss(sdf_pred, gt_sdf)
                        dice_loss = cl_dice(targets_onehot,output_softmax,test_batch)
                        image_processing_dice_loss = cl_dice(targets_onehot,avg_output_one_hot.float(),test_batch)

                        #loss = cross_entropy_loss + sdf_loss + dice_loss
                        loss = cross_entropy_loss  + dice_loss

                        test_cross_entropy_loss += cross_entropy_loss.item()
                        #test_sdf_loss += sdf_loss.item()
                        test_dice_loss +=  dice_loss.item()
                        test_image_processing_dice_loss += image_processing_dice_loss.item()
                        test_loss += loss.item()

                        # if epoch %10 == 0:
                        if not os.path.exists(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch))):
                            os.mkdir(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch)))
                        if not os.path.exists(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch), 'sdf_map')):
                            os.mkdir(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch), 'sdf_map'))
                        for i in range(len(output)):
                            plt.axis('off')
                            plt.imsave(f'{save_parent_dir}/test/Epoch {epoch}/result_{test_batch}.png',
                                       output[i].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=2)
                            plt.close()
                            b0_pred, b1_pred = compute_betti_numbers(output[i])
                            b0_img_prc, b1_img_prc = compute_betti_numbers(avg_output[i])
                            b0_gt, b1_gt = compute_betti_numbers(labels[i])
                            betti_error_pred = abs(b0_pred - b0_gt) + abs(b1_pred - b1_gt)
                            betti_error_img_prc = abs(b0_img_prc - b0_gt) + abs(b1_img_prc - b1_gt)
                            iou_dl, iou_img_prc = visulaize_iou_metrics(raw_images=images[i], epoch=epoch,
                                                                        batch=test_batch,
                                                                        predicted_structures=output[i], gt=labels[i],
                                                                        image_processing_result=avg_output[i],
                                                                        betti_error_pred=betti_error_pred,
                                                                        betti_error_img_prc=betti_error_img_prc,
                                                                        save_path=f'{save_parent_dir}/test')
                            test_iou_dl += iou_dl
                            test_iou_img_prc += iou_img_prc

                            # sdf_vis = sdf_pred[i].permute(1, 2, 0).detach().cpu().numpy()
                            # # vmax = max(abs(sdf_vis.min()), abs(sdf_vis.max()))
                            # fig, ax = plt.subplots(figsize=(6, 5))
                            # im = ax.imshow(sdf_vis, cmap='seismic', vmin=-1, vmax=1)
                            # cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                            # plt.savefig(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch), 'sdf_map',
                            #                          f'sdf_{test_batch}'))
                            # plt.close(fig)
                            test_batch += 1

                    print(f'Test Loss for Epoch {epoch}', test_loss / len(test_dataloader))
                    print('Test IoU for deep learning', test_iou_dl / len(test_idx))
                    print('Test IoU for image processing ', test_iou_img_prc / len(test_idx))

                run.log({'epoch': epoch,
                         'alpha': alpha,
                         'learning_rate': scheduler.get_last_lr()[0],
                         'Train_Total_Loss': train_loss / len(train_dataloader),
                         'Test_Total_Loss': test_loss / len(test_dataloader),
                         'Val_Total_Loss': val_loss / len(val_dataloader),
                         'Train_CE_Loss': train_cross_entropy_loss / len(train_dataloader),'Val_CE_Loss': val_cross_entropy_loss / len(val_dataloader),'Test_CE_Loss': test_cross_entropy_loss / len(test_dataloader),
                         'Train IoU DL': train_iou_dl / len(train_idx),
                         'Train IoU_img': train_iou_img_prc / len(train_idx),
                         'Test IoU DL': test_iou_dl / len(test_idx),
                         'Test IoU_img': test_iou_img_prc / len(test_idx),
                         'Val IoU DL': val_iou_dl / len(val_idx), 'Val IoU_img': val_iou_img_prc / len(val_idx),
                         'Train DL Dice Loss': train_dice_loss / len(train_dataloader),
                         'Val DL Dice Loss': val_dice_loss / len(val_dataloader),
                         'Test DL Dice Loss': test_dice_loss / len(test_dataloader),
                         'Train Img Prc Dice Loss': train_img_proc_dice_loss / len(train_dataloader),
                         'Val Img Prc Dice Loss': val_image_processing_dice_loss / len(val_dataloader),
                         'Test Img Prc Dice Loss': test_image_processing_dice_loss / len(test_dataloader),
                         # 'Train SDF Loss': train_sdf_loss / len(train_dataloader),
                         # 'Val SDF Loss': val_sdf_loss / len(val_dataloader),
                         # 'Test_SDF_Loss': test_sdf_loss / len(test_dataloader),
                         })

        wandb.finish()

def extract_foreground_mask(image):

    R = image[:,:,0]
    # G = image[:,:,1]
    # B = image[:,:,2]
    mask = (R ==2)
    return mask.astype(float)



def visulaize_iou_metrics(raw_images, epoch,batch,predicted_structures, gt,image_processing_result,betti_error_pred,betti_error_img_prc,tolerance=0,save_path='Simple_Conv_Grayscale_output'):

    red_extracted_structures = extract_foreground_mask(predicted_structures.unsqueeze(dim = -1).detach().cpu().numpy())
    red_gt_structures = extract_foreground_mask(gt.unsqueeze(dim = -1).detach().cpu().numpy())
    red_image_processing_structures = extract_foreground_mask(image_processing_result.permute(1,2,0).detach().cpu().numpy())

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
    plt.close()

if __name__ == "__main__":
    train_grayscale()
