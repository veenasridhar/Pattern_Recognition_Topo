import os
import numpy as np
import torchvision
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader,Subset,random_split
from torchinfo import summary
import cv2
from input_pipeline_binary import *
from models import simple_Conv
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from cldice_loss_binary import *
from torch.utils.data import ConcatDataset
from visualizations.thesis_visualisation import *
from logger import setup_logger
import logging

KERNEL_SIZE = 3
BATCH_SIZE = 1
ACCUM_STEPS = 4
FOLDS = 5
N=5
lr = 0.001
epochs = 101
weight_decay = 1e-4
data_dir = 'data/Batches'
ground_truth_dir = 'data/Ground_Truth'
image_processing_dir = 'data/Image Processing Results'
aug_data_dir = 'data/Augmented TO Data'

opento_batches_dir = 'OpenTO_Consensus_Dataset_reduced_thickness_cldice/subbatches'
opento_gt_dir = 'OpenTO_Consensus_Dataset_reduced_thickness_cldice/ground_truth'
opento_img_prc_dir = 'OpenTO_Consensus_Dataset_reduced_thickness_cldice/image_proc'



def train_grayscale(model,train_dataloader,val_dataloader,test_dataloader,fold):

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(name="train_grayscale", log_file="logs/train_grayscale.log", level=logging.INFO)
    save_parent_dir = f'CNN_RailwayTO_FOLD_{fold}'

    model = model.to(device)
    logger.info('The training parameters for the module are %d', sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer1 = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay= weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=epochs)

    crossentropy_loss = torch.nn.CrossEntropyLoss().to(device)
    cl_dice = soft_dice_cldice()
    best_loss = float('inf')

    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="veenasridhar96-university-of-stuttgart",
        project='Masterthesis Curvess',
        name=f'CNN RailwayTO FOLD_{fold}',
        config={
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "loss": 'CE + cldice',
            "architecture": "CNN",
            "dataset": "Railway TO",
            "fold": fold,
        }, )


    for epoch in range(epochs):
        
        model.train()
        logger.info('Epoch: %d, learning rate: %f', epoch, lr)
        
        train_loss = 0.0
        train_cross_entropy_loss = 0.0
        train_cldice_loss =0.0
        train_img_proc_cldice_loss = 0.0
        train_cldice_image_prc = 0.0
        train_cldice_dl = 0.0
        train_clrec_img_prc = 0.0
        train_clrec_dl = 0.0
        train_clprec_img_prc = 0.0
        train_clprec_dl = 0.0 
        
        # if epoch < 150:             # For weighing ce_loss and sdf loss --- 150,250
        #     alpha = epoch / 150
        # else:
        #     alpha =  1
        #alpha = epoch/epochs

        accum_steps = ACCUM_STEPS  # Number of steps to accumulate gradients over
        optimizer1.zero_grad()
        train_batch = 0

        for i,( images, labels, avg_output) in enumerate(train_dataloader):

            images = images.to(device)
            labels = labels.to(device)
            avg_output = avg_output.to(device)
            avg_output_classes = avg_output
            
            avg_output_one_hot = F.one_hot(avg_output, num_classes=2).permute(0, 3, 1, 2).to(device)

            output_logits  = model(images)

            output_softmax = torch.softmax(output_logits, dim=1)
            output = torch.argmax(output_logits,dim=1)                          # Using argmax since the output is only raw logits and argmax makes you choose the maximum of the labels. So the 2 channels (2 classes) is converted to 1 channel the max value.
            targets_onehot = F.one_hot(labels, num_classes=2) # (B, H, W, C)
            targets_onehot = targets_onehot.permute(0, 3, 1, 2)  # (B, C, H, W)
            targets_onehot = targets_onehot.float()  # long to float

            cross_entropy_loss = crossentropy_loss(output_logits, labels)
            #sdf_loss = F.l1_loss(sdf_pred, gt_sdf)
            cldice_loss = cl_dice(targets_onehot,output_softmax)
            img_processing_cldice_loss = cl_dice(targets_onehot,avg_output_one_hot.float())
            
            loss = cross_entropy_loss + cldice_loss
            train_img_proc_cldice_loss += img_processing_cldice_loss.item()
            train_cross_entropy_loss += cross_entropy_loss.item()
            train_cldice_loss += cldice_loss.item()

            # Gradient accumulation
            (loss/accum_steps).backward()
            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer1.step()
                optimizer1.zero_grad()

            train_loss += loss.item()
            
            for j in range(len(output)):
                
                predicted_binary = (output[j].detach().cpu().numpy() == 1).astype(np.float32)
                gt_binary = (labels[j].detach().cpu().numpy() == 1).astype(np.float32)
                img_prc_binary = (avg_output[j].squeeze(0).detach().cpu().numpy() == 1).astype(np.float32)
                predicted_skeleton = skeletonize(predicted_binary > 0).astype(np.float32)                       # Can use skimage.morphology.skeletonize for metric calculation
                gt_skeleton = skeletonize(gt_binary > 0).astype(np.float32)                                     
                img_prc_skeleton = skeletonize(img_prc_binary > 0).astype(np.float32)

                cl_prec = np.sum(predicted_skeleton * gt_binary) / (np.sum(predicted_skeleton) + 1e-7)
                cl_rec = np.sum(gt_skeleton * predicted_binary) / (np.sum(gt_skeleton) + 1e-7)
                cl_dice_dl = 2 * cl_prec * cl_rec / (cl_prec + cl_rec)

                cl_prec_i = np.sum(img_prc_skeleton * gt_binary) / (np.sum(img_prc_skeleton) + 1e-7)
                cl_rec_i = np.sum(gt_skeleton * img_prc_binary) / (np.sum(gt_skeleton) + 1e-7)
                cl_dice_img_prc = 2 * cl_prec_i * cl_rec_i / (cl_prec_i + cl_rec_i)

                
                if not os.path.exists(os.path.join(save_parent_dir)):
                    os.mkdir(os.path.join(save_parent_dir))
                if not os.path.exists(os.path.join(save_parent_dir,'train')):
                    os.mkdir(os.path.join(save_parent_dir,'train'))
                if not os.path.exists(os.path.join(save_parent_dir,'train', 'Epoch '+str(epoch))):
                    os.mkdir(os.path.join(save_parent_dir,'train', 'Epoch '+str(epoch)))
                if not os.path.exists(os.path.join(save_parent_dir, 'train', 'Epoch ' + str(epoch), 'Metric Comparison')):
                    os.mkdir(os.path.join(save_parent_dir, 'train', 'Epoch ' + str(epoch), 'Metric Comparison'))
            
                #iou,img_prc_iou =  calculate_iou_metrics(raw_images=images[j], epoch=epoch, batch=train_batch, predicted_structures = output[j],
                #                                        gt = labels[j],image_processing_result=avg_output[j],betti_error_pred=cl_dice_dl,betti_error_img_prc=cl_dice_img_prc,save_path= f'{save_parent_dir}/train')
                
                if train_batch <50:
                    group = {
                        'inputs': images[j],
                        'gt': labels[j].detach().cpu().numpy(),
                        'pred': output[j].detach().cpu().numpy(),
                        'baseline': avg_output[j].squeeze(0).detach().cpu().numpy(),
                        'label': f'Batch {train_batch}',
                        # clDice already computed above — just pass them in
                        'cld_pred': cl_dice_dl,
                        'prec_pred': cl_prec,
                        'rec_pred': cl_rec,
                        'cld_base': cl_dice_img_prc,
                        'prec_base': cl_prec_i,
                        'rec_base': cl_rec_i,
                        #'iou_dl':iou,
                        #'iou_img_prc': img_prc_iou
                    }
                    visualize_comparison_with_overlay(
                        groups=[group],
                        structure_val=1,  # label index for structure class
                        save_path=f'{save_parent_dir}/train/Epoch {epoch}/Metric Comparison',
                        batch=train_batch
                    )
                    #plot2_overlay(labels[i].detach().cpu().numpy(),output[i].detach().cpu().numpy(),avg_output[i].squeeze(0).detach().cpu().numpy(),(cl_dice_dl,cl_prec,cl_rec),
                    #              (cl_dice_img_prc,cl_prec_i,cl_rec_i),save_path=f'{save_parent_dir}/train/Epoch {epoch}/IoU Metric',batch = train_batch)
                
                #train_iou_dl += iou
                #train_iou_img_prc += img_prc_iou
                train_cldice_image_prc += cl_dice_img_prc
                train_cldice_dl += cl_dice_dl
                train_clrec_img_prc += cl_rec_i
                train_clrec_dl += cl_rec
                train_clprec_img_prc += cl_prec_i
                train_clprec_dl += cl_prec
                
                train_batch += 1

        scheduler.step()
        logger.info(f'Epoch %d | Train Loss: %.4f', epoch, train_loss / len(train_dataloader))

        # Starting a val loop
        if epoch % 10 == 0:
            model.eval()
            
            with torch.no_grad():
                val_loss = 0.0
                val_cross_entropy_loss = 0.0
                val_cldice_loss = 0.0  
                # val_iou_dl = 0.0
                # val_iou_img_prc = 0.0
                val_img_proc_cldice_loss = 0.0
                val_cldice_image_prc = 0.0
                val_cldice_dl = 0.0
                val_clrec_img_prc = 0.0
                val_clrec_dl = 0.0
                val_clprec_img_prc = 0.0
                val_clprec_dl = 0.0

                val_batch = 0
                for images, labels, avg_output in val_dataloader:
                    
                    images = images.to(device)
                    labels = labels.to(device)
                    avg_output = avg_output.to(device)
                    avg_output_classes = avg_output
                    
                    avg_output_one_hot = F.one_hot(avg_output_classes, num_classes=2).permute(0, 3, 1, 2).to(device)

                    #output_logits = model(images, avg_output_one_hot, alpha)
                    output_logits = model(images)
                    output_softmax = torch.softmax(output_logits.detach(), dim=1)
                    output = torch.argmax(output_logits, dim=1)                         # Using argmax since the output is only raw logits and argmax makes you choose the maximum of the labels. So the 3 channels (3 classes) is converted to 1 channel the max value.
                    
                    targets_onehot = F.one_hot(labels, num_classes=2)   # (B, H, W, C)
                    targets_onehot = targets_onehot.permute(0, 3, 1, 2) # (B, C, H, W)
                    targets_onehot = targets_onehot.float()             # Long to float 


                    cross_entropy_loss = crossentropy_loss(output_logits, labels)
                    #sdf_loss = F.l1_loss(sdf_pred, gt_sdf)
                    cldice_loss = cl_dice(targets_onehot,output_softmax)
                    img_processing_cldice_loss = cl_dice(targets_onehot,avg_output_one_hot.float())
                    
                    loss = cross_entropy_loss + cldice_loss

                    val_img_proc_cldice_loss += img_processing_cldice_loss.item()
                    val_cross_entropy_loss += cross_entropy_loss.item()
                    val_cldice_loss +=  cldice_loss.item()
                    val_loss += loss.item()

                    # if epoch % 10 == 0:
                    
                    for i in range(len(output)):
                        
                        predicted_binary = (output[i].detach().cpu().numpy() == 1).astype(np.float32)
                        gt_binary = (labels[i].detach().cpu().numpy() == 1).astype(np.float32)
                        img_prc_binary = (avg_output[i].squeeze(0).detach().cpu().numpy()==1).astype(np.float32)
                        
                        predicted_skeleton = skeletonize(predicted_binary > 0).astype(np.float32)                   #Can use skimage.morphology.skeletonize for metric calculation
                        gt_skeleton = skeletonize(gt_binary > 0).astype(np.float32)
                        img_prc_skeleton = skeletonize(img_prc_binary > 0).astype(np.float32)

                        cl_prec = np.sum(predicted_skeleton * gt_binary) / (np.sum(predicted_skeleton) + 1e-7)
                        cl_rec = np.sum(gt_skeleton * predicted_binary) / (np.sum(gt_skeleton) + 1e-7)
                        cl_dice_dl = 2 * cl_prec * cl_rec / (cl_prec + cl_rec)

                        cl_prec_i = np.sum(img_prc_skeleton * gt_binary) / (np.sum(img_prc_skeleton) + 1e-7)
                        cl_rec_i = np.sum(gt_skeleton * img_prc_binary) / (np.sum(gt_skeleton) + 1e-7)
                        cl_dice_img_prc = 2 * cl_prec_i * cl_rec_i / (cl_prec_i + cl_rec_i)
                
                        # iou_dl, iou_img_prc = visulaize_iou_metrics(raw_images=images[i], epoch=epoch,
                        #                                                 batch=val_batch,
                        #                                                 predicted_structures=output[i],
                        #                                                 gt=labels[i],
                        #                                                 image_processing_result=avg_output[i],
                        #                                                 betti_error_pred=cl_dice_dl,
                        #                                                 betti_error_img_prc=cl_dice_img_prc,
                        #                                                 save_path=f'{save_parent_dir}/val')
                        
                        if val_batch < 50:
                            if not os.path.exists(os.path.join(save_parent_dir,'val')):
                                os.mkdir(os.path.join(save_parent_dir,'val'))
                            if not os.path.exists(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch))):
                                os.mkdir(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch)))
                            if not os.path.exists(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch), 'Metric Comparison')):
                                os.mkdir(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch), 'Metric Comparison'))
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
                                # 'iou_dl':iou_dl,
                                # 'iou_img_prc': iou_img_prc,
                            }
                            
                            #Plot individual output images for val batches
                            plt.axis('off')
                            plt.imsave(f'{save_parent_dir}/val/Epoch {epoch}/result_{val_batch}.png',
                                    output[i].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
                            plt.close()

                            visualize_comparison_with_overlay(
                                groups=[group],
                                structure_val=1,  # your label index for structure class
                                save_path=f'{save_parent_dir}/val/Epoch {epoch}/Metric Comparison',
                                batch=val_batch
                            )
                            # plot2_overlay(labels[i].detach().cpu().numpy(), output[i].detach().cpu().numpy(),
                            #               avg_output[i].squeeze(0).detach().cpu().numpy(),
                            #               (cl_dice_dl, cl_prec, cl_rec),
                            #               (cl_dice_img_prc, cl_prec_i, cl_rec_i),
                            #               save_path=f'{save_parent_dir}/val/Epoch {epoch}',
                            #               batch=val_batch)
                        
                        #val_iou_dl += iou_dl
                        #val_iou_img_prc += iou_img_prc
                        val_cldice_image_prc +=cl_dice_img_prc
                        val_cldice_dl += cl_dice_dl
                        val_clrec_img_prc += cl_rec_i
                        val_clrec_dl += cl_rec
                        val_clprec_img_prc += cl_prec_i
                        val_clprec_dl += cl_prec

                        val_batch += 1

        logger.info(f'Epoch %d | Val Loss: %.4f', epoch, val_loss / len(val_dataloader))
        epoch_cldice_loss = val_loss /len(val_dataloader.dataset)
        if epoch_cldice_loss <= best_loss:
            best_loss = epoch_cldice_loss
            torch.save(model,f'CNN_chkpoint_fold_{fold}.pth')
            logger.info(f'Model Saved with test loss %.4f', best_loss)

            #Test loop
            model.eval()
            
            with torch.no_grad():
                test_loss = 0.0
                test_cross_entropy_loss = 0.0
                test_cldice_loss = 0.0
                # test_iou_dl = 0.0
                # test_iou_img_prc = 0.0
                test_cldice_img_proc = 0.0
                test_cldice_dl = 0.0
                test_image_processing_cldice_loss = 0.0
                test_clrec_img_prc = 0.0
                test_clrec_dl = 0.0
                test_clprec_img_prc = 0.0
                test_clprec_dl = 0.0
            
                test_batch = 0
                for images, labels, avg_output in test_dataloader:
                    
                    images = images.to(device)
                    labels = labels.to(device)
                    avg_output = avg_output.to(device)
                    
                    avg_output_classes = avg_output
                    avg_output_one_hot = F.one_hot(avg_output_classes, num_classes=2).permute(0, 3, 1, 2).to(device)

                    #output_logits = model(images, avg_output_one_hot.float(), alpha)
                    output_logits = model(images)
                    output_softmax = torch.softmax(output_logits.detach(), dim=1)
                    output = torch.argmax(output_logits,dim=1)  # Using argmax since the output is only raw logits and argmax makes you choose the maximum of the labels. So the 3 channels (3 classes) is converted to 1 channel the max value.
                    
                    targets_onehot = F.one_hot(labels, num_classes=2)    # (B, H, W, C)
                    targets_onehot = targets_onehot.permute(0, 3, 1, 2)  # (B, C, H, W)
                    targets_onehot = targets_onehot.float()              # Long to float
                    
                    cross_entropy_loss = crossentropy_loss(output_logits, labels)
                    #sdf_loss = F.l1_loss(sdf_pred, gt_sdf)
                    cldice_loss = cl_dice(targets_onehot,output_softmax)
                    image_processing_cldice_loss = cl_dice(targets_onehot,avg_output_one_hot.float())
                    
                    loss = cross_entropy_loss + cldice_loss
                    test_image_processing_cldice_loss += image_processing_cldice_loss.item()
                    test_cross_entropy_loss += cross_entropy_loss.item()
                    test_cldice_loss +=  cldice_loss.item()
                    test_loss += loss.item()
                    
                    for i in range(len(output)):
                        
                        predicted_binary = (output[i].detach().cpu().numpy() == 1).astype(np.float32)
                        gt_binary = (labels[i].detach().cpu().numpy() == 1).astype(np.float32)
                        img_prc_binary = (avg_output[i].squeeze(0).detach().cpu().numpy()==1).astype(np.float32)
                        predicted_skeleton = skeletonize(predicted_binary > 0).astype(np.float32)                       # Use skimage.morphology.skeletonize for metric calculation
                        gt_skeleton = skeletonize(gt_binary > 0).astype(np.float32)
                        img_prc_skeleton = skeletonize(img_prc_binary > 0).astype(np.float32)

                        cl_prec = np.sum(predicted_skeleton * gt_binary) / (np.sum(predicted_skeleton) + 1e-7)
                        cl_rec = np.sum(gt_skeleton * predicted_binary) / (np.sum(gt_skeleton) + 1e-7)
                        cl_dice_dl = 2 * cl_prec * cl_rec / (cl_prec + cl_rec)

                        cl_prec_i = np.sum(img_prc_skeleton * gt_binary) / (np.sum(img_prc_skeleton) + 1e-7)
                        cl_rec_i = np.sum(gt_skeleton * img_prc_binary) / (np.sum(gt_skeleton) + 1e-7)
                        cl_dice_img_prc = 2 * cl_prec_i * cl_rec_i / (cl_prec_i + cl_rec_i)

                        # iou_dl, iou_img_prc = visulaize_iou_metrics(raw_images=images[i], epoch=epoch,
                        #                                              batch=test_batch,
                        #                                              predicted_structures=output[i],
                        #                                              gt=labels[i],
                        #                                              image_processing_result=avg_output[i],
                        #                                              betti_error_pred=cl_dice_dl,
                        #                                              betti_error_img_prc=cl_dice_img_prc,
                        #                                              save_path=f'{save_parent_dir}/val')

                        if not os.path.exists(os.path.join(save_parent_dir,'test')):
                            os.mkdir(os.path.join(save_parent_dir,'test'))
                        if not os.path.exists(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch))):
                            os.mkdir(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch)))
                        if not os.path.exists(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch), 'Metric Comparison')):
                            os.mkdir(os.path.join(save_parent_dir, 'test', 'Epoch ' + str(epoch), 'Metric Comparison'))
                        group = {
                            'inputs': images[i],
                            'gt': labels[i].detach().cpu().numpy(),
                            'pred': output[i].detach().cpu().numpy(),
                            'baseline': avg_output[i].squeeze(0).detach().cpu().numpy(),
                            'label': f'Batch {test_batch}',
                            # clDice already computed above — just pass them in
                            'cld_pred': cl_dice_dl,
                            'prec_pred': cl_prec,
                            'rec_pred': cl_rec,
                            'cld_base': cl_dice_img_prc,
                            'prec_base': cl_prec_i,
                            'rec_base': cl_rec_i,
                            # 'iou_dl':iou_dl,
                            # 'iou_img_prc': iou_img_prc,
                        }
                        visualize_comparison_with_overlay(
                            groups=[group],
                            structure_val=1,  # your label index for structure class
                            save_path=f'{save_parent_dir}/test/Epoch {epoch}/Metric Comparison',
                            batch=test_batch
                        )
                        # plot2_overlay(labels[i].detach().cpu().numpy(), output[i].detach().cpu().numpy(),
                        #               avg_output[i].squeeze(0).detach().cpu().numpy(),
                        #               (cl_dice_dl, cl_prec, cl_rec),
                        #               (cl_dice_img_prc, cl_prec_i, cl_rec_i),
                        #               save_path=f'{save_parent_dir}/test/Epoch {epoch}/Metric Comparison',
                        #               batch=test_batch)
                            
                        # test_iou_dl += iou_dl
                        # test_iou_img_prc += iou_img_prc
                        test_cldice_dl += cl_dice_dl
                        test_cldice_img_proc += cl_dice_img_prc
                        test_clrec_img_prc += cl_rec_i
                        test_clrec_dl += cl_rec
                        test_clprec_img_prc += cl_prec_i
                        test_clprec_dl += cl_prec
                        
                        test_batch += 1

            logger.info(f'Epoch: %d | Test Loss: %.4f', epoch, test_loss / len(test_dataloader))


            run.log({'epoch': epoch,
                    'fold': fold,
                    'learning_rate': scheduler.get_last_lr()[0],
                    'Train_Total_Loss': train_loss / len(train_dataloader),
                    'Test_Total_Loss': test_loss / len(test_dataloader),
                    'Val_Total_Loss': val_loss / len(val_dataloader),
                    'Train_CE_Loss': train_cross_entropy_loss / len(train_dataloader),
                    'Test_CE_Loss': test_cross_entropy_loss / len(test_dataloader),
                    'Val_CE_Loss': val_cross_entropy_loss / len(val_dataloader),
                    # 'Train IoU': train_iou_dl / len(train_dataloader.dataset),
                    # 'Train IoU_img': train_iou_img_prc / len(train_dataloader.dataset),
                    # 'Test IoU': test_iou_dl / len(test_dataloader.dataset),
                    # 'Test IoU_img': test_iou_img_prc /len(test_dataloader.dataset),
                    # 'Val IoU': val_iou_dl / len(val_dataloader.dataset), 
                    # 'Val IoU img': val_iou_img_prc / len(val_dataloader.dataset),
                    'Train_Img_Processing_Dice Loss': train_img_proc_cldice_loss/len(train_dataloader),
                    'Val_Img_Proccessing_Dice Loss': val_img_proc_cldice_loss/len(val_dataloader),
                    'Test_Img_Processing_Dice Loss': test_image_processing_cldice_loss/len(test_dataloader),
                    'Train Dice Loss': train_cldice_loss / len(train_dataloader), 'Test_Dice_Loss': test_cldice_loss / len(test_dataloader),
                    'Val Dice Loss': val_cldice_loss/len(val_dataloader),
                    'Train_Img_Processing_Cldice': train_cldice_image_prc / len(train_dataloader.dataset),
                    'Val_Img_Proccessing_Cldice': val_cldice_image_prc / len(val_dataloader.dataset),
                    'Test_Img_Processing_Cldice': test_cldice_img_proc / len(test_dataloader.dataset),
                    'Train Cldice DL': train_cldice_dl / len(train_dataloader.dataset),
                    'Test Cldice DL': test_cldice_dl / len(test_dataloader.dataset),
                    'Val Cldice DL': val_cldice_dl / len(val_dataloader.dataset),
                    'Train clrec Img_prc': train_clrec_img_prc/len(train_dataloader.dataset), 'Train clrec DL': train_clrec_dl/len(train_dataloader.dataset),
                    'Val clrec Img_prc': val_clrec_img_prc/len(val_dataloader.dataset), 'Val clrec DL': val_clrec_dl /len(val_dataloader.dataset),
                    'Test clrec Img_prc': test_clrec_img_prc/len(test_dataloader.dataset), 'Test clrec DL': test_clrec_dl/len(test_dataloader.dataset),
                    'Train cplrec Img_prc': train_clprec_img_prc/len(train_dataloader.dataset), 'Train clprec DL': train_clprec_dl/len(train_dataloader.dataset),
                    'Val cplrec Img_prc': val_clprec_img_prc/len(val_dataloader.dataset), 'Val clprec DL': val_clprec_dl /len(val_dataloader.dataset),
                    'Test cplrec Img_prc': test_clprec_img_prc/len(test_dataloader.dataset), 'Test clprec DL': test_clprec_dl/len(test_dataloader.dataset),
                    })

    wandb.finish()


if __name__ == "__main__":
    train_grayscale()
