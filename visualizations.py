import cv2
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
import os
import matplotlib
#matplotlib.use('Agg')  # To avoid the tkinter error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
from image_processing_approach import *
from metrics import iou_metrics
from scipy.ndimage.morphology import binary_dilation

def postprocess_result(reconstructed_image):

    t = [(1.0, 0.0, 0.0),  # full red
         (0.5, 0.5, 0.5),  # full grey
         (1.0, 1.0, 1.0), ]  # full white
    t = torch.tensor(t, dtype=torch.float)  # The shape of the tensor is (T,C)
    # print(t.shape)
    reconstructed = reconstructed_image.permute(0, 2, 3, 1).detach().cpu().unsqueeze(-2)  # This reshapes the tensor to be (B,H,W,1,C)
    print(reconstructed.shape)

    dist = ((reconstructed - t) ** 2).sum(dim=-1)  # This first subtracts (H,W,1,C) - (T,C) and the result is summed along dim(-1) thus resulting in (H,W,T)
    min_dis_index = dist.argmin(-1)
    print('The shape of min distances',min_dis_index.shape)
    reconstructed = t[min_dis_index]
    print('The shape of the reconstructed image',reconstructed.shape)
    return reconstructed

def extract_foreground_mask(image):
    channels = (image.shape)[-1]
    print(channels)
    if channels == 1:
        grayscale_channel = image[:,:,0]
        mask = (grayscale_channel ==1)
        print(mask.min(),mask.max())
    elif channels == 3:
        channel1 = image[:,:,0]
        channel2 = image[:,:,1]
        channel3 = image[:,:,2]
        mask = (channel1 >= 0.7) & (channel2  >= 0.7) & (channel3 >= 0.7)
        # R = image[:, :, 0]
        # G = image[:, :, 1]
        # mask = (R < 0.2)
    return mask.astype(float)

def visulaize_iou_metrics(raw_images, epoch,batch,predicted_structures, gt,image_processing_result,tolerance=0,save_path='Simple_Conv_Grayscale_output'):

    print(predicted_structures.shape)
    print(image_processing_result.shape)
    red_extracted_structures = extract_foreground_mask(predicted_structures.permute(1,2,0).detach().cpu().numpy())
    #red_extracted_structures = extract_foreground_mask(predicted_structures.unsqueeze(-1).detach().cpu().numpy())
    red_gt_structures = extract_foreground_mask(gt.permute(1,2,0).detach().cpu().numpy())
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


    return iou_dilated,iou_dilated_image_processing,dilated_overlap,img_proc_dilated_overlap

def visulaize_images(multi_image_mode, batch, epoch, reconstructed_image, original, gt_image, image_processing_result, iou_dilated,iou_dilated_image_processing,dilated_overlap,
                                         img_proc_dilated_overlap, save_path, index,test=False):

    reconstructed_image = torch.clamp(reconstructed_image, 0, 1)
    original = torch.clamp(original, 0, 1)

    if multi_image_mode:
        gt_image = torch.clamp(gt_image, 0, 1)
        #reconstructed = postprocess_result(reconstructed_image)

        #for i in range(len(reconstructed_image)):
        if test:
            #image_processing_result = cv2.imread(os.path.join('data/Final Data/Image Processing Results Grayscale','test',f'result_{index}.png'))
            #plt.imsave(os.path.join(save_path, f'output_test_images_{epoch}_{batch}_{i}.png'),reconstructed_image.detach().cpu().numpy()*0.5,cmap='gray')
            plt.imsave(os.path.join(save_path, f'output_test_images_{epoch}_{batch}_{index}.png'), reconstructed_image.permute(1, 2, 0).detach().cpu().numpy())
            #plt.imsave(os.path.join(save_path, f'output_test_images_{index}.png'),reconstructed_image[i].permute(1, 2, 0).detach().cpu().numpy())
            #index = index+1
            #plt.imsave(os.path.join(save_path, f'output_test_images_{epoch}_{batch}_{i}.png'),reconstructed_image[i])
            # individual_reconstruction = random_reconstruction[i].squeeze(0)
            # individual_reconstruction= torch.clamp(individual_reconstruction, 0.0,1.0)
            # plt.imsave(os.path.join(save_path, f'random_recon/random_reconstruction_{epoch}_{batch}_{i}.png'),individual_reconstruction.permute(1, 2, 0).detach().cpu().numpy())
        else:
            #image_processing_result = cv2.imread(os.path.join('data/Final Data/Image Processing Results Grayscale', 'train', f'result_{index}.png'))
            #print('Image Processing Result',image_processing_result.shape)
            #plt.imsave(os.path.join(save_path, f'output_images_{epoch}_{batch}_{i}.png'),
            #           reconstructed_image[i].detach().cpu().numpy()*0.5,cmap='gray')
            plt.imsave(os.path.join(save_path, f'output_images_{epoch}_{batch}_{index}.png'), reconstructed_image.permute(1, 2, 0).detach().cpu().numpy())
            #plt.imsave(os.path.join(save_path, f'output_images_{index}.png'),reconstructed_image[i].permute(1, 2, 0).detach().cpu().numpy())
            #index+=1
            #plt.imsave(os.path.join(save_path, f'output_images_{epoch}_{batch}.png'), reconstructed_image)

        #for i in range(len(original)):
        fig = plt.figure(figsize=(16, 8))
        grid_spec = fig.add_gridspec(2, 5, hspace=0.1, wspace=0.1)

        for j in range(5):
            ax = fig.add_subplot(grid_spec[0, j])
            print(original.shape)
            orig_img = original[j].permute(1, 2, 0).detach().cpu().numpy()
            ax.imshow(orig_img)
            ax.axis('off')

        ax_center = fig.add_subplot(grid_spec[1, 1])
        #recon_images = reconstructed_image.detach().cpu().numpy()
        recon_images = reconstructed_image.permute(1, 2, 0).detach().cpu().numpy()
        ax_center.imshow(recon_images,cmap='gray')
        ax_center.axis('off')
        ax_center.set_title('Reconstructed Image')

        ax_left = fig.add_subplot(grid_spec[1, 0])
        gt_image_visual = gt_image.permute(1, 2, 0).detach().cpu().numpy()
        ax_left.imshow(gt_image_visual,cmap='gray')
        ax_left.axis('off')
        ax_left.set_title('Ground Truth')

        ax_left = fig.add_subplot(grid_spec[1, 2])
        #print(image_processing_result.shape)
        img_prc_result = image_processing_result.permute(1, 2, 0).detach().cpu().numpy()
        ax_left.imshow(img_prc_result, cmap='gray')
        ax_left.axis('off')
        ax_left.set_title('Pixelwise Averaging Result')

        ax_left = fig.add_subplot(grid_spec[1, 3])
        #img_prc_result = image_processing_result[i].permute(1, 2, 0).detach().cpu().numpy()
        ax_left.imshow(dilated_overlap)
        ax_left.axis('off')
        ax_left.set_title(f'IoU DL: {iou_dilated:.2f}')

        ax_left = fig.add_subplot(grid_spec[1, 4])
        # img_prc_result = image_processing_result[i].permute(1, 2, 0).detach().cpu().numpy()
        ax_left.imshow(img_proc_dilated_overlap)
        ax_left.axis('off')
        ax_left.set_title(f'IoU Pixelwise Avergaing: {iou_dilated_image_processing:.2f}')

        # ax_left = fig.add_subplot(grid_spec[1, 3])
        # ax_left.imshow(image_processing_result, cmap='gray')
        # ax_left.axis('off')
        # ax_left.set_title('Image Processing Result')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'reconstructed_images_{epoch}_{batch}_{index}.png'), dpi=150)
        plt.close()
    else:

        num_images = reconstructed_image.shape[0]
        fig, axes = plt.subplots(2, num_images, figsize=(16, 8))
        # Original Image
        if num_images == 1:
            orig_img = original[0].permute(1, 2, 0).detach().cpu().numpy()
            axes[0].imshow(orig_img)
            axes[0].set_title(f'Original {0 + 1}')
            axes[0].axis('off')

            # Reconstruction Images
            recon_images = reconstructed_image[0].detach().cpu().numpy()
            #recon_images = reconstructed_image[0].permute(1, 2, 0).detach().cpu().numpy()
            axes[1].imshow(recon_images,cmap='gray')
            #ssim_metric = ssim(torch.unsqueeze(reconstructed_image[0], 0), torch.unsqueeze(original[0], 0))
            #axes[1].set_title(f'SSIM {ssim_metric:.2f}')
            axes[1].axis('off')
        else:
            for j in range(num_images):
                orig_img = original[j].permute(1, 2, 0).detach().cpu().numpy()
                axes[0, j].imshow(orig_img,cmap='gray')
                axes[0, j].set_title(f'Original {j + 1}')
                axes[0, j].axis('off')

                # Reconstruction Images
                recon_images = reconstructed_image[j].detach().cpu().numpy()
                #recon_images = reconstructed_image[j].permute(1, 2, 0).detach().cpu().numpy()
                axes[1, j].imshow(recon_images,cmap='gray')
                #ssim_metric = ssim(torch.unsqueeze(reconstructed_image[j],0),torch.unsqueeze(original[j],0))
                #axes[1, j].set_title(f'SSIM {ssim_metric:.2f}')
                axes[1, j].axis('off')

        if test:
            plt.tight_layout()
            plt.savefig(os.path.join(save_path,f'reconstructed_test_{epoch}_{batch}.png'), dpi=150)
            plt.close()
        else:
            plt.tight_layout()
            plt.savefig(os.path.join(save_path,f'reconstructed_{epoch}_{batch}.png'), dpi=150)
            plt.close()

        '''for i in range(len(reconstructed_image)):
            if test:
                plt.imsave(os.path.join(save_path, f'output_test_images_{epoch}_{batch}_{i}.png'),
                           reconstructed_image[i].permute(1, 2, 0).detach().cpu().numpy())
            else:
                plt.imsave(os.path.join(save_path, f'output_images_{epoch}_{batch}_{i}.png'),
                           reconstructed_image[i].permute(1, 2, 0).detach().cpu().numpy())'''

def visulaize_images_individual(batch,epoch, reconstructed_image, original, gt_image, save_path, index,test=False):

    reconstructed_image = torch.clamp(reconstructed_image, 0, 1)
    original = torch.clamp(original, 0, 1)

    num_images = reconstructed_image.shape[0]
    fig, axes = plt.subplots(2, num_images, figsize=(16, 8))
    # Original Image
    if num_images == 1:
        orig_img = original[0].permute(1, 2, 0).detach().cpu().numpy()
        axes[0].imshow(orig_img)
        axes[0].set_title(f'Original {0 + 1}')
        axes[0].axis('off')

        # Reconstruction Images
        #recon_images = reconstructed_image[0].detach().cpu().numpy()
        recon_images = reconstructed_image[0].permute(1, 2, 0).detach().cpu().numpy()
        axes[1].imshow(recon_images,cmap='gray')
        iou_metric, overlap_image = iou_metrics(recon_images, orig_img, tolerance=0)
        axes[1].set_title(f'IoU Metric {iou_metric:.2f}')
        axes[1].axis('off')
    else:
        for j in range(num_images):
            orig_img = original[j].permute(1, 2, 0).detach().cpu().numpy()
            axes[0, j].imshow(orig_img,cmap='gray')
            axes[0, j].set_title(f'Original {j + 1}')
            axes[0, j].axis('off')

            # Reconstruction Images
            #recon_images = reconstructed_image[j].detach().cpu().numpy()
            recon_images = reconstructed_image[j].permute(1, 2, 0).detach().cpu().numpy()
            axes[1, j].imshow(recon_images,cmap='gray')
            iou_metric,overlap_image = iou_metrics(recon_images, orig_img,tolerance=0)
            axes[1, j].set_title(f'IoU Metric {iou_metric:.2f}')
            axes[1, j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,f'reconstructed_{epoch}_{batch}.png'), dpi=150)
    plt.close()




def visualize_latent_space( individual_latent , averaged_latent,gt_latent,epoch,batch,test,val):

    individual_latent_numpy = individual_latent.detach().cpu().numpy() # The shape is (10,4,32,32)
    individual_latents_reshaped = individual_latent_numpy.reshape(len(individual_latent_numpy),-1)
    averaged_latent = averaged_latent.detach().cpu().numpy()
    averaged_latent_reshaped = averaged_latent.reshape(len(averaged_latent),-1)
    gt_latents_numpy = gt_latent.detach().cpu().numpy()
    gt_latents_reshaped = gt_latents_numpy.reshape(len(gt_latents_numpy),-1)

    pca = PCA(n_components=2)
    individual_latents_2d = pca.fit_transform(individual_latents_reshaped)

    fig, ax = plt.subplots(figsize=(5, 5))
    ind_labels = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    scatter = ax.scatter(individual_latents_2d[:, 0], individual_latents_2d[:, 1], c=ind_labels, cmap='viridis', s=50, label='individual latent')

    gt_labels = np.array([1, 2])
    gt_latent_2d = pca.fit_transform(gt_latents_reshaped)
    gt_scatter = ax.scatter(gt_latent_2d[:, 0], gt_latent_2d[:, 1], c=gt_labels, cmap='viridis', s=300, edgecolors='black', marker='*', label='ground truth')

    avg_latents_labels = np.array([1,2])
    avg_latent_2d = pca.fit_transform(averaged_latent_reshaped)
    avg_acatter = ax.scatter(avg_latent_2d[:,0],avg_latent_2d[:,1], c= avg_latents_labels,cmap='viridis', s= 200,label='avg_latent')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('VAE Latent Space')
    if test:
        plt.savefig(f'./latent_visualisation/test_images/image_{epoch}_{batch}.png', dpi=150)
    elif val:
        plt.savefig(f'./latent_visualisation/val_images/image_{epoch}_{batch}.png', dpi=150)
    else:
        plt.savefig(f'./latent_visualisation/image_{epoch}_{batch}.png', dpi=150)
    #plt.show()

    #avg_labels = np.array([1, 2])

    '''ind_tsne = TSNE(n_components=2,init='pca',random_state=42,perplexity=9)
    avg_tsne = TSNE(n_components=2,init='pca',random_state=42,perplexity=1)
    gt_tsne = TSNE(n_components=2,init='pca',random_state=42,perplexity=1)
    ind_latent_embedded = ind_tsne.fit_transform(individual_latents_reshaped)
    avg_latent_embedded = avg_tsne.fit_transform(averaged_latent_reshaped)
    gt_embedded = gt_tsne.fit_transform(gt_latents_reshaped)
    
    
    avg_scatter = ax.scatter(avg_latent_embedded[:,0],avg_latent_embedded[:,1],c=avg_labels, cmap='viridis', s=300, edgecolors='black', label='average latent')'''


    #plt.scatter(ind_latent_embedded[:,0],ind_latent_embedded[:,1],c=ind_labels)
    #plt.scatter(avg_latent_embedded[:,0],avg_latent_embedded[:,1],c=avg_labels)
    #plt.colorbar(scatter,ax=ax, label ='Class')
    #plt.title('Latent Visualisation')
    #plt.show()

def visualize_latent_space_individual( z_individual):

    z_individual = z_individual.detach().cpu().numpy()          # (B,C,H,W) = (16,4,32,32)
    print('The shape of individual latents',z_individual.shape)
    z_reshaped = z_individual.reshape(len(z_individual),-1)
    pca = PCA(n_components=2)
    ind_latent_embedded = pca.fit_transform(z_reshaped)
    '''fig, ax = plt.subplots(figsize=(5, 5))
    scatter = ax.scatter(ind_latent_embedded[:, 0], ind_latent_embedded[:, 1], s=50, label='individual latent')
    plt.title('Latent Visualisation')
    plt.show()'''
    return ind_latent_embedded,pca

def get_latent_plot(individual_latent , averaged_latent, gt_latent):
    individual_latent_numpy = individual_latent.detach().cpu().numpy()  # The shape is (5,4,32,32)
    #print('The shape of individual latents',individual_latent_numpy.shape)
    individual_latents_reshaped = individual_latent_numpy.reshape(len(individual_latent_numpy), -1)
    #print('Reshaped individual latents',individual_latents_reshaped.shape)
    averaged_latent = np.expand_dims(averaged_latent.detach().cpu().numpy(),0)
    #print('The shape of averaged latents',averaged_latent.shape)
    averaged_latent_reshaped = averaged_latent.reshape(len(averaged_latent), -1)
    #print('Reshaped average latents',averaged_latent_reshaped.shape)
    gt_latents_numpy = np.expand_dims(gt_latent.detach().cpu().numpy(),0)
    #print('The shape of gt_latents',gt_latents_numpy.shape)
    gt_latents_reshaped = gt_latents_numpy.reshape(len(gt_latents_numpy), -1)
    #print('REshaped gt',gt_latents_reshaped.shape)

    all_latents = np.concatenate([individual_latents_reshaped,averaged_latent_reshaped,gt_latents_reshaped],axis=0)

    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_latents)
    '''print(all_2d.mean())
    print(all_2d.std())
    individual_latent_2d = all_2d[:5]
    avg_2d = all_2d[5]
    gt_2d = all_2d[6]'''

    return all_2d,pca

def visualize_latent_and_images(multi_image_mode, batch, epoch, reconstructed_image, original, gt_image, individual_latent , averaged_latent, gt_latent, save_path, test=False):
    reconstructed_image = torch.clamp(reconstructed_image, 0, 1)
    original = torch.clamp(original, 0, 1)

    ssim = StructuralSimilarityIndexMeasure().to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    if multi_image_mode:

        gt_image = torch.clamp(gt_image, 0, 1)

        for i in range(len(reconstructed_image)):
            if test:
                plt.imsave(os.path.join(save_path, f'output_test_images_{epoch}_{batch}_{i}.png'),
                           reconstructed_image[i].permute(1, 2, 0).detach().cpu().numpy())
                # individual_reconstruction = random_reconstruction[i].squeeze(0)
                # individual_reconstruction= torch.clamp(individual_reconstruction, 0.0,1.0)
                # plt.imsave(os.path.join(save_path, f'random_recon/random_reconstruction_{epoch}_{batch}_{i}.png'),individual_reconstruction.permute(1, 2, 0).detach().cpu().numpy())
            else:
                plt.imsave(os.path.join(save_path, f'output_images_{epoch}_{batch}_{i}.png'),
                           reconstructed_image[i].permute(1, 2, 0).detach().cpu().numpy())

        for i in range(2):
            fig = plt.figure(figsize=(16, 8))
            grid_spec = fig.add_gridspec(2, 5, hspace=0.1, wspace=0.05)

            for j in range(5):
                ax = fig.add_subplot(grid_spec[0, j])
                orig_img = original[i][j].permute(1, 2, 0).detach().cpu().numpy()
                ax.imshow(orig_img)
                ax.axis('off')

            ssim_loss = ssim(reconstructed_image, gt_image)
            ax_center = fig.add_subplot(grid_spec[1, 2])
            recon_images = reconstructed_image[i].permute(1, 2, 0).detach().cpu().numpy()
            ax_center.imshow(recon_images)
            ax_center.axis('off')
            ax_center.set_title(f'SSIM: {ssim_loss:.2f}')

            ax_left = fig.add_subplot(grid_spec[1, 1])
            # print(type(gt_image), gt_image.shape)
            gt_image_visual = gt_image[i].permute(1, 2, 0).detach().cpu().numpy()
            ax_left.imshow(gt_image_visual)
            ax_left.axis('off')
            ax_left.set_title(f'Ground Truth Image')

            start_index = i*5 ; end_index = start_index+5
            individual_latents_2d, avg_latent_2d, gt_latent_2d,pca = get_latent_plot(individual_latent[start_index:end_index] , averaged_latent[i], gt_latent[i])
            ax_right = fig.add_subplot(grid_spec[1, 3])
            scatter = ax_right.scatter(individual_latents_2d[:, 0], individual_latents_2d[:, 1],  s=50, label='individual latent')
            gt_scatter = ax_right.scatter(gt_latent_2d[0], gt_latent_2d[1],   s=300, edgecolors='black', marker='*', label='ground truth')
            avg_scatter = ax_right.scatter(avg_latent_2d[0], avg_latent_2d[1],  marker='x', s=200, label='avg_latent')
            ax_right.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
            ax_right.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
            ax_right.set_title('VAE Latent Space')

            for l in [0, 4]:
                ax = fig.add_subplot(grid_spec[1, l])
                ax.axis("off")

            #plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'reconstructed_images_{epoch}_{batch}_{i}.png'), dpi=150)
            plt.close()


    else:

        num_images = reconstructed_image.shape[0]
        fig, axes = plt.subplots(2, num_images, figsize=(16, 8))
        # Original Image
        for j in range(num_images):
            orig_img = original[j].permute(1, 2, 0).detach().cpu().numpy()
            axes[0, j].imshow(orig_img)
            axes[0, j].set_title(f'Original {j + 1}')
            axes[0, j].axis('off')

            # Reconstruction Images
            recon_images = reconstructed_image[j].permute(1, 2, 0).detach().cpu().numpy()
            axes[1, j].imshow(recon_images)
            axes[1, j].set_title(f'Recon {j + 1}')
            axes[1, j].axis('off')
        if test:
            plt.tight_layout()
            plt.savefig(f'./output_images/individual_images/test_images/reconstructed_test_{epoch}_{batch}.png',
                        dpi=150)
            plt.close()
        else:
            plt.tight_layout()
            plt.savefig(f'./output_images/individual_images/reconstructed_images_{epoch}_{batch}.png', dpi=150)
            plt.close()

        for i in range(len(reconstructed_image)):
            if test:
                plt.imsave(os.path.join('./output_images/individual_images/test_images/',
                                        f'output_test_images_{epoch}_{batch}_{i}.png'),
                           reconstructed_image[i].permute(1, 2, 0).detach().cpu().numpy())
            else:
                plt.imsave(os.path.join('./output_images/individual_images/', f'output_images_{epoch}_{batch}_{i}.png'),
                           reconstructed_image[i].permute(1, 2, 0).detach().cpu().numpy())




def visualize_feature_maps(images, feature_maps, reconstructed, averaged_mu,multi_image_mode= True):

    if multi_image_mode:
        print('Visualising feature maps')
        batch_size = images.shape[0]
        num_imgs = images.shape[1]
        num_channels = feature_maps.shape[2]

        for i in range(batch_size):
            fig, ax = plt.subplots(nrows=5, ncols=6, figsize=(16, 8))

            for k in range(num_imgs+1):

                if k < num_imgs:
                    orig_img = images[i][k].permute(1, 2, 0).detach().cpu().numpy()
                    ax[0,k].imshow(orig_img)
                    ax[0,k].axis('off')
                else:
                    recon_image = reconstructed[i].permute(1, 2, 0).detach().cpu().numpy()
                    ax[0,k].imshow(recon_image)
                    ax[0, k].axis('off')


                for j in range(num_channels):
                    if k < num_imgs:
                        ax[j+1, 2].set_title(f'Channel {j}')
                        feature_map = feature_maps[i][k][j].detach().cpu()
                        ax[j+1,k].imshow(feature_map,cmap='gray')
                        ax[j+1,k].axis('off')
                    else:
                        feature_map = averaged_mu[i][j].detach().cpu()
                        ax[j + 1, k].imshow(feature_map, cmap='gray')
                        ax[j + 1, k].axis('off')

            plt.show()
    else:
        batch_size = images.shape[0]
        print(images.shape)
        print(feature_maps.shape)

        fig,ax = plt.subplots(nrows = 5,ncols=batch_size,figsize=(16,10))

        for i in range(batch_size):
            orig_img = images[i].permute(1, 2, 0).detach().cpu().numpy()
            ax[0, i].imshow(orig_img)
            ax[0, i].axis('off')

            for j in range(1,5):
                for k in range(batch_size):
                    feature_map = feature_maps[k][j-1].detach().cpu()
                    ax[j,k].imshow(feature_map,cmap='gray')
                    ax[j,k].axis('off')

        plt.show()





def latent_space_exploration(individual_mu, averaged_mu,target_mu,model):
    print('Latent exploration')

    for i in range(1):
        fig1, ax1 = plt.subplots(5,5,figsize = (16,8))
        fig1.suptitle('Latent Figures')
        start_index = i * 5
        end_index = start_index + 5
        individual_latents_2d, pca = get_latent_plot(individual_mu[start_index:end_index],averaged_mu[i], target_mu[i])
        #individual_latents_2d,pca = visualize_latent_space_individual(individual_mu)
        # Moving around in the learned manifold of the averaged latent space
        for i,x_val in enumerate(np.linspace(individual_latents_2d[:,0].min(), individual_latents_2d[:,0].max(), 5)):
            for j,y_val in enumerate(np.linspace(individual_latents_2d[:,1].min(), individual_latents_2d[:,1].max(), 5)):
                print('The xval and yval is ',x_val,y_val)
                z_inversed = pca.inverse_transform([[x_val,y_val]])
                z_inversed = torch.tensor(z_inversed,dtype=torch.float32,device= torch.device('cuda'))
                #print('The shape of z_inversed',z_inversed.shape)
                z_spatial = z_inversed.view(1,4,32,32)
                print('The shape of z_inversed', z_spatial.shape)
                with torch.no_grad():
                    reconstructed = model.decode(z_spatial).sample
                print('The shape of the reconstructed',reconstructed.shape)
                ax1[i,j].imshow(reconstructed[0].permute(1,2,0).detach().cpu().numpy())
                ax1[i,j].axis('off')
        plt.tight_layout()
        plt.show()


def visualize_agg_feature_map(images, common_feature_map,reconstructed):
        print('Visualising feature maps')
        batch_size = images.shape[0]
        num_imgs = images.shape[1]
        print(common_feature_map.shape)
        num_channels = common_feature_map.shape[1]

        for i in range(batch_size):
            fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(16, 8))

            for k in range(num_imgs):
                orig_img = images[i][k].permute(1, 2, 0).detach().cpu().numpy()
                ax[0, k].imshow(orig_img)
                ax[0, k].axis('off')

            for j in range(num_channels):
                ax[1,j].set_title(f'Channel {j}')
                feature_map = common_feature_map[i][j].detach().cpu()
                ax[1, j].imshow(feature_map, cmap='gray')
                ax[1, j].axis('off')

            ax[1, 4].set_title(f'Reconstructed')
            ax[1,4].imshow(reconstructed[i].permute(1,2,0).detach().cpu().numpy())
            ax[1,4].axis('off')

            plt.show()


def visualize_temp(batch, epoch, original_reconstruction, target_reconstruction, reconstructed, input_latents, common_feature_map,save_path):

    batch_size = original_reconstruction.shape[0]
    num_imgs = original_reconstruction.shape[1]
    print(input_latents[0].shape)
    print(common_feature_map.shape)
    num_channels = common_feature_map.shape[1]

    for i in range(batch_size):
        fig = plt.figure(figsize = (10,6),layout='constrained')
        gsp = gridspec.GridSpec(5,6,figure=fig,wspace=0,hspace=0)

        for j in range(num_imgs):
            ax = fig.add_subplot(gsp[0,j])
            ax.imshow(original_reconstruction[i][j].permute(1,2,0).detach().cpu().numpy())
            ax.axis('off')
            ax.set_title(f'Original Image {j}')

            for k in range(num_channels):
                #Feature maps of input latent of all images (channel wise)
                ax = fig.add_subplot(gsp[k+1, j])
                ind_latent = input_latents[i][j][k].detach().cpu().numpy()
                ax.imshow(ind_latent,cmap='viridis')
                ax.axis('off')

        ax = fig.add_subplot(gsp[0, num_imgs])
        ax.imshow(reconstructed[i].permute(1,2,0).detach().cpu().numpy())
        ax.axis('off')
        ax.set_title(f'Reconstructed Image {epoch}')

        for k in range(num_channels):
            ax = fig.add_subplot(gsp[k+1, num_imgs])
            common_fm = common_feature_map[i][k].detach().cpu().numpy()
            ax.imshow(common_fm, cmap='viridis')
            ax.axis('off')
            #plt.colorbar(common_fm,ax=ax)

        plt.savefig(os.path.join(save_path, f'reconstructed_images_{epoch}_{batch}_{i}.png'), dpi=150)
        plt.close()

def channelwise_visualisation(batch, epoch, original_reconstruction, target_reconstruction, reconstructed, input_latents, common_feature_map, model , save_path):

    batch_size = original_reconstruction.shape[0]
    num_imgs = original_reconstruction.shape[1]
    print('Original Reconstruction shape',original_reconstruction.shape)

    print('Input Latents',input_latents[0].shape)
    print('Input Latent keys ',input_latents.keys())
    print(common_feature_map.keys())
    #print('Common Feature map',common_feature_map.shape)
    #num_channels = common_feature_map[0].shape
    #print(num_channels)

    for i in range(batch_size):
        fig = plt.figure(figsize=(10, 6), layout='constrained')
        gsp = gridspec.GridSpec(5, 6, figure=fig, wspace=0, hspace=0)

        for j in range(num_imgs):
            ax = fig.add_subplot(gsp[0, j])
            ax.imshow(original_reconstruction[i][j].permute(1, 2, 0).detach().cpu().numpy())
            ax.axis('off')
            ax.set_title(f'Original Image {j}')

            for k in range(4):
                # Feature maps of input latent of all images (channel wise)
                ax = fig.add_subplot(gsp[k + 1, j])
                ind_latent = input_latents[k][i][j].permute(1,2,0).detach().cpu().numpy()
                ax.imshow(ind_latent)
                ax.axis('off')

        ax = fig.add_subplot(gsp[0, num_imgs])
        ax.imshow(reconstructed[i].permute(1, 2, 0).detach().cpu().numpy())
        ax.axis('off')
        ax.set_title(f'Reconstructed Image {epoch}')

        for k in range(4):
            ax = fig.add_subplot(gsp[k + 1, num_imgs])
            common_fm = common_feature_map[k][i].permute(1,2,0).detach().cpu().numpy()
            ax.imshow(common_fm)
            ax.axis('off')
            # plt.colorbar(common_fm,ax=ax)

        plt.savefig(os.path.join(save_path, f'reconstructed_images_{epoch}_{batch}_{i}.png'), dpi=150)
        plt.close()


def visualize_interpolation(images, z,vae):
    z1 = z[0][0]
    z2 = z[0][1]
    z3 = z[0][2]
    z4 = z[0][3]
    z5 = z[0][4]

    #z = z.reshape(-1, *z.shape[2:])  # (B*N,C,H,W) = (10,3,256,256)
    #original = vae.decode(z).sample

    z_inter = 0.25 * z1 + 0.25 * z2 +0.25 * z3 + 0.25 * z4 + 0.25 * z5
    print(z_inter.shape)
    decoded = vae.decode(z_inter.unsqueeze(0)).sample
    print(decoded.shape)


    fig = plt.figure(figsize=(16, 8))
    grid_spec = fig.add_gridspec(2, 5, hspace=0.1, wspace=0.1)

    for j in range(5):
        ax = fig.add_subplot(grid_spec[0, j])
        orig_img = images[0][j].permute(1, 2, 0).detach().cpu().numpy()
        ax.imshow(orig_img)
        ax.axis('off')

    ax = fig.add_subplot(grid_spec[1, 2])
    decoded_img = decoded[0].permute(1, 2, 0).detach().cpu().numpy()
    ax.imshow(decoded_img)
    ax.axis('off')

    plt.show()





