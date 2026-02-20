from input_pipeline import *
from model_training.training import *
from utils import *
from model_training.model import *
from transfer_learning import *
#from VQ_VAE_transfer_learning import *
import cv2
from image_processing_approach import best_classical_method,create_ground_truth


def main():
    #Util function to batch and preprocess (crop and resize) the images.
    #debatching('./batched_data/unprocessed',preprocessing=True)

    #Getting Red and orange masks from the images.
    '''image_directory = './batched_data/test'
    for batch_id,batch in enumerate(os.listdir(image_directory)):
        images = []
        #if int(batch[-2:]) == 61:
            #print(batch[-2:])
        for image in os.listdir(os.path.join(image_directory,batch)):
            #print(os.path.join(image_directory,batch,image))
            images.append(cv2.imread(os.path.join(image_directory,batch,image)))
        print(len(images))
        get_red_orange_masks(images,batch)'''

    
    #Sanity Check on created continous masks
    '''image =cv2.imread('./batched_data_red_orange/Batch 2/image_0.png')
    np.savetxt("image_pixels.txt", image[:,:,2], fmt="%.1f")
    print(image[256,512])
    print(image[:,:,2].min(),image[:,:,2].max())'''

    #Image Processing - the baseline approach & prepares the ground truth images
    # source_directory = './batched_data_red_orange'
    # all_batches = False
    # if all_batches:
    #     for batch in os.listdir(source_directory):
    #         create_ground_truth(os.path.join(source_directory,batch),batch)
    # else:
    #     batch_directory = './batched_data_red_orange/Batch 4'
    #     create_ground_truth(batch_directory,'Batch 4')

    '''batch_directory = './batched_data_red_orange/test'
    #batches = 'Batch 50'
    for batches in os.listdir(batch_directory):
        create_ground_truth(os.path.join(batch_directory,batches),batches)'''
    '''for images in os.listdir(batch_directory):
        image_stack.append(cv2.imread(os.path.join(batch_directory,images)))'''
    

    #VAE Fine Tuning
    vae_tuner = VAEFineTuner(model_name='stabilityai/sd-vae-ft-mse')
    train_loader = vae_tuner.prepare_data('./batched_data_red_orange/train','./Ground Truth Images/train')
    test_loader = vae_tuner.prepare_data('./batched_data_red_orange/test/','./Ground Truth Images/test')
    #gt_data = preprocess_gt('./ground_truth')
    vae_tuner.finetune(train_loader,test_loader)
    
    #VQVAE Fine Tuning
    '''vq_vae_tuner = VQVAEFineTuner(prebatched_images=False)
    train_loader,test_loader = vq_vae_tuner.prepare_data('./batched_data/train/','./batched_data/test/',4, 256)
    gt_data = preprocess_gt('./ground_truth')
    vq_vae_tuner.finetune(train_loader,test_loader,gt_data)'''

    

if __name__ == "__main__":
    main()