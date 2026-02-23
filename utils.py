import os
import shutil
import torchvision
import cv2
import numpy as np

# Debatching the images and creating a collection of train images
def debatching(batch_directory,preprocessing=False):
    destination_directory = './batched_data/unprocessed/Processed'
    #img_index = len(os.listdir(destination_directory))
    for batch in os.listdir(batch_directory):
        img_index=0
        for image in os.listdir(os.path.join(batch_directory,batch)):
            if str(image).endswith('.png'):
                if preprocessing:
                    if not os.path.exists(os.path.join(destination_directory,batch)):
                        os.mkdir(os.path.join(destination_directory,batch))
                    decoded_image = cv2.imread(os.path.join(batch_directory,batch,image))
                    resized_image = crop_resize(decoded_image)
                    cv2.imwrite(os.path.join(destination_directory,batch,f'image{img_index}.png'),resized_image)
                #shutil.move(os.path.join(batch_directory,batch,image),os.path.join(destination_directory,f'image{img_index}.png'))
                    img_index +=1


def crop_resize(image):
    image = image[250:435,50:1550]
    #image = image[:,500:1100]
    #image = cv2.resize(image,(256,256))
    #print(image.type)
    cv2.imshow('Image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return image


def rename(dir):
    img_index = 94
    for image in os.listdir(dir):
        if str(image).endswith('.png'):
                image = cv2.imread(os.path.join(dir,image))
                resized_image = cv2.resize(image,(1024,200))
                cv2.imshow('Image',resized_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                #image = cv2.resize(image, (512,512))
                #cv2.imwrite(os.path.join(dir,f'image{img_index}.png'),image)
                #os.rename(os.path.join(dir,f'image{img_index}.png'),os.path.join(dir,f'image{img_index+94}.png'))
                img_index+=1


def get_red_orange_masks(raw_images,batch):
    red_masks = []
    for index,image in enumerate(raw_images):
        print(index)
        # Convert to HSV color space for better separability
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define HSV color ranges for red and orange 
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])

        #Red needs two ranges as red wraps around in the HSV schema so it covers both ends of the range. --- No change one range is enough
        red_lower2 = np.array([170,100,100])
        red_upper2 = np.array([180,255,255])

        orange_lower = np.array([5,50,50])
        orange_upper = np.array([15,255,255])

        lower_white = np.array([255, 255, 255]) 
        upper_white = np.array([255, 255, 255])

        # Create the mask
        white_mask = cv2.inRange(image, lower_white, upper_white)

        # Create masks
        mask_red = cv2.inRange(image_hsv, red_lower1, red_upper1)
        #mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
        #mask_red = cv2.bitwise_or(mask_red1,mask_red2)

        mask_orange = cv2.inRange(image_hsv,orange_lower,orange_upper)
        mask = cv2.bitwise_or(mask_red,mask_orange)
        
        
        
        colored_mask = np.zeros_like(image)


        # Black → Gray
        
        colored_mask[:] = [128, 128, 128]

        colored_mask[white_mask==255] = [255,255,255]
        
        colored_mask[mask==255] = image[mask ==255]

        '''cv2.imshow('Colored Mask',colored_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''



        if os.path.isdir(os.path.join('./batched_data_red_orange/train','Batch 43')):
            cv2.imwrite(os.path.join('./batched_data_red_orange/train','Batch 43',f'image_{index}.png'),colored_mask)
        else:
            os.mkdir(os.path.join('./batched_data_red_orange/train','Batch 43'))
            cv2.imwrite(os.path.join('./batched_data_red_orange/train','Batch 43',f'image_{index}.png'),colored_mask)

        '''coords_of_interest = np.where(mask_red == 255)
        # White → Red
        for w_pixel,h_pixel in zip(*coords_of_interest):
            print(image[w_pixel,h_pixel])
            colored_mask[w_pixel,h_pixel] = image[w_pixel,h_pixel]   # OpenCV uses BGR
        #reconstructed_image = np.zeros((extracted_structures.shape[0], extracted_structures.shape[1],3),dtype=np.uint8)
        #reconstructed_image[extracted_structures == 0] = [169, 169, 169]'''

        


            