import os
import shutil
import torchvision
import cv2

# Debatching the images and creating a collection of train images
def debatching(batch_directory,preprocessing=False):
    destination_directory = 'C:\\Study\\Old Code\\Pattern_Recognition_Topo\\batched_data\\unprocessed'
    img_index = len(os.listdir(destination_directory))
    for batch in os.listdir(batch_directory):
        for image in os.listdir(os.path.join(batch_directory,batch)):
            if str(image).endswith('.png'):
                if preprocessing:
                    decoded_image = cv2.imread(os.path.join(batch_directory,batch,image))
                    resized_image = crop_resize(decoded_image)
                    cv2.imwrite(os.path.join(destination_directory,batch,f'image{img_index}.png'),resized_image)
                #shutil.move(os.path.join(batch_directory,batch,image),os.path.join(destination_directory,f'image{img_index}.png'))
                    img_index +=1


def crop_resize(image):
    image = image[100:550,50:1550]
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


            