import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import torchvision
import os
import json
import matplotlib.pyplot as plt

class SimpleCustomBatch(Dataset):

    def __init__(self, image_directory, output_file=None, transform=None,target_transform = None):

        self.image_directory = image_directory
        self.transform =transform
        

    # Override the __getitem__() method
    def __getitem__(self,index):

        image_path = os.path.join(self.image_directory,f'image{index}'+'.png')
        image = torchvision.io.decode_image(image_path)
        if self.transform:
            image = self.transform(image)
        return image

        #Just to check the input images to the model
        '''image2 = image.permute(1,2,0)
        image2 = image2.clip(0,1)
        #print('The shape of the image is ',image2.shape)        
        plt.imsave(f'transformed_train_images\\Img_{index}.png',image2)'''
        #Parallely stack all the images

    def __len__(self):
        return len(os.listdir(self.image_directory))
    

def show_image(image, label):
    image_show = image.permute(1,2,0) 
    plt.figure(figsize=(6,8))
    plt.title(label)
    plt.imshow(image_show)
    plt.show()


def input_pipeline():
    config_file = open('./config.json')
    config_data = json.load(config_file)

    #Fetching the train and test images from folders
    train_images = config_data['train']['image_directory']
    test_images = config_data['test']['image_directory']

    #Image Transformations
    train_transformations = v2.Compose([
        v2.ToDtype(torch.float32, scale = True), # Scale = true already scales the iamges between 0,1.
        v2.Resize((256,256)),
        v2.ToTensor(),
        #v2.Normalize(mean=(0.5,), std= (1.0,)),
    ])

    test_transformations = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((256,256)),
        v2.ToTensor(),
        #v2.Normalize(mean=(0.5,), std= (1.0,)),
    ])

    #Create a Dataset
    train_dataset = SimpleCustomBatch(image_directory=train_images,transform=train_transformations)
    #train_subset = torch.utils.data.Subset(train_dataset,range(10))
    test_dataset = SimpleCustomBatch(image_directory =test_images,transform=test_transformations)

    '''print('The length of dataset',len(train_dataset))
    for i, image in enumerate(train_dataset):
        print(image.dtype)'''

    #Transform the Dataset into DataLoader 
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=config_data['batch_size'],shuffle=True, num_workers=2, pin_memory=True)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=config_data['batch_size'],shuffle=False, num_workers=2, pin_memory=True)

    # Testing the batches of data for all the steps done above 
    '''first_img_batch = next(iter(train_dataloader))
    print("The shape of first batch of images",first_img_batch.shape)
    #print("The shape of first batch of labels",first_labels_batch.shape)
    image01 = first_img_batch[0]
    image01 = image01.squeeze()         # Removes the dimensions with value 1
    show_image(image01)'''
    return train_dataloader,test_dataloader


