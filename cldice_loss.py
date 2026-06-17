import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class SoftSkeletonize(torch.nn.Module):

    def __init__(self,num_iters =5):
        super(SoftSkeletonize,self).__init__()
        self.num_iters = num_iters

    def soft_erode(self,img):

        p1 = -F.max_pool2d(-img,kernel_size=(3,1),stride=(1,1),padding=(1,0))
        p2 = -F.max_pool2d(-img,kernel_size=(1,3),stride=(1,1),padding=(0,1))

        return torch.min(p1,p2)

    def soft_dilate(self,img):

        return F.max_pool2d(img,kernel_size=(3,3),stride=(1,1),padding=(1,1))

    def soft_open(self,img):

        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):

        img1 = self.soft_open(img)
        skel = F.relu(img - img1)

        #Iteratively add to the existing skeleton
        for j in range(self.num_iters):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def forward(self,img):
        return self.soft_skel(img)


def dice_coeff(reconstructed, target):

    # reconstructed_softmax = torch.softmax(reconstructed,dim=1)
    # print(reconstructed_softmax.shape)
    # reconstructed_white_areas = reconstructed_softmax[:,2]
    reconstructed_white_areas = reconstructed
    # cv2.imshow('White Structures',reconstructed_white_areas[0].unsqueeze(dim=-1).detach().cpu().numpy())
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    target_white_areas = (target==2).float()
    # cv2.imshow('Target White Structures', target_white_areas[0].unsqueeze(dim=-1).detach().cpu().numpy())
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    intersection = (reconstructed_white_areas * target_white_areas).sum(dim =(1,2))
    #print('Intersection: ', intersection)
    union = (reconstructed_white_areas + target_white_areas).sum(dim =(1,2))
    smooth = 1e-6
    dice = 1 - ((2 * intersection + smooth) / (union + smooth)).mean()
    return dice

class soft_dice_cldice(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, smooth = 1., exclude_background=True):
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha
        self.soft_skeletonize = SoftSkeletonize(num_iters=5)
        self.exclude_background = exclude_background

    def forward(self, y_true, y_pred,index):
        if self.exclude_background:
            y_true = y_true[:, 2:, :, :]
            y_pred = y_pred[:, 2:, :, :]
        dice = dice_coeff(y_true, y_pred)
        skel_pred = self.soft_skeletonize(y_pred)
        print(skel_pred.shape)
        skel_true = self.soft_skeletonize(y_true)
        fig, ax = plt.subplots(1,2,figsize=(10,5))

        ax[0].imshow(skel_true[0].permute(1,2,0).detach().cpu().numpy())
        ax[0].axis('off')

        ax[1].imshow(skel_pred[0].permute(1,2,0).detach().cpu().numpy())
        ax[1].axis('off')
        plt.savefig(f'./Skeletons_gt/skeleton_{index}.png',)
        plt.close()
        # plt.show()
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return (1.0-self.alpha)*dice+self.alpha*cl_dice
