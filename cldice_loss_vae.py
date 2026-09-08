import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class SoftSkeletonize(nn.Module):
    """Differentiable morphological skeletonization (soft-clDice, Shit et al. 2021)."""

    def __init__(self, num_iters: int = 5):
        super().__init__()
        self.num_iters = num_iters

    def soft_erode(self, img):
        p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
        p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
        return torch.min(p1, p2)

    def soft_dilate(self, img):
        return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def forward(self, img):
        skel = F.relu(img - self.soft_open(img))
        for _ in range(self.num_iters):
            img = self.soft_erode(img)
            delta = F.relu(img - self.soft_open(img))
            skel = skel + F.relu(delta - skel * delta)
        return skel


# def _check_mask(x, name):
#     """Guard against every representation bug this pipeline has produced so far."""
#     assert x.dim() == 4 and x.shape[1] == 1, \
#         f"{name}: expected (B,1,H,W), got {tuple(x.shape)}"
#     assert x.numel() > 0, \
#         f"{name}: empty tensor — channel selected twice? shape {tuple(x.shape)}"
#     assert x.min() >= 0.0 and x.max() <= 1.0, \
#         f"{name}: values in [{x.min():.3f}, {x.max():.3f}] — expected [0,1] probabilities, got logits?"


def soft_dice(member_prob, member_gt, smooth=1e-6):
    # """Soft Dice COEFFICIENT (higher = better), computed per sample and averaged.

    # member_prob: (B,1,H,W) float in [0,1] — probs[:, 2:3] from the trainer
    # member_gt:   (B,1,H,W) float {0,1}    — (target_label == 2).float().unsqueeze(1)
    # """
    # _check_mask(member_prob, "member_prob")
    # _check_mask(member_gt, "member_gt")
    # assert member_prob.shape == member_gt.shape

    # print('The shape of member gt', member_gt.shape)
    # print('The shape of member prob', member_prob.shape)

    intersection = (member_prob * member_gt).sum(dim=(1, 2, 3))
    denom        = (member_prob + member_gt).sum(dim=(1, 2, 3))
    return ((2.0 * intersection + smooth) / (denom + smooth)).mean()


class soft_dice_cldice(nn.Module):

    def __init__(self, alpha: float = 0.5, skel_iters: int = 5, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iters=skel_iters)

    def forward(self, member_prob, member_gt):
        # _check_mask(member_prob, "member_prob")
        # _check_mask(member_gt, "member_gt")
        # assert member_prob.shape == member_gt.shape

        dice_loss = 1.0 - soft_dice(member_prob, member_gt)

        skel_pred = self.soft_skeletonize(member_prob)
        skel_true = self.soft_skeletonize(member_gt)

        tprec = (torch.sum(skel_pred * member_gt) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(skel_true * member_prob) + self.smooth) / (torch.sum(skel_true) + self.smooth)
        cldice_loss = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)

        return (1.0 - self.alpha) * dice_loss + self.alpha * cldice_loss