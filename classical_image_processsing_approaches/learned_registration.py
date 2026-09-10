"""
registered_consensus.py

Differentiable groupwise registration front-end + consensus network for
topology optimization (TO) consensus extraction.

Pipeline per batch of N (=5) TO images:
    1. Build an emergent template (initially the pixel mean of the batch).
    2. A FlowNet-Simple style FlowHead predicts a dense per-pixel flow
       for each image relative to the template (the localization network).
    3. The flow is hard-capped to +/- max_disp pixels via tanh
       (genuine-alignment-vs-fabrication rule).
    4. An STN-style warp (base identity grid + normalized flow ->
       F.grid_sample) aligns each image to the template. Fully
       differentiable, so gradients from the downstream consensus loss
       train the registration too.
    5. Optionally iterate: rebuild the template as the mean of the
       warped images (unbiased emergent-template construction).
    6. The warped stack goes into ConsensusNet, which predicts the
       consensus structure map.

STN mapping:
    FlowHead                     -> localization network (dense, not affine)
    base_grid + normalized flow  -> grid generator
    F.grid_sample                -> grid sampler

Usage:
    model = RegisteredConsensus(n_inputs=5, max_disp=8.0, n_template_iters=2)
    out = model(x)          # x: [B, N, 1, H, W], values in [0, 1]
    out["consensus"]        # [B, 1, H, W]
    out["warped"]           # [B, N, 1, H, W]
    out["flows"]            # [B, N, 2, H, W]  (pixel units, capped)
    out["template"]         # [B, 1, H, W]

    loss = (consensus_loss(out["consensus"], target)
            + lam_reg    * registration_loss(out["warped"], out["template"])
            + lam_smooth * flow_smoothness(out["flows"]))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import os
import numpy as np
import torchvision
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader,Subset
from torchinfo import summary
import cv2
#from input_pipeline_discrete import *
from input_pipeline import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_dilation
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from cldice_loss import *
from metrics import compute_betti_numbers
import collections
from torch.utils.data import ConcatDataset
from visualizations.thesis_visualisation import *
from plot_registration_steps import *

# ----------------------------------------------------------------------
# 1. FlowHead  --  FlowNet-Simple style dense flow predictor (localization net)
# ----------------------------------------------------------------------
class FlowHead(nn.Module):
    """
    Predicts a dense 2-channel flow field aligning `moving` to `template`.

    Input : concat([moving, template], dim=1)  ->  [B, 2, H, W]
    Output: raw flow                            ->  [B, 2, H, W]

    Encoder-decoder with skip connections (FlowNet-Simple flavour). The
    final conv is zero-initialised so the module starts as the identity
    warp (zero flow) -- the standard STN/VoxelMorph stability trick that
    also guards against the identity-collapse/explode failure mode.
    """

    def __init__(self, base_ch: int = 32):
        super().__init__()

        def conv(i, o, k=3, s=1):
            return nn.Sequential(
                nn.Conv2d(i, o, k, stride=s, padding=k // 2),
                nn.LeakyReLU(0.1, inplace=True),
            )

        # encoder (downsample x2 twice)
        self.enc0 = conv(2, base_ch)                 # H
        self.enc1 = conv(base_ch, base_ch * 2, s=2)  # H/2
        self.enc2 = conv(base_ch * 2, base_ch * 4, s=2)  # H/4
        self.bott = conv(base_ch * 4, base_ch * 4)   # H/4

        # decoder (upsample back, with skips)
        self.up1 = conv(base_ch * 4 + base_ch * 2, base_ch * 2)  # after cat enc1
        self.up0 = conv(base_ch * 2 + base_ch, base_ch)          # after cat enc0

        # zero-initialised flow prediction -> identity warp at init
        self.flow = nn.Conv2d(base_ch, 2, 3, padding=1)
        nn.init.zeros_(self.flow.weight)
        nn.init.zeros_(self.flow.bias)

    def forward(self, moving, template):
        x = torch.cat([moving, template], dim=1)     # [B, 2, H, W]

        e0 = self.enc0(x)                            # H
        e1 = self.enc1(e0)                           # H/2
        e2 = self.enc2(e1)                           # H/4
        b = self.bott(e2)                            # H/4

        d1 = F.interpolate(b, scale_factor=2, mode="bilinear",
                           align_corners=True)        # H/2
        d1 = self.up1(torch.cat([d1, e1], dim=1))

        d0 = F.interpolate(d1, scale_factor=2, mode="bilinear",
                           align_corners=True)        # H
        d0 = self.up0(torch.cat([d0, e0], dim=1))

        return self.flow(d0)                         # [B, 2, H, W], raw flow


# ----------------------------------------------------------------------
# 2. warp  --  STN grid generator + differentiable bilinear sampler
# ----------------------------------------------------------------------
def warp(img, raw_flow, max_disp: float):
    """
    Apply an STN-style dense warp to `img` using `raw_flow`.

    img      : [B, 1, H, W]
    raw_flow : [B, 2, H, W]  (channel 0 = x/col disp, channel 1 = y/row disp)
    returns  : (warped_img [B, 1, H, W], capped_flow [B, 2, H, W] in pixels)

    - tanh caps displacement to +/- max_disp pixels (no fabricated long-range
      correspondence).
    - flow (pixels) is converted to grid_sample's normalised [-1, 1] units.
    - grid_sample is a "pull" resampler and is differentiable wrt the grid,
      so gradients reach raw_flow and hence FlowHead's weights.
    """
    B, _, H, W = img.shape

    # hard displacement cap
    flow = max_disp * torch.tanh(raw_flow)           # pixels, capped

    # identity base grid in normalised coords; align_corners=True everywhere
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, H, device=img.device),
        torch.linspace(-1, 1, W, device=img.device),
        indexing="ij",
    )
    base = torch.stack((xs, ys), dim=-1)             # [H, W, 2] -> (x, y)
    base = base.unsqueeze(0).expand(B, -1, -1, -1)   # [B, H, W, 2]

    # pixel flow -> normalised grid units (1 px = 2/(size-1))
    fx = flow[:, 0] / (W - 1) * 2.0                  # [B, H, W]
    fy = flow[:, 1] / (H - 1) * 2.0
    grid = base + torch.stack((fx, fy), dim=-1)      # [B, H, W, 2]

    out = F.grid_sample(img, grid, mode="bilinear",
                        padding_mode="border", align_corners=True)
    return out, flow


# ----------------------------------------------------------------------
# 3. ConsensusNet  --  downstream head over the aligned stack
# ----------------------------------------------------------------------
class ConsensusNet(nn.Module):
    """
    Small conv head that maps the warped/aligned stack (concatenated over
    the N inputs) to a single-channel consensus logit map.

    Input : [B, N, H, W]  (N aligned images stacked on the channel axis)
    Output: [B, 1, H, W]  consensus logits (apply sigmoid outside)
    """

    def __init__(self, n_inputs: int,  n_classes: int = 3, base_ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_inputs, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch, n_classes, 1),
        )

    def forward(self, aligned_stack):
        return self.net(aligned_stack)               # [B, 1, H, W] logits


# ----------------------------------------------------------------------
# 4. RegisteredConsensus  --  full module
# ----------------------------------------------------------------------
class RegisteredConsensus(nn.Module):
    def __init__(self,n_inputs=5, n_classes=3, max_disp=5.0, n_template_iters=2, base_ch=32):
        super().__init__()
        self.n_inputs = n_inputs
        self.max_disp = max_disp
        self.n_template_iters = max(1, n_template_iters)

        self.flow_head = FlowHead(base_ch=base_ch)
        self.consensus = ConsensusNet(n_inputs=n_inputs, base_ch=base_ch)

    def _align_to_template(self, x, template):
        """
        Warp every image in the batch-of-N to the current template.

        x        : [B, N, 1, H, W]
        template : [B, 1, H, W]
        returns  : warped [B, N, 1, H, W], flows [B, N, 2, H, W]
        """
        B, N, _, H, W = x.shape
        warped, flows = [], []
        for i in range(N):
            mov = x[:, i]                            # [B, 1, H, W]
            raw = self.flow_head(mov, template)      # [B, 2, H, W]
            w, f = warp(mov, raw, self.max_disp)
            warped.append(w)
            flows.append(f)
        warped = torch.stack(warped, dim=1)          # [B, N, 1, H, W]
        flows = torch.stack(flows, dim=1)            # [B, N, 2, H, W]
        return warped, flows

    def forward(self, x):
        """
        x : [B, N, 1, H, W]  in [0, 1]
        """

        if x.dim() == 4:
            x = x.unsqueeze(2)                    # [B,N,H,W] -> [B,N,1,H,W]
        print('The shape of the input', x.shape)
        B, N, _, H, W = x.shape
        
        assert N == self.n_inputs, f"expected N={self.n_inputs}, got {N}"

        # emergent template: start from the pixel mean of the batch
        template = x.mean(dim=1)                      # [B, 1, H, W]

        warped, flows = None, None
        for _ in range(self.n_template_iters):
            warped, flows = self._align_to_template(x, template)
            # unbiased template update: mean of the aligned stack
            template = warped.mean(dim=1)            # [B, 1, H, W]

        # consensus over the aligned stack
        aligned_stack = warped.squeeze(2)            # [B, N, H, W]
        consensus = self.consensus(aligned_stack)    # [B, 1, H, W] logits

        return {
            "consensus": consensus,                  # logits
            "warped": warped,                        # [B, N, 1, H, W]
            "flows": flows,                          # [B, N, 2, H, W] pixels
            "template": template,                    # [B, 1, H, W]
        }


# ----------------------------------------------------------------------
# 5. Losses
# ----------------------------------------------------------------------
def registration_loss(warped, template):
    """MSE of each aligned image against the emergent template."""
    return F.mse_loss(warped, template.unsqueeze(1).expand_as(warped))


def flow_smoothness(flows):
    """
    Total-variation style smoothness on the flow fields.
    flows : [B, N, 2, H, W]
    """
    dx = flows[..., :, 1:] - flows[..., :, :-1]
    dy = flows[..., 1:, :] - flows[..., :-1, :]
    return dx.abs().mean() + dy.abs().mean()

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

def grouped_split(group_of_item, fold, ratios=(0.7, 0.2, 0.1), seed_base=0):
    group_of_item = np.asarray(group_of_item)
    groups = np.unique(group_of_item)
    sizes = {g: int((group_of_item == g).sum()) for g in groups}
    total = len(group_of_item)
    targets = np.array(ratios) * total            # target sample counts: train/test/val [84, 21, 10.5]

    rng = np.random.default_rng(seed_base + fold)  # Range Generator
    order = rng.permutation(groups)             #[ 2 10  3 12  0  4  7  5 16 13 14 11  6  9 17  8  1 15] - different main batch order for  each fold

    assigned = {0: set(), 1: set(), 2: set()}      # 0=train, 1=test, 2=val
    counts = np.zeros(3)
    for g in order:
        deficit = (targets - counts) / targets     # who's most under quota
        s = int(np.argmax(deficit))
        assigned[s].add(g)
        counts[s] += sizes[g]

    idx = lambda s: np.array([i for i, g in enumerate(group_of_item) if g in assigned[s]])
    train_idx, test_idx, val_idx = idx(0), idx(1), idx(2)  # train, val,test

    id_to_group = {0: 'Anowas V4 Bottom', 1: 'Anowas V4 Side View' ,2:'Anowas V4 Top', 3:'Bottom Inside View V3', 4:'Bottom View MW',
                   5: 'Bottom View V3',6: 'Front View V3', 7:'Intermediate Floor V3', 8: 'Jermias Side', 9:'Jermias_bottom', 10: 'Rear View MW',
                   11: 'Side View MW', 12: 'Side View V3', 13: 'Top View V3', 14: 'gen WK Front', 15: 'gen WK Side', 16: 'gen WK Top', 17: 'gn WK Bottom'}
    for name, split_idx in [("train", train_idx), ("test", test_idx), ("val", val_idx)]:
        ids = sorted(set(group_of_item[split_idx].tolist()))
        print(f"{name}: {len(split_idx)} sub-batches across {len(ids)} views")
        for i in ids:
             print(f"    {id_to_group[i]}")
        split_log = {name: [id_to_group[i] for i in sorted(set(group_of_item[idx].tolist()))]
                     for name, idx in [("train", train_idx), ("test", test_idx), ("val", val_idx)]}
    with open(f"split_fold{fold}.json", "w") as f:
        json.dump(split_log, f, indent=2)

    assert not (set(group_of_item[train_idx]) & set(group_of_item[test_idx]))
    assert not (set(group_of_item[train_idx]) & set(group_of_item[val_idx]))
    assert not (set(group_of_item[test_idx]) & set(group_of_item[val_idx]))

    return train_idx, test_idx, val_idx


def train_grayscale():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size=1

    single_image_reconstruction = False

    data_dir = 'data/Batches'
    ground_truth_dir = 'data/Ground_Truth'
    image_processing_dir = 'data/Image Processing Results'

    # data_dir = 'data/Bigger Batches of Data_original_size'
    # ground_truth_dir = 'data/Edited Ground Truth_ Encoded'
    # image_processing_dir = 'data/Image Processing Original Size Grayscale'
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

        #model = simple_Conv(5, 3, kernel_size=3, stride=1, padding=1)
        #model = torch.load('./convolutional_network_chkpoint/convolutional_chkpoint.pth',weights_only=False)
        model = RegisteredConsensus(n_inputs=5, n_classes=3, max_disp=3.0, n_template_iters=2)
        model = model.to(device)
        # print(summary(model))

        train_dataset = TopoDatasetImg(items,train_idx,ground_truth_dir= ground_truth_dir, img_proc_dir = image_processing_dir, augment= True)
        val_dataset = TopoDatasetImg(items,val_idx,ground_truth_dir= ground_truth_dir, img_proc_dir = image_processing_dir, augment=False)
        test_dataset = TopoDatasetImg(items,test_idx,ground_truth_dir= ground_truth_dir, img_proc_dir = image_processing_dir,augment=False)

        # opento_dataset_dir = create_image_groups('data/Consensus Dataset/subbatches')
        # opento_gt_dir = 'data/Consensus Dataset/ground_truth'
        # opento_img_prc_dir = 'data/Consensus Dataset/image_proc' 
        # openTO_train_dataset = OpenTOConsensusDatasetdisc(opento_dataset_dir,opento_gt_dir,opento_img_prc_dir, augment = True)

        # train_combined = ConcatDataset([train_dataset, openTO_train_dataset])
        
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=4,pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=4,pin_memory=True)

        lr = 0.001
        epochs = 250

        optimizer1 = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay = 1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=300)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer1,gamma= 0.95)

        #weights = torch.tensor([0.25, 0.35, 0.5])
        #crossentropy_loss = torch.nn.CrossEntropyLoss(weight=weights).to(device)
        crossentropy_loss = torch.nn.CrossEntropyLoss().to(device)
        cl_dice = soft_dice_cldice()

        save_parent_dir = f'Learned Registration Results_original size_wOpenTOData{fold}'
        best_loss = float('inf')

        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="veenasridhar96-university-of-stuttgart",
            name='Learned Registration Topology Optimization (No Residual Config)',
            config={
                "learning_rate": lr,
                "epochs": epochs,
                "loss": 'CE+CLDice',
                "architecture": "CNN",
                "dataset": "TO",
                "epochs": epochs,
                "fold": fold,
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

                out = model(images)
                output_logits = out["consensus"]
                                                      

                output_softmax = torch.softmax(output_logits, dim=1)
                output = torch.argmax(output_logits,dim=1)                          # Using argmax since the output is only raw logits and argmax makes you choose the maximum of the labels. So the 3 channels (3 classes) is converted to 1 channel the max value.
                targets_onehot = F.one_hot(labels, num_classes=3) # (B, H, W, C)
                targets_onehot = targets_onehot.permute(0, 3, 1, 2)  # (B, C, H, W)
                targets_onehot = targets_onehot.float()  # scatter_ gives long by default


                cross_entropy_loss = crossentropy_loss(output_logits, labels)
                #sdf_loss = F.l1_loss(sdf_pred, gt_sdf)
                dice_loss = cl_dice(targets_onehot,output_softmax)
                img_processing_dice_loss = cl_dice(targets_onehot,avg_output_one_hot.float())
                
                reg_loss    = registration_loss(out["warped"], out["template"])
                smooth_loss = flow_smoothness(out["flows"])
                loss = cross_entropy_loss + dice_loss + reg_loss + smooth_loss
                #loss = 0.4* cross_entropy_loss + 0.6* dice_loss
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
                    if not os.path.exists(os.path.join(save_parent_dir)):
                        os.mkdir(os.path.join(save_parent_dir))
                    if not os.path.exists(os.path.join(save_parent_dir,'train')):
                        os.mkdir(os.path.join(save_parent_dir,'train'))
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
                        # plt.imsave(f'{save_parent_dir}/train/Epoch {epoch}/result_{train_batch}.png',output[i].detach().cpu().numpy(),cmap='gray',vmin=0,vmax=2)
                        # plt.close()
                        
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

                        # dl_sdf, normalised_dl_sdf= sdf_distance_function(output[i].detach().cpu().numpy())
                        # img_prc_sdf,normalised_img_prc_sdf = sdf_distance_function(avg_output[i].squeeze(0).detach().cpu().numpy())
                        # gt_sdf, normlaised_gt_sdf = sdf_distance_function(labels[i].detach().cpu().numpy())
                        # dl_sdf_mse = np.mean((dl_sdf - gt_sdf) ** 2)
                        # img_prc_sdf_mse = np.mean((img_prc_sdf - gt_sdf) ** 2)
                        # print(dl_sdf_mse.shape)
                        
                        iou,img_prc_iou =  visulaize_iou_metrics(raw_images=images[i], epoch=epoch, batch=train_batch, predicted_structures = output[i],
                                                                  gt = labels[i],image_processing_result=avg_output[i],betti_error_pred=cl_dice_dl,betti_error_img_prc=cl_dice_img_prc,save_path= f'{save_parent_dir}/train')
                        if i <10:
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
                                'iou_dl':iou,
                                'iou_img_prc': img_prc_iou
                            }
                            visualize_comparison_with_overlay(
                                groups=[group],
                                structure_val=2,  # your label index for structure class
                                save_path=f'{save_parent_dir}/train/Epoch {epoch}/IoU Metric',
                                batch=train_batch
                            )
                        
                        plot_registration_steps(out["warped"][i].detach().cpu().numpy(),# "warped": warped, [B, N, 1, H, W]
                        out["flows"][i].detach().cpu().numpy(),     # "flows": flows, [B, N, 2, H, W] pixels                                                       
                        out["template"][i].detach().cpu().numpy(),  # "template": template, [B, 1, H, W]
                        save_path=f'{save_parent_dir}/train/Epoch {epoch}/registration_steps',batch = train_batch)
                        
                        
                        # plot2_overlay(labels[i].detach().cpu().numpy(),output[i].detach().cpu().numpy(),avg_output[i].squeeze(0).detach().cpu().numpy(),(cl_dice_dl,cl_prec,cl_rec),
                        #               (cl_dice_img_prc,cl_prec_i,cl_rec_i),save_path=f'{save_parent_dir}/train/Epoch {epoch}/IoU Metric',batch = train_batch)
                        
                        train_iou_dl += iou
                        train_iou_img_prc += img_prc_iou
                        train_cl_dice_image_prc += cl_dice_img_prc
                        train_cl_dice_dl += cl_dice_dl
                        
                        train_batch += 1

            scheduler.step()
            print(f'Train Loss for Epoch {epoch}', train_loss / len(train_dataloader))
            # print('Train IoU for deep learning', train_iou_dl / len(train_idx))
            # print('Train IoU for image processing ', train_iou_img_prc / len(train_idx))

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

                        out = model(images)
                        output_logits = out["consensus"]
                        output_softmax = torch.softmax(output_logits.detach(), dim=1)
                        output = torch.argmax(output_logits,
                                              dim=1)  # Using argmax since the output is only raw logits and argmax makes you choose the maximum of the labels. So the 3 channels (3 classes) is converted to 1 channel the max value.
                        
                        targets_onehot = F.one_hot(labels, num_classes=3)  # (B, H, W, C)
                        targets_onehot = targets_onehot.permute(0, 3, 1, 2)  # (B, C, H, W)
                        targets_onehot = targets_onehot.float()

                        
                        cross_entropy_loss = crossentropy_loss(output_logits, labels)

                        #sdf_loss = F.l1_loss(sdf_pred, gt_sdf)
                        dice_loss = cl_dice(targets_onehot,output_softmax)
                        img_processing_dice_loss = cl_dice(targets_onehot,avg_output_one_hot.float())

                        reg_loss    = registration_loss(out["warped"], out["template"])
                        smooth_loss = flow_smoothness(out["flows"])
                        loss = 0.4*cross_entropy_loss + 0.6*dice_loss + 0.1*reg_loss + 0.01*smooth_loss
                        
                        #loss = 0.3* cross_entropy_loss + 0.7* dice_loss

                        val_img_proc_dice_loss += img_processing_dice_loss
                        # val_cross_entropy_loss += cross_entropy_loss.item()
                        #val_sdf_loss = sdf_loss.item()
                        val_dice_loss +=  dice_loss
                        val_loss += loss.item()

                        # if epoch % 10 == 0:
                        if not os.path.exists(os.path.join(save_parent_dir,'val')):
                            os.mkdir(os.path.join(save_parent_dir,'val'))
                        if not os.path.exists(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch))):
                            os.mkdir(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch)))
                        if not os.path.exists(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch), 'sdf_map')):
                            os.mkdir(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch), 'sdf_map'))
                        if not os.path.exists(
                                os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch), 'IoU Metric')):
                            os.mkdir(os.path.join(save_parent_dir, 'val', 'Epoch ' + str(epoch), 'IoU Metric'))
                        for i in range(len(output)):
                            # plt.axis('off')
                            # plt.imsave(f'{save_parent_dir}/val/Epoch {epoch}/result_{val_batch}.png',
                            #            output[i].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=2)
                            # plt.close()
                            
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
                                'iou_dl':iou_dl,
                                'iou_img_prc': iou_img_prc,
                            }
                            visualize_comparison_with_overlay(
                                groups=[group],
                                structure_val=2,  # your label index for structure class
                                save_path=f'{save_parent_dir}/val/Epoch {epoch}/IoU Metric',
                                batch=val_batch
                            )
                            # plot2_overlay(labels[i].detach().cpu().numpy(), output[i].detach().cpu().numpy(),
                            #               avg_output[i].squeeze(0).detach().cpu().numpy(),
                            #               (cl_dice_dl, cl_prec, cl_rec),
                            #               (cl_dice_img_prc, cl_prec_i, cl_rec_i),
                            #               save_path=f'{save_parent_dir}/val/Epoch {epoch}/IoU Metric',
                            #               batch=val_batch)
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
            # print('Val IoU for deep learning', val_iou_dl / len(val_idx))
            # print('Val IoU for image processing ', val_iou_img_prc / len(val_idx))

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

                        out = model(images)
                        output_logits = out["consensus"]
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
                        dice_loss = cl_dice(targets_onehot,output_softmax)
                        image_processing_dice_loss = cl_dice(targets_onehot,avg_output_one_hot.float())
                        # boundary_loss = sdf_loss(output_softmax, gt_sdf_white)
                        # loss = cross_entropy_loss + sdf_loss
                        reg_loss    = registration_loss(out["warped"], out["template"])
                        smooth_loss = flow_smoothness(out["flows"])
                        loss = 0.4*cross_entropy_loss + 0.6*dice_loss + 0.1*reg_loss + 0.01*smooth_loss
                        test_image_processing_dice_loss += image_processing_dice_loss
                        # test_cross_entropy_loss += cross_entropy_loss.item()
                        #test_sdf_loss += sdf_loss.item()
                        test_dice_loss +=  dice_loss
                        test_loss += loss.item()

                        # if epoch %10 == 0:
                        if not os.path.exists(os.path.join(save_parent_dir,'test')):
                            os.mkdir(os.path.join(save_parent_dir,'test'))
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

                            iou_dl, iou_img_prc = visulaize_iou_metrics(raw_images=images[i], epoch=epoch,
                                                                         batch=test_batch,
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
                                'iou_dl':iou_dl,
                                'iou_img_prc': iou_img_prc,
                            }
                            visualize_comparison_with_overlay(
                                groups=[group],
                                structure_val=2,  # your label index for structure class
                                save_path=f'{save_parent_dir}/test/Epoch {epoch}/IoU Metric',
                                batch=test_batch
                            )
                            # plot2_overlay(labels[i].detach().cpu().numpy(), output[i].detach().cpu().numpy(),
                            #               avg_output[i].squeeze(0).detach().cpu().numpy(),
                            #               (cl_dice_dl, cl_prec, cl_rec),
                            #               (cl_dice_img_prc, cl_prec_i, cl_rec_i),
                            #               save_path=f'{save_parent_dir}/test/Epoch {epoch}/IoU Metric',
                            #               batch=test_batch)
                            # iou_dl, iou_img_prc = visulaize_iou_metrics(raw_images=images[i], epoch=epoch,
                            #                                             batch=test_batch,
                            #                                             predicted_structures=output[i], gt=labels[i],
                            #                                             image_processing_result=avg_output[i],
                            #                                             betti_error_pred=cl_dice_dl,
                            #                                             betti_error_img_prc=cl_dice_img_prc,
                            #                                             save_path=f'{save_parent_dir}/test')
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
                # print('Test IoU for deep learning', test_iou_dl / len(test_idx))
                # print('Test IoU for image processing ', test_iou_img_prc / len(test_idx))

                if test_loss / len(test_dataloader) < best_loss:
                    best_loss = test_loss / len(test_dataloader)
                    torch.save(model,'./convolutional_network_chkpoint/convolutional_chkpoint.pth')
                    print(f'Model Saved with test loss {test_loss / len(test_dataloader):.4f}')

                run.log({'epoch': epoch,
                        'alpha': alpha,
                        'fold': fold,
                        # 'learning_rate': scheduler.get_last_lr()[0],
                        'Train_Total_Loss': train_loss / len(train_dataloader),
                        'Test_Total_Loss': test_loss / len(test_dataloader),
                        'Val Loss': val_loss / len(val_dataloader),
                        # 'Train_CE_Loss': train_cross_entropy_loss / len(train_dataloader),'Test_CE_Loss': test_cross_entropy_loss / len(test_dataloader),
                        'Train IoU': train_iou_dl / len(train_dataloader.dataset),
                        'Train IoU_img': train_iou_img_prc / len(train_dataloader.dataset),
                        'Test IoU': test_iou_dl / len(test_dataloader.dataset),
                        'Test IoU_img': test_iou_img_prc /len(test_dataloader.dataset),
                        'Val IoU': val_iou_dl / len(val_dataloader.dataset), 'Val IoU img': val_iou_img_prc / len(val_dataloader.dataset),
                        'Train_Img_Processing_Dice Loss': train_img_proc_dice_loss/len(train_dataloader),
                        'Val_Img_Proccessing_Dice Loss': val_img_proc_dice_loss/len(val_dataloader),
                        'Test_Img_Processing_Dice Loss': test_image_processing_dice_loss/len(test_dataloader),
                        'Train Dice Loss': train_dice_loss / len(train_dataloader), 'Test_Dice_Loss': test_dice_loss / len(test_dataloader),
                        'Val Dice Loss': val_dice_loss/len(val_dataloader),

                        'Train_Img_Processing_Cldice': train_cl_dice_image_prc / len(train_dataloader.dataset),
                        'Val_Img_Proccessing_Cldice': val_cl_dice_image_prc / len(val_dataloader.dataset),
                        'Test_Img_Processing_Cldice': test_cl_dice_img_proc / len(test_dataloader.dataset),
                        'Train Cldice DL': train_cl_dice_dl / len(train_dataloader.dataset),
                        'Test Cldice DL': test_cl_dice_dl / len(test_dataloader.dataset),
                        'Val Cldice DL': val_cl_dice_dl / len(val_dataloader.dataset),
                        })

        wandb.finish()

def extract_red_mask(image):
    #print(image.min(),image.max())
    R = image
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


    print('The unique of predicted structures',predicted_structures.shape)
    print('The unique of gt',gt.shape)
    print('The shpe of image processing result',image_processing_result.shape)
    red_extracted_structures = extract_red_mask(predicted_structures.detach().cpu().numpy())
    red_gt_structures = extract_red_mask(gt.detach().cpu().numpy())
    red_image_processing_structures = extract_red_mask(image_processing_result.detach().cpu().numpy())

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

    #visulaize_iou_results(raw_images,epoch,batch,predicted_structures,gt,dilated_overlap,red_image_processing_structures,img_proc_dilated_overlap,iou_dilated,iou_dilated_image_processing,betti_error_pred,betti_error_img_prc,save_path)

    return iou_dilated,iou_dilated_image_processing

# ----------------------------------------------------------------------
# 6. Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    train_grayscale()


   
    # out = model(x)

    # print("consensus:", tuple(out["consensus"].shape))
    # print("warped   :", tuple(out["warped"].shape))
    # print("flows    :", tuple(out["flows"].shape))
    # print("template :", tuple(out["template"].shape))
    # print("flow |max| (px):", out["flows"].abs().max().item())

    # target = (torch.rand(B, 1, H, W) > 0.5).float()
    # loss = (F.binary_cross_entropy_with_logits(out["consensus"], target)
    #         + 0.1 * registration_loss(out["warped"], out["template"])
    #         + 0.01 * flow_smoothness(out["flows"]))
    # loss.backward()

    # g = model.flow_head.flow.weight.grad
    # print("loss:", round(loss.item(), 4))
    # print("flow head grad present:", g is not None and g.abs().sum().item() > 0)