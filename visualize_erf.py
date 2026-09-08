import torch
import torch.nn.functional as F
import numpy as np
from simple_conv_approach import simple_Conv
from input_pipeline_binary import *
from torchinfo import summary


def get_input_grad(model, samples):
    outputs = model(samples)              # (B, num_classes, H, W), H×W == input size
    _, _, h, w = outputs.size()
    center = outputs[:, :, h // 2, w // 2].sum()   # center pixel, summed over classes + batch
    grad = torch.autograd.grad(center, samples)[0]
    grad = F.relu(grad)
    grad_map = grad.sum((0, 1))           # -> (H_in, W_in)
    return grad_map.cpu().numpy()


def build_erf_map(model, dataloader, device, num_images=50):
    model.eval()
    accumulated = None
    seen = 0
    for samples, _, _ in dataloader:
        print('The sample shape', samples.shape)
        samples = samples.to(device)
        samples.requires_grad = True
        model.zero_grad()
        contrib = get_input_grad(model, samples)
        if np.isnan(np.sum(contrib)):
            continue
        accumulated = contrib if accumulated is None else accumulated + contrib
        seen += 1
        if seen >= num_images:
            break
    accumulated = accumulated / seen
    np.save('erf_matrix_39px_latest.npy', accumulated)   # <-- this is what analyze_erf.py reads
    return accumulated

# model = torch.load('CNN Results/AugmentedRailwayTO_Convolutional_Network_chkpoint/convolutional_network_chkpoint_0.pth', weights_only = False)
# print(summary(model))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size=1

data_dir = 'data/Batches'
ground_truth_dir = 'data/Ground_Truth'
image_processing_dir = 'data/Image Processing Results'
aug_data_dir = 'data/Augmented TO Data'
    
    
batches_list, items = build_dataset_items(data_dir)                   # batches_list is the list of batches as main batch indices [7,7,11,11,11,11,....]
#                                                                      # items is the list of  full address of each batch

# build augmented items ONCE (same main-batch ids as batches_list)
aug_batches_list, aug_items = build_dataset_items(aug_data_dir)

# # print('Total Sub Batches ', len(items))
# # print("Number of Main Batches:", len(set(batches_list)))
# # print("Sub-Batches per Main Batch:", collections.Counter(batches_list.tolist()))

for fold in range(1):

    train_idx, val_idx, test_idx = grouped_split(batches_list, fold=fold, ratios=[0.7, 0.2, 0.1])

    # print('Train Index', train_idx)
    # print('Test Index', test_idx)
    # print('Val Index', val_idx)

    # # which MAIN BATCHES are in train this fold
    train_groups = set(np.asarray(batches_list)[train_idx].tolist())

    # # augmented sub-batches whose main batch is a train batch -> train only
    aug_train_idx = np.array(
            [i for i, g in enumerate(aug_batches_list) if g in train_groups]
        )


    # # safety: no augmented val/test batch can leak into train
    val_groups  = set(np.asarray(batches_list)[val_idx].tolist())
    test_groups = set(np.asarray(batches_list)[test_idx].tolist())
    assert not (set(aug_batches_list[aug_train_idx].tolist()) & (val_groups | test_groups)), \
        "augmented train batches overlap val/test!"


    #model = simple_Conv(5, 2, kernel_size=3, stride=1, padding=1)
    model = torch.load(f'CNN Results/AugmentedRailwayTO_Convolutional_Network_chkpoint/convolutional_network_chkpoint_0.pth',weights_only=False)
    model = model.to(device)
    #print(summary(model))

    print('The training parameters for the module are',sum(p.numel() for p in model.parameters() if p.requires_grad))

    train_dataset = TopoDatasetImg(items,train_idx,ground_truth_dir= ground_truth_dir, img_proc_dir = image_processing_dir, augment= True)
    val_dataset = TopoDatasetImg(items,val_idx,ground_truth_dir= ground_truth_dir, img_proc_dir = image_processing_dir, augment=False)
    val_subset = torch.utils.data.Subset(val_dataset, [7])
    test_dataset = TopoDatasetImg(items,test_idx,ground_truth_dir= ground_truth_dir, img_proc_dir = image_processing_dir,augment=False)

    # # augmented dataset -> TRAIN ONLY, points at augmented GT/result dirs
    aug_train_dataset = TopoDatasetImg(aug_items, aug_train_idx,
                                        ground_truth_dir=ground_truth_dir,
                                        img_proc_dir=image_processing_dir,
                                        augment=True)
    
    

    train_combined = ConcatDataset([train_dataset,aug_train_dataset])


    train_dataloader = DataLoader(train_combined, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,num_workers=4,pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=4,pin_memory=True)

    build_erf_map(model, val_dataloader, device)