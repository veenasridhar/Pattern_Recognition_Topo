import cv2
import numpy as np
import os
import shutil
import torch
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt


def convert_to_sdf(batch_dir,batch):
    # index = 0
    # global_pos_max = 0
    # global_neg_min = float ('inf')
    # for batch in os.listdir('data/Final Data/batched_data_grayscale_resized/train'):
    #     for image_path in os.listdir(os.path.join(batch_dir, batch)):
    #         image = cv2.imread(os.path.join(batch_dir, batch, image_path),cv2.IMREAD_GRAYSCALE)
    #         image = image/255.0
    #         image_labels = convert_labels(image)
    #         un_normalised_sdf = sdf_distance_function(image_labels)
    #         global_pos_max = max(global_pos_max,un_normalised_sdf.max())
    #         global_neg_min = min(global_neg_min,un_normalised_sdf.min())
    #
    # print(f'Dataset Stats: Max: {global_pos_max} Min: {global_neg_min}')

    global_pos_max = 194.6535
    global_neg_min = -21.0

    for set in ['test']:
        #for batch in os.listdir(f'data/Final Data/batched_data_grayscale_resized/{set}'):

            batch = ''
            for image_path in os.listdir(os.path.join(f'data/Final Data/gt_grayscale_resized/{set}', batch)):
                index = (image_path.split('.')[0]).split(' ')[-1]
                print(index)
                if int(index) == 9:
                    print('Processing ',image_path)
                    image = cv2.imread(os.path.join(f'data/Final Data/gt_grayscale_resized/{set}', batch, image_path),cv2.IMREAD_GRAYSCALE)
                    image = image/255.0
                    image_labels = convert_labels(image)
                    un_normalised_sdf = sdf_distance_function(image_labels)
                    normalised_sdf = np.zeros_like(un_normalised_sdf, dtype=np.float32)
                    pos_mask = un_normalised_sdf > 0
                    neg_mask = un_normalised_sdf < 0
                    normalised_sdf[pos_mask] = un_normalised_sdf[pos_mask] / global_pos_max
                    normalised_sdf[neg_mask] = un_normalised_sdf[neg_mask] / abs(global_neg_min)   # Important as negative and negative cancels out
                    visualise_sdf(image_labels, image_labels, neg_mask, pos_mask, un_normalised_sdf, normalised_sdf)

                    if not os.path.exists(os.path.join(f'data/Final Data/gt_data_sdf/{set}',batch)):
                        os.mkdir(os.path.join(f'data/Final Data/gt_data_sdf/{set}',batch))
                    if not os.path.exists(os.path.join(f'data/Final Data/gt_data_sdf/vis/{set}', batch)):
                        os.mkdir(os.path.join(f'data/Final Data/gt_data_sdf/vis/{set}', batch))
                    np.save(os.path.join(f'data/Final Data/gt_data_sdf/{set}',batch,f'image_{index}.npy' ), normalised_sdf)
                    fig, ax = plt.subplots(figsize=(6, 6))
                    im = ax.imshow(normalised_sdf, cmap='RdBu', vmin=-1, vmax=1)
                    fig.colorbar(im, ax=ax, label='normalized signed distance')
                    ax.axis('off')
                    fig.savefig(os.path.join(f'data/Final Data/gt_data_sdf/vis/{set}', batch,f'image_{index}.png'), dpi=150, bbox_inches='tight')
                    plt.close(fig)



def convert_labels( input_label):
    input = input_label
    # print('Before decoding',torch.unique(input,return_counts = True))
    input[input_label == 0.0] = 0
    input[input_label == 1.0] = 2
    input[(input_label >= 0.25) & (input_label < 0.75)] = 1  # since floating point precision might vary
    # print('After decoding',torch.unique(input, return_counts=True))
    return input.astype(np.long)



def sdf_distance_function(label_map,num_classes=1):

    sdf = np.zeros_like(label_map, dtype=np.float32)

    #for class_id in [2]:
    class_id =2
    mask = (label_map == class_id)          # Mask with all white structures = True
    #print('The mask shape', mask.shape, 'mask values', np.unique(mask,return_counts=True))

    if not mask.any() or mask.all():
        return sdf
        #sdf[class_id] = 0.0
        #continue

    print(mask.shape)
    distance_inside = distance_transform_edt(mask)
    #print('Distance Inside',distance_inside.max(),distance_inside.min())
    distance_outside = distance_transform_edt(~mask)
    # print('Distance Outside',distance_outside.max(),distance_outside.min())

    sdf = distance_outside - distance_inside                #This is a Numpy Array

    #sdf = (dist_out * (~mask) - (dist_in - 1) * mask).astype(np.float32) # This has zerolevel set

    #sdf[class_id] = distance_outside - distance_inside

    print('The sdf type is ',type(sdf),sdf.max(),sdf.min())

    # Inside pixel:  dist_inside < 0, dist_outside = 0  → negative
    # Outside pixel: dist_inside = 0, dist_outside > 0  → positive
    # Boundary pixel: both = 0                          → zero

    # normalized = np.zeros_like(sdf)
    #
    # pos_mask = sdf > 0  # outside pixels
    # neg_mask = sdf < 0  # inside pixels
    #
    # if pos_mask.any():
    #     pos_max = sdf[pos_mask].max()
    #     normalized[pos_mask] = sdf[pos_mask] / pos_max  # → [0, +1]
    #
    # if neg_mask.any():
    #     neg_min = abs(sdf[neg_mask].min())
    #     normalized[neg_mask] = sdf[neg_mask] / neg_min  # → [-1, 0]
    #
    # print(pos_max,neg_min)

    #visualise_sdf(label_map, mask, distance_inside, distance_outside, sdf, normalized=None)

    # boundary stays 0
    return sdf


def visualise_sdf(label_map,mask,dist_in,dist_out,raw_sdf,normalized):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('SDF Step by Step', fontsize=14)

    # # 1. Label map
    axes[0, 0].imshow(label_map, cmap='gray', vmin=0, vmax=2)
    axes[0, 0].set_title('Label map\n(0=black, 1=gray, 2=white)')
    axes[0, 0].axis('off')
    #
    # # 2. Binary mask
    axes[0, 1].imshow(mask.astype(np.float32), cmap='gray')
    axes[0, 1].set_title('Binary mask\n(white class only)')
    axes[0, 1].axis('off')

    # # 3. dist_inside
    im = axes[0, 2].imshow(dist_in, cmap='hot')
    axes[0, 2].set_title('dist_inside\n(0 outside, +d inside)')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    # 4. dist_outside
    im = axes[1, 0].imshow(dist_out, cmap='hot')
    axes[1, 0].set_title('dist_outside\n(0 inside, +d outside)')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    # 5. Raw SDF
    vmax = max(abs(raw_sdf.min()), abs(raw_sdf.max()))
    im = axes[1, 1].imshow(raw_sdf, cmap='seismic', vmin=-vmax, vmax=vmax)
    axes[1,1].contour(raw_sdf, levels=[0], colors="k", linewidths=1)
    axes[1, 1].set_title('Raw SDF\n(red=inside +, blue=outside -)')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    # 6. Normalized SDF
    im = axes[1, 2].imshow(normalized, cmap='seismic', vmin=-1, vmax=1)
    axes[1, 2].contour(normalized, levels=[0], colors="k", linewidths=1)
    axes[1, 2].set_title('Normalized SDF [-1, +1]\n(red=inside, blue=outside, white=boundary)')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    plt.savefig('sdf_visualization1.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved to sdf_visualization.png")

    # Print stats
    print(f"\nSDF stats:")
    print(f"  Raw     — min: {raw_sdf.min():.2f}  max: {raw_sdf.max():.2f}")
    print(f"  Norm    — min: {normalized.min():.2f}  max: {normalized.max():.2f}")
    print(f"  Inside  pixels: {mask.sum()}")
    print(f"  Outside pixels: {(~mask).sum()}")
    print(f"  Boundary pixels (SDF=0): {(raw_sdf == 0).sum()}")

#convert_to_sdf('data/Final Data/batched_data_grayscale_original size/test',batch=0)
#
# for gt in os.listdir('data/Final Data/gt_grayscale_resized/test'):
#     index = (gt.split('.')[0]).split(' ')[-1]
#     print('index:', index)
#     label_image = cv2.imread(os.path.join('data/Final Data/gt_grayscale_resized/test', gt),cv2.IMREAD_GRAYSCALE)
#     label_image = label_image/255.0
#     label_image = convert_labels(label_image)
#     sdf_result = sdf_distance_function(label_image)
#     np.save(f'data/Final Data/sdf_dir/test/sdf_{index}.npy',sdf_result)


#
# for  sdf in os.listdir('data/Final Data/sdf_dir/test'):
#     sdf_numpy = np.load(f'data/Final Data/sdf_dir/test/{sdf}')
#     plt.imshow(sdf_numpy,cmap='hot')
#     plt.show()
# for image_path in os.listdir('data/unbatched_resized/train'):
#      if image_path.endswith('image 229.png'):
#          image_resized = cv2.resize(cv2.imread(os.path.join('data/unbatched_resized/train', image_path)), (246, 246))
#          image = cv2.copyMakeBorder(image_resized, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value =(255, 255, 255))
#          cv2.imwrite(os.path.join('data/unbatched_resized/train', image_path), image)

# index = 0
# for batch in os.listdir('data/Final Data/batched_data_grayscale_resized/test'):
#     for images in os.listdir(os.path.join('data/Final Data/batched_data_grayscale_resized/test',batch)):
#         if images.endswith('.png'):
#             shutil.copy(os.path.join('data/Final Data/batched_data_grayscale_resized/test',batch,images), f'data/Final Data/unbatched_data_grayscale/test/image {index}.png')
#             index += 1

'''img_dir = 'batched_data_resized_red/train/Batch 0/image_0.png'
image = torchvision.io.decode_image(img_dir,torchvision.io.ImageReadMode.RGB)
image = image.float()/255.0
print(image.unique())'''

# Convert the gray in ground truth to more brighter gray
# for images in os.listdir('./Ground Truth Images Resized/test'):
#     image = cv2.imread(os.path.join('./Ground Truth Images Resized/test',images))
#     image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     gray_lower = np.array([0,0,80])
#     gray_upper = np.array([179,50,200])
#
#     gray_mask = cv2.inRange(image_hsv, gray_lower, gray_upper)
#
#     image[gray_mask==255] = [128,128,128]
#
#     cv2.imwrite(os.path.join('./Ground Truth Images Resized_red/test',images),image)


def encode_densities(image_dir):

    #for batches in os.listdir(image_dir):
    batches = ''
    for image_path in os.listdir(os.path.join(image_dir, batches)):

        image = cv2.imread(os.path.join(image_dir, batches, image_path))
        image_hsv = image
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #print(np.where(image == 84))
        #print(image[240,29])
        red_lower1 = np.array([0, 100, 100])
        red_upper = np.array([10, 255, 255])

        orange_lower = np.array([5, 50, 50])
        orange_upper = np.array([15, 255, 255])

        red_lower2 = np.array([170, 100, 100])
        red_upper2 = np.array([180, 255, 255])


        white_range = np.array([255,255,255])

        grey_range = np.array([128,128,128])
        red_mask1 = cv2.inRange(image_hsv, red_lower1, red_upper)
        orange_mask = cv2.inRange(image_hsv,orange_lower,orange_upper)
        #ed_mask2 = cv2.inRange(image_hsv, red_lower2, red_upper2)
        #red_mask = cv2.bitwise_or(red_mask1,red_mask2)
        white_mask = cv2.inRange(image,white_range,white_range)
        grey_mask = cv2.inRange(image, grey_range, grey_range)
        mask = cv2.bitwise_or(red_mask1,orange_mask)


        output = np.zeros_like(torch.tensor(image_hsv),dtype=np.float32)
        output[grey_mask == 255] = 0
        output[mask==255] = 1
        output[white_mask == 255] = 0.5

        if os.path.exists(os.path.join('data/Encoded/unbatched_grayscale_data/test',batches)):
            cv2.imwrite(os.path.join('data/Encoded/unbatched_grayscale_data/test',batches,image_path),output*255)
        else:
            os.mkdir(os.path.join('data/Encoded/unbatched_grayscale_data/test',batches))
            cv2.imwrite(os.path.join('data/Encoded/unbatched_grayscale_data/test', batches, image_path), output*255)


#encode_densities('data/unbatched_resized/test')
#image_dir = './batched_data_resized/train'
'''target_dir = './batched_data_resized_red/train'
for batches in os.listdir(image_dir):
    for image_path in os.listdir(os.path.join(image_dir,batches)):
        image = cv2.imread(os.path.join(image_dir,batches,image_path))
        image_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_mask = cv2.inRange(image_hsv,red_lower1,red_upper1)

        orange_lower = np.array([5, 50, 50])
        orange_upper = np.array([15, 255, 255])
        orange_mask = cv2.inRange(image_hsv, orange_lower, orange_upper)

        mask = cv2.bitwise_or(red_mask,orange_mask)
        image[mask ==255] = [0,0,255]

        cv2.imwrite(os.path.join(target_dir,batches,image_path),image)'''

'''image_dir = './unbatched_resized/test'
i = 435
train_transformations = v2.Compose([
    v2.ToImage(),
    #v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.6)),
    v2.RandomPosterize(bits=4,p=1),
    v2.RandomHorizontalFlip(p=1),
    #v2.RandomRotation(degrees=[0, 5]),
    v2.ToDtype(torch.float32, scale=True),  # Scale = true already scales the iamges between 0,1.
    v2.Resize((256, 256)),
    v2.ToTensor(),

])
transformations = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),  # Scale = true already scales the iamges between 0,1.
    v2.Resize((256, 256)),
    v2.ToTensor(),
])'''


'''for image_path in os.listdir(image_dir):
    #print(image_path)
    image = cv2.imread(os.path.join(image_dir, image_path))

    aug_image = train_transformations(image)
    aug_image = aug_image.permute(1,2,0).numpy()
    aug_image_to_save = (aug_image * 255).astype(np.uint8)

    # FIX 2: Convert RGB → BGR because cv2 expects BGR
    #aug_image_to_save = cv2.cvtColor(aug_image_to_save, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(image_dir, f'image {i}.png'),aug_image_to_save)
    cv2.imwrite(os.path.join('gt_unbatched_resized/train', f'gt {i}.png'),image)
    i = i+1
    print(image.shape, aug_image.shape)
    (score,diff) = skimage.metrics.structural_similarity(image,aug_image,full=True,channel_axis=2,
    data_range=1.0)
    print(score)
    cv2.imshow('image', image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imshow('aug image',aug_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    diff = (diff *255).astype('uint8')
    #diff_image = cv2.merge([diff,diff,diff])
    #print(diff.shape)
    cv2.imshow('Diff',diff)
    cv2.waitKey()
    cv2.destroyAllWindows()'''
    #cv2.imshow('Augmented Image',image.permute(1,2,0).numpy())