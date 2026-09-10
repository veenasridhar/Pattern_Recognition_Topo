import os
from diffusers import AutoencoderKL
from input_pipeline import *
import torch
import torchvision
import umap
import matplotlib.pyplot as plt


def extract_latent_codes():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')
    vae = AutoencoderKL.from_pretrained('./finetuned_vae/individually_trained_models',use_safetensors=True) # Forces the model to use pretrained weights
    vae = vae.to(device)
    train_transformations = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # Scale = true already scales the images between 0,1.
        v2.Resize((256, 256)),
        v2.ToTensor(),
        # v2.Normalize(mean=(0.5,), std= (1.0,)),
    ])

    latent_codes = []
    labels = []

    for batch_idx, image_batch in enumerate(os.listdir('data/batched_data_resized_red/train')):

        for images in os.listdir(os.path.join('data/batched_data_resized_red/train',image_batch)):

            image = torchvision.io.decode_image(os.path.join(os.path.join('data/batched_data_resized_red/train',image_batch),images),torchvision.io.ImageReadMode.RGB)
            image = train_transformations(image)
            image = image.to(device)
            z = vae.encode(image.unsqueeze(0).to(device)).latent_dist.mean

            z_flat = z.mean(dim=[2,3]).squeeze().cpu().detach().numpy()
            latent_codes.append(z_flat)
            print(z_flat.shape)
            labels.append(batch_idx)

    reducer = umap.UMAP(n_components=2,n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(latent_codes)

    fig,ax = plt.subplots(figsize=(12,8))

    n_families = len(set(labels))
    family_colours = plt.cm.tab10(np.linspace(0, 1, n_families))

    plotted_types = set()

    for i,(x,y) in enumerate(embedding):
        fam = labels[i]
        colour = family_colours[fam % n_families]

        ax.scatter(x, y, c=[colour], marker='o', s=30, alpha=0.7,  edgecolors='black')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('UMAP of Latent Space', fontsize=13)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    plt.tight_layout()
    #plt.savefig(f'latent_space_{method}.png', dpi=150, bbox_inches='tight')
    plt.show()


extract_latent_codes()



