from input_pipeline import *
from model_training.training import *
from utils import *
from model_training.model import *
from transfer_learning import *
from VQ_VAE_transfer_learning import *


def main():
    #TransferLearning()
    #debatching('C:\\Study\\Old Code\\Pattern_Recognition_Topo\\batched_data\\unprocessed',preprocessing=True)
    #train_dataloader,test_dataloader = input_pipeline()
    #training(train_dataloader,test_dataloader)

    #VAE Fine Tuning
    #vae_tuner = VAEFineTuner(model_name='stabilityai/sd-vae-ft-mse')
    #train_loader = vae_tuner.prepare_data('./batched_data/train/')
    #test_loader = vae_tuner.prepare_data('./batched_data/test/')
    #vae_tuner.finetune(train_loader,test_loader)

    #VQVAE Fine Tuning
    vq_vae_tuner = VQVAEFineTuner(prebatched_images=False)
    train_loader,test_loader = vq_vae_tuner.prepare_data('./data/train/','./data/test/',4, 256)
    vq_vae_tuner.finetune(train_loader,test_loader)



if __name__ == "__main__":
    main()