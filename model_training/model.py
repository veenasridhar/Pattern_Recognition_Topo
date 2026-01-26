from torch import nn
import torchvision
from torchinfo import summary

class AutoEncoder(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = Encoder(3,1024)
        self.decoder = Decoder(1024)
    
    def forward(self,x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

class Encoder(nn.Module):
    
    def __init__(self, input_shape=3, latent_dim = 1024):
        super().__init__()
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(in_channels= input_shape, out_channels =32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels= 32, out_channels =64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.encoder_block2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels= 128, kernel_size=3, stride=1, padding =1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels = 128, out_channels= 256, kernel_size=3, stride=1, padding =1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels = 256, out_channels= 512, kernel_size=3, stride=1, padding =1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.linear_block = nn.Linear(in_features = 512*(256//32)*(256//32),out_features = latent_dim)
        #self.linear_block = nn.Linear(in_features = 512*(512//32)*(512//32),out_features = latent_dim)

    def forward(self, x):
        x = self.encoder_block1(x)
        x = self.encoder_block2(x)
        #print('The shape before flattening',x.shape)
        x = x.view(x.size(0), -1)  # Flatten
        #print('The shape after flattening', x.shape)
        z = self.linear_block(x)
        #print('The shape of latent dimensions', z.shape)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim=1028):
        super().__init__()

        self.linear_block2 = nn.Linear(latent_dim, 512*8*8)
        #self.linear_block2 = nn.Linear(latent_dim, 512*16*16)

        self.decoder_block = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1,padding=1 ),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1,padding=1 ),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1,padding=1 ),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2,padding=1),

            #nn.Sigmoid()            #Sigmoid activation function is required to keep the outputs between 0 and 1.No need Sigmoid because BCEWithLogitsLoss
        )
    
    def forward(self,z):
        x = self.linear_block2(z)
        #print('After linear block in the decoder',x.shape)
        #x = x.view(-1, 256, 16, 16)  # Reshape to spatial
        x = x.view(-1, 512, 8, 8)
        #print('After reshaping to spatial in decoder',x.shape)
        x = self.decoder_block(x)
        return x



'''class TransferLearning_Model(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        self.encoder_layer0 = nn.Sequential(resnet.conv1,resnet.bn1, resnet.relu) #Avoiding the max pool from the first resnet block
        #Ouput: 64x256x256
        self.encoder_layer1 = resnet.layer1
        #Output: 64x128x128 
        self.encoder_layer2 = resnet.layer2
        #Output: 128x64x64
        self.encoder_layer3 = resnet.layer3
        #Output: 256x32x32
        self.encoder_layer4 = resnet.layer4
        #Output: 512x16x16



    def forward(self,x):
        x= self.encoder_layer0(x)
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        x = self.encoder_layer4(x)
        return x

def TransferLearning():
    #resnet = torchvision.models.get_model("resnet18")
    model = TransferLearning_Model()
    summary(model,input_size=(32,3,256,256))'''



