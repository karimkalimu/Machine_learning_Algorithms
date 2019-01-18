from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision
import numpy as np
import matplotlib.pyplot as plt 

workers = 2
batchSize = 128
imageSize = 32 # if you change this number you have to adjust the network to suit it
Zsize = 100 #Latent space size
nChannel = 16
nEpochs = 25 
lr = 0.0002 #learning rate
beta1 = 0.5

nc=1 #input channel size

manualSeed = 19 #to reproduce  the same result

print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

cudnn.benchmark = True

    
dataroot = '../../data/mnist'
dataset = dset.MNIST(root=dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size= batchSize,
                                         shuffle=True, num_workers=int(workers))

device = torch.device("cpu")
Zsize = int(Zsize)
nChannel = int(nChannel)

# custom weights initialization called on Generator_nn and Discriminator_nn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            #The formula to measure the output shape for transpose = stride*(input - 1) + kernel - 2 * padding
            nn.ConvTranspose2d(Zsize, nChannel * 8, 4, 1, 0, bias=False),#output shape = 4 x 4, 128 channel 
            nn.BatchNorm2d(nChannel * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(nChannel * 8, nChannel * 4, 4, 2, 1, bias=False),#output shape = 8 x 8, 64 channel 
            nn.BatchNorm2d(nChannel * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(nChannel * 4, nChannel * 2, 4, 2, 1, bias=False),#output shape = 16 x 16, 32 channel
            nn.BatchNorm2d(nChannel * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(nChannel * 2, 1, 4, 2, 1, bias=False),#output shape = 32 x 32, 1 channel
            nn.Tanh()

        )

    def forward(self, input):
      
        output = self.main(input)
        
        return output


Generator_nn = Generator().to(device)
Generator_nn.apply(weights_init)

print(Generator_nn)


class Discriminator(nn.Module):
    def __init__(self,):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            #nc : input channel size
            nn.Conv2d(nc, nChannel, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nChannel, nChannel * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nChannel * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nChannel * 2, nChannel * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nChannel * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nChannel * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        
        output = self.main(input)

        return output.view(-1, 1).squeeze(1)


Discriminator_nn = Discriminator().to(device)
Discriminator_nn.apply(weights_init)

print(Discriminator_nn)

criterion = nn.BCELoss()

fixed_noise = torch.randn(batchSize, Zsize, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(Discriminator_nn.parameters(), lr=lr, betas=(beta1, 0.9999))
optimizerG = optim.Adam(Generator_nn.parameters(), lr=lr, betas=(beta1, 0.9999))


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = np.squeeze(inp)
    mean = np.array(0.5)
    std = np.array(0.5)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated





for epoch in range(nEpochs):
    for i, images in enumerate(dataloader, 0):
      

        #Update Discriminator_nn: maximize log(D(x)) + log(1 - D(G(z)))###

        # train with real
        Discriminator_nn.zero_grad()
        real_images = images[0].to(device)
        batch_size = real_images.size(0)
        label = torch.full((batch_size,), real_label, device=device)
        output = Discriminator_nn(real_images)
        Loss_D_real = criterion(output, label)
        Loss_D_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, Zsize, 1, 1, device=device)
        fake = Generator_nn(noise)
        label.fill_(fake_label)
        output = Discriminator_nn(fake.detach())
        Loss_D_fake = criterion(output, label)
        Loss_D_fake.backward()
        D_G_z1 = output.mean().item()
        Loss_D = Loss_D_real + Loss_D_fake
        
        optimizerD.step()

        #Update Generator_nn: maximize log(D(G(z)))###
        
        Generator_nn.zero_grad()
        label.fill_(real_label) 
        output = Discriminator_nn(fake.detach())
        Loss_G = criterion(output, label)
        Loss_G.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        
        if  i % 20 ==  0:
            showme = Generator_nn(fixed_noise)
            print("epoch [",epoch,"/",nEpochs,"]", i, len(dataloader),
                  "Loss Discriminator :",Loss_D.item(),
                  "Loss Generator :",Loss_G.item())
            print( "D(x):",D_x,
                  "G(x):",D_G_z1,"/", D_G_z2)
            images = showme.detach().data[:1].numpy()
            imshow(images)

            
            
            
PATH_SAVE='/Pytorch_GAN_MNSIT.pt'
torch.save(Generator_nn.state_dict(), PATH_SAVE)