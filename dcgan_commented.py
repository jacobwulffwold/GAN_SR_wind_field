# Deep Convolutional GANs

# Importing the libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets import *


# Defining the weights_init function that takes as input a neural network m and that will initialize all its weights.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Defining the generator


class G(nn.Module):  # We introduce a class to define the generator.
    def __init__(
        self,
    ):  # We introduce the __init__() function that will define the architecture of the generator.
        super(G, self).__init__()  # We inherit from the nn.Module tools.
        self.main = nn.Sequential(  # We create a meta module of a neural network that will contain a sequence of modules (convolutions, full connections, etc.).
            # input is Z, going into a convolution
            nn.ConvTranspose2d(
                in_channels=nz,
                out_channels=ngf * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),  # first index is HR_data.shape[0] aka nz, second index is ngf * 8, where ngf relates to the size of the feature maps, i.e. 64
            nn.BatchNorm2d(
                ngf * 8
            ),  # We normalize all the features along the dimension of the batch.
            nn.ReLU(True),  # We apply a ReLU rectification to break the linearity.
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(
                ngf * 8, ngf * 4, 4, 2, 1, bias=False
            ),  # We add another inversed convolution.
            nn.BatchNorm2d(ngf * 4),  # We normalize again.
            nn.ReLU(True),  # We apply another ReLU.
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(
                ngf * 4, ngf * 2, 4, 2, 1, bias=False
            ),  # We add another inversed convolution.
            nn.BatchNorm2d(ngf * 2),  # We normalize again.
            nn.ReLU(True),  # We apply another ReLU.
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(
                ngf * 2, ngf, 4, 2, 1, bias=False
            ),  # We add another inversed convolution.
            nn.BatchNorm2d(ngf),  # We normalize again.
            nn.ReLU(True),  # We apply another ReLU.
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(
                ngf, 3, 4, 2, 1, bias=False
            ),  # We add another inversed convolution.
            nn.Tanh()  # We apply a Tanh rectification to break the linearity and stay between -1 and +1.
            # state size. 3 x 64 x 64
        )

    def forward(
        self, input
    ):  # We define the forward function that takes as argument an input that will be fed to the neural network, and that will return the output containing the generated images.
        output = self.main(
            input
        )  # We forward propagate the signal through the whole neural network of the generator defined by self.main.
        return output  # We return the output containing the generated images.


# Creating the generator
netG = G()  # We create the generator object.
netG.apply(weights_init)  # We initialize all the weights of its neural network.

# Defining the discriminator


class D(nn.Module):  # We introduce a class to define the discriminator.
    def __init__(
        self,
    ):  # We introduce the __init__() function that will define the architecture of the discriminator.
        super(D, self).__init__()  # We inherit from the nn.Module tools.
        self.main = nn.Sequential(  # We create a meta module of a neural network that will contain a sequence of modules (convolutions, full connections, etc.).
            # input is (nc) x 128 x 128
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),  # We apply a LeakyReLU.
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # We add another convolution.
            nn.BatchNorm2d(ndf * 2),  # ndf * 2
            nn.LeakyReLU(0.2, inplace=True),  # We apply another LeakyReLU.
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(
                ndf * 2, ndf * 4, 4, 2, 1, bias=False
            ),  # We add another convolution.
            nn.BatchNorm2d(ndf * 4),  # We normalize again.
            nn.LeakyReLU(0.2, inplace=True),  # We apply another LeakyReLU.
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(
                ndf * 4, ndf * 8, 4, 2, 1, bias=False
            ),  # We add another convolution.
            nn.BatchNorm2d(ndf * 8),  # We normalize again.
            nn.LeakyReLU(0.2, inplace=True),  # We apply another LeakyReLU.
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),  # We add another convolution.
            nn.Sigmoid(),  # We apply a Sigmoid rectification to break the linearity and stay between 0 and 1.
        )

    def forward(
        self, input
    ):  # We define the forward function that takes as argument an input that will be fed to the neural network, and that will return the output which will be a value between 0 and 1.
        output = self.main(
            input
        )  # We forward propagate the signal through the whole neural network of the discriminator defined by self.main.
        return output.view(
            -1
        )  # We return the output which will be a value between 0 and 1.


# Creating the discriminator
netD = D()  # We create the discriminator object.
netD.apply(weights_init)  # We initialize all the weights of its neural network.


# Training the DCGANs
criterion = (
    nn.BCELoss()
)  # We create a criterion object that will measure the error between the prediction and the target.
# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(
    netD.parameters(), lr=0.0002, betas=(0.5, 0.999)
)  # We create the optimizer object of the discriminator.
optimizerG = optim.Adam(
    netG.parameters(), lr=0.0002, betas=(0.5, 0.999)
)  # We create the optimizer object of the generator.


netD = netD.float()

for epoch in range(num_epochs):  # Loop over the dataset multiple times
    for i, data in enumerate(trainloader):
        # 1st Step: Updating the weights of the neural network of the discriminator
        netD.zero_grad()  # We initialize to 0 the gradients of the discriminator with respect to the weights.

        # Training the discriminator with a real image of the dataset
        LR, HR = data  # features, labels
        HR = Variable(HR)  # wrap it in a variable
        target = Variable(torch.ones(HR.size()[0]))  # We get the target.
        output = netD(
            HR.float()
        )  # We forward propagate this real image into the neural network of the discriminator to get the prediction (a value between 0 and 1).
        errD_HR = criterion(
            output, target
        )  # We compute the loss between the predictions (output) and the target (equal to 1).

        # Training the discriminator with a fake image generated by the generator
        LR = Variable(LR)
        target = Variable(torch.zeros(HR.size()[0]))  # We get the target.
        output = netD(
            LR.float().detach()
        )  # We forward propagate the fake generated images into the neural network of the discriminator to get the prediction (a value between 0 and 1).
        errD_LR = criterion(
            output, target
        )  # We compute the loss between the prediction (output) and the target (equal to 0).

        # Backpropagating the total error
        errD = errD_HR + errD_LR  # We compute the total error of the discriminator.
        errD.backward()  # We backpropagate the loss error by computing the gradients of the total error with respect to the weights of the discriminator.
        optimizerD.step()  # We apply the optimizer to update the weights according to how much they are responsible for the loss error of the discriminator.

        # 2nd Step: Updating the weights of the neural network of the generator
        netG.zero_grad()  # We initialize to 0 the gradients of the generator with respect to the weights.
        target = Variable(torch.ones(HR.size()[0]))  # We get the target.
        output = netD(
            LR
        )  # We forward propagate the fake generated images into the neural network of the discriminator to get the prediction (a value between 0 and 1).
        errG = criterion(
            output, target
        )  # We compute the loss between the prediction (output between 0 and 1) and the target (equal to 1).
        errG.backward()  # We backpropagate the loss error by computing the gradients of the total error with respect to the weights of the generator.
        optimizerG.step()  # We apply the optimizer to update the weights according to how much they are responsible for the loss error of the generator.

        # 3rd Step: Printing the losses and saving the real images and the generated images of the minibatch every 100 steps
        """
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, num_epochs, i, len(trainloader), errD.data[0], errG.data[0])) # We print les losses of the discriminator (Loss_D) and the generator (Loss_G).
        if i % 100 == 0: # Every 100 steps:
            vutils.save_image(HR, '%s/real_samples.png' % "./results", normalize = True) # We save the real images of the minibatch.
            LR = netG(noise) # We get our fake generated images.
            vutils.save_image(LR.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True) # We also save the fake generated images of the minibatch.
        """
