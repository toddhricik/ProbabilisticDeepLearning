'''
This U-Net attempts to recreate the model described in section "C" of the "Extra Information" section in the paper "Denoising Diffusion Probabilistic Model" by Ho et al. 2020. Also see section B "Experimental Details" for more information on how the models were set up and used in specific experiments.

Note that Section 4 "Experiments of Ho et al. use a non-masked pixelCNN++ backbone.
'''

import torch
import torch.nn as nn
import residualBlocks as rb

class unet_DPPM_32(nn.Module):
    def __init__(self):
        super().__init__()
        # According to Ho et al. DPPM paper, they use two residual blocks in each convolutional layer of the U-Net
        self.convLayer1 = nn.ModuleList(
                                [rb.residualBlock_PixelCNN(inputchannelCount=3, outputchannelCount=3)
                                 for i in range(2)])
        self.convLayer2 = nn.ModuleList(
                                [rb.residualBlock_PixelCNN(inputchannelCount=3, outputchannelCount=3)
                                 for i in range(2)])
        self.convLayer3 = nn.ModuleList(
                                [rb.residualBlock_PixelCNN(inputchannelCount=3, outputchannelCount=3)
                                 for i in range(2)])
        self.convLayer4 = nn.ModuleList(
                                [rb.residualBlock_PixelCNN(inputchannelCount=3, outputchannelCount=3)
                                 for i in range(2)])
     
    def forward(self, x, t):
        outputChannelCount = 
        # Next they do a tensorflow 1.x tf.contrib.layers.group_norm(x) using 'norm1' as the layer name
        # Group norm statistics provide the averages of 32 groups of feature maps.
        x = self.groupNormalization(x)
        # After group normalization Ho et al. then do a ReLU activation
        x = nn.ReLU()(x)
        # After doing a ReLU() activation, do the first purple and green boxes in the pixelcnn++ paper
        # g0 = conv2d()(x)
        # gx0 = some function of x and g0
        # p0 = conv2d()(gx0)
        # Now enter the first convoltuional layer of the U-Net encoder
        #d1g1 = self.convLayer1()(g0)
        d1p1 = None
        # Now enter the second residual layer of the U-Net encoder
        d2 = self.convLayer2()(d1)
        # There should be a self-attention layer hear (requires a time embedding????)
        pass
        # Now enter the third convoltuional layer of the U-Net encoder
        d3 = self.convLayer3()(d2)
        # Now enter the fourth residual layer of the U-Net encoder
        d4 = self.convLayer4()(d3)
        # Now
        e3 = None
        # Now
        e2 = None
        # Now
        e1 = None
        # Now
        e4 = None

