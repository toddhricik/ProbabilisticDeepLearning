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
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=1, padding='same', bias=True, groups=1)
        self.ReLU = nn.ReLU()

        self.convLayer1 = nn.ModuleList(
                                [rb.residualBlock_DDPM(inputchannelCount=256, outputchannelCount=256),
                                 for i in range(2)])
        self.downSample1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding='valid', groups=1, bias=True)
        self.convLayer2 = nn.ModuleList(
                                [rb.residualBlock_DPPM(inputchannelCount=256, outputchannelCount=256)
                                 for i in range(2)])
        self.convLayer3 = nn.ModuleList(
                                [rb.residualBlock_PixelCNN(inputchannelCount=256, outputchannelCount=256)
                                 for i in range(2)])
        self.convLayer4 = nn.ModuleList(
                                [rb.residualBlock_PixelCNN(inputchannelCount=256, outputchannelCount=256)
                                 for i in range(2)])
        self.crossConv1 = nn.Conv2d(in_channels=, out_channels=, kernel_size=1, stride=1, padding='same', bias=True, groups=1)
    def forward(self, x, t):
        # Ho et al. First do a 2d convolution on the input to create enough feature maps (which i think is 256) to go into the first convolutional residual block of the 
        x = self.conv1(x)
        # Next they do a tensorflow 1.x tf.contrib.layers.group_norm(x) using 'norm1' as the layer name
        # Group norm statistics provide the averages of 32 groups of feature maps.
        x = self.groupNormalization(x)
        # After group normalization Ho et al. then do a ReLU activation
        x = nn.ReLU(x)
        # Now enter the first convolutional layer that makes up the U part of the U-Net encoder.
        # The feature maps should all be (32x32) before running convLayer1
        r1 = self.convLayer1(x)
        # Now downsample via convolution in order to get to the next lower resolution.
        # The feature maps should all be (16x16) after call to downSample returns.
        d1 = self.downSample1(r1)
        # Now enter the second convolutional layer that makes up the U part of the U-Net encoder.
        # The feature maps should all be (16x16) at this step.
        r2 = self.convLayer2(d1)
        # There should be a self-attention layer here (requires a time embedding????)
        a1 = None
        # Now downsample via convolution in order to get to the next lower resolution.
        # The result of downsampling should be a set of (8x8) feature maps
        d2 = self.downSample1(a1)
        # Now enter the third convoltuional layer of the U-Net encoder.
        # The feature maps should all be (8x8) at this step.
        r3 = self.convLayer3(d2)
        # Now we enter the "flat" "bottom" part of the U-Net which consists of resnet -> attention -> resnet.
        # See "Downsampling section of model definition in unet.py at https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py"
        # Go "right" at the "bottom" of the U-Net by adding a residual block
        c1 = self.crossConv1(r3)
        # Go "right" again by adding a self attention block 
        c2 = self.attention2(c1)
        # Go "right" again by adding a residual block
        c3 = self.crossConv2(c2)         
        # Now enter the first convolutional residual layer at the bottom of the of the U-Net decoder.
        
        # See "Downsampling section of model definition in unet.py at https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py"
        
