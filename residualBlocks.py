from torch import nn

# 1
# This residual block is based upon the diagram in Figure 5 for the PixelCNN demonstrated
# by Aaron van den Oord, 2016 arXiv:160106759v3 

# 2 
# This residual block aims to reproduce their tensorflow 1.x implementation at https://github.com/hojonathanho/diffusion

# Their may be conflicting implementation details that only satisfy either "1" or "2" above.
# Here I follow the diagram of Ho et al. residual blocks in Figure 5 of the article referenced "1" at the top
class residualBlock_PixelCNN(nn.Module):
    def __init__(self, inputChanels, outputChannels=None, ):
        super(residualBlock_PixelCNN, self).__init__()
        # Process parameters given to this module
        self.inputChannels = inputChanels
        self.outputChannels = outputChannels

        # Initialize the model components
        self.groupNormalization = NotImplemented
        # you should have 2h feature maps at this point according to "1" above.
        # Ho et al. first do a 2D convolution named 'conv1' in their tensorflow 1.x code
        # using their conv2d(...) custom function implemented in their nn.py from the repo in "2" above.
        # Note: They may have mistakenly used a filter size of (3,3) for their filter "W" and a "stride" of 1
        # because it is a default in their code (see their unet.py and nn.py files in their github repo). The filter size should be (1,1) according to the description
        # of the residual blocks in their paper.
        # They use an optional "bias" tf.variable. I have included it.
        self.residualBlock =    nn.Sequential(
                                nn.Conv2d(in_channels=self.inputChannels, out_channels=self.outputChannels, kernel_size=1, stride=1, padding=None, bias=True),
                                # You should have h feature maps at this point according to "1" above
                                # After the first 1x1 convolution, they do a ReLU activation
                                nn.ReLU(),
                                # Next they do a 3x3 convolution
                                nn.Conv2d(in_channels=self.inputChannels/2, out_channels=self.outputChannels/2, kernel_size=3, stride=1, padding=None, bias=True),
                                # You should have h feature maps at this point according to "1" above
                                # Next they do a ReLU()
                                nn.ReLU(),
                                # Next they do a 1x1 convolution
                                nn.Conv2d(in_channels=self.inputChanels, out_channels=self.outputChannels, kernel_size=1, stride=1, padding=None, bias=True),
                                # Next the do a ReLU()                            
                                nn.ReLU(),
                                # You should have 2h feature maps at this point according to "1" above
                            )

    def forward(self, x):
        # Ho et al. 2020 first make a copy of the input
        original_x = x
        # Next they do a tensorflow 1.x tf.contrib.layers.group_norm(x) using 'norm1' as the layer name
        # Group norm statistics provide the averages of 32 groups of feature maps.
        x = self.groupNormalization(x)
        # After group normalization Ho et al. then do a ReLU activation
        x = nn.ReLU()(x)
        # After doing a ReLU() activation, they use a residual block
        x = self.residualBlock(x)
        # This is the final sumation of the original x and out
        x += original_x
        # All done return 
        return x
    

