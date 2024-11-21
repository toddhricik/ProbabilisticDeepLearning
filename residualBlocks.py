import sys
import torch.nn as nn

# 1
# This residual block is based upon the diagram in Figure 5 for the PixelCNN demonstrated
# by Aaron van den Oord in "Pixel Recurrent Networks", 2016 arXiv:160106759v3 

# 2 
# This residual block aims to reproduce their tensorflow 1.x implementation at "https://github.com/hojonathanho/diffusion"

# Their may be conflicting implementation details that only satisfy either "1" or "2" above.
# Note: Ho et al. may have mistakenly used a filter size of (3,3) for their filter "W" and a "stride" of 1 in their conv2d custom function
# because it is a default in their code (see their unet.py and nn.py files in their github repo). The filter size should be (1,1) according to the description when compared to the residual blocks in their paper.


# Here I follow the diagram of Ho et al. residual blocks in Figure 5 of the article referenced "1" at the top
class residualBlock_PixelCNN(nn.Module):
    def __init__(self, inputChannelCount=3, outputChannelCount=10):
        super(residualBlock_PixelCNN, self).__init__()
       
        # First make sure you received arguments for all params of __init__
        if inputChannelCount==None or outputChannelCount==None:
            print("One or more of the parameters to class residualBlock_PixelCNN is 'None'.")
            sys.exit(664)

        # Process parameters given to this module
        self.inputChannelCount = inputChannelCount
        self.outputChannelCount = outputChannelCount

        # Define some layers
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding='same', bias=True, groups=1)
        self.ReLU = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same', bias=True, groups=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding='same', bias=True, groups=1)


    def forward(self, x):
        # Ho et al. 2020 first make a copy of the input
        original_x = x
        x = self.conv1(x)
        # You should have h feature maps at this point according to "1" above
        # After the first 1x1 convolution, they do a ReLU activation
        x = self.ReLU(x)
        # Next they do a 3x3 convolution
        x =self.conv2(x)
        # You should have h feature maps at this point according to "1" above
        # Next they do a ReLU()
        x = self.ReLU(x)
        # Next they do a 1x1 convolution
        x = self.conv3(x)
        # Next the do a ReLU()                            
        x = self.ReLU(x)
        # You should have 2h feature maps at this point according to "1" above
        # This is the final sumation of the original x and out
        x += original_x
        # All done return 
        return x
    

