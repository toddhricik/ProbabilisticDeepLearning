from torch import nn
import residualBlocks as rb

class pixelCNN:
    def __init__(self, numResidualBlocks=5):
        super(pixelCNN,self).__init__()
        self.numResidualBlocks = numResidualBlocks
        self.residualBlock_PixelCNN = rb.residualBlock_PixelCNN(inputChanels=3, outputChannels=3, stride=1, downsample=None) 
    
    def forward(self, x):
        # Step 1: The first layer is a 7x7 conv using Mask A
        # x = maskA()(x) or something like that
        x = nn.Conv2d(inputChannels=3, outputChannels=3, shape=7, stride=1)(x)
        # Step 2: Several residual blocks follow the first layer
        x = nn.ModuleList([self.residualBlock_PixelCNN(inputChanels=3, outputChannels=3, stride=1, downsample=None)  for i in range(self.numResidualBlocks)]) 
        # Step 3: A 3x3 convolution using Mask B follows the residual blocks from step 2 
        x = nn.Conv2d(inputChannels=3, outputChannels=3, shape=3, stride=1)(x)
        # Step 4: ReLU is applied to the result of step 3
        x = nn.ReLU(x)
        # Step 5: A 1x1 convolution is used after ReLU is applied in step 4.
        x = nn.Conv2d(inputChannels=3, outputChannels=3, shape=1, stride=1)(x)
        # Step 6:   For Natural Images, use a separate 256-way softmax for each channel {R, G, B}.
        #           For MNIST images, use a sigmoid activation.
        # END



