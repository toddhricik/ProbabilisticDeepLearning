from torch import nn

class residualBlock_PixelCNN(nn.Module):
    
    def __init__(self, inputChanels, outputChannels, stride=1, downsample=None):
        super(residualBlock_PixelCNN, self).__init__()
        # Initialize the model
        self.model =    nn.Sequential(
                            nn.ReLU(),
                            nn.Conv2d(inputChanels, outputChannels, kernel_size=1, stride=stride),
                            nn.Conv2d(inputChanels, outputChannels, kernel_size=3, stride=stride),
                            nn.Conv2d(inputChanels, outputChannels, kernel_size=1, stride=stride),
                        )

    def forward(self, x):
        original_x = x
        out = self.model(x)
        out += original_x 
        return out