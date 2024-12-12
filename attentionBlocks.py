import torch
import torch.nn as nn

# 1
# This attention block is based upon the diagram in Figure 5 for the PixelCNN demonstrated
# by Aaron van den Oord in "Pixel Recurrent Networks", 2016 arXiv:160106759v3 

# 2 
# This attention block aims to reproduce their tensorflow 1.x implementation at "https://github.com/hojonathanho/diffusion/diffusion_tf/models/unet.py"




# Here I follow the diagram of Ho et al. attention blocks in Figure 5 of the article referenced "1" at the top
