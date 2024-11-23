import torch

# simpleLossHoDPP is based upon equation 14 of Ho et al. 2020 DPPM paper
def simpleLossHoDPPM(actual, prediction, type='mse'):
    if type == 'mse':
        # Basic MSE of the prediction vs the actual
        loss = torch.pow(torch.mse(prediction - actual), 2)
        return loss
    