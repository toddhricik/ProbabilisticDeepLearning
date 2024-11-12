import torch

class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(1. - self.alpha_cum_prod)

    def add_noise(self, original, noise, t):
        original_shape = original.shape
        bath_size = original_shape[0]
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minu_alhpa_cum_prod[t].reshape(batch_size)

        for _ in range(len(original_shape)-1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod = swqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

            return sqrt_alpha_cum_prod * original + sqrt_one_minus_alpha_cum_prod * noise
