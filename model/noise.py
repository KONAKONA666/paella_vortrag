import torch 

class NoiseScheduler:
    def __init__(self, c_r, max_positions=10000) -> None:
        self.c_r = c_r
        self.max_positions = max_positions
        self.num_tokens = 8192
    def gamma(self, r):
        return (r * torch.pi / 2).cos()
    def get_noise(self, r, x):
        r = self.gamma(r)[:, None, None]
        mask = torch.bernoulli(r*torch.ones_like(x)).round().long()
        random_x =torch.randint_like(x, 0, self.num_tokens)
        return mask, random_x
    def add_noise(self, x, r):
        mask, noise = self.get_noise(r, x)
        return x*(1-mask) + noise*mask