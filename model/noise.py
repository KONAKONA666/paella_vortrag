import torch 
import math

class NoiseScheduler:
    def __init__(self, c_r, max_positions=10000) -> None:
        self.c_r = c_r
        self.max_positions = max_positions
        self.num_tokens = 8192
    def get_embedding(self, r):
        half_dim = self.c_r // 2
        r = self.gamma(r) * self.max_positions
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None]*emb[None, :]
        emb = torch.cat([emb.cos(), emb.sin()], dim=-1)
        return emb.to(r.dtype) 
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