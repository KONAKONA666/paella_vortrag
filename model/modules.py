import torch
import torch.nn as nn
# (0, 1, 2, 3) -> (0, 2, 3, 1)
#[bs, ch, h, w] -> [bs, h, w, ch]
def channel_to_last(x):
    return x.permute(0, 2, 3, 1)
def channel_to_first(x):
    return x.permute(0, 3, 2, 1)


class ModulatedLayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(features, eps=eps)
        self.gamma = nn.Parameter(torch.randn(1,1,1))
        self.beta = nn.Parameter(torch.randn(1,1,1))
    def forward(self, x, w=None):
        x = self.ln(x)
        if w is not None:
            return self.gamma*w*x + self.beta*w
        return x


class ResBlock(nn.Module):
    def __init__(self, channels, hidden_dim, cond, skip=0) -> None:
        super().__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.cond = cond
        self.skip = skip

        self.depthwise_conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels 
        )
        self.layer_norm = ModulatedLayerNorm(channels)
        self.channelwise = nn.Sequential(
            nn.Linear(channels + skip, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels)
        )
        self.gamma = nn.Parameter(1e-6*torch.ones(channels), requires_grad=True)
        self.cond_linear = nn.Linear(cond, channels)
    def forward(self, x, s, skip=None): #skip: [bs, ch, h, w]
        residual = x
        x = self.depthwise_conv(x) # [bs, ch, h, w] -> [bs, ch, h, w]
        s = self.cond_linear(channel_to_last(s)) # [bs, cond_dim, 1,1] -> [bs, 1, 1, cond_dim]
        x = self.layer_norm(channel_to_last(x), s) # x: [bs, ch, h, w] -> [bs, h, w, ch]
        if skip is not None:
            x = torch.cat([x, channel_to_last(skip)], dim=-1)
        x = self.channelwise(x)
        x = self.gamma*x
        return channel_to_first(x) + residual


class UpBlock(nn.Module):
    def __init__(self, n, in_channels, out_channels, cond, skip, add_upsample=True):
        super().__init__()
        self.n = n
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_upsample = add_upsample
        self.skip_block = ResBlock(in_channels, in_channels*4, cond, skip)
        self.res_blocks = nn.ModuleList()
        for _ in range(self.n - 1):
            self.res_blocks.append(ResBlock(in_channels, in_channels*4, cond))
        if add_upsample:
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
    def forward(self, x, c, skip=None):
        x = self.skip_block(x, c, skip)
        for i, block in enumerate(self.res_blocks):
            x = block(x, c)
        if self.add_upsample:
            x = self.upsample(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, n, in_channels, out_channels, cond, add_downsample=True) -> None:
        super().__init__()
        self.n = n
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond = cond
        self.res_blocks = nn.ModuleList()
        self.add_downsample = add_downsample
        for _ in range(self.n):
            self.res_blocks.append(
                ResBlock(out_channels, out_channels*4, cond)
            )
        if add_downsample:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, c):
        if self.add_downsample:
            x = self.downsample(x) # [bs, in_channels, h, w] ->  [bs, out_channels, h/2, w/2]
        for block in self.res_blocks:
            x = block(x, c)
        return x

class UNet2D(nn.Module):
    def __init__(self, num_tokens=8192, block_outputs=[320, 640, 1280], num_layers=[4,8,16], c_r=64, condition_dim=1024) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.block_outputs = block_outputs
        self.num_layers = num_layers
        self.c_r = c_r
        self.condition_dim = condition_dim
        self.embeding = nn.Embedding(num_tokens, block_outputs[0])
        self.down_block = nn.ModuleList()
        out_channels = block_outputs[0]
        for i, (channels, n) in enumerate(zip(block_outputs, num_layers)):
            is_first_block = i == 0
            in_channels = out_channels
            out_channels = channels
            self.down_block.append(DownBlock(n, in_channels, out_channels, condition_dim+c_r, not is_first_block))
        self.up_blocks = nn.ModuleList()
        skip = 0
        for i, (channel, n) in enumerate(zip(reversed(block_outputs), reversed(num_layers))):
            is_last_block = i + 1 == len(block_outputs)
            in_channels = block_outputs[-(i+1)]
            out_channels = block_outputs[-(i+2)] if not is_last_block else in_channels
            self.up_blocks.append(UpBlock(n, in_channels, out_channels, c_r+condition_dim, skip, not is_last_block))
            skip = out_channels
        self.clf = nn.Conv2d(block_outputs[0], num_tokens, kernel_size=1) # [bs, num_tokens, 32, 32]

    def forward(self, x, c, r): #[bs, ch, h, w], #[bs, condition_dim], #[bs, c_r] -> [bs, condtion_dim+c_r]
        x = channel_to_first(self.embeding(x))
        print(x.shape)
        c = torch.cat([c, r], dim=-1)[:, :, None, None] # [bs, condtion_dim+c_] -> [bs, condtion_dim+c_r, 1, 1]
        hidden_states = []
        for i, block in enumerate(self.down_block):
            x = block(x, c)
            hidden_states.append(x)
        hidden_states = reversed(hidden_states)
        x = self.up_blocks[0](next(hidden_states), c)
        for i, block in enumerate(self.up_blocks[1:]):
            x = block(x, c, next(hidden_states))
        return self.clf(x)



