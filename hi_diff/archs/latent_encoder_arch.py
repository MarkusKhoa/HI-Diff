import torch
import torch.nn as nn
from einops import rearrange
from basicsr.utils.registry import ARCH_REGISTRY

class MLP(nn.Module):
    def __init__(self,
                 num_patches,
                 embed_dims,
                 patch_expansion,
                 channel_expansion,
                 **kwargs):

        super(MLP, self).__init__()

        patch_mix_dims = int(patch_expansion * num_patches)
        channel_mix_dims = int(channel_expansion * embed_dims)

        self.patch_mixer = nn.Sequential(
            nn.Linear(num_patches, patch_mix_dims),
            nn.GELU(),
            nn.Linear(patch_mix_dims, num_patches),
        )

        self.channel_mixer = nn.Sequential(
            nn.Linear(embed_dims, channel_mix_dims),
            nn.GELU(),
            nn.Linear(channel_mix_dims, embed_dims),
        )

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

    def forward(self, x):
        x = x + self.patch_mixer(self.norm1(x).transpose(1,2)).transpose(1,2)
        x = x + self.channel_mixer(self.norm2(x))

        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DynamicConv(nn.Module):
    """
    Dynamic Convolution Layer: Generates convolutional kernels dynamically based on input features.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, reduction=4):
        super(DynamicConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Kernel generation network
        self.kernel_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, out_channels * kernel_size * kernel_size, kernel_size=1),
        )

        # Normalization for dynamic kernels
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()

        # Generate dynamic kernels
        kernels = self.kernel_gen(x)  # Shape: [B, out_channels * kernel_size^2, 1, 1]
        kernels = kernels.view(batch_size, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        kernels = self.softmax(kernels.view(batch_size, -1)).view_as(kernels)

        # Apply dynamic convolution
        x_unfold = nn.functional.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.view(batch_size, self.in_channels, self.kernel_size * self.kernel_size, -1)
        out = torch.einsum('bocij,bicj->boij', kernels, x_unfold)
        out = out.view(batch_size, self.out_channels, height, width)

        return out


@ARCH_REGISTRY.register()
class latent_encoder_gelu(nn.Module):
    def __init__(self, in_chans=6, embed_dim=64, block_num=4, stage=1, group=4, patch_expansion=0.5, channel_expansion=4):
        super(latent_encoder_gelu, self).__init__()

        assert in_chans == int(6 // stage), "in channel size is wrong"

        self.group = group

        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        self.conv1 = nn.Sequential(
            DynamicConv(in_chans * 16, embed_dim, 3, 1, 1),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList()
        for i in range(block_num):
            block = nn.Sequential(
                DynamicConv(embed_dim, embed_dim, 3, 1, 1),
                nn.GELU(),
                DynamicConv(embed_dim, embed_dim, 3, 1, 1),
            )
            self.blocks.append(block)

        self.conv2 = DynamicConv(embed_dim, embed_dim, 3, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((group, group))
        self.mlp = MLP(num_patches=group * group, embed_dims=embed_dim, patch_expansion=patch_expansion, channel_expansion=channel_expansion)
        self.end = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
        )

    def forward(self, inp_img, gt=None):
        if gt is not None:
            x = torch.cat([gt, inp_img], dim=1)
        else:
            x = inp_img

        x = self.pixel_unshuffle(x)
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x) + x
        x = self.pool(self.conv2(x))
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.mlp(x)
        x = self.end(x)
        return x


@ARCH_REGISTRY.register()
class latent_encoder_lrelu(nn.Module):
    def __init__(self, in_chans=6, embed_dim=64, block_num=4, stage=1, group=4, patch_expansion=0.5, channel_expansion=4):
        super(latent_encoder_lrelu, self).__init__()

        assert in_chans == int(6 // stage), "in channel size is wrong"

        self.group = group

        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        self.conv1 = nn.Sequential(
            DynamicConv(in_chans * 16, embed_dim, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
        )

        self.blocks = nn.ModuleList()
        for i in range(block_num):
            block = nn.Sequential(
                DynamicConv(embed_dim, embed_dim, 3, 1, 1),
                nn.LeakyReLU(0.1, True),
                DynamicConv(embed_dim, embed_dim, 3, 1, 1),
            )
            self.blocks.append(block)

        self.conv2 = nn.Sequential(
            DynamicConv(embed_dim, embed_dim * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            DynamicConv(embed_dim * 2, embed_dim * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            DynamicConv(embed_dim * 2, embed_dim * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
        )
        self.pool = nn.AdaptiveAvgPool2d((group, group))
        self.mlp = MLP(num_patches=group * group, embed_dims=embed_dim * 4, patch_expansion=patch_expansion, channel_expansion=channel_expansion)
        self.end = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim * 4),
            nn.LeakyReLU(0.1, True),
        )

    def forward(self, inp_img, gt=None):
        if gt is not None:
            x = torch.cat([gt, inp_img], dim=1)
        else:
            x = inp_img

        x = self.pixel_unshuffle(x)
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x) + x
        x = self.pool(self.conv2(x))
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.mlp(x)
        x = self.end(x)
        return x