"""
AFNONet model adapted from FourCastNet for OceanCastNet reproduction.

Original: https://github.com/NVlabs/FourCastNet (networks/afnonet.py)
Reference: https://github.com/NVlabs/AFNO-transformer

Modifications for OCN repro:
- Removed PrecipNet (not needed for wave prediction)
- Removed dependency on utils.img_utils (PeriodicPad2d)
- Removed dependency on params object — uses direct kwargs
- Input/Output channels configurable for wind→wave prediction
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torch.nn.modules.container import Sequential
from torch.utils.checkpoint import checkpoint_sequential
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
 

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFNO2D(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01,
                 hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, \
            f"hidden_size {hidden_size} should be divisible by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(
            2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(
            2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(
            2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(
            2, self.num_blocks, self.block_size))

    def forward(self, x):
        # Convert to FP32 for FFT precision.  Keep reference AFTER conversion
        # so the original BF16 tensor can be freed (bias points to FP32 copy).
        dtype = x.dtype
        x = x.float()
        bias = x

        B, H, W, C = x.shape

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros(
            [B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor],
            device=x.device)
        o1_imag = torch.zeros(
            [B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor],
            device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo',
                         x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real,
                         self.w1[0]) -
            torch.einsum('...bi,bio->...bo',
                         x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag,
                         self.w1[1]) +
            self.b1[0]
        )

        o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo',
                         x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag,
                         self.w1[0]) +
            torch.einsum('...bi,bio->...bo',
                         x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real,
                         self.w1[1]) +
            self.b1[1]
        )

        o2_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = (
            torch.einsum('...bi,bio->...bo',
                         o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes],
                         self.w2[0]) -
            torch.einsum('...bi,bio->...bo',
                         o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes],
                         self.w2[1]) +
            self.b2[0]
        )

        o2_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = (
            torch.einsum('...bi,bio->...bo',
                         o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes],
                         self.w2[0]) +
            torch.einsum('...bi,bio->...bo',
                         o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes],
                         self.w2[1]) +
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, H, W // 2 + 1, C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.type(dtype)

        return x + bias


class Block(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            double_skip=True,
            num_blocks=8,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = AFNO2D(dim, num_blocks, sparsity_threshold, hard_thresholding_fraction)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        # Gradient checkpointing on AFNO filter — trades ~20% compute for
        # ~16× memory reduction across layers (only 1 layer's FFT intermediates
        # are kept at a time instead of all 16).
        x = torch.utils.checkpoint.checkpoint(self.filter, x, use_reentrant=False)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = torch.utils.checkpoint.checkpoint(self.mlp, x, use_reentrant=False)
        x = self.drop_path(x)
        x = x + residual
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16),
                 in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class AFNONet(nn.Module):
    """
    AFNONet model for wind→wave prediction.

    Adapted from FourCastNet for OceanCastNet reproduction.
    Non-autoregressive: input is wind+pressure at t and t+1,
    output is wave field (Hs, Tm, DM) at t+1.

    Args:
        img_size: (H, W) of the input grid
        patch_size: (pH, pW) patch dimensions
        in_chans: number of input channels (default 6: u10,v10,mslp at t,t+1)
        out_chans: number of output channels (default 3: Hs,Tm,DM)
        embed_dim: embedding dimension
        depth: number of AFNO blocks
        mlp_ratio: MLP hidden dimension multiplier
        drop_rate: dropout rate
        drop_path_rate: stochastic depth rate
        num_blocks: number of frequency blocks in AFNO2D
        sparsity_threshold: softshrink threshold for AFNO2D
        hard_thresholding_fraction: fraction of frequency modes to keep
    """

    def __init__(
            self,
            img_size=(20, 20),
            patch_size=(4, 4),
            in_chans=6,
            out_chans=3,
            embed_dim=768,
            depth=12,
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.,
            num_blocks=16,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        norm_layer = lambda d: nn.LayerNorm(d, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.h = img_size[0] // patch_size[0]
        self.w = img_size[1] // patch_size[1]

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate,
                  drop_path=dpr[i], norm_layer=norm_layer,
                  num_blocks=self.num_blocks,
                  sparsity_threshold=sparsity_threshold,
                  hard_thresholding_fraction=hard_thresholding_fraction)
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(
            embed_dim, out_chans * patch_size[0] * patch_size[1], bias=False)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        x = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.img_size[0] // self.patch_size[0],
            w=self.img_size[1] // self.patch_size[1],
        )
        return x


if __name__ == "__main__":
    # Test with HK region grid size (example: 20x20 at 0.05 deg resolution)
    model = AFNONet(img_size=(20, 20), patch_size=(4, 4),
                    in_chans=6, out_chans=3, embed_dim=384, depth=8)
    sample = torch.randn(2, 6, 20, 20)
    result = model(sample)
    print(f"Input shape: {sample.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
