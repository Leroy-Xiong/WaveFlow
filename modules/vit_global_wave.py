"""
ViTGlobalWaveModel: Vision Transformer for global regular-grid ocean wave forecasting.

Operates on a 720×1440 grid (0.25° global, pole-deduplicated) with 16×16 patches
→ 45×90 = 2025 tokens. ERA5 conditioning is fused via pixel-level channel
concatenation before patchification (the regular-grid analog of direct_concat).

Designed for fair comparison with PatchedGraphTransformer_Huge:
same backbone (24 blocks, hidden=1024, heads=16), ~460M total params.
"""

import torch
import torch.nn as nn

from modules.vit_wave import PatchEmbed2D
from modules.graph_transformer import TransformerBlock, TimestepEmbedder


class ViTGlobalWaveModel(nn.Module):
    def __init__(
        self,
        img_size=(720, 1440),
        patch_size=16,
        in_channels=1,
        out_channels=1,
        era5_channels=3,
        era5_num_frames=4,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qk_norm=True,
        use_layer_scale=True,
        layer_scale_init=1e-5,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.era5_channels = era5_channels
        self.era5_num_frames = era5_num_frames

        patch_in_channels = in_channels + era5_channels * era5_num_frames
        self.patch_embed = PatchEmbed2D(
            img_size=img_size, patch_size=patch_size,
            in_channels=patch_in_channels, embed_dim=hidden_size,
        )
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qk_norm=qk_norm,
                use_layer_scale=use_layer_scale,
                layer_scale_init=layer_scale_init,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x, t, era5_data=None, **unused_kwargs):
        """
        Args:
            x:   (B, C_in, H, W) noisy wave grid at diffusion time t.
            t:   (B,) diffusion timestep in [0, 1].
            era5_data: (B, T, 3, H, W) ERA5 frames.

        Returns:
            (B, C_out, H, W) predicted x (clean sample).
        """
        B, C, H, W = x.shape

        if era5_data is not None:
            era5_flat = era5_data.reshape(B, self.era5_num_frames * self.era5_channels, H, W)
            inp = torch.cat([x, era5_flat], dim=1)
        else:
            inp = x

        tokens = self.patch_embed(inp) + self.pos_embed

        c = self.t_embedder(t)

        prores_alphas = unused_kwargs.get('prores_alphas', None)
        for i, block in enumerate(self.blocks):
            alpha = prores_alphas[i] if prores_alphas is not None else None
            tokens = block(tokens, c, prores_alpha=alpha)

        tokens = self.norm(tokens)
        patches = self.head(tokens)

        gH, gW = self.patch_embed.grid_size
        P = self.patch_size
        patches_2d = patches.reshape(
            B, gH, gW, P, P, self.out_channels
        ).permute(0, 5, 1, 3, 2, 4).contiguous()
        out = patches_2d.reshape(B, self.out_channels, gH * P, gW * P)

        if out.shape[2] != H or out.shape[3] != W:
            out = out[:, :, :H, :W]

        return out


def ViTGlobalWave_Huge(**kwargs):
    return ViTGlobalWaveModel(depth=24, hidden_size=1024, num_heads=16, **kwargs)


def ViTGlobalWave_Large(**kwargs):
    return ViTGlobalWaveModel(depth=16, hidden_size=512, num_heads=8, **kwargs)


def ViTGlobalWave_Base(**kwargs):
    return ViTGlobalWaveModel(depth=12, hidden_size=384, num_heads=6, **kwargs)
