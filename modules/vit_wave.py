"""
ViTWaveModel: Vision Transformer for regular-grid ocean wave forecasting.

Processes a 75×107 regular grid (~0.00935° resolution over HK waters) with
the same transformer backbone and cross-attention ERA5 fusion as the existing
PatchedGraphTransformer, enabling a controlled comparison between irregular
mesh and regular grid representations.

Patch size 4 → 19×27 = 513 patches (≈ 512 in the existing model).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.graph_transformer import (
    ERA5Encoder,
    TimestepEmbedder,
    TransformerBlockWithCrossAttn,
)


# ---------------------------------------------------------------------------
# 2D Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbed2D(nn.Module):
    """Convert a regular grid into a sequence of patch tokens.

    Input:  (B, C, H, W)
    Output: (B, num_patches, embed_dim)
    """

    def __init__(self, img_size=(75, 107), patch_size=4, in_channels=5,
                 embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            (img_size[0] + patch_size - 1) // patch_size,
            (img_size[1] + patch_size - 1) // patch_size,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # Pad to exact multiple of patch_size
        H, W = x.shape[2], x.shape[3]
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # (left, right, top, bottom) → pad right & bottom

        x = self.proj(x)  # (B, embed_dim, gH, gW)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


# ---------------------------------------------------------------------------
# Main ViT model
# ---------------------------------------------------------------------------

class ViTWaveModel(nn.Module):
    """Vision Transformer for regular-grid conditional wave forecasting.

    Parameters
    ----------
    img_size : tuple
        (H, W) of the input grid.  Default (75, 107).
    patch_size : int
        Patch size for 2D patchification.  Default 4.
    in_channels : int
        Input channels after normalisation.  Default 5 (hs, tp, dm_sin, dm_cos, elev).
    out_channels : int
        Output channels.  Default 5 (same as input for diffusion velocity prediction).
    hidden_size : int
        Transformer hidden dimension.  Default 384 (Base).
    depth : int
        Number of transformer blocks.  Default 12 (Base).
    num_heads : int
        Attention heads.  Default 6 (Base).
    mlp_ratio : float
        MLP hidden ratio.  Default 4.0.
    era5_in_channels : int
        ERA5 input channels.  Default 3 (u10, v10, mslp).
    era5_num_frames : int
        Number of ERA5 time frames.  Default 8.
    era5_encoder_type : str
        'cnn' or 'vit' for ERA5 encoder type.  Default 'cnn'.
    dropout : float
        Dropout rate.  Default 0.0.
    qk_norm : bool
        Whether to use QK LayerNorm.  Default False.
    use_tod_conditioning : bool
        Whether to enable absolute-time tidal conditioning (time_mlp → adaLN).
    tod_injection : str
        Injection point: 'trunk' or 'decoder'. Kept for config parity with
        PatchedGraphTransformer; ViTWaveModel has a single shared condition
        vector c for all blocks (no separate decoder layer), so both values
        behave identically ('trunk' semantics).
    time_feat_dim : int
        Dimensionality of time_feats input.  Default 26 (= TIME_FEAT_DIM).
    """

    def __init__(
        self,
        img_size=(75, 107),
        patch_size=4,
        in_channels=5,
        out_channels=5,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        era5_in_channels=3,
        era5_num_frames=8,
        era5_encoder_type='cnn',
        dropout=0.0,
        qk_norm=False,
        use_layer_scale=False,
        use_tod_conditioning=False,
        tod_injection='trunk',
        time_feat_dim=26,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.use_layer_scale = use_layer_scale
        self.use_tod_conditioning = use_tod_conditioning
        self.tod_injection = tod_injection

        # Patch embedding
        self.patch_embed = PatchEmbed2D(
            img_size=img_size, patch_size=patch_size,
            in_channels=in_channels, embed_dim=hidden_size,
        )
        self.num_patches = self.patch_embed.num_patches

        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Timestep embedder
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Absolute-time tidal conditioning (v3): time_feats → time_mlp → add to c
        if use_tod_conditioning:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_feat_dim, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )

        # ERA5 encoder
        if era5_encoder_type == 'cnn':
            self.era5_encoder = ERA5Encoder(
                in_channels=era5_in_channels,
                embed_dim=hidden_size,
                num_frames=era5_num_frames,
            )
        else:
            from modules.graph_transformer import ERA5ViTEncoder
            self.era5_encoder = ERA5ViTEncoder(
                in_channels=era5_in_channels,
                embed_dim=hidden_size,
                num_frames=era5_num_frames,
            )
        self.era5_encoder_type = era5_encoder_type

        # Transformer blocks with cross-attention
        self.blocks = nn.ModuleList([
            TransformerBlockWithCrossAttn(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                qk_norm=qk_norm,
                use_layer_scale=use_layer_scale,
            )
            for _ in range(depth)
        ])

        # Final LayerNorm
        self.norm = nn.LayerNorm(hidden_size)

        # Prediction head: patch token → flattened patch pixels
        self.patch_out_dim = patch_size * patch_size * out_channels
        self.head = nn.Linear(hidden_size, self.patch_out_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # Xavier uniform for linear layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Timestep embedder: normal init
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Time-of-day MLP: normal init + zero-out last layer (zero tod contribution at start)
        if self.use_tod_conditioning:
            nn.init.normal_(self.time_mlp[0].weight, std=0.02)
            nn.init.constant_(self.time_mlp[-1].weight, 0)
            nn.init.constant_(self.time_mlp[-1].bias, 0)

        # Zero-out adaLN modulation layers in transformer blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Prediction head: trunc_normal (standard ViT)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, t, era5_data=None, tod=None, time_feats=None):
        """
        Args:
            x:   (B, C, H, W)  normalised wave grid at diffusion time t.
            t:   (B,)          diffusion timestep in [0, 1].
            era5_data: (B, T, 3, H_era, W_era) or None.
            tod: (B,)          deprecated — accepted for backward compat, ignored.
            time_feats: (B, time_feat_dim) absolute-time sin/cos tidal features
                        (active path when use_tod_conditioning=True).

        Returns:
            (B, C, H, W)  predicted velocity (or score) with same shape as x.
        """
        B, C, H, W = x.shape
        target_H, target_W = self.img_size

        # Patch embedding + position encoding
        tokens = self.patch_embed(x)                # (B, P, D)
        tokens = tokens + self.pos_embed

        # Timestep embedding
        c = self.t_embedder(t)                      # (B, D)

        # Absolute-time tidal conditioning (v3): ungated addition to c
        if self.use_tod_conditioning and time_feats is not None:
            time_emb = self.time_mlp(time_feats)    # (B, D)
            c = c + time_emb

        # ERA5 context tokens
        if era5_data is not None:
            context = self.era5_encoder(era5_data)  # (B, T*L, D)
        else:
            context = None

        # Transformer blocks with adaLN-Zero + cross-attention
        for block in self.blocks:
            tokens = block(tokens, c, context)

        tokens = self.norm(tokens)

        # Prediction head → patch pixels
        patches = self.head(tokens)                 # (B, P, patch_dim)

        # Reconstruct image from patches using fold
        gH, gW = self.patch_embed.grid_size
        P = self.patch_size
        patches_2d = patches.reshape(
            B, gH, gW, P, P, self.out_channels
        ).permute(0, 5, 1, 3, 2, 4).contiguous()
        # (B, C, gH, P, gW, P) → (B, C, gH*P, gW*P)
        out = patches_2d.reshape(B, self.out_channels, gH * P, gW * P)

        # Crop back to original image size
        if out.shape[2] != target_H or out.shape[3] != target_W:
            out = out[:, :, :target_H, :target_W]

        return out


# ---------------------------------------------------------------------------
# Model size configs
# ---------------------------------------------------------------------------

def ViTWave_Small(**kwargs):
    return ViTWaveModel(depth=8, hidden_size=256, num_heads=4, **kwargs)


def ViTWave_Base(**kwargs):
    return ViTWaveModel(depth=12, hidden_size=384, num_heads=6, **kwargs)


def ViTWave_Large(**kwargs):
    return ViTWaveModel(depth=16, hidden_size=512, num_heads=8, **kwargs)
