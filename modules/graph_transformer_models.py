import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Mlp

# Shared components (single source: modules.graph_attention) + node-level Transformer blocks

from modules.graph_attention import (
    modulate,
    TimestepEmbedder,
    CoordinateEmbedder,
    compute_extreme_feats,
    ERA5Encoder,
    ERA5ViTEncoder,
    ERA5TemporalEncoder,
    LayerScale,
    QKLayerNorm,
    StableAttention,
    CrossAttention,
)


class TransformerBlock(nn.Module):
    """
    A Transformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
    This structure is agnostic to whether the input is image patches or graph nodes.
    
    Supports ProRes (Progressive Residual Warmup) for stable pretraining.
    Reference: https://arxiv.org/abs/2603.05369
    
    For BF16 stability in large models, uses:
    - StableAttention with QK Normalization
    - LayerScale for residual connections
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qk_norm=True, use_layer_scale=True, layer_scale_init=1e-5, dropout=0.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Use StableAttention with QK normalization for BF16 stability
        self.attn = StableAttention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm, dropout=dropout, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=dropout)
        
        # LayerScale for stable deep network training
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.ls1 = LayerScale(hidden_size, init_value=layer_scale_init)
            self.ls2 = LayerScale(hidden_size, init_value=layer_scale_init)
        
        # Condition only on Time (no class labels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, prores_alpha=None):
        """
        Args:
            x: (B, N, C) Input features
            c: (B, C) Time embedding
            prores_alpha: Optional scalar for ProRes residual warmup. When provided,
                          residual connections are scaled by this value.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Apply ProRes scaling to residual connections if enabled
        if prores_alpha is not None:
            attn_out = self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            if self.use_layer_scale:
                attn_out = self.ls1(attn_out)
            x = x + prores_alpha * gate_msa.unsqueeze(1) * attn_out
            
            mlp_out = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            if self.use_layer_scale:
                mlp_out = self.ls2(mlp_out)
            x = x + prores_alpha * gate_mlp.unsqueeze(1) * mlp_out
        else:
            attn_out = self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            if self.use_layer_scale:
                attn_out = self.ls1(attn_out)
            x = x + gate_msa.unsqueeze(1) * attn_out
            
            mlp_out = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            if self.use_layer_scale:
                mlp_out = self.ls2(mlp_out)
            x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


class TransformerBlockWithCrossAttn(nn.Module):
    """
    Enhanced Transformer block with Cross-Attention for ERA5 conditioning.
    Combines Self-Attention (node-to-node), Cross-Attention (node-to-ERA5), and MLP.
    
    Supports ProRes (Progressive Residual Warmup) for stable pretraining.
    Reference: https://arxiv.org/abs/2603.05369
    
    For BF16 stability in large models, uses:
    - StableAttention with QK Normalization
    - LayerScale for residual connections
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qk_norm=True, use_layer_scale=True, layer_scale_init=1e-5, dropout=0.0, **block_kwargs):
        super().__init__()

        # 1. Self-Attention: Inter-node interactions
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = StableAttention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm, dropout=dropout, **block_kwargs)

        # 2. Cross-Attention: Node-to-ERA5 interactions
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm, dropout=dropout)

        # 3. MLP: Feed-forward network
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=dropout)
        
        # LayerScale for stable deep network training
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.ls1 = LayerScale(hidden_size, init_value=layer_scale_init)
            self.ls_cross = LayerScale(hidden_size, init_value=layer_scale_init)
            self.ls2 = LayerScale(hidden_size, init_value=layer_scale_init)
        
        # AdaLN modulation with 9 parameters:
        # 6 for Self-Attn + MLP (shift, scale, gate each)
        # 3 for Cross-Attn (shift, scale, gate)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, c, context, prores_alpha=None):
        """
        Args:
            x: (B, N_nodes, C) Graph node features
            c: (B, C) Time embedding
            context: (B, L_era5, C) ERA5 encoded features
            prores_alpha: Optional scalar for ProRes residual warmup. When provided,
                          residual connections are scaled by this value.
        Returns:
            x: (B, N_nodes, C) Updated node features
        """
        # Split adaLN parameters
        params = self.adaLN_modulation(c).chunk(9, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_cross, scale_cross, gate_cross = params
        
        # Apply ProRes scaling to residual connections if enabled
        if prores_alpha is not None:
            # 1. Self-Attention Block
            attn_out = self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            if self.use_layer_scale:
                attn_out = self.ls1(attn_out)
            x = x + prores_alpha * gate_msa.unsqueeze(1) * attn_out
            
            # 2. Cross-Attention Block (Query from nodes, Key/Value from ERA5)
            x_norm = modulate(self.norm_cross(x), shift_cross, scale_cross)
            cross_out = self.cross_attn(x_norm, context)
            if self.use_layer_scale:
                cross_out = self.ls_cross(cross_out)
            x = x + prores_alpha * gate_cross.unsqueeze(1) * cross_out
            
            # 3. MLP Block
            mlp_out = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            if self.use_layer_scale:
                mlp_out = self.ls2(mlp_out)
            x = x + prores_alpha * gate_mlp.unsqueeze(1) * mlp_out
        else:
            # 1. Self-Attention Block
            attn_out = self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            if self.use_layer_scale:
                attn_out = self.ls1(attn_out)
            x = x + gate_msa.unsqueeze(1) * attn_out
            
            # 2. Cross-Attention Block (Query from nodes, Key/Value from ERA5)
            x_norm = modulate(self.norm_cross(x), shift_cross, scale_cross)
            cross_out = self.cross_attn(x_norm, context)
            if self.use_layer_scale:
                cross_out = self.ls_cross(cross_out)
            x = x + gate_cross.unsqueeze(1) * cross_out
            
            # 3. MLP Block
            mlp_out = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            if self.use_layer_scale:
                mlp_out = self.ls2(mlp_out)
            x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x


class GraphFinalLayer(nn.Module):
    """
    The final layer of GraphTransformer.
    Projects hidden states back to data dimension (e.g., 1 for wave height).
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# Main Model: GraphTransformer

class GraphTransformer(nn.Module):
    """
    Diffusion/Interpolant model for Unstructured Grid Data (Graph/Mesh)
    with ERA5 meteorological conditioning via Cross-Attention.
    
    Supports ProRes (Progressive Residual Warmup) for stable pretraining.
    Reference: https://arxiv.org/abs/2603.05369
    """
    def __init__(
        self,
        node_coords,                # (Num_Nodes, 2) numpy array or tensor
        in_channels=1,              # e.g., 1 for Significant Wave Height
        hidden_size=384,            # Reduced size compared to images to save memory
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        learn_sigma=True,
        use_cross_attn=True,        # Enable Cross-Attention for ERA5 conditioning
        era5_encoder_type='cnn',    # Encoder version: 'cnn' or 'vit'
        era5_patch_size=4,          # Patch size for ViT encoder
        era5_num_frames=1,          # Number of ERA5 time steps
        coord_encoding_type='fourier',  # Coordinate encoding type: 'fourier' or 'spherical_3d'
        use_prores=False,           # Enable ProRes (Progressive Residual Warmup)
        prores_warmup_epochs=1,     # T parameter: warmup epochs for layer 1
        use_tod_conditioning=False, # Enable time-of-day (tidal) conditioning
        time_feat_dim=26,           # Dim of absolute-time feature vector (wave_dataset.TIME_FEAT_DIM)
        tod_injection='trunk',      # 'trunk': time features feed all adaLN blocks (v3);
                                    # 'decoder': trunk stays time-free, time features only
                                    # reach the final layer.
        use_extreme_conditioning=False,  # Enable ERA5 storm-intensity conditioning
        extreme_feats_dim=6,        # Dim of storm-intensity feature vector
        # Large model stability features
        qk_norm=True,               # QK Layer Normalization for BF16 stability
        use_layer_scale=True,       # LayerScale for deep network stability
        layer_scale_init=1e-5,      # LayerScale initial value
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads
        self.use_cross_attn = use_cross_attn
        self.era5_num_frames = era5_num_frames
        self.depth = depth
        self.use_prores = use_prores
        self.prores_warmup_epochs = prores_warmup_epochs
        self.use_tod_conditioning = use_tod_conditioning
        self.tod_injection = tod_injection
        self.use_extreme_conditioning = use_extreme_conditioning
        self.extreme_feats_dim = extreme_feats_dim
        self.qk_norm = qk_norm
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init = layer_scale_init

        # Store coordinates as a fixed buffer (normalized to [0,1] for Fourier encoding)
        if isinstance(node_coords, np.ndarray):
            node_coords = torch.from_numpy(node_coords).float()
        self.register_buffer("node_coords", node_coords) # shape: (7788, 2), raw lat/lon
        self.num_nodes = node_coords.shape[0]

        # Normalize coordinates to [0, 1] for stable Fourier encoding
        coords_min = node_coords.min(dim=0).values
        coords_max = node_coords.max(dim=0).values
        coords_norm = (node_coords - coords_min) / (coords_max - coords_min + 1e-8)
        self.register_buffer("coords_norm", coords_norm)  # (7788, 2) in [0,1]

        # 1. Node Input Embedding: Projects scalar/vector data to hidden_dim
        self.x_embedder = nn.Linear(in_channels, hidden_size)

        # 2. Coordinate Embedding: Projects normalized (lat,lon) to hidden_dim
        # scale=1.0 matches PatchedGraphTransformer; normalized coords avoid aliasing
        self.pos_embedder = CoordinateEmbedder(
            in_dims=2,
            hidden_size=hidden_size,
            scale=1.0,
            encoding_type=coord_encoding_type,
            num_nodes=self.num_nodes
        )

        # 3. Timestep Embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # 3b. Absolute-time tidal feature conditioning (optional, v3)
        if self.use_tod_conditioning:
            # sin/cos features at tidal-constituent periods projected by a
            # plain MLP, added to the diffusion-t embedding (ungated).
            # NOTE: the legacy [0,1) tod scalar + zero-init tod_scale path was
            # removed — the model learned to ignore it (scale stayed ~1e-4),
            # and its unused parameters break DDP when time_feats is supplied.
            self.time_mlp = nn.Sequential(
                nn.Linear(time_feat_dim, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )

        # 3c. ERA5 storm-intensity conditioning (optional)
        # Explicit per-sample scalars (max wind, min pressure, ...) injected
        # alongside the timestep embedding so the model knows "this is a storm".
        if self.use_extreme_conditioning:
            self.extreme_mlp = nn.Sequential(
                nn.Linear(extreme_feats_dim, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )

        # 4. ERA5 Encoder (only if using cross-attention)
        if self.use_cross_attn:
            if era5_encoder_type == 'vit':
                self.era5_encoder = ERA5ViTEncoder(in_channels=3, embed_dim=hidden_size, patch_size=era5_patch_size, num_heads=num_heads, num_frames=era5_num_frames)
            else:
                self.era5_encoder = ERA5Encoder(in_channels=3, embed_dim=hidden_size, num_frames=era5_num_frames)

        # 5. Transformer Blocks
        if self.use_cross_attn:
            self.blocks = nn.ModuleList([
                TransformerBlockWithCrossAttn(
                    hidden_size, num_heads, mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm, use_layer_scale=use_layer_scale, layer_scale_init=layer_scale_init
                ) for _ in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                TransformerBlock(
                    hidden_size, num_heads, mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm, use_layer_scale=use_layer_scale, layer_scale_init=layer_scale_init
                ) for _ in range(depth)
            ])

        # 6. Final Layer
        self.final_layer = GraphFinalLayer(hidden_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in Transformer blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, era5_data=None, prores_alphas=None, tod=None, time_feats=None, extreme_feats=None):
        """
        Forward pass of GraphTransformer with ERA5 conditioning.

        Args:
            x: (B, Num_Nodes, C) tensor. E.g., (Batch, 7788, 1)
               Node-wise features (e.g., wave height at each grid point).
            t: (B,) tensor of diffusion timesteps.
            era5_data: (B, 3, 40, 40) tensor of ERA5 meteorological data.
                       Channels: [wind_u, wind_v, pressure]
                       Required if use_cross_attn=True, ignored otherwise.
            prores_alphas: (num_layers,) tensor of pre-computed alpha values for ProRes.
                           Each value is alpha(l, t) = min(t / (T * l), 1).
                           If None, ProRes is disabled (for inference).
            tod: (B,) tensor of normalized time-of-day values [0, 1).
                 Deprecated — accepted for backward compatibility but ignored.
            time_feats: (B, time_feat_dim) tensor of absolute-time sin/cos features
                        (tidal constituents + annual cycle). Preferred conditioning
                        input when use_tod_conditioning=True.
            extreme_feats: (B, K) optional pre-computed storm-intensity features.
                           If None and use_extreme_conditioning=True, computed
                           internally from era5_data via compute_extreme_feats.

        Returns:
            pred: (B, Num_Nodes, Out_Channels) Predicted noise or data
        """
        # 1. Embed Input Data
        x = self.x_embedder(x) # (B, Num_Nodes, hidden_size)

        # 2. Embed Position
        # Fourier encoding: use normalized coords for stable frequencies
        # Spherical_3d encoding: use raw lat/lon in degrees
        if self.pos_embedder.encoding_type in ('fourier', 'learnable'):
            pos_emb = self.pos_embedder(self.coords_norm)
        else:
            pos_emb = self.pos_embedder(self.node_coords)
        x = x + pos_emb.unsqueeze(0) # Broadcast to batch dim

        # 3. Embed Time
        t_emb = self.t_embedder(t)  # (B, hidden_size)

        # 3b. Embed absolute-time tidal features (v3) and choose injection point
        if self.use_tod_conditioning and time_feats is not None:
            time_emb = self.time_mlp(time_feats)  # (B, hidden_size), ungated
            if self.tod_injection == 'decoder':
                # Trunk stays time-free; time features only reach the final layer
                c_trunk = t_emb
                c_dec = t_emb + time_emb
            else:  # 'trunk'
                c_trunk = t_emb + time_emb
                c_dec = c_trunk
        else:
            c_trunk = t_emb
            c_dec = t_emb

        # 3c. Embed ERA5 storm-intensity features
        if self.use_extreme_conditioning:
            if extreme_feats is None:
                extreme_feats = compute_extreme_feats(era5_data, with_trend=(self.extreme_feats_dim > 6))
            if extreme_feats is None:
                extreme_feats = torch.zeros(x.shape[0], self.extreme_feats_dim, device=x.device, dtype=x.dtype)
            c_trunk = c_trunk + self.extreme_mlp(extreme_feats)
            c_dec = c_dec + self.extreme_mlp(extreme_feats)

        # 4. Encode ERA5 Context (if using cross-attention)
        if self.use_cross_attn:
            if era5_data is None:
                raise ValueError("era5_data is required when use_cross_attn=True")
            context = self.era5_encoder(era5_data) # (B, 100, hidden_size)
        else:
            context = None

        # 5. Transformer Blocks with optional ProRes
        for layer_idx, block in enumerate(self.blocks):
            # Get pre-computed ProRes alpha for this layer if enabled
            prores_alpha = None
            if self.use_prores and prores_alphas is not None:
                prores_alpha = prores_alphas[layer_idx]
            
            if self.use_cross_attn:
                x = block(x, c_trunk, context, prores_alpha=prores_alpha)
            else:
                x = block(x, c_trunk, prores_alpha=prores_alpha)

        # 6. Final Projection
        x = self.final_layer(x, c_dec) # (B, Num_Nodes, out_channels)

        # 7. Split Sigma if learning variance
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=-1)
            
        return x

# Example Configuration

def GraphTransformer_Small(node_coords, use_cross_attn=True, era5_encoder_type='cnn', era5_patch_size=4, era5_num_frames=1, qk_norm=True, use_layer_scale=True, layer_scale_init=1e-5, **kwargs):
    """
    Smaller model to handle O(N^2) attention on 7788 nodes.
    Suitable for limited GPU memory.
    """
    return GraphTransformer(
        node_coords,
        depth=8,
        hidden_size=256,
        num_heads=4,
        use_cross_attn=use_cross_attn,
        era5_encoder_type=era5_encoder_type,
        era5_patch_size=era5_patch_size,
        era5_num_frames=era5_num_frames,
        qk_norm=qk_norm,
        use_layer_scale=use_layer_scale,
        layer_scale_init=layer_scale_init,
        **kwargs
    )

def GraphTransformer_Base(node_coords, use_cross_attn=True, era5_encoder_type='cnn', era5_patch_size=4, era5_num_frames=1, qk_norm=True, use_layer_scale=True, layer_scale_init=1e-5, **kwargs):
    """
    Base model with moderate size.
    """
    return GraphTransformer(
        node_coords,
        depth=12,
        hidden_size=384,
        num_heads=6,
        use_cross_attn=use_cross_attn,
        era5_encoder_type=era5_encoder_type,
        era5_patch_size=era5_patch_size,
        era5_num_frames=era5_num_frames,
        qk_norm=qk_norm,
        use_layer_scale=use_layer_scale,
        layer_scale_init=layer_scale_init,
        **kwargs
    )


