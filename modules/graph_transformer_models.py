import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Attention, Mlp
import torch.nn.functional as F

# --------------------------------------------------------
# Helper Functions
# --------------------------------------------------------

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# --------------------------------------------------------
# Embedding Layers
# --------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        # Use unsqueeze instead of indexing for torch.compile compatibility
        args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class CoordinateEmbedder(nn.Module):
    """
    Embeds spatial coordinates into vector representations.
    Supports three encoding modes:
    - 'fourier': Fourier feature mapping for planar coordinates
    - 'spherical_3d': Convert lat/lon to 3D Cartesian coordinates for spherical geometry
    - 'learnable': Learnable positional embeddings (no coordinate input)
    
    For geographical coordinates (latitude, longitude), the spherical_3d mode is recommended
    as it properly handles the spherical nature of Earth and avoids periodicity issues.
    The learnable mode is useful when coordinate information is not reliable or when
    you want the model to learn spatial relationships from data.
    """
    def __init__(self, in_dims, hidden_size, scale=1.0, encoding_type='fourier', num_nodes=None):
        super().__init__()
        self.in_dims = in_dims
        self.encoding_type = encoding_type
        self.num_nodes = num_nodes
        
        if encoding_type == 'spherical_3d':
            # For spherical conversion, input should be (lat, lon) in degrees
            # Output is 3D Cartesian coordinates normalized to unit sphere
            self.use_spherical = True
            self.use_learnable = False
            embed_input_dim = 3  # x, y, z in 3D space
        elif encoding_type == 'learnable':
            # Learnable positional embeddings - no coordinate input needed
            self.use_spherical = False
            self.use_learnable = True
            if num_nodes is None:
                raise ValueError("num_nodes must be specified when using 'learnable' encoding_type")
            # Create learnable embedding table: (num_nodes, hidden_size)
            self.pos_embed = nn.Parameter(torch.randn(num_nodes, hidden_size) * 0.02)
            embed_input_dim = hidden_size  # Not used, but kept for consistency
        else:
            # Traditional Fourier feature mapping
            self.use_spherical = False
            self.use_learnable = False
            # Random Gaussian matrix for Fourier feature mapping
            # We project coordinate dims (2) -> hidden_size // 2
            self.register_buffer("B", torch.randn((in_dims, hidden_size // 2)) * scale)
            embed_input_dim = hidden_size
        
        # MLP is only used for fourier and spherical_3d modes
        if encoding_type != 'learnable':
            self.mlp = nn.Sequential(
                nn.Linear(embed_input_dim, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )

    def forward(self, coords):
        """
        coords: (N_nodes, 2) where columns are [latitude, longitude] in degrees
                For 'learnable' mode, this parameter is ignored (can be None)
        return: (N_nodes, hidden_size)
        """
        if self.use_learnable:
            # Return learnable positional embeddings
            # coords is ignored in this mode
            return self.pos_embed
        elif self.use_spherical:
            # Convert spherical coordinates (lat, lon) to 3D Cartesian
            # Input: coords in degrees [lat, lon]
            # Output: 3D Cartesian coordinates on unit sphere
            lat_rad = torch.deg2rad(coords[:, 0])  # latitude in radians
            lon_rad = torch.deg2rad(coords[:, 1])  # longitude in radians
            
            # Spherical to Cartesian conversion
            x = torch.cos(lat_rad) * torch.cos(lon_rad)
            y = torch.cos(lat_rad) * torch.sin(lon_rad)
            z = torch.sin(lat_rad)
            
            # Stack into 3D coordinates: (N_nodes, 3)
            cart_coords = torch.stack([x, y, z], dim=-1)
            
            # Pass through MLP
            emb = self.mlp(cart_coords)
        else:
            # Fourier feature mapping: sin(2*pi*x*B), cos(2*pi*x*B)
            x_proj = (2 * torch.pi * coords) @ self.B  # (N_nodes, hidden/2)
            emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1) # (N_nodes, hidden)
            emb = self.mlp(emb)
        
        return emb


# --------------------------------------------------------
# ERA5 extreme-intensity features (storm conditioning)
# --------------------------------------------------------

def compute_extreme_feats(era5_data):
    """
    Extract per-sample storm-intensity scalars from the normalized ERA5 field.

    Args:
        era5_data: (B, T, 3, H, W) or (B, 3, H, W) normalized ERA5 data.
                   Channels: [wind_u, wind_v, pressure], each normalized to [-1, 1].

    Returns:
        feats: (B, 6) tensor of [u_max, v_max, wind_max, wind_mean, p_min, p_mean]
               aggregated over all frames and grid points. Higher wind / lower
               pressure => stronger storm. Computed identically at train and
               sample time, so no distribution shift.
    """
    if era5_data is None:
        return None
    if era5_data.ndim == 4:
        era5_data = era5_data.unsqueeze(1)  # (B, 1, 3, H, W)
    u = era5_data[:, :, 0]
    v = era5_data[:, :, 1]
    p = era5_data[:, :, 2]
    wind = torch.sqrt(u * u + v * v)  # (B, T, H, W)
    feats = torch.stack([
        u.abs().amax(dim=(1, 2, 3)),
        v.abs().amax(dim=(1, 2, 3)),
        wind.amax(dim=(1, 2, 3)),
        wind.mean(dim=(1, 2, 3)),
        p.amin(dim=(1, 2, 3)),
        p.mean(dim=(1, 2, 3)),
    ], dim=-1)  # (B, 6)
    return feats


class ERA5Encoder(nn.Module):
    """
    Encodes ERA5 meteorological data (B, T, 3, H, W) into context tokens.
    Uses CNN to extract spatial patterns and flatten into sequence features.
    
    Note: The input spatial dimensions (H, W) must be divisible by 4 due to two pooling layers.
    The resulting sequence length L will be T * (H/4) * (W/4).
    """
    def __init__(self, in_channels=3, embed_dim=384, num_frames=1):
        super().__init__()
        self.num_frames = num_frames
        # CNN to extract spatial features from ERA5 data
        # Input: (B*T, 3, H, W) -> Output: (B*T, embed_dim, H/4, W/4)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),  # HxW -> H/2 x W/2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),  # H/2 x W/2 -> H/4 x W/4
            nn.Conv2d(128, embed_dim, kernel_size=1)  # Project to embed_dim
        )
        self.norm = nn.LayerNorm(embed_dim)
        
        # Temporal embedding to distinguish different time steps
        if num_frames > 1:
            self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, 1, embed_dim))
            torch.nn.init.trunc_normal_(self.temporal_embed, std=.02)

    def forward(self, x):
        """
        Args:
            x: (B, T, 3, H, W) or (B, 3, H, W) ERA5 meteorological data
               Channels: [wind_u, wind_v, pressure]
        Returns:
            feat_seq: (B, T*L, embed_dim) Flattened context tokens, where L = (H/4)*(W/4)
        """
        if x.ndim == 4:
            # If single frame, add temporal dimension: (B, 3, H, W) -> (B, 1, 3, H, W)
            x = x.unsqueeze(1)
            
        B, T, C, H, W = x.shape
        
        # Fold batch and time: (B*T, C, H, W)
        x = x.reshape(B * T, C, H, W)
        
        feat_map = self.cnn(x)  # (B*T, embed_dim, H/4, W/4)
        
        # Flatten spatial dimensions into sequence: (B*T, L, embed_dim)
        BT, C_out, H_out, W_out = feat_map.shape
        feat_seq = feat_map.flatten(2).transpose(1, 2)
        
        # Add temporal embedding
        feat_seq = feat_seq.reshape(B, T, -1, C_out) # (B, T, L, C_out)
        if hasattr(self, 'temporal_embed'):
            # self.temporal_embed: (1, T, 1, C_out)
            feat_seq = feat_seq + self.temporal_embed
            
        # Flatten to (B, T*L, C_out)
        feat_seq = feat_seq.reshape(B, T * (H_out * W_out), C_out)
        
        return self.norm(feat_seq)


class ERA5ViTEncoder(nn.Module):
    """
    Encodes ERA5 meteorological data (B, T, 3, H, W) into context tokens using patch embedding only (no transformer blocks).
    """
    def __init__(self, in_channels=3, embed_dim=384, patch_size=4, input_size=(40, 40), depth=4, num_heads=6, mlp_ratio=4.0, num_frames=1):
        super().__init__()
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Calculate spatial sequence length L = (H/P) * (W/P).
        self.num_patches = (input_size[0] // patch_size) * (input_size[1] // patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Temporal embedding to distinguish different time steps
        if num_frames > 1:
            self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, 1, embed_dim))
            torch.nn.init.trunc_normal_(self.temporal_embed, std=.02)
        
        # Note: Transformer blocks removed - using patch embedding only
        
        # Initialize pos_embed
        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        """
        Args:
            x: (B, T, 3, H, W) or (B, 3, H, W) ERA5 meteorological data
        Returns:
            feat_seq: (B, T*L, embed_dim) Context tokens
        """
        if x.ndim == 4:
            x = x.unsqueeze(1)
            
        B, T, C, H, W = x.shape
        
        # Fold batch and time: (B*T, C, H, W)
        x = x.reshape(B * T, C, H, W)
        
        x = self.patch_embed(x)  # (B*T, embed_dim, H/P, W/P)
        BT, C_out, H_p, W_p = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B*T, L, C)
        
        # Add spatial positional embedding
        # Use interpolation to adapt pos_embed to different input sizes
        L = x.shape[1]
        if L != self.num_patches:
            # Interpolate pos_embed to match the actual sequence length
            pos_embed = self.pos_embed.transpose(1, 2)  # (1, embed_dim, num_patches)
            pos_embed = torch.nn.functional.interpolate(
                pos_embed, size=L, mode='linear', align_corners=False
            )
            pos_embed = pos_embed.transpose(1, 2)  # (1, L, embed_dim)
            x = x + pos_embed
        else:
            x = x + self.pos_embed
        
        # Add temporal embedding
        x = x.reshape(B, T, -1, C_out) # (B, T, L, C_out)
        if hasattr(self, 'temporal_embed'):
            x = x + self.temporal_embed
            
        # Re-flatten to sequence: (B, T*L, C_out)
        x = x.reshape(B, T * (H_p * W_p), C_out)
            
        return x


# --------------------------------------------------------
# ERA5 Temporal Encoder (1D CNN for frame compression)
# --------------------------------------------------------

class ERA5TemporalEncoder(nn.Module):
    """
    Compresses multi-frame ERA5 data along the time axis using shared-weight 1D CNN.

    Reduces 8 frames × 3 channels = 24 input dims per node → out_channels (e.g. 8)
    Applied per-node with shared weights across all 7788 nodes.
    """
    def __init__(self, in_channels=3, num_frames=8, hidden=32, out_channels=8):
        super().__init__()
        self.num_frames = num_frames
        self.in_channels = in_channels

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),    # (B*N, hidden, 8) → (B*N, hidden, 1)
            nn.Flatten(1),              # (B*N, hidden)
            nn.Linear(hidden, out_channels),
            nn.SiLU(),
        )

    def forward(self, era5_on_nodes):
        """
        Args:
            era5_on_nodes: (B, N, 3*T) concatenated ERA5 data on wave nodes
        Returns:
            (B, N, out_channels) temporally compressed features
        """
        B, N, C = era5_on_nodes.shape
        # Reshape to per-node time series: (B*N, num_frames, 3)
        x = era5_on_nodes.reshape(B * N, self.num_frames, self.in_channels)
        # Conv1d expects (batch, channels, length): (B*N, 3, 8)
        x = x.permute(0, 2, 1)
        x = self.net(x)  # (B*N, out_channels)
        x = x.reshape(B, N, -1)  # (B, N, out_channels)
        return x


# --------------------------------------------------------
# Cross-Attention Module
# --------------------------------------------------------

class LayerScale(nn.Module):
    """
    LayerScale: Learnable per-channel scaling for residual connections.
    Crucial for training deep transformers (ViT-22B, PaLM, etc.).
    
    Initialized with small value (e.g., 1e-5) to prevent gradient explosion
    in early training, then learned during training.
    
    Reference: Going deeper with Image Transformers (https://arxiv.org/abs/2103.17239)
    """
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))
    
    def forward(self, x):
        return x * self.gamma


class QKLayerNorm(nn.Module):
    """
    QK Layer Normalization for attention stability.
    Normalizes Q and K before computing attention scores.
    
    This is critical for BF16 training stability in large models.
    Reference: ViT-22B (https://arxiv.org/abs/2302.05442)
    
    Uses nn.LayerNorm internally for efficiency (leverages fused kernels).
    """
    def __init__(self, head_dim, eps=1e-6):
        super().__init__()
        # Use PyTorch's LayerNorm for efficiency (fused kernels)
        self.norm = nn.LayerNorm(head_dim, eps=eps)
    
    def forward(self, x):
        # x: (B, H, N, D)
        # LayerNorm expects last dimension to be normalized
        # Input shape is (B, H, N, D), normalize over D
        return self.norm(x)


class StableAttention(nn.Module):
    """
    Attention with QK Normalization for numerical stability.
    Replaces timm's Attention for BF16 training in large models.
    
    Key features:
    - QK LayerNorm before attention computation
    - Compatible with FlashAttention/SDPA
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        # QK Normalization - critical for BF16 stability
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = QKLayerNorm(self.head_dim)
            self.k_norm = QKLayerNorm(self.head_dim)
    
    def forward(self, x):
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, D)
        
        # Apply QK normalization for stability
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # Use scaled_dot_product_attention (supports FlashAttention)
        # Note: When QK norm is applied, we don't need the scale factor
        x = F.scaled_dot_product_attention(q, k, v)
        
        # Reshape and project
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        # QK Normalization for stability
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = QKLayerNorm(self.head_dim)
            self.k_norm = QKLayerNorm(self.head_dim)

    def forward(self, x, context):
        B, N, C = x.shape
        _, L, _ = context.shape

        # 1. Project Q, K, V
        # shape: (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # shape: (B, L, 2, num_heads, head_dim) -> (2, B, num_heads, L, head_dim)
        kv = self.kv_proj(context).reshape(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Apply QK normalization for stability
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 2. Flash Attention / SDPA
        # shape: (B, num_heads, N, head_dim)
        out = F.scaled_dot_product_attention(q, k, v)
        
        # 3. Output Projection
        # (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim) -> (B, N, C)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        
        return out


# --------------------------------------------------------
# Core Graph Transformer Block
# --------------------------------------------------------

from modules.graph_attention import (
    modulate,
    TimestepEmbedder,
    CoordinateEmbedder,
    compute_extreme_feats,
    ERA5Encoder,
    ERA5ViTEncoder,
    LayerScale,
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


# --------------------------------------------------------
# Main Model: GraphTransformer
# --------------------------------------------------------

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

# --------------------------------------------------------
# Example Configuration
# --------------------------------------------------------

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


