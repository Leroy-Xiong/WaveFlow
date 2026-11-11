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

def compute_extreme_feats(era5_data, with_trend=False):
    """
    Extract per-sample storm-intensity scalars from the normalized ERA5 field.

    Args:
        era5_data: (B, T, 3, H, W) or (B, 3, H, W) normalized ERA5 data.
                   Channels: [wind_u, wind_v, pressure], each normalized to [-1, 1].
        with_trend: If True, append 4 temporal features (wind_slope, pmin_slope,
                    drift_h, drift_w) -> (B, 10). If False, legacy (B, 6) output
                    (default, keeps old checkpoints compatible).

    Returns:
        feats: (B, 6) or (B, 10) tensor of
               [u_max, v_max, wind_max, wind_mean, p_min, p_mean] + optional
               [wind_slope, pmin_slope, drift_h, drift_w]
               - First 6: aggregated over all frames and grid points
                 (higher wind / lower pressure => stronger storm).
               - wind_slope / pmin_slope: intensity trend between the first and
                 last third of the frames (storm strengthening / filling).
               - drift_h / drift_w: normalized displacement of the storm center
                 (argmax of wind) from the first to the last frame.
               Trend/drift are zeroed when fewer than 2 frames are available.
               Computed identically at train and sample time, so no
               distribution shift.
    """
    if era5_data is None:
        return None
    if era5_data.ndim == 4:
        era5_data = era5_data.unsqueeze(1)  # (B, 1, 3, H, W)
    u = era5_data[:, :, 0]
    v = era5_data[:, :, 1]
    p = era5_data[:, :, 2]
    wind = torch.sqrt(u * u + v * v)  # (B, T, H, W)
    feats = [
        u.abs().amax(dim=(1, 2, 3)),
        v.abs().amax(dim=(1, 2, 3)),
        wind.amax(dim=(1, 2, 3)),
        wind.mean(dim=(1, 2, 3)),
        p.amin(dim=(1, 2, 3)),
        p.mean(dim=(1, 2, 3)),
    ]
    if with_trend:
        B, T = wind.shape[0], wind.shape[1]
        if T >= 2:
            wf = wind.amax(dim=(2, 3))      # (B, T) per-frame max wind
            pf = p.amin(dim=(2, 3))         # (B, T) per-frame min pressure
            n3 = max(1, T // 3)
            feats.append(wf[:, -n3:].mean(dim=1) - wf[:, :n3].mean(dim=1))   # wind_slope
            feats.append(pf[:, -n3:].mean(dim=1) - pf[:, :n3].mean(dim=1))   # pmin_slope
            H, W = wind.shape[2], wind.shape[3]
            flat = wind.reshape(B, T, H * W)
            idx = flat.argmax(dim=2)        # (B, T) flat index of storm center
            hi = (idx // W).float() / max(H, 1)
            wi = (idx % W).float() / max(W, 1)
            feats.append(hi[:, -1] - hi[:, 0])  # drift_h
            feats.append(wi[:, -1] - wi[:, 0])  # drift_w
        else:
            feats.extend([torch.zeros(wind.shape[0], device=wind.device, dtype=wind.dtype)] * 4)
    return torch.stack(feats, dim=-1)  # (B, 6) or (B, 10)


def compute_extreme_feats_from_nodes(era5_on_nodes, num_frames=1):
    """
    Node-level variant of compute_extreme_feats for direct_concat fusion,
    where ERA5 is interpolated onto wave nodes instead of kept on a grid.

    Args:
        era5_on_nodes: (B, N, 3*T) normalized ERA5 on wave nodes, frame-major
                       per node: [u(t0), v(t0), p(t0), u(t1), v(t1), p(t1), ...].
        num_frames: Number of ERA5 time frames T.

    Returns:
        feats: (B, 6) [u_max, v_max, wind_max, wind_mean, p_min, p_mean]
               aggregated over all nodes and frames. Matches the first 6
               outputs of compute_extreme_feats (no trend/drift: node order
               carries no spatial layout).
    """
    if era5_on_nodes is None:
        return None
    B, N, D = era5_on_nodes.shape
    x = era5_on_nodes.reshape(B, N, num_frames, D // num_frames)
    u = x[..., 0]
    v = x[..., 1]
    p = x[..., 2]
    wind = torch.sqrt(u * u + v * v)  # (B, N, T)
    feats = [
        u.abs().amax(dim=(1, 2)),
        v.abs().amax(dim=(1, 2)),
        wind.amax(dim=(1, 2)),
        wind.mean(dim=(1, 2)),
        p.amin(dim=(1, 2)),
        p.mean(dim=(1, 2)),
    ]
    return torch.stack(feats, dim=-1)  # (B, 6)


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
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, dropout=0.0, **kwargs):
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

        # Attention-weight dropout (MC-dropout UQ baseline); train-mode only
        self.attn_drop = float(dropout)

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
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        
        # Reshape and project
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, dropout=0.0):
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

        # Attention-weight dropout (MC-dropout UQ baseline); train-mode only
        self.attn_drop = float(dropout)

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
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop if self.training else 0.0)

        # 3. Output Projection
        # (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim) -> (B, N, C)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        
        return out


# --------------------------------------------------------
# Core Graph Transformer Block
# --------------------------------------------------------

