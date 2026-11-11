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
        feats: (B, 10) tensor of
               [u_max, v_max, wind_max, wind_mean, p_min, p_mean,
                wind_slope, pmin_slope, drift_h, drift_w]
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
    B, T = wind.shape[0], wind.shape[1]
    if T >= 2:
        wf = wind.amax(dim=(2, 3))      # (B, T) per-frame max wind
        pf = p.amin(dim=(2, 3))         # (B, T) per-frame min pressure
        n3 = max(1, T // 3)
        feats.append(wf[:, -n3:].mean(dim=1) - wf[:, :n3].mean(dim=1))   # wind_slope
        feats.append(pf[:, -n3:].mean(dim=1) - pf[:, :n3].mean(dim=1))   # pmin_slope
        H, W = wind.shape[2], wind.shape[3]
        idx = wind.argmax(dim=(2, 3))   # (B, T) flat index of storm center
        hi = (idx // W).float() / max(H, 1)
        wi = (idx % W).float() / max(W, 1)
        feats.append(hi[:, -1] - hi[:, 0])  # drift_h
        feats.append(wi[:, -1] - wi[:, 0])  # drift_w
    else:
        feats.extend([torch.zeros(B, device=wind.device, dtype=wind.dtype)] * 4)
    return torch.stack(feats, dim=-1)  # (B, 10)


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

        # Attention-weight dropout (MC-dropout UQ baseline); applied only in
        # train mode (SDPA dropout_p is a functional arg, so gate on self.training)
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

from modules.graph_attention import (
    modulate,
    TimestepEmbedder,
    CoordinateEmbedder,
    compute_extreme_feats,
    compute_extreme_feats_from_nodes,
    ERA5Encoder,
    ERA5ViTEncoder,
    ERA5TemporalEncoder,
    LayerScale,
)
from modules.graph_transformer_models import TransformerBlock, TransformerBlockWithCrossAttn


class PointNetPatchEncoder(nn.Module):
    """
    Encoder: Encodes multiple points within a Patch into a single Token.
    Input: Points in Patch (Local coords + features)
    Output: Patch Token
    """
    def __init__(self, in_channels, hidden_size):
        super().__init__()
        # in_channels is typically: 2 (dx, dy) + 1 (wave_height) = 3
        
        # 1. Per-point MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU()
        )
        
        # 2. Aggregation Projector (Concat Max + Mean -> Hidden)
        self.proj = nn.Linear(hidden_size * 2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        """
        x: (B, Num_Patches, Max_Points, In_Channels)
        mask: (B, Num_Patches, Max_Points) - True if valid point, False if padding
        """
        # 1. Point-wise feature extraction
        # (B, P, M, C) -> (B, P, M, H)
        point_feats = self.mlp(x)
        
        # 2. Aggregation (Max + Mean)
        if mask is not None:
            # Mask preparation for broadcasting: (B, P, M, 1)
            mask_expanded = mask.unsqueeze(-1)
            
            # Fill padding with very small value for Max Pooling
            feats_for_max = point_feats.masked_fill(~mask_expanded, -1e9)
            # Fill padding with 0 for Mean Pooling
            feats_for_mean = point_feats.masked_fill(~mask_expanded, 0)
            
            # Count valid points per patch
            valid_counts = mask_expanded.sum(dim=2).clamp(min=1)
        else:
            feats_for_max = point_feats
            feats_for_mean = point_feats
            valid_counts = x.shape[2]

        # Pooling
        feat_max = torch.max(feats_for_max, dim=2)[0] # (B, P, H)
        feat_mean = torch.sum(feats_for_mean, dim=2) / valid_counts # (B, P, H)
        
        # 3. Combine
        token = torch.cat([feat_max, feat_mean], dim=-1) # (B, P, 2H)
        token = self.proj(token) # (B, P, H)
        return self.norm(token)


class ImplicitPatchDecoder(nn.Module):
    """
    Decoder: Query-based decoding.
    Input: Patch Token + Query Point Local Coords
    Output: Predicted Value at Query Point
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_token = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # Input to MLP: Token (H) + Local Coords (2)
        self.decoder_mlp = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, out_channels)
        )
        
        # Time modulation for the decoder (similar to FinalLayer)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, tokens, local_coords, c):
        """
        tokens: (B, Num_Patches, H)
        local_coords: (B, Num_Patches, Max_Points, 2) - The (dx, dy) to query
        c: (B, H) - Time embedding
        """
        B, P, M, _ = local_coords.shape

        # 1. Modulate Token with Time
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        # Apply modulation to tokens (broadcast over patches)
        # tokens: (B, P, H), shift/scale: (B, 1, H) via broadcast
        tokens = modulate(self.norm_token(tokens), shift, scale)
        
        # 2. Expand tokens to match points
        # (B, P, H) -> (B, P, 1, H) -> (B, P, M, H)
        tokens_expanded = tokens.unsqueeze(2).expand(-1, -1, M, -1)
        
        # 3. Concatenate with Local Coordinates
        # (B, P, M, H+2)
        decoder_input = torch.cat([tokens_expanded, local_coords], dim=-1)
        
        # 4. Predict
        # (B, P, M, Out_Channels)
        return self.decoder_mlp(decoder_input)


class SimplePatchEncoder(nn.Module):
    """
    Simple MLP Encoder: Flatten patch points and encode to tokens.
    Input: (B, P, M, C) -> Flatten to (B, P, M*C) -> MLP to (B, P, H)
    """
    def __init__(self, in_channels, hidden_size, max_points):
        super().__init__()
        self.max_points = max_points
        # MLP: (M*C) -> H
        self.mlp = nn.Sequential(
            nn.Linear(max_points * in_channels, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        """
        x: (B, Num_Patches, Max_Points, In_Channels)
        mask: (B, Num_Patches, Max_Points) - Not used in this simple version
        """
        B, P, M, C = x.shape
        # Flatten: (B, P, M, C) -> (B, P, M*C)
        x_flat = x.reshape(B, P, M * C)
        
        # MLP encoding: (B, P, M*C) -> (B, P, H)
        token = self.mlp(x_flat)
        
        return self.norm(token)


class SimplePatchDecoder(nn.Module):
    """
    Simple MLP Decoder: Decode tokens back to flattened points, then reshape.
    Input: (B, P, H) -> MLP to (B, P, M*C) -> Reshape to (B, P, M, C)
    
    Optionally encodes local coordinates and adds to tokens before decoding.
    """
    def __init__(self, hidden_size, out_channels, max_points, use_coord_encoding=False, local_coord_dim=2, mlp_depth=2, use_cond=False):
        super().__init__()
        self.max_points = max_points
        self.out_channels = out_channels
        self.use_coord_encoding = use_coord_encoding
        self.local_coord_dim = local_coord_dim
        # When use_cond=True, the condition embedding c (e.g. diffusion-t + tidal
        # time features) is added to every patch token before decoding. This is
        # used by the 'decoder' tod-injection mode to keep time conditioning out
        # of the shared trunk. Default False preserves the original behavior
        # (c ignored), so old checkpoints are unaffected.
        self.use_cond = use_cond
        
        # Build MLP with configurable depth
        # mlp_depth=2: H -> H -> M*C (default, original behavior)
        # mlp_depth=3: H -> H -> H -> M*C
        # mlp_depth=4: H -> H -> H -> H -> M*C
        layers = []
        for i in range(mlp_depth - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_size, max_points * out_channels))
        self.decoder_mlp = nn.Sequential(*layers)
        
        # Optional: Encode local coordinates (B, P, M*local_coord_dim) -> (B, P, H)
        if self.use_coord_encoding:
            self.coord_encoder = nn.Sequential(
                nn.Linear(max_points * local_coord_dim, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size)
            )

    def forward(self, tokens, local_coords, c):
        """
        tokens: (B, Num_Patches, H)
        local_coords: (B, Num_Patches, Max_Points, local_coord_dim) - Local coordinates
        c: (B, H) - Condition embedding. Only used when use_cond=True;
                    ignored otherwise (backward-compatible default).
        """
        B, P, H = tokens.shape
        M = local_coords.shape[2]

        # Fuse condition embedding into every patch token
        if self.use_cond and c is not None:
            tokens = tokens + c.unsqueeze(1)
        
        # Optionally encode local coordinates and add to tokens
        if self.use_coord_encoding:
            # local_coords: (B, P, M, local_coord_dim)
            # Flatten: (B, P, M, local_coord_dim) -> (B, P, M*local_coord_dim)
            local_coords_flat = local_coords.reshape(B, P, -1)
            
            # Encode: (B, P, M*local_coord_dim) -> (B, P, H)
            coord_embed = self.coord_encoder(local_coords_flat)
            
            # Add to tokens
            tokens = tokens + coord_embed
        
        # MLP decoding: (B, P, H) -> (B, P, M*C)
        output_flat = self.decoder_mlp(tokens)
        
        # Reshape: (B, P, M*C) -> (B, P, M, C)
        output = output_flat.reshape(B, P, M, self.out_channels)
        
        return output


# --------------------------------------------------------
# Main Model: PatchedGraphTransformer
# --------------------------------------------------------

class PatchedGraphTransformer(nn.Module):
    """
    Patched Graph Transformer for efficient processing of large unstructured grids.
    
    Supports ProRes (Progressive Residual Warmup) for stable pretraining.
    Reference: https://arxiv.org/abs/2603.05369
    
    For BF16 stability in large models, supports:
    - QK Layer Normalization (qk_norm)
    - LayerScale for residual connections (use_layer_scale)
    """
    def __init__(
        self,
        patch_centroids,            # (Num_Patches, 2) - Global positions
        local_coords_template,      # (Num_Patches, Max_Points, 2/3/0) - Local coordinates template (2D, 3D, or none)
        mask_template,              # (Num_Patches, Max_Points) - Validity mask template
        cluster_labels,             # (N,) - Cluster assignment for each node
        in_channels=1,              
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        use_cross_attn=True,
        era5_encoder_type='cnn',    # Encoder version: 'cnn' or 'vit'
        era5_patch_size=4,          # Patch size for ViT encoder
        era5_num_frames=1,          # Number of ERA5 time steps
        coord_encoding_type='fourier',  # Coordinate encoding type: 'fourier' or 'spherical_3d'
        use_decoder_coord_encoding=False,  # Whether to encode local coords in decoder
        decoder_mlp_depth=2,        # MLP depth in SimplePatchDecoder (default: 2 for backward compatibility)
        multimodal_fusion_type='cross_attn',  # Fusion type: 'cross_attn' or 'direct_concat'
        era5_channels=3,            # Number of ERA5 channels (u_wind, v_wind, pressure)
        use_era5_temporal_encoder=False,  # Enable 1D CNN temporal compression for direct_concat
        era5_temporal_out_channels=8,     # Output channels from temporal encoder
        use_prores=False,           # Enable ProRes (Progressive Residual Warmup)
        prores_warmup_epochs=1,     # T parameter: warmup epochs for layer 1
        use_tod_conditioning=False, # Enable time-of-day (tidal) conditioning
        time_feat_dim=26,           # Dim of absolute-time feature vector (wave_dataset.TIME_FEAT_DIM)
        tod_injection='trunk',      # Where to inject tidal time features:
                                    #   'trunk'   - added to t_emb feeding all adaLN blocks (v3 behavior)
                                    #   'decoder' - trunk stays time-free (diffusion-t only); time features
                                    #               are fused only in the patch decoder, so hs/tp/dm are
                                    #               not modulated by the tidal signal
        use_extreme_conditioning=False,  # Enable ERA5 storm-intensity conditioning
        extreme_feats_dim=6,        # Dim of storm-intensity feature vector
        # Large model stability features
        qk_norm=True,               # QK Layer Normalization for BF16 stability
        use_layer_scale=True,       # LayerScale for deep network stability
        layer_scale_init=1e-5,      # LayerScale initial value
        # UQ baselines
        dropout=0.0,                # Attention/MLP dropout (MC-dropout baseline); 0.0 = off
        out_channels=None,          # Output channels; None = in_channels (CFM).
                                    # NLL regression baseline uses 2*in_channels (mu, log_sigma)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels else in_channels
        self.dropout_rate = float(dropout)
        self.use_cross_attn = use_cross_attn
        self.hidden_size = hidden_size
        self.era5_num_frames = era5_num_frames
        self.use_decoder_coord_encoding = use_decoder_coord_encoding
        self.decoder_mlp_depth = decoder_mlp_depth
        self.multimodal_fusion_type = multimodal_fusion_type
        self.era5_channels = era5_channels
        self.use_era5_temporal_encoder = use_era5_temporal_encoder
        self.era5_temporal_out_channels = era5_temporal_out_channels
        self.depth = depth
        self.use_prores = use_prores
        self.prores_warmup_epochs = prores_warmup_epochs
        self.use_tod_conditioning = use_tod_conditioning
        self.tod_injection = tod_injection
        self.use_extreme_conditioning = use_extreme_conditioning
        self.extreme_feats_dim = extreme_feats_dim
        # Store stability features config
        self.qk_norm = qk_norm
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init = layer_scale_init
        
        # Validate fusion type
        if multimodal_fusion_type not in ['cross_attn', 'direct_concat']:
            raise ValueError(f"Invalid multimodal_fusion_type: {multimodal_fusion_type}. "
                           f"Must be 'cross_attn' or 'direct_concat'.")
    
        # Register Patch Centroids (Fixed Anchors)
        if isinstance(patch_centroids, np.ndarray):
            patch_centroids = torch.from_numpy(patch_centroids).float()
        self.register_buffer("patch_centroids", patch_centroids) 
        self.num_patches = patch_centroids.shape[0]
            
        # Register Local Coordinates Template and Mask
        if isinstance(local_coords_template, np.ndarray):
            local_coords_template = torch.from_numpy(local_coords_template).float()
        if isinstance(mask_template, np.ndarray):
            mask_template = torch.from_numpy(mask_template)
        self.register_buffer("local_coords_template", local_coords_template)
        self.register_buffer("mask_template", mask_template)
            
        # Store cluster labels and create node-to-cluster mappings
        if isinstance(cluster_labels, np.ndarray):
            cluster_labels = torch.from_numpy(cluster_labels).long()
        self.register_buffer("cluster_labels", cluster_labels)
        self.num_nodes = cluster_labels.shape[0]
        self.max_points = local_coords_template.shape[1]
        
        # Determine local coordinate dimension (2 for 2D, 3 for 3D, 0 for none)
        self.local_coord_dim = local_coords_template.shape[2]
            
        # Create optimized node-to-patch mapping tensor for fast GPU indexing
        # node_to_patch_indices: [num_nodes] - which patch each node belongs to
        # patch_node_indices: [num_patches, max_points] - global node indices organized by patch
        # patch_node_counts: [num_patches] - number of valid nodes in each patch
            
        patch_node_indices = torch.full((self.num_patches, self.max_points), -1, dtype=torch.long)
        patch_node_counts = torch.zeros(self.num_patches, dtype=torch.long)
            
        for cluster_id in range(self.num_patches):
            node_indices = torch.where(cluster_labels == cluster_id)[0]
            n_points = len(node_indices)
            patch_node_indices[cluster_id, :n_points] = node_indices
            patch_node_counts[cluster_id] = n_points
            
        self.register_buffer("patch_node_indices", patch_node_indices)
        self.register_buffer("patch_node_counts", patch_node_counts)
        
        # Precomputed static indexing tensors for _organize_into_patches /
        # _restore_from_patches. These never change after init, so computing
        # them once avoids per-forward torch.where / boolean-mask indexing
        # (the latter forces a GPU->CPU nonzero() sync every forward pass).
        # persistent=False keeps them out of the state_dict, so checkpoints
        # stay compatible with earlier runs.
        patch_valid_mask = patch_node_indices >= 0                                  # [P, M]
        # -1 -> 0 for safe gather; masked out afterwards by patch_valid_mask_f
        patch_safe_indices = patch_node_indices.clamp_min(0)                        # [P, M]
        flat_indices = patch_node_indices.reshape(-1)                               # [P*M]
        flat_valid = patch_valid_mask.reshape(-1)                                   # [P*M]
        restore_positions = flat_valid.nonzero(as_tuple=True)[0]                    # [n_valid]
        restore_node_indices = flat_indices[flat_valid]                             # [n_valid]
        self.register_buffer("patch_valid_mask_f", patch_valid_mask.float(), persistent=False)
        self.register_buffer("patch_safe_indices", patch_safe_indices, persistent=False)
        self.register_buffer("restore_positions", restore_positions, persistent=False)
        self.register_buffer("restore_node_indices", restore_node_indices, persistent=False)
            
        # ---------------- Layers ----------------
            
        # 1. Patch Encoder (Simple MLP)
        # Input to encoder depends on fusion type
        self.era5_temporal_encoder = None
        if self.multimodal_fusion_type == 'direct_concat':
            if self.use_era5_temporal_encoder:
                # Temporal CNN compresses 24→out_channels before concatenation
                self.era5_temporal_encoder = ERA5TemporalEncoder(
                    in_channels=self.era5_channels,
                    num_frames=self.era5_num_frames,
                    out_channels=self.era5_temporal_out_channels,
                )
                encoder_in_channels = self.in_channels + self.local_coord_dim + self.era5_temporal_out_channels
            else:
                encoder_in_channels = self.in_channels + self.local_coord_dim + (self.era5_channels * self.era5_num_frames)
        else:
            # Cross attention: Input is only Local coords + Wave Data
            encoder_in_channels = self.in_channels + self.local_coord_dim
        self.patch_encoder = SimplePatchEncoder(in_channels=encoder_in_channels, hidden_size=hidden_size, max_points=self.max_points)
            
        # 2. Global Position Embedding (Fourier on Centroids)
        # Support three encoding modes: 'fourier', 'spherical_3d', 'learnable'
        self.global_pos_embedder = CoordinateEmbedder(
            in_dims=2, 
            hidden_size=hidden_size, 
            scale=1.0,
            encoding_type=coord_encoding_type,  # Use config parameter
            num_nodes=self.num_patches  # Required for 'learnable' mode
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

        # 4. ERA5 Encoder (only for cross_attn fusion type)
        if self.use_cross_attn and self.multimodal_fusion_type == 'cross_attn':
            if era5_encoder_type == 'vit':
                self.era5_encoder = ERA5ViTEncoder(in_channels=3, embed_dim=hidden_size, patch_size=era5_patch_size, num_heads=num_heads, num_frames=era5_num_frames)
            else:
                self.era5_encoder = ERA5Encoder(in_channels=3, embed_dim=hidden_size, num_frames=era5_num_frames)

        # 5. Transformer Backbone
        if self.use_cross_attn and self.multimodal_fusion_type == 'cross_attn':
            self.blocks = nn.ModuleList([
                TransformerBlockWithCrossAttn(
                    hidden_size, num_heads, mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm, use_layer_scale=use_layer_scale, layer_scale_init=layer_scale_init,
                    dropout=self.dropout_rate
                ) for _ in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                TransformerBlock(
                    hidden_size, num_heads, mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm, use_layer_scale=use_layer_scale, layer_scale_init=layer_scale_init,
                    dropout=self.dropout_rate
                ) for _ in range(depth)
            ])

        # 6. Patch Decoder (Simple MLP)
        # Output: Predicted wave height
        self.patch_decoder = SimplePatchDecoder(
            hidden_size, 
            self.out_channels, 
            max_points=self.max_points,
            use_coord_encoding=self.use_decoder_coord_encoding,
            local_coord_dim=self.local_coord_dim,
            mlp_depth=self.decoder_mlp_depth,
            use_cond=(self.use_tod_conditioning and self.tod_injection == 'decoder')
        )

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
    
    def _organize_into_patches(self, x):
        """
        Organize node data into patches based on clustering.
        Optimized version using vectorized operations to avoid Python loops.
        
        Args:
            x: (B, N, C) - Node-wise features where N is number of nodes
        
        Returns:
            (B, P, M, C) - Patch-organized data where:
                P = Num_Patches
                M = Max_Points per patch
        """
        B, N, C = x.shape
        
        # Static indexing tensors precomputed at init (patch_safe_indices has
        # -1 replaced by 0; invalid positions are zeroed by the mask below).
        indices_expanded = self.patch_safe_indices.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, C)
        
        # Gather operation: [B, N, C] -> [B, P, M, C]
        x_expanded = x.unsqueeze(1).expand(-1, self.num_patches, -1, -1)  # [B, P, N, C]
        x_patches = torch.gather(x_expanded, 2, indices_expanded)  # [B, P, M, C]
        
        # Zero out invalid positions (view-based broadcast, no per-call allocation)
        x_patches = x_patches * self.patch_valid_mask_f[None, :, :, None]
        
        return x_patches
    
    def _restore_from_patches(self, x_patches):
        """
        Restore patch-organized data back to node-wise format.
        Fully vectorized version without loops.
        
        Args:
            x_patches: (B, P, M, C) - Patch-organized predictions
        
        Returns:
            (B, N, C) - Node-wise predictions where N is number of nodes
        """
        B, P, M, C = x_patches.shape
        
        # Initialize node tensor
        x_nodes = torch.zeros(
            (B, self.num_nodes, C),
            dtype=x_patches.dtype,
            device=x_patches.device
        )
        
        # Flatten patches and select valid slots with precomputed static
        # positions (index_select instead of boolean-mask indexing — the
        # latter forces a GPU->CPU nonzero() sync on every forward pass).
        x_patches_flat = x_patches.reshape(B, -1, C)  # [B, P*M, C]
        valid_data = x_patches_flat.index_select(1, self.restore_positions)  # [B, n_valid, C]
        
        # Scatter data back to node positions
        indices = self.restore_node_indices.unsqueeze(0).unsqueeze(-1).expand(B, -1, C)
        x_nodes.scatter_(1, indices, valid_data)
        
        return x_nodes

    def forward(self, x, t, era5_data=None, era5_on_nodes=None, prores_alphas=None, tod=None, time_feats=None, extreme_feats=None):
        """
        Forward pass with automatic patch organization and restoration.

        Args:
            x: (B, Num_Nodes, C) - Node-wise wave data (same as GraphTransformer)
            t: (B,) - Timesteps
            era5_data: (B, 3, H, W) - ERA5 meteorological data on regular grid (optional)
                      Used when multimodal_fusion_type='cross_attn'
            era5_on_nodes: (B, Num_Nodes, era5_channels * num_frames) - ERA5 data interpolated
                          to wave node positions (optional). Used when multimodal_fusion_type='direct_concat'
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
            pred: (B, Num_Nodes, Out_Channels) - Node-wise predictions (same as GraphTransformer)

        Note:
            This method now has the same input/output interface as GraphTransformer.
            Internally, it organizes data into patches, processes them, and restores
            the output back to node-wise format.
        """
        # Convert input from (B, N, C) to (B, P, M, C)
        x_patches = self._organize_into_patches(x)
        B, P, M, C = x_patches.shape
        
        # Expand local_coords and mask to batch size
        # local_coords_template: (P, M, local_coord_dim) -> (B, P, M, local_coord_dim)
        # mask_template: (P, M) -> (B, P, M)
        local_coords = self.local_coords_template.unsqueeze(0).expand(B, -1, -1, -1)
        mask = self.mask_template.unsqueeze(0).expand(B, -1, -1)
        
        # --- 1. Encoding Stage ---
        
        if self.multimodal_fusion_type == 'direct_concat':
            # Direct concatenation fusion: ERA5 data is already interpolated to node positions
            if era5_on_nodes is None:
                raise ValueError("era5_on_nodes is required when multimodal_fusion_type='direct_concat'")

            # Grid ERA5 is unavailable under direct_concat; derive storm-intensity
            # features from the raw node-interpolated ERA5 before temporal compression.
            if self.use_extreme_conditioning and extreme_feats is None and self.extreme_feats_dim == 6:
                extreme_feats = compute_extreme_feats_from_nodes(era5_on_nodes, num_frames=self.era5_num_frames)

            # Apply temporal encoder to compress 24-dim → temporal_out_channels
            if self.era5_temporal_encoder is not None:
                era5_on_nodes = self.era5_temporal_encoder(era5_on_nodes)

            # Organize ERA5 data into patches
            era5_patches = self._organize_into_patches(era5_on_nodes)

            # Concatenate Local Coords + Wave Data + ERA5 Data
            if self.local_coord_dim > 0:
                encoder_input = torch.cat([local_coords, x_patches, era5_patches], dim=-1)
            else:
                encoder_input = torch.cat([x_patches, era5_patches], dim=-1)
        else:
            # Cross attention fusion: only Local Coords + Wave Data
            if self.local_coord_dim > 0:
                encoder_input = torch.cat([local_coords, x_patches], dim=-1)
            else:
                encoder_input = x_patches
        
        # Get Patch Tokens via PointNet
        # tokens: (B, P, hidden_size)
        tokens = self.patch_encoder(encoder_input, mask)
        
        # Add Global Position Embedding (based on Centroids)
        # global_pos: (P, hidden_size)
        global_pos = self.global_pos_embedder(self.patch_centroids)
        tokens = tokens + global_pos.unsqueeze(0) # Broadcast to B
        
        # Embed Time
        t_emb = self.t_embedder(t)  # (B, hidden_size)

        # Embed absolute-time tidal features (v3) and choose injection point
        c_trunk = t_emb
        c_dec = t_emb
        if self.use_tod_conditioning and time_feats is not None:
            time_emb = self.time_mlp(time_feats)  # ungated
            if self.tod_injection == 'decoder':
                # Trunk stays time-free; time features only reach the decoder
                c_dec = t_emb + time_emb
            else:  # 'trunk'
                c_trunk = t_emb + time_emb
                c_dec = c_trunk

        # Embed ERA5 storm-intensity features
        if self.use_extreme_conditioning:
            if extreme_feats is None:
                extreme_feats = compute_extreme_feats(era5_data, with_trend=(self.extreme_feats_dim > 6))
            if extreme_feats is None:
                extreme_feats = torch.zeros(x.shape[0], self.extreme_feats_dim, device=x.device, dtype=x.dtype)
            extreme_emb = self.extreme_mlp(extreme_feats)  # ungated
            c_trunk = c_trunk + extreme_emb
            c_dec = c_dec + extreme_emb

        # Encode ERA5 Context (only for cross_attn fusion)
        context = None
        if self.use_cross_attn and self.multimodal_fusion_type == 'cross_attn' and era5_data is not None:
            context = self.era5_encoder(era5_data)

        # --- 2. Processing Stage (Transformer) ---
        
        for layer_idx, block in enumerate(self.blocks):
            # Get pre-computed ProRes alpha for this layer if enabled
            prores_alpha = None
            if self.use_prores and prores_alphas is not None:
                prores_alpha = prores_alphas[layer_idx]
            
            if self.use_cross_attn and self.multimodal_fusion_type == 'cross_attn':
                tokens = block(tokens, c_trunk, context, prores_alpha=prores_alpha)
            else:
                tokens = block(tokens, c_trunk, prores_alpha=prores_alpha)
        
        # --- 3. Decoding Stage ---
        
        # Query the decoder at the specific local_coords
        # We use the processed tokens + the same local_coords we started with
        pred_patches = self.patch_decoder(tokens, local_coords, c_dec)  # (B, P, M, C)
        
        # Convert output from (B, P, M, C) back to (B, N, C)
        pred = self._restore_from_patches(pred_patches)
            
        return pred


def PatchedGraphTransformer_Tiny(patch_centroids, local_coords_template, mask_template, cluster_labels, use_cross_attn=True, era5_encoder_type='cnn', era5_patch_size=4, era5_num_frames=1, multimodal_fusion_type='cross_attn', **kwargs):
    """
    Tiny PatchedGraphTransformer model for size-sweep experiments.
    Configuration: depth=6, hidden_size=192, num_heads=3 (~5M params).

    Args:
        patch_centroids: (Num_Patches, 2) - Global positions of patch centers
        local_coords_template: (Num_Patches, Max_Points, 2) - Local coordinates template
        mask_template: (Num_Patches, Max_Points) - Validity mask template
        cluster_labels: (N,) - Cluster assignment for each node
        use_cross_attn: Whether to use cross attention with ERA5 data
        era5_encoder_type: Encoder version: 'cnn' or 'vit'
        era5_patch_size: Patch size for ViT encoder
        era5_num_frames: Number of ERA5 time steps
        multimodal_fusion_type: Fusion type: 'cross_attn' or 'direct_concat'
        **kwargs: Additional arguments passed to PatchedGraphTransformer

    Returns:
        PatchedGraphTransformer model with tiny architecture
    """
    return PatchedGraphTransformer(
        patch_centroids,
        local_coords_template,
        mask_template,
        cluster_labels,
        depth=6,
        hidden_size=192,
        num_heads=3,
        use_cross_attn=use_cross_attn,
        era5_encoder_type=era5_encoder_type,
        era5_patch_size=era5_patch_size,
        era5_num_frames=era5_num_frames,
        multimodal_fusion_type=multimodal_fusion_type,
        **kwargs
    )


def PatchedGraphTransformer_Small(patch_centroids, local_coords_template, mask_template, cluster_labels, use_cross_attn=True, era5_encoder_type='cnn', era5_patch_size=4, era5_num_frames=1, multimodal_fusion_type='cross_attn', **kwargs):
    """
    Smaller PatchedGraphTransformer model for limited GPU memory.
    Reduces depth and hidden size while maintaining patch-based processing.
    
    Args:
        patch_centroids: (Num_Patches, 2) - Global positions of patch centers
        local_coords_template: (Num_Patches, Max_Points, 2) - Local coordinates template
        mask_template: (Num_Patches, Max_Points) - Validity mask template
        cluster_labels: (N,) - Cluster assignment for each node
        use_cross_attn: Whether to use cross attention with ERA5 data
        era5_encoder_type: Encoder version: 'cnn' or 'vit'
        era5_patch_size: Patch size for ViT encoder
        era5_num_frames: Number of ERA5 time steps
        multimodal_fusion_type: Fusion type: 'cross_attn' or 'direct_concat'
        **kwargs: Additional arguments passed to PatchedGraphTransformer
    
    Returns:
        PatchedGraphTransformer model with smaller architecture
    """
    return PatchedGraphTransformer(
        patch_centroids,
        local_coords_template,
        mask_template,
        cluster_labels,
        depth=8,
        hidden_size=256,
        num_heads=4,
        use_cross_attn=use_cross_attn,
        era5_encoder_type=era5_encoder_type,
        era5_patch_size=era5_patch_size,
        era5_num_frames=era5_num_frames,
        multimodal_fusion_type=multimodal_fusion_type,
        **kwargs
    )


def PatchedGraphTransformer_Base(patch_centroids, local_coords_template, mask_template, cluster_labels, use_cross_attn=True, era5_encoder_type='cnn', era5_patch_size=4, era5_num_frames=1, multimodal_fusion_type='cross_attn', **kwargs):
    """
    Base PatchedGraphTransformer model with moderate size.
    Balanced configuration for performance and memory usage.
    
    Args:
        patch_centroids: (Num_Patches, 2) - Global positions of patch centers
        local_coords_template: (Num_Patches, Max_Points, 2) - Local coordinates template
        mask_template: (Num_Patches, Max_Points) - Validity mask template
        cluster_labels: (N,) - Cluster assignment for each node
        use_cross_attn: Whether to use cross attention with ERA5 data
        era5_encoder_type: Encoder version: 'cnn' or 'vit'
        era5_patch_size: Patch size for ViT encoder
        era5_num_frames: Number of ERA5 time steps
        multimodal_fusion_type: Fusion type: 'cross_attn' or 'direct_concat'
        **kwargs: Additional arguments passed to PatchedGraphTransformer
    
    Returns:
        PatchedGraphTransformer model with base architecture
    """
    return PatchedGraphTransformer(
        patch_centroids,
        local_coords_template,
        mask_template,
        cluster_labels,
        depth=12,
        hidden_size=384,
        num_heads=6,
        use_cross_attn=use_cross_attn,
        era5_encoder_type=era5_encoder_type,
        era5_patch_size=era5_patch_size,
        era5_num_frames=era5_num_frames,
        multimodal_fusion_type=multimodal_fusion_type,
        **kwargs
    )


def PatchedGraphTransformer_Large(patch_centroids, local_coords_template, mask_template, cluster_labels, use_cross_attn=True, era5_encoder_type='cnn', era5_patch_size=4, era5_num_frames=1, multimodal_fusion_type='cross_attn', **kwargs):
    """
    Large PatchedGraphTransformer model for high-performance scenarios.
    Larger configuration with more depth and wider hidden dimensions.
    
    Args:
        patch_centroids: (Num_Patches, 2) - Global positions of patch centers
        local_coords_template: (Num_Patches, Max_Points, 2) - Local coordinates template
        mask_template: (Num_Patches, Max_Points) - Validity mask template
        cluster_labels: (N,) - Cluster assignment for each node
        use_cross_attn: Whether to use cross attention with ERA5 data
        era5_encoder_type: Encoder version: 'cnn' or 'vit'
        era5_patch_size: Patch size for ViT encoder
        era5_num_frames: Number of ERA5 time steps
        multimodal_fusion_type: Fusion type: 'cross_attn' or 'direct_concat'
        **kwargs: Additional arguments passed to PatchedGraphTransformer
    
    Returns:
        PatchedGraphTransformer model with large architecture
    """
    return PatchedGraphTransformer(
        patch_centroids,
        local_coords_template,
        mask_template,
        cluster_labels,
        depth=16,
        hidden_size=512,
        num_heads=8,
        use_cross_attn=use_cross_attn,
        era5_encoder_type=era5_encoder_type,
        era5_patch_size=era5_patch_size,
        era5_num_frames=era5_num_frames,
        multimodal_fusion_type=multimodal_fusion_type,
        **kwargs
    )


def PatchedGraphTransformer_Huge(patch_centroids, local_coords_template, mask_template, cluster_labels, use_cross_attn=True, era5_encoder_type='cnn', era5_patch_size=4, era5_num_frames=1, multimodal_fusion_type='cross_attn', **kwargs):
    """
    Huge PatchedGraphTransformer model with ViT-Large scale parameters.
    Configuration: depth=24, hidden_size=1024, num_heads=16
    ~300M+ parameters for maximum capacity and performance.
    
    Args:
        patch_centroids: (Num_Patches, 2) - Global positions of patch centers
        local_coords_template: (Num_Patches, Max_Points, 2) - Local coordinates template
        mask_template: (Num_Patches, Max_Points) - Validity mask template
        cluster_labels: (N,) - Cluster assignment for each node
        use_cross_attn: Whether to use cross attention with ERA5 data
        era5_encoder_type: Encoder version: 'cnn' or 'vit'
        era5_patch_size: Patch size for ViT encoder
        era5_num_frames: Number of ERA5 time steps
        multimodal_fusion_type: Fusion type: 'cross_attn' or 'direct_concat'
        **kwargs: Additional arguments passed to PatchedGraphTransformer
    
    Returns:
        PatchedGraphTransformer model with huge architecture (ViT-Large scale)
    """
    return PatchedGraphTransformer(
        patch_centroids,
        local_coords_template,
        mask_template,
        cluster_labels,
        depth=24,
        hidden_size=1024,
        num_heads=16,
        use_cross_attn=use_cross_attn,
        era5_encoder_type=era5_encoder_type,
        era5_patch_size=era5_patch_size,
        era5_num_frames=era5_num_frames,
        multimodal_fusion_type=multimodal_fusion_type,
        **kwargs
    )


def PatchedGraphTransformer_Giant(patch_centroids, local_coords_template, mask_template, cluster_labels, use_cross_attn=True, era5_encoder_type='cnn', era5_patch_size=4, era5_num_frames=1, multimodal_fusion_type='cross_attn', **kwargs):
    """
    Giant PatchedGraphTransformer model with scale beyond ViT-Large.
    Configuration: depth=32, hidden_size=1280, num_heads=20
    ~1.2B+ parameters for extreme capacity and state-of-the-art performance.
    
    Args:
        patch_centroids: (Num_Patches, 2) - Global positions of patch centers
        local_coords_template: (Num_Patches, Max_Points, 2) - Local coordinates template
        mask_template: (Num_Patches, Max_Points) - Validity mask template
        cluster_labels: (N,) - Cluster assignment for each node
        use_cross_attn: Whether to use cross attention with ERA5 data
        era5_encoder_type: Encoder version: 'cnn' or 'vit'
        era5_patch_size: Patch size for ViT encoder
        era5_num_frames: Number of ERA5 time steps
        multimodal_fusion_type: Fusion type: 'cross_attn' or 'direct_concat'
        **kwargs: Additional arguments passed to PatchedGraphTransformer
    
    Returns:
        PatchedGraphTransformer model with giant architecture (beyond ViT-Large scale)
    """
    return PatchedGraphTransformer(
        patch_centroids,
        local_coords_template,
        mask_template,
        cluster_labels,
        depth=32,
        hidden_size=1280,
        num_heads=20,
        use_cross_attn=use_cross_attn,
        era5_encoder_type=era5_encoder_type,
        era5_patch_size=era5_patch_size,
        era5_num_frames=era5_num_frames,
        multimodal_fusion_type=multimodal_fusion_type,
        **kwargs
    )



# --------------------------------------------------------
# Simulate Data Loading and Run Tests
# --------------------------------------------------------

if __name__ == "__main__":
    # Settings
    B = 2
    N_PATCHES = 256
    MAX_POINTS = 35 # Assume each Patch has at most 35 points
    HIDDEN = 128
    T_ERA5 = 4 # 4 time points
    
    # 1. Simulate centroid data (Offline Preprocessing)
    centroids = np.random.rand(N_PATCHES, 2).astype(np.float32)
    
    # 2. Initialize model
    cluster_labels = torch.arange(N_PATCHES).repeat_interleave(MAX_POINTS)
    for p_size in [4, 8]:
        print(f"\nTesting with ERA5 patch_size: {p_size}, num_frames: {T_ERA5}")
        model = PatchedGraphTransformer(
            patch_centroids=centroids,
            local_coords_template=torch.zeros(N_PATCHES, MAX_POINTS, 2),
            mask_template=torch.ones(N_PATCHES, MAX_POINTS, dtype=torch.bool),
            cluster_labels=cluster_labels.numpy(),
            hidden_size=HIDDEN,
            depth=4, # For demonstration only
            num_heads=4,
            era5_encoder_type='vit',
            era5_patch_size=p_size,
            era5_num_frames=T_ERA5
        )
        
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        
        # 3. Simulate input data (Batch)
        # x: Wave height data (B, N)
        x = torch.randn(B, N_PATCHES * MAX_POINTS, 1)
        
        # t: Timesteps
        t = torch.rand(B)
        
        # era5: Meteorological data (B, T, 3, 40, 40)
        era5 = torch.randn(B, T_ERA5, 3, 40, 40)
        
        # 4. Forward
        output = model(x, t, era5_data=era5)
        
        # Check context shape (T * L, hidden_size)
        # For 40x40 and p_size=4, L=100. With T=4, total tokens = 400.
        context = model.era5_encoder(era5)
        print(f"ERA5 Encoder output shape: {context.shape}")
        print(f"Output shape: {output.shape}")
        
        expected_tokens = T_ERA5 * (40 // p_size) * (40 // p_size)
        assert context.shape[1] == expected_tokens, f"Expected {expected_tokens} tokens, got {context.shape[1]}"
        print(f"Forward pass for patch_size {p_size} and num_frames {T_ERA5} successful.")