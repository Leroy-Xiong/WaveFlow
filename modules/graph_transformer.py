"""
Graph / patch transformer model components (re-export shim).

Implementation has been split into:
  - modules.graph_attention:        shared attention/embedding/ERA5-encoder components
  - modules.graph_transformer_models: node-based GraphTransformer family
  - modules.patch_transformer:      PatchedGraphTransformer family

This module re-exports everything for backward compatibility with
existing ``from modules.graph_transformer import ...`` statements.
"""
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
from modules.graph_transformer_models import (
    TransformerBlock,
    TransformerBlockWithCrossAttn,
    GraphFinalLayer,
    GraphTransformer,
    GraphTransformer_Small,
    GraphTransformer_Base,
)
from modules.patch_transformer import (
    PointNetPatchEncoder,
    ImplicitPatchDecoder,
    SimplePatchEncoder,
    SimplePatchDecoder,
    PatchedGraphTransformer,
    PatchedGraphTransformer_Tiny,
    PatchedGraphTransformer_Small,
    PatchedGraphTransformer_Base,
    PatchedGraphTransformer_Large,
    PatchedGraphTransformer_Huge,
    PatchedGraphTransformer_Giant,
)

__all__ = [
    "modulate",
    "TimestepEmbedder",
    "CoordinateEmbedder",
    "compute_extreme_feats",
    "ERA5Encoder",
    "ERA5ViTEncoder",
    "ERA5TemporalEncoder",
    "LayerScale",
    "QKLayerNorm",
    "StableAttention",
    "CrossAttention",
    "TransformerBlock",
    "TransformerBlockWithCrossAttn",
    "GraphFinalLayer",
    "GraphTransformer",
    "GraphTransformer_Small",
    "GraphTransformer_Base",
    "PointNetPatchEncoder",
    "ImplicitPatchDecoder",
    "SimplePatchEncoder",
    "SimplePatchDecoder",
    "PatchedGraphTransformer",
    "PatchedGraphTransformer_Tiny",
    "PatchedGraphTransformer_Small",
    "PatchedGraphTransformer_Base",
    "PatchedGraphTransformer_Large",
    "PatchedGraphTransformer_Huge",
    "PatchedGraphTransformer_Giant",
]
