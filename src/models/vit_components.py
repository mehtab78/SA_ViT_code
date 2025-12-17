"""
Vision Transformer Components for SA-ViT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

def get_sinusoidal_encoding(num_positions: int, embed_dim: int) -> torch.Tensor:
    """Generate sinusoidal positional encoding."""
    position = torch.arange(num_positions).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
    
    pos_encoding = torch.zeros(num_positions, embed_dim)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    
    return pos_encoding.unsqueeze(0)

class PatchEmbedding(nn.Module):
    """Convert spectrogram images into patch embeddings."""
    
    def __init__(self, image_size: int = 128, patch_size: int = 16, in_channels: int = 1, embed_dim: int = 128):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch projection using convolution
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # x: (B, 1, 128, 128) -> (B, embed_dim, 8, 8)
        x = self.projection(x)
        
        # Flatten spatial dimensions: (B, embed_dim, 64) -> (B, 64, embed_dim)
        x = x.flatten(2).transpose(-1, -2)
        return x

class Attention(nn.Module):
    """Attention mechanism."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
            
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    """MLP block for Transformer."""
    
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, 
                 out_features: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * 2)
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """Single Transformer block with pre-norm architecture."""
    
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, mlp_ratio: float = 2.0, 
                 dropout: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim, dropout)
        
        # Proper drop path handling
        self.drop_path = nn.Identity() if drop_path <= 0 else nn.Dropout(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CrossAttentionGating(nn.Module):
    """Cross-attention layer to fuse visual and semantic features."""
    
    def __init__(self, visual_dim: int, semantic_dim: int, fusion_dim: int, 
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.visual_dim = visual_dim
        self.semantic_dim = semantic_dim
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(semantic_dim, fusion_dim, bias=False)  # Semantic -> Query
        self.W_k = nn.Linear(visual_dim, fusion_dim, bias=False)   # Visual -> Key
        self.W_v = nn.Linear(visual_dim, fusion_dim, bias=False)   # Visual -> Value
        
        # Final projection after attention
        self.W_o = nn.Linear(fusion_dim, visual_dim)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(visual_dim + semantic_dim, visual_dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm_visual = nn.LayerNorm(visual_dim)
        self.norm_semantic = nn.LayerNorm(semantic_dim)

    def forward(self, visual_features: torch.Tensor, semantic_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: (B, N_patches, D_v)
            semantic_features: (B, D_s) broadcastable to (B, 1, D_s) -> (B, N_patches, D_s)
        Returns:
            Fused features: (B, N_patches, D_v)
        """
        B, N, D_v = visual_features.shape
        # semantic_features: (B, D_s) -> (B, 1, D_s) -> (B, N, D_s)
        semantic_expanded = semantic_features.unsqueeze(1).expand(-1, N, -1)
        
        # Normalize
        visual_norm = self.norm_visual(visual_features)
        semantic_norm = self.norm_semantic(semantic_expanded)
        
        # Get Q, K, V
        Q = self.W_q(semantic_norm)  # (B, N, D_f)
        K = self.W_k(visual_norm)    # (B, N, D_f)
        V = self.W_v(visual_norm)    # (B, N, D_f)
        
        # Scaled Dot-Product Attention
        scale = (self.fusion_dim // self.num_heads) ** -0.5
        Q = Q.view(B, N, self.num_heads, -1).transpose(1, 2)  # (B, H, N, D_f/H)
        K = K.view(B, N, self.num_heads, -1).transpose(1, 2)  # (B, H, N, D_f/H)
        V = V.view(B, N, self.num_heads, -1).transpose(1, 2)  # (B, H, N, D_f/H)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, N, N)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (B, H, N, D_f/H)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, self.fusion_dim)  # (B, N, D_f)
        
        # Final output projection
        attended_visual = self.W_o(attn_output)  # (B, N, D_v)
        attended_visual = self.dropout(attended_visual)
        
        # Gating mechanism
        gate_input = torch.cat([visual_features, semantic_features.unsqueeze(1).expand(-1, N, -1)], dim=-1)  # (B, N, D_v + D_s)
        gate_signal = self.gate(gate_input)  # (B, N, D_v)
        
        # Apply gating to the attended visual features
        gated_output = gate_signal * attended_visual  # (B, N, D_v)
        
        # Residual connection with original visual features
        fused_features = visual_features + gated_output  # (B, N, D_v)
        
        return fused_features

class VisualBranch(nn.Module):
    """Visual Branch of SA-ViT."""
    
    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int, 
                 num_heads: int, num_layers: int, dropout: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(get_sinusoidal_encoding(num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Proper stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=2.0, dropout=dropout, drop_path=dpr[i])
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the Visual Branch."""
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer layers
        for blk in self.blocks:
            x = blk(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Return CLS token output for classification, or all features if requested
        cls_output = x[:, 0]  # (B, embed_dim)
        features = x if return_features else None
        
        return cls_output, features