"""
SA-ViT Model: Semantically-Augmented Vision Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Union

from .vit_components import (
    PatchEmbedding,
    Attention,
    MLP,
    TransformerBlock,
    VisualBranch,
    CrossAttentionGating,
    get_sinusoidal_encoding
)

class SAViT(nn.Module):
    """SA-ViT Model: Semantically-Augmented Vision Transformer."""
    
    def __init__(self, image_size: int, patch_size: int, embed_dim: int, num_heads: int, 
                 num_layers: int, num_patterns: int, num_classes: int, dropout: float = 0.1, 
                 drop_path: float = 0.0):
        super().__init__()
        
        self.visual_branch = VisualBranch(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=1,  # Spectrograms are single channel
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            drop_path=drop_path
        )
        
        self.cross_attention_gating = CrossAttentionGating(
            visual_dim=embed_dim,
            semantic_dim=num_patterns,  # Dimension of the pattern vector
            fusion_dim=embed_dim,  # Can be same as visual_dim or different
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Temperature scaling parameter for calibration
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, spectrogram: torch.Tensor, semantic_vector: torch.Tensor, 
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass of the SA-ViT model.
        
        Args:
            spectrogram: (B, C, H, W) - Spectrogram input
            semantic_vector: (B, K) - Pattern probability vector
            return_attention: If true, return attention weights
            
        Returns:
            outputs: (B, num_classes) - Logits
            aux_info: Optional dictionary with auxiliary outputs
        """
        # Pass spectrogram through Visual Branch to get initial visual features
        # We need the features for all patches to apply cross-attention
        visual_cls, visual_features_all = self.visual_branch(spectrogram, return_features=True)
        visual_patch_features = visual_features_all[:, 1:]  # Patch features for cross-attention: (B, N_patches, D)
        
        # Apply Cross-Attention Gating to fuse visual and semantic features
        fused_patch_features = self.cross_attention_gating(visual_patch_features, semantic_vector)
        
        # Aggregate fused features (e.g., mean pooling over patches)
        aggregated_features = torch.mean(fused_patch_features, dim=1)  # (B, D)
        
        # Combine CLS token with aggregated features
        final_features = visual_cls + aggregated_features  # (B, D)
        
        # Pass through classifier
        logits = self.classifier(final_features)  # (B, num_classes)
        
        # Apply temperature scaling for potential calibration during inference
        scaled_logits = logits / self.temperature
        
        aux_info = {"attention": None} if return_attention else None
        return scaled_logits, aux_info