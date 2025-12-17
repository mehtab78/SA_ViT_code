#!/usr/bin/env python3
"""
Quick validation script for SA-ViT model
"""

import argparse
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import SAViT
from utils import evaluate_model, setup_matplotlib_for_plotting
from sklearn.metrics import accuracy_score
import numpy as np

def quick_validation(model_path: str, device: str = 'auto'):
    """Quick validation of model on sample data."""
    
    # Setup device
    if device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Create model
    model = SAViT(
        image_size=128,
        patch_size=16,
        embed_dim=128,
        num_heads=4,
        num_layers=6,
        num_patterns=3,
        num_classes=2,
        dropout=0.1,
        drop_path=0.1
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model loaded successfully")
    print(f"  Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        # Create sample data
        sample_spec = torch.randn(4, 1, 128, 128).to(device)  # Batch of 4
        sample_semantic = torch.randn(4, 3).to(device)  # Batch of 4, 3 patterns
        
        try:
            outputs, aux = model(sample_spec, sample_semantic)
            print(f"✓ Forward pass successful")
            print(f"  Output shape: {outputs.shape}")
            print(f"  Min logit: {outputs.min().item():.4f}")
            print(f"  Max logit: {outputs.max().item():.4f}")
            
            # Test probabilities
            probs = torch.softmax(outputs, dim=1)
            print(f"  Min probability: {probs.min().item():.4f}")
            print(f"  Max probability: {probs.max().item():.4f}")
            
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            return False
    
    # Test model components
    print("\nTesting model components...")
    
    try:
        # Test visual branch
        visual_cls, visual_features = model.visual_branch(sample_spec, return_features=True)
        print(f"✓ Visual branch: {visual_cls.shape}, {visual_features.shape}")
        
        # Test cross-attention
        fused_features = model.cross_attention_gating(
            visual_features[:, 1:],  # Remove CLS token
            sample_semantic
        )
        print(f"✓ Cross-attention: {fused_features.shape}")
        
        # Test classifier
        classifier_output = model.classifier(visual_cls)
        print(f"✓ Classifier: {classifier_output.shape}")
        
    except Exception as e:
        print(f"✗ Component test failed: {e}")
        return False
    
    # Test parameter counting
    print(f"\nModel Statistics:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Memory usage estimate
    memory_mb = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)  # 4 bytes per float32
    print(f"  Estimated memory usage: {memory_mb:.1f} MB")
    
    print("\n" + "="*50)
    print("✓ QUICK VALIDATION PASSED")
    print("Model is ready for training/inference!")
    print("="*50)
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Quick validation of SA-ViT model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default='auto', help="Device to use (auto/cpu/cuda)")
    args = parser.parse_args()
    
    success = quick_validation(args.model_path, args.device)
    
    if not success:
        print("\n❌ VALIDATION FAILED")
        print("Please check the model file and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()