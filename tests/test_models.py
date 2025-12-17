"""
Tests for SA-ViT Models
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import SAViT, PatchEmbedding, Attention, TransformerBlock
from rules import RuleEngine, CandleProperties, create_candle

class TestModels:
    """Test model components."""
    
    def setup_method(self):
        """Setup for each test."""
        self.device = torch.device("cpu")  # Use CPU for tests
        self.batch_size = 2
        self.image_size = 128
        self.patch_size = 16
        self.embed_dim = 64  # Smaller for faster tests
        self.num_heads = 4
        self.num_layers = 2  # Fewer layers for tests
        self.num_patterns = 3
        self.num_classes = 2
    
    def test_patch_embedding(self):
        """Test patch embedding layer."""
        patch_embed = PatchEmbedding(
            image_size=self.image_size,
            patch_size=self.patch_size,
            in_channels=1,
            embed_dim=self.embed_dim
        ).to(self.device)
        
        # Create sample input
        x = torch.randn(self.batch_size, 1, self.image_size, self.image_size).to(self.device)
        
        # Forward pass
        output = patch_embed(x)
        
        # Check output shape
        expected_patches = (self.image_size // self.patch_size) ** 2
        assert output.shape == (self.batch_size, expected_patches, self.embed_dim)
    
    def test_attention(self):
        """Test attention mechanism."""
        attention = Attention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=0.1
        ).to(self.device)
        
        # Create sample input
        x = torch.randn(self.batch_size, 10, self.embed_dim).to(self.device)
        
        # Forward pass
        output = attention(x)
        
        # Check output shape
        assert output.shape == x.shape
    
    def test_transformer_block(self):
        """Test transformer block."""
        transformer_block = TransformerBlock(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=0.1
        ).to(self.device)
        
        # Create sample input
        x = torch.randn(self.batch_size, 10, self.embed_dim).to(self.device)
        
        # Forward pass
        output = transformer_block(x)
        
        # Check output shape
        assert output.shape == x.shape
    
    def test_savit_model(self):
        """Test full SA-ViT model."""
        model = SAViT(
            image_size=self.image_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            num_patterns=self.num_patterns,
            num_classes=self.num_classes,
            dropout=0.1,
            drop_path=0.1
        ).to(self.device)
        
        # Create sample inputs
        spectrogram = torch.randn(self.batch_size, 1, self.image_size, self.image_size).to(self.device)
        semantic_vector = torch.randn(self.batch_size, self.num_patterns).to(self.device)
        
        # Forward pass
        outputs, aux = model(spectrogram, semantic_vector)
        
        # Check output shape
        assert outputs.shape == (self.batch_size, self.num_classes)
        assert aux is not None

class TestRuleEngine:
    """Test rule engine and pattern detection."""
    
    def setup_method(self):
        """Setup for each test."""
        self.rule_engine = RuleEngine()
    
    def test_candle_properties(self):
        """Test candlestick properties."""
        # Create test candle
        ohlc = np.array([100.0, 105.0, 98.0, 103.0])  # Open, High, Low, Close
        candle = create_candle(ohlc)
        
        # Test properties
        assert candle.open == 100.0
        assert candle.high == 105.0
        assert candle.low == 98.0
        assert candle.close == 103.0
        assert candle.is_bullish == True  # Close > Open
        assert candle.is_bearish == False
        assert candle.body_size == 3.0
        assert candle.total_range == 7.0
        assert candle.upper_shadow == 2.0
        assert candle.lower_shadow == 2.0
    
    def test_pattern_detection(self):
        """Test pattern detection logic."""
        # Create test candles for evening star pattern
        c1_ohlc = np.array([100.0, 105.0, 98.0, 104.0])  # Bullish
        c2_ohlc = np.array([104.5, 106.0, 103.0, 104.2])  # Small body
        c3_ohlc = np.array([104.0, 105.0, 99.0, 101.0])  # Bearish
        
        c1 = create_candle(c1_ohlc)
        c2 = create_candle(c2_ohlc)
        c3 = create_candle(c3_ohlc)
        
        # Test pattern detection
        advance_block = self.rule_engine.detect_advance_block(c1, c2, c3)
        doji_star = self.rule_engine.detect_doji_star(c2, c3, 2.0)
        evening_star = self.rule_engine.detect_evening_star(c1, c2, c3, 2.0)
        
        # Basic sanity checks
        assert isinstance(advance_block, float)
        assert isinstance(doji_star, float)
        assert isinstance(evening_star, float)
        assert 0.0 <= advance_block <= 1.0
        assert 0.0 <= doji_star <= 1.0
        assert 0.0 <= evening_star <= 1.0
    
    def test_pattern_vector(self):
        """Test pattern vector calculation."""
        # Create test OHLC window
        ohlc_window = np.array([
            [100.0, 105.0, 98.0, 103.0],
            [103.5, 106.0, 102.0, 104.0],
            [104.0, 105.0, 99.0, 101.0]
        ])
        
        # Calculate pattern vector
        pattern_vector = self.rule_engine.calculate_pattern_vector(ohlc_window)
        
        # Check shape and values
        assert pattern_vector.shape == (3,)  # 3 patterns
        assert pattern_vector.dtype == torch.float32
        assert torch.all(pattern_vector >= 0.0)
        assert torch.all(pattern_vector <= 1.0)

class TestUtils:
    """Test utility functions."""
    
    def test_parameter_counting(self):
        """Test parameter counting."""
        from src.utils import count_parameters, count_trainable_parameters
        
        model = SAViT(
            image_size=128,
            patch_size=16,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            num_patterns=3,
            num_classes=2,
            dropout=0.1,
            drop_path=0.1
        )
        
        total_params = count_parameters(model)
        trainable_params = count_trainable_parameters(model)
        
        # Basic sanity checks
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params

if __name__ == "__main__":
    pytest.main([__file__])