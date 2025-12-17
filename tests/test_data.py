"""
Tests for SA-ViT Data Module
"""

import pytest
import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import SpectrogramGenerator, DataPreprocessor, CryptoDataset, load_crypto_data

class TestSpectrogramGenerator:
    """Test spectrogram generation."""
    
    def setup_method(self):
        """Setup for each test."""
        self.generator = SpectrogramGenerator(image_size=64, num_scales=32)  # Smaller for tests
    
    def test_generate_spectrogram(self):
        """Test spectrogram generation."""
        # Create sample time series
        time_series = np.random.randn(100)
        
        # Generate spectrogram
        spectrogram = self.generator.generate(time_series)
        
        # Check output
        assert isinstance(spectrogram, torch.Tensor)
        assert spectrogram.shape == (1, 64, 64)  # (channels, height, width)
        assert spectrogram.dtype == torch.float32
        assert torch.all(spectrogram >= 0.0)
        assert torch.all(spectrogram <= 1.0)
    
    def test_normalization(self):
        """Test spectrogram normalization."""
        # Test with constant series
        constant_series = np.ones(50)
        spectrogram = self.generator.generate(constant_series)
        
        # Should be normalized (though may have some numerical precision issues)
        assert torch.allclose(spectrogram, torch.zeros_like(spectrogram), atol=1e-6)

class TestDataPreprocessor:
    """Test data preprocessing."""
    
    def setup_method(self):
        """Setup for each test."""
        self.preprocessor = DataPreprocessor(iqr_multiplier=1.5)
    
    def test_normalize(self):
        """Test normalization."""
        # Create sample DataFrame
        df = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [95, 96, 97, 98, 99],
            'Close': [102, 103, 104, 105, 106],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        # Normalize
        normalized = self.preprocessor.normalize(df, ['Open', 'High', 'Low', 'Close'])
        
        # Check that values are in [0, 1] range
        assert normalized[['Open', 'High', 'Low', 'Close']].min().min() >= 0.0
        assert normalized[['Open', 'High', 'Low', 'Close']].max().max() <= 1.0
    
    def test_remove_outliers(self):
        """Test outlier removal."""
        # Create DataFrame with outliers
        df = pd.DataFrame({
            'Open': [100, 101, 102, 500, 104],  # 500 is outlier
            'High': [105, 106, 107, 600, 109],  # 600 is outlier
            'Low': [95, 96, 97, 98, 99],
            'Close': [102, 103, 104, 105, 106]
        })
        
        # Remove outliers
        cleaned = self.preprocessor.remove_outliers_iqr(df, ['Open', 'High', 'Low', 'Close'])
        
        # Check that outliers are clipped
        assert cleaned['Open'].max() < 500
        assert cleaned['High'].max() < 600
    
    def test_preprocess_pipeline(self):
        """Test full preprocessing pipeline."""
        # Create sample DataFrame
        df = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [95, 96, 97, 98, 99],
            'Close': [102, 103, 104, 105, 106],
            'Volume': [1000, 1100, 1200, 1300, 1400],
            'Date': pd.date_range('2023-01-01', periods=5)
        })
        
        # Preprocess
        preprocessed = self.preprocessor.preprocess(df, ticker='TEST')
        
        # Check output
        assert len(preprocessed) == len(df)
        assert 'Ticker' not in preprocessed.columns  # Should not have ticker column yet

class TestDataLoading:
    """Test data loading functions."""
    
    def test_load_crypto_data_mock(self):
        """Test crypto data loading with mock."""
        # This would test the actual loading function
        # For now, just test that the function exists and can be called
        assert callable(load_crypto_data)
        
        # Test with empty list (should return empty DataFrame)
        result = load_crypto_data([], '2023-01-01', '2023-01-05')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

class TestCryptoDataset:
    """Test CryptoDataset class."""
    
    def setup_method(self):
        """Setup for each test."""
        # Create minimal components for testing
        self.preprocessor = DataPreprocessor()
        self.spectrogram_gen = SpectrogramGenerator(image_size=32, num_scales=16)
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        # This would require actual data, so we'll just test that the class exists
        assert hasattr(CryptoDataset, '__init__')
        assert hasattr(CryptoDataset, '__len__')
        assert hasattr(CryptoDataset, '__getitem__')

if __name__ == "__main__":
    pytest.main([__file__])