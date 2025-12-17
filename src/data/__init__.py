"""
Data Loading and Preprocessing Module
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, Subset
import warnings
warnings.filterwarnings("ignore")

class SpectrogramGenerator:
    """Generate spectrograms from time series using Morlet Wavelets."""
    
    def __init__(self, image_size: int = 128, wavelet: str = 'morl', num_scales: int = 128):
        self.image_size = image_size
        self.wavelet = wavelet
        self.num_scales = num_scales
        
        try:
            import pywt
            import cv2
            self.pywt = pywt
            self.cv2 = cv2
        except ImportError as e:
            raise ImportError(f"Required packages not installed: {e}")
        
        self.scales = np.arange(1, num_scales + 1)

    def generate(self, time_series: np.ndarray) -> 'torch.Tensor':
        """Generate spectrogram from time series."""
        import torch
        
        # Continuous Wavelet Transform using Morlet wavelet
        coefficients, _ = self.pywt.cwt(time_series, self.scales, self.wavelet)
        
        # Calculate power spectrum (magnitude squared)
        power = np.abs(coefficients) ** 2
        
        # Resize to target image size
        resized = self.cv2.resize(power, (self.image_size, self.image_size), interpolation=self.cv2.INTER_LINEAR)
        
        # Normalize to [0, 1] range
        min_val = resized.min()
        max_val = resized.max()
        if max_val - min_val > 1e-8:
            resized = (resized - min_val) / (max_val - min_val)
        else:
            resized = np.zeros_like(resized)
        
        # Convert to tensor with channel dimension
        spectrogram = torch.tensor(resized, dtype=torch.float32).unsqueeze(0)
        
        return spectrogram

class DataPreprocessor:
    """Data preprocessing pipeline."""
    
    def __init__(self, iqr_multiplier: float = 1.5, normalize_range: Tuple[float, float] = (0, 1)):
        self.iqr_multiplier = iqr_multiplier
        self.normalize_range = normalize_range
        self.scalers = {}  # Store scalers per ticker

    def remove_outliers_iqr(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        df = df.copy()
        for col in columns:
            # Explicitly convert the column to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            
            if pd.isna(Q1) or pd.isna(Q3):
                continue
                
            IQR = Q3 - Q1
            lower_bound = Q1 - self.iqr_multiplier * IQR
            upper_bound = Q3 + self.iqr_multiplier * IQR
            
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        return df

    def normalize(self, df: pd.DataFrame, columns: List[str], fit: bool = True, ticker: str = 'default') -> pd.DataFrame:
        """Normalize columns to [0, 1] range using MinMaxScaler."""
        df = df.copy()
        scaler_key = ticker

        if fit:
            self.scalers[scaler_key] = MinMaxScaler(feature_range=self.normalize_range)
            normalized_data = self.scalers[scaler_key].fit_transform(df[columns])
            df.loc[:, columns] = normalized_data
        else:
            if scaler_key not in self.scalers:
                raise ValueError(f"No scaler found for ticker '{ticker}'. Must fit first.")
            df[columns] = self.scalers[scaler_key].transform(df[columns])

        return df

    def preprocess(self, df: pd.DataFrame, ticker: str = 'default') -> pd.DataFrame:
        """Apply the full preprocessing pipeline."""
        price_columns = ['Open', 'High', 'Low', 'Close']
        df = df.copy()
        
        # Fill NaN values
        df = df.ffill().fillna(method='bfill')
        
        # Remove outliers using IQR
        df = self.remove_outliers_iqr(df, price_columns)
        
        # Normalize prices and volume
        df = self.normalize(df, price_columns + ['Volume'], fit=True, ticker=ticker)

        return df

class CryptoDataset(Dataset):
    """Dataset for SA-ViT training on cryptocurrency data."""
    
    def __init__(self, tickers: List[str], start_date: str, end_date: str, 
                 window_size: int = 60, image_size: int = 128, 
                 prediction_step: int = 1, preprocessor: DataPreprocessor = None,
                 spectrogram_generator: SpectrogramGenerator = None):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        self.image_size = image_size
        self.prediction_step = prediction_step
        self.preprocessor = preprocessor
        self.spectrogram_generator = spectrogram_generator
        
        self.data = self._load_and_preprocess_data()
        self.samples = self._create_samples()

    def _load_and_preprocess_data(self) -> pd.DataFrame:
        """Load data for specified tickers and apply preprocessing."""
        combined_df = pd.DataFrame()
        
        for ticker in self.tickers:
            print(f"Loading data for {ticker}...")
            try:
                ticker_data = yf.download(ticker, start=self.start_date, end=self.end_date)
                if ticker_data.empty:
                    print(f"Warning: No data found for {ticker}")
                    continue
                    
                ticker_data.reset_index(inplace=True)
                ticker_data['Ticker'] = ticker
                combined_df = pd.concat([combined_df, ticker_data], ignore_index=True)
                
            except Exception as e:
                print(f"Error loading {ticker}: {e}")
                continue
        
        if combined_df.empty:
            raise ValueError("No data was loaded for any of the specified tickers.")
            
        print(f"Loaded raw data for {len(self.tickers)} tickers, shape: {combined_df.shape}")
        
        # Apply preprocessing per ticker
        processed_data = []
        for ticker in self.tickers:
            ticker_df = combined_df[combined_df['Ticker'] == ticker].copy()
            if not ticker_df.empty:
                ticker_df = self.preprocessor.preprocess(ticker_df, ticker=ticker)
                processed_data.append(ticker_df)
        
        final_df = pd.concat(processed_data, ignore_index=True)
        print(f"Preprocessed data shape: {final_df.shape}")
        return final_df

    def _create_samples(self) -> List[Dict]:
        """Create sliding window samples from the data."""
        import torch
        
        samples = []
        
        for ticker in self.tickers:
            ticker_data = self.data[self.data['Ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('Date').reset_index(drop=True)
            
            if len(ticker_data) < self.window_size + self.prediction_step:
                print(f"Warning: Insufficient data for {ticker}")
                continue
                
            # Create sliding windows
            for window_start_idx in range(len(ticker_data) - self.window_size - self.prediction_step + 1):
                window_end_idx = window_start_idx + self.window_size
                future_idx = window_end_idx + self.prediction_step - 1
                
                # Ensure future index is valid
                if future_idx >= len(ticker_data):
                    break
                    
                window = ticker_data.iloc[window_start_idx:window_end_idx][['Open', 'High', 'Low', 'Close']].values
                dates = ticker_data.iloc[window_start_idx:window_end_idx]['Date'].values
                future_close = ticker_data.iloc[future_idx]['Close']
                current_close = ticker_data.iloc[window_end_idx - 1]['Close']
                
                # Define target: 1 if future price is higher, 0 otherwise
                target = 1 if future_close > current_close else 0
                
                samples.append({
                    'idx': len(samples),
                    'window': window,
                    'close_series': window[:, 3],  # Only Close prices for spectrogram
                    'ohlc': window,  # OHLC for rule engine
                    'target': target,
                    'ticker': ticker,
                    'date': dates[-1]  # Date of the last day in the window
                })
        
        print(f"Created {len(samples)} samples")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple['torch.Tensor', 'torch.Tensor', 'torch.Tensor']:
        """Get a sample: (spectrogram, raw_ohlc, target)"""
        import torch
        
        sample = self.samples[idx]
        
        # Generate spectrogram
        spectrogram = self.spectrogram_generator.generate(sample['close_series'])
        
        # Raw OHLC data for the rule engine
        raw_ohlc = torch.tensor(sample['ohlc'], dtype=torch.float32)
        
        # Target label
        target = torch.tensor(sample['target'], dtype=torch.long)
        
        return spectrogram, raw_ohlc, target

def create_temporal_splits(dataset: Dataset, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[Subset, Subset, Subset]:
    """Create train/validation/test splits respecting temporal order."""
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    indices = list(range(total_size))
    
    # Temporal splits assume data is sorted by date
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices)
    )

def load_crypto_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Load cryptocurrency data from Yahoo Finance."""
    combined_df = pd.DataFrame()
    
    for ticker in tickers:
        print(f"Loading {ticker} from {start_date} to {end_date}...")
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if not data.empty:
                data.reset_index(inplace=True)
                data['Ticker'] = ticker
                combined_df = pd.concat([combined_df, data], ignore_index=True)
                print(f"✓ Loaded {len(data)} days of data for {ticker}")
            else:
                print(f"✗ No data found for {ticker}")
        except Exception as e:
            print(f"✗ Error loading {ticker}: {e}")
    
    return combined_df