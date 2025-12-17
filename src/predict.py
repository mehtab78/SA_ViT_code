"""
Prediction Module for SA-ViT
"""

import torch
import argparse
from pathlib import Path
import yfinance as yf
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Import our modules
from models import SAViT
from data import DataPreprocessor, SpectrogramGenerator
from rules import RuleEngine
from utils import calculate_semantic_vectors, setup_matplotlib_for_plotting

def predict_single_ticker(model, ticker: str, days: int = 60, device: torch.device = None) -> dict:
    """Make a prediction for a single ticker."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Initialize components
        preprocessor = DataPreprocessor()
        spectrogram_gen = SpectrogramGenerator()
        rule_engine = RuleEngine()
        
        print(f"Fetching data for {ticker}...")
        
        # Fetch data
        hist = yf.download(ticker, period=f"{days + 10}d")  # Get extra for safety
        if hist.empty or len(hist) < days:
            return {'error': f'Insufficient historical data for {ticker}'}
        
        # Prepare data
        latest_data_raw = hist.tail(days)[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
        latest_data = preprocessor.preprocess(latest_data_raw.copy(), ticker=ticker)
        
        # Prepare input tensors
        ohlc_window = latest_data[['Open', 'High', 'Low', 'Close']].values
        close_series = ohlc_window[:, 3]  # Only Close for spectrogram
        
        spectrogram = spectrogram_gen.generate(close_series).unsqueeze(0).to(device)
        raw_ohlc_tensor = torch.tensor(ohlc_window, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Calculate semantic vector
        semantic_vector = calculate_semantic_vectors(raw_ohlc_tensor, rule_engine, device)
        
        # Run model
        model.eval()
        with torch.no_grad():
            outputs, _ = model(spectrogram, semantic_vector)
            probabilities = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
            predicted_class = np.argmax(probabilities)
        
        # Format results
        prediction_map = {0: 'DOWN', 1: 'UP'}
        pattern_vector = rule_engine.calculate_pattern_vector(ohlc_window)
        
        result = {
            'ticker': ticker,
            'prediction': prediction_map[predicted_class],
            'confidence': {
                'down': float(probabilities[0]),
                'up': float(probabilities[1])
            },
            'patterns_detected': {
                name: float(score) 
                for name, score in zip(rule_engine.pattern_names, pattern_vector.numpy())
            },
            'latest_close': float(hist['Close'].iloc[-1]),
            'prediction_probabilities': {
                'DOWN': float(probabilities[0]),
                'UP': float(probabilities[1])
            },
            'technical_indicators': {
                'price_change_1d': float((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]),
                'price_change_7d': float((hist['Close'].iloc[-1] - hist['Close'].iloc[-8]) / hist['Close'].iloc[-8]) if len(hist) >= 8 else None,
                'volatility_7d': float(hist['Close'].iloc[-8:].std()) if len(hist) >= 8 else None
            }
        }
        
        return result
        
    except Exception as e:
        return {'error': str(e)}

def predict_multiple_tickers(model, tickers: list, days: int = 60, device: torch.device = None) -> dict:
    """Make predictions for multiple tickers."""
    results = {}
    
    for ticker in tickers:
        print(f"\nPredicting for {ticker}...")
        result = predict_single_ticker(model, ticker, days, device)
        results[ticker] = result
        
        if 'error' not in result:
            print(f"✓ {ticker}: {result['prediction']} (Confidence: {result['confidence'][result['prediction'].lower()]:.1%})")
            patterns = [name for name, score in result['patterns_detected'].items() if score > 0.5]
            if patterns:
                print(f"  Patterns detected: {', '.join(patterns)}")
            else:
                print(f"  No strong patterns detected")
        else:
            print(f"✗ {ticker}: {result['error']}")
    
    return results

def main_prediction(args):
    """Main prediction function."""
    # Setup matplotlib
    setup_matplotlib_for_plotting()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Create model (using default configuration, you may want to load from config)
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
    print(f"✓ Model loaded (Epoch: {checkpoint['epoch']})")
    
    # Make predictions
    if ',' in args.ticker:
        tickers = [t.strip() for t in args.ticker.split(',')]
    else:
        tickers = [args.ticker]
    
    results = predict_multiple_tickers(model, tickers, args.days, device)
    
    # Save results
    output_path = Path(f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
    import json
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Prediction Results Summary:")
    print("=" * 50)
    
    for ticker, result in results.items():
        if 'error' not in result:
            print(f"{ticker:8}: {result['prediction']:4} "
                  f"(UP: {result['confidence']['up']:.1%} | "
                  f"DOWN: {result['confidence']['down']:.1%})")
            print(f"           Latest Price: ${result['latest_close']:.2f}")
        else:
            print(f"{ticker:8}: ERROR - {result['error']}")
    
    print("=" * 50)
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions with SA-ViT")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--ticker", type=str, default="BTC-USD", help="Cryptocurrency ticker(s)")
    parser.add_argument("--days", type=int, default=60, help="Number of days for prediction")
    args = parser.parse_args()
    
    main_prediction(args)