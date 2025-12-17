#!/usr/bin/env python3
"""
SA-ViT: Semantically-Augmented Vision Transformer for Cryptocurrency Prediction

Main entry point for the SA-ViT project.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.train import main_training
from src.predict import main_prediction
from src.validate import main_validation

def create_parser():
    """Create argument parser for command line interface."""
    parser = argparse.ArgumentParser(
        description="SA-ViT: Semantically-Augmented Vision Transformer for Cryptocurrency Prediction"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the SA-ViT model")
    train_parser.add_argument(
        "--config", 
        type=str, 
        default="config/default.yaml",
        help="Path to configuration file"
    )
    train_parser.add_argument(
        "--epochs", 
        type=int, 
        help="Number of training epochs (overrides config)"
    )
    train_parser.add_argument(
        "--batch-size", 
        type=int, 
        help="Training batch size (overrides config)"
    )
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions with trained model")
    predict_parser.add_argument(
        "--model-path", 
        type=str, 
        required=True,
        help="Path to trained model checkpoint"
    )
    predict_parser.add_argument(
        "--ticker", 
        type=str, 
        default="BTC-USD",
        help="Cryptocurrency ticker to predict"
    )
    predict_parser.add_argument(
        "--days", 
        type=int, 
        default=60,
        help="Number of days to use for prediction"
    )
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate model on test data")
    validate_parser.add_argument(
        "--model-path", 
        type=str, 
        required=True,
        help="Path to trained model checkpoint"
    )
    validate_parser.add_argument(
        "--config", 
        type=str, 
        default="config/default.yaml",
        help="Path to configuration file"
    )
    
    return parser

def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    try:
        if args.command == "train":
            main_training(args)
        elif args.command == "predict":
            main_prediction(args)
        elif args.command == "validate":
            main_validation(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()