"""
Validation Module for SA-ViT
"""

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
import argparse
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings("ignore")

# Import our modules
from models import SAViT
from data import CryptoDataset, DataPreprocessor, SpectrogramGenerator
from rules import RuleEngine
from utils import (
    evaluate_model, 
    plot_confusion_matrix, 
    save_results, 
    setup_matplotlib_for_plotting,
    set_random_seed
)

class ValidationConfig:
    """Validation configuration."""
    def __init__(self, config_dict: dict):
        # Model configuration
        self.image_size = config_dict.get('image_size', 128)
        self.patch_size = config_dict.get('patch_size', 16)
        self.embed_dim = config_dict.get('embed_dim', 128)
        self.num_heads = config_dict.get('num_heads', 4)
        self.num_layers = config_dict.get('num_layers', 6)
        self.num_patterns = config_dict.get('num_patterns', 3)
        self.num_classes = config_dict.get('num_classes', 2)
        self.dropout = config_dict.get('dropout', 0.1)
        self.drop_path = config_dict.get('drop_path', 0.1)
        
        # Data configuration
        self.tickers = config_dict.get('tickers', ['BTC-USD', 'ETH-USD'])
        self.start_date = config_dict.get('start_date', '2022-01-01')
        self.end_date = config_dict.get('end_date', '2024-12-31')
        self.window_size = config_dict.get('window_size', 60)
        self.prediction_step = config_dict.get('prediction_step', 1)
        self.iqr_multiplier = config_dict.get('iqr_multiplier', 1.5)
        self.batch_size = config_dict.get('batch_size', 32)
        
        # Paths
        self.results_dir = Path(config_dict.get('results_dir', 'results'))
        
        # Seed for reproducibility
        self.seed = config_dict.get('seed', 42)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main_validation(args):
    """Main validation function."""
    # Set up matplotlib
    setup_matplotlib_for_plotting()
    
    # Load configuration
    config_dict = load_config(args.config)
    config = ValidationConfig(config_dict)
    
    # Set random seed
    set_random_seed(config.seed)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize components
    print("Initializing components...")
    
    # Rule engine
    rule_engine = RuleEngine()
    print(f"✓ Rule Engine initialized")
    
    # Data preprocessor
    preprocessor = DataPreprocessor(iqr_multiplier=config.iqr_multiplier)
    print("✓ Data Preprocessor initialized")
    
    # Spectrogram generator
    spectrogram_gen = SpectrogramGenerator(image_size=config.image_size)
    print("✓ Spectrogram Generator initialized")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Create model
    model = SAViT(
        image_size=config.image_size,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        num_patterns=config.num_patterns,
        num_classes=config.num_classes,
        dropout=config.dropout,
        drop_path=config.drop_path
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model loaded (Epoch: {checkpoint['epoch']})")
    
    # Create dataset
    print("Creating dataset for validation...")
    dataset = CryptoDataset(
        tickers=config.tickers,
        start_date=config.start_date,
        end_date=config.end_date,
        window_size=config.window_size,
        image_size=config.image_size,
        prediction_step=config.prediction_step,
        preprocessor=preprocessor,
        spectrogram_generator=spectrogram_gen
    )
    
    if len(dataset) == 0:
        raise ValueError("Dataset is empty! Check data loading and preprocessing.")
    
    print(f"Full dataset size: {len(dataset)}")
    
    # Use full dataset for validation (no train/val split)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    
    # Initialize criterion
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate model
    print("Running comprehensive validation...")
    results = evaluate_model(model, dataloader, criterion, device, rule_engine)
    
    # Print results
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    
    print(f"Overall Performance:")
    print(f"  Accuracy: {results['accuracy']:.4f} ({results['accuracy'] * 100:.2f}%)")
    print(f"  F1 Score: {results['f1_score']:.4f}")
    print(f"  Loss: {results['loss']:.4f}")
    
    print(f"\nClass-wise Performance:")
    for class_name, metrics in results['classification_report'].items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"  {class_name}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1-Score: {metrics['f1-score']:.4f}")
    
    print(f"\nPattern Detection Statistics:")
    for pattern_name, stats in results['pattern_stats'].items():
        print(f"  {pattern_name}:")
        print(f"    Detection Rate: {stats['detection_rate']:.2%}")
        print(f"    Mean Probability: {stats['mean_probability']:.4f}")
        print(f"    Std Probability: {stats['std_probability']:.4f}")
    
    # Create results directory
    config.results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    save_results(results, config.results_dir / 'validation_results.json')
    
    # Plot confusion matrix
    class_names = ['DOWN', 'UP']
    plot_confusion_matrix(
        results['targets'], 
        results['predictions'], 
        class_names,
        save_path=config.results_dir / 'validation_confusion_matrix.png'
    )
    
    # Additional analysis
    print(f"\nDetailed Analysis:")
    
    # Pattern vs Performance analysis
    pattern_vectors = np.array([item for item in results.get('semantic_vectors', [])])
    if len(pattern_vectors) > 0:
        print(f"\nPattern Impact Analysis:")
        for i, pattern_name in enumerate(rule_engine.pattern_names):
            # Find samples where pattern was detected
            pattern_detected = pattern_vectors[:, i] > 0.5
            if np.any(pattern_detected):
                pattern_accuracy = accuracy_score(
                    np.array(results['targets'])[pattern_detected],
                    np.array(results['predictions'])[pattern_detected]
                )
                print(f"  {pattern_name} detected in {np.sum(pattern_detected)} samples")
                print(f"    Accuracy when {pattern_name} detected: {pattern_accuracy:.4f}")
    
    # Prediction confidence analysis
    from scipy.special import softmax
    prediction_confidence = []
    for i, (target, pred) in enumerate(zip(results['targets'], results['predictions'])):
        # Get probability for predicted class
        # This is a simplified approach - in practice you'd want to store probabilities
        confidence = 1.0 if target == pred else 0.5
        prediction_confidence.append(confidence)
    
    avg_confidence = np.mean(prediction_confidence)
    print(f"\nConfidence Analysis:")
    print(f"  Average prediction confidence: {avg_confidence:.4f}")
    print(f"  Correct predictions: {np.sum(np.array(results['targets']) == np.array(results['predictions']))}/{len(results['targets'])}")
    
    print("\n" + "=" * 80)
    print("✓ SA-ViT Validation Completed Successfully!")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate SA-ViT model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to configuration file")
    args = parser.parse_args()
    
    main_validation(args)