"""
Training Module for SA-ViT
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import argparse
from pathlib import Path
import yaml
import warnings
from typing import Tuple, Dict, List, Any
warnings.filterwarnings("ignore")

# Import our modules
from models import SAViT
from data import CryptoDataset, DataPreprocessor, SpectrogramGenerator, create_temporal_splits
from rules import RuleEngine
from utils import (
    calculate_semantic_vectors, 
    evaluate_model, 
    save_checkpoint, 
    plot_training_curves, 
    save_results,
    set_random_seed,
    setup_matplotlib_for_plotting
)
from research_outputs import ResearchOutputs

class TrainingConfig:
    """Training configuration."""
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
        
        # Training configuration
        self.epochs = config_dict.get('epochs', 15)
        self.batch_size = config_dict.get('batch_size', 32)
        self.learning_rate = config_dict.get('learning_rate', 1e-4)
        self.weight_decay = config_dict.get('weight_decay', 0.01)
        self.patience = config_dict.get('patience', 8)
        self.seed = config_dict.get('seed', 42)
        
        # Paths
        self.checkpoints_dir = Path(config_dict.get('checkpoints_dir', 'checkpoints'))
        self.results_dir = Path(config_dict.get('results_dir', 'results'))
        self.logs_dir = Path(config_dict.get('logs_dir', 'logs'))

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                   optimizer: optim.Optimizer, device: torch.device, 
                   rule_engine: RuleEngine) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    
    for batch_idx, (spectrogram, raw_ohlc, target) in enumerate(pbar):
        spectrogram = spectrogram.to(device)
        raw_ohlc = raw_ohlc.to(device)
        target = target.to(device)
        
        # Calculate semantic vectors using the rule engine
        semantic_vectors_batch = calculate_semantic_vectors(raw_ohlc, rule_engine, device)
        
        optimizer.zero_grad()
        outputs, _ = model(spectrogram, semantic_vectors_batch)
        loss = criterion(outputs, target)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Collect predictions and targets for accuracy calculation
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        
        # Update progress bar
        from sklearn.metrics import accuracy_score
        current_acc = accuracy_score(all_targets, all_preds)
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{current_acc:.4f}'
        })
    
    epoch_loss = running_loss / len(dataloader)
    acc = accuracy_score(all_targets, all_preds)
    
    return epoch_loss, acc

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
               device: torch.device, rule_engine: RuleEngine, config: TrainingConfig) -> Dict:
    """Full training loop with validation."""
    
    # Setup training
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = model.state_dict()
    
    # Create directories
    config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting training loop...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on device: {device}")
    
    for epoch in range(config.epochs):
        # Training
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, rule_engine)
        
        # Validation
        val_results = evaluate_model(model, val_loader, criterion, device, rule_engine)
        val_loss = val_results['loss']
        val_acc = val_results['accuracy']
        val_f1 = val_results['f1_score']
        
        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | "
              f"LR: {current_lr:.2e}")
        
        # Early Stopping Logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            # Save best model checkpoint
            checkpoint_path = config.checkpoints_dir / 'best_model.pth'
            save_checkpoint(model, optimizer, epoch, val_results, checkpoint_path)
            print(f"\t-> Saved best model (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.patience:
            print(f"\nEarly stopping after {epoch+1} epochs")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    print("=" * 60)
    print(f"Training complete. Best Val Acc: {best_val_acc:.4f}")
    
    return history

def main_training(args):
    """Main training function."""
    # Set up matplotlib
    setup_matplotlib_for_plotting()
    
    # Load configuration
    if args.config:
        config_dict = load_config(args.config)
    else:
        # Load project default config if available
        default_conf_path = Path(__file__).resolve().parents[1] / 'config' / 'default.yaml'
        if default_conf_path.exists():
            config_dict = load_config(str(default_conf_path))
        else:
            config_dict = {}
    
    # Override config with command line arguments
    if args.epochs:
        config_dict['epochs'] = args.epochs
    if args.batch_size:
        config_dict['batch_size'] = args.batch_size
    # NOTE: patience is read exclusively from the YAML config (no CLI override)
    
    config = TrainingConfig(config_dict)
    
    # Set random seed
    set_random_seed(config.seed)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize components
    print("Initializing components...")
    
    # Rule engine
    rule_engine = RuleEngine()
    print(f"✓ Rule Engine initialized with patterns: {rule_engine.pattern_names}")
    
    # Data preprocessor
    preprocessor = DataPreprocessor(iqr_multiplier=config.iqr_multiplier)
    print("✓ Data Preprocessor initialized")
    
    # Spectrogram generator
    spectrogram_gen = SpectrogramGenerator(image_size=config.image_size)
    print("✓ Spectrogram Generator initialized")
    
    # Create dataset
    print("Creating dataset...")
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
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create splits
    train_dataset, val_dataset, test_dataset = create_temporal_splits(
        dataset, train_ratio=0.7, val_ratio=0.15
    )
    
    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    
    # Create model
    print("Creating SA-ViT model...")
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
    
    # Test model
    sample_spec = torch.randn(2, 1, config.image_size, config.image_size).to(device)
    sample_semantic = torch.randn(2, config.num_patterns).to(device)
    
    with torch.no_grad():
        outputs, _ = model(sample_spec, sample_semantic)
    
    print(f"✓ Model created successfully")
    print(f"Model output shape: {outputs.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train the model
    print("\nStarting SA-ViT training...")
    history = train_model(model, train_loader, val_loader, device, rule_engine, config)
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    criterion = nn.CrossEntropyLoss()
    test_results = evaluate_model(model, test_loader, criterion, device, rule_engine)
    
    print(f"\nTest Set Results:")
    print(f"Test Loss: {test_results['loss']:.4f}")
    print(f"Test Accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy'] * 100:.1f}%)")
    print(f"Test F1 Score: {test_results['f1_score']:.4f}")
    
    # Initialize research outputs system
    research_outputs = ResearchOutputs(config.results_dir, paper_title="SA-ViT")
    
    # Generate comprehensive research outputs
    print("\nGenerating research outputs...")
    
    # 1. Publication-quality training curves
    research_outputs.plot_publication_curves(history, "sa_vit_training_curves")
    
    # 2. Confusion matrices (raw and normalized)
    if 'predictions' in test_results and 'targets' in test_results:
        research_outputs.plot_confusion_matrix_publication(
            test_results['targets'], 
            test_results['predictions'],
            class_names=['DOWN', 'UP'],
            save_name="sa_vit_confusion_matrix"
        )
        
        # 3. Calibration curve
        if 'probabilities' in test_results:
            ece = research_outputs.plot_calibration_curve(
                test_results['targets'],
                test_results['probabilities'],
                save_name="sa_vit_calibration"
            )
            test_results['ece'] = ece
    
    # 4. Performance comparison table (baseline vs enhanced)
    performance_results = {
        'Rule-Based (Uzun et al.)': {
            'accuracy': 0.532,
            'precision': 0.530,
            'recall': 0.535,
            'f1': 0.85,
            'notes': 'High precision, limited context'
        },
        'ViT-Spectrogram (Zeng et al.)': {
            'accuracy': 0.584,
            'precision': 0.580,
            'recall': 0.588,
            'f1': 0.58,
            'notes': 'No semantic priors'
        },
        'SA-ViT (This Work)': {
            'accuracy': test_results['accuracy'],
            'precision': test_results.get('precision', 0.0),
            'recall': test_results.get('recall', 0.0),
            'f1': test_results['f1_score'],
            'notes': 'Cross-attention fusion + fuzzy rules + semantic augmentation'
        }
    }
    
    research_outputs.generate_performance_table(performance_results, "sa_vit_performance_comparison")
    
    # 5. Save comprehensive metrics
    comprehensive_metrics = {
        'model_name': 'SA-ViT',
        'accuracy': test_results['accuracy'],
        'precision': test_results.get('precision', 0.0),
        'recall': test_results.get('recall', 0.0),
        'f1': test_results['f1_score'],
        'ece': test_results.get('ece', 0.0),
        'loss': test_results['loss'],
        'pattern_stats': test_results.get('pattern_stats', {}),
        'classification_report': test_results.get('classification_report', {}),
        'confusion_matrix': test_results.get('confusion_matrix', []),
        'training_history': history,
        'config': {k: str(v) if hasattr(v, '__dict__') else v 
                  for k, v in config.__dict__.items()},
        'dataset_info': {
            'tickers': config.tickers,
            'start_date': config.start_date,
            'end_date': config.end_date,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'test_size': len(test_dataset)
        }
    }
    
    research_outputs.save_comprehensive_metrics(
        'Enhanced_SA_ViT', 
        comprehensive_metrics,
        'sa_vit_comprehensive_results'
    )
    
    # 6. Generate research paper summary
    research_outputs.generate_paper_summary(performance_results)
    
    # 7. Save traditional results (for backward compatibility)
    results = {
        'history': history,
        'test_results': test_results,
        'config': config.__dict__
    }
    
    save_results(results, config.results_dir / 'training_results.json')
    
    # 8. Save model checkpoint with metadata
    checkpoint_metadata = {
        'model_name': 'SA-ViT',
        'test_accuracy': test_results['accuracy'],
        'test_f1': test_results['f1_score'],
        'training_config': {k: str(v) if hasattr(v, '__dict__') else v 
                           for k, v in config.__dict__.items()},
        'dataset_info': {
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'test_size': len(test_dataset)
        }
    }
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'metadata': checkpoint_metadata,
        'test_results': test_results
    }, config.checkpoints_dir / 'sa_vit_model.pth')
    
    print("\n" + "=" * 80)
    print("✓ SA-ViT Training Completed Successfully!")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SA-ViT model")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Training batch size")
    args = parser.parse_args()
    
    main_training(args)