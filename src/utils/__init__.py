"""
Utility Functions for SA-ViT
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

def setup_matplotlib_for_plotting():
    """Setup matplotlib for plotting with proper configuration."""
    plt.switch_backend("Agg")
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")
    
    # Use commonly available fonts to avoid font warnings
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        plt.rcParams["font.sans-serif"] = [
            "DejaVu Sans",           # Default fallback
            "Arial",                 # Common alternative
            "Liberation Sans",       # Linux alternative
            "sans-serif"             # Generic fallback
        ]
    
    plt.rcParams["axes.unicode_minus"] = False

def calculate_semantic_vectors(raw_ohlc: torch.Tensor, rule_engine, device: torch.device) -> torch.Tensor:
    """Calculate semantic vectors for a batch of OHLC data."""
    batch_size = raw_ohlc.size(0)
    semantic_vectors = []
    
    for i in range(batch_size):
        vec = rule_engine.calculate_pattern_vector(raw_ohlc[i].cpu())
        semantic_vectors.append(vec)
    
    return torch.stack(semantic_vectors).to(device)

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, metrics: Dict, save_path: Path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   checkpoint_path: Path, device: torch.device) -> Dict:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['metrics']

def count_parameters(model: nn.Module) -> int:
    """Count total number of parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_trainable_parameters(model: nn.Module) -> int:
    """Count number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_summary(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict:
    """Get model summary information."""
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Register forward hook to capture layer information
    layer_info = {}
    
    def hook_fn(module, input, output):
        layer_info[module.__class__.__name__] = {
            'output_shape': list(output.shape),
            'num_parameters': count_parameters(module)
        }
    
    hooks = []
    for module in model.modules():
        hook = module.register_forward_hook(hook_fn)
        hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        model(dummy_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return {
        'total_parameters': count_parameters(model),
        'trainable_parameters': count_trainable_parameters(model),
        'layer_info': layer_info
    }

def plot_confusion_matrix(y_true: List[int], y_pred: List[int], 
                         class_names: List[str] = None, 
                         save_path: Optional[Path] = None):
    """Plot confusion matrix."""
    setup_matplotlib_for_plotting()
    
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()

def plot_training_curves(history: Dict[str, List[float]], 
                        save_path: Optional[Path] = None):
    """Plot training and validation curves."""
    setup_matplotlib_for_plotting()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy curves
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
    if 'val_acc' in history:
        axes[0, 1].plot(history['val_acc'], label='Validation Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 Score (if available)
    if 'val_f1' in history:
        axes[1, 0].plot(history['val_f1'], label='Validation F1 Score')
        axes[1, 0].set_title('Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Learning Rate (if available)
    if 'lr' in history:
        axes[1, 1].plot(history['lr'], label='Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()

def evaluate_model(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
                  criterion: nn.Module, device: torch.device, rule_engine) -> Dict:
    """Comprehensive model evaluation with probability outputs for calibration analysis."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probabilities = []
    all_semantic_vectors = []
    
    with torch.no_grad():
        for spectrogram, raw_ohlc, target in dataloader:
            spectrogram = spectrogram.to(device)
            raw_ohlc = raw_ohlc.to(device)
            target = target.to(device)
            
            semantic_vectors_batch = calculate_semantic_vectors(raw_ohlc, rule_engine, device)
            outputs, _ = model(spectrogram, semantic_vectors_batch)
            loss = criterion(outputs, target)
            
            running_loss += loss.item()
            
            # Get predictions and probabilities
            probabilities = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of positive class (UP)
            all_semantic_vectors.extend(semantic_vectors_batch.cpu().numpy())
    
    # Calculate metrics
    epoch_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    precision, recall, _, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    
    # Pattern analysis
    pattern_vectors = np.array(all_semantic_vectors)
    pattern_stats = {}
    for i, pattern_name in enumerate(rule_engine.pattern_names):
        pattern_stats[pattern_name] = {
            'detection_rate': float(np.mean(pattern_vectors[:, i] > 0.5)),
            'mean_probability': float(np.mean(pattern_vectors[:, i])),
            'std_probability': float(np.std(pattern_vectors[:, i]))
        }
    
    # Classification report
    class_report = classification_report(all_targets, all_preds, 
                                       target_names=['DOWN', 'UP'], 
                                       output_dict=True)
    
    return {
        'loss': epoch_loss,
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probabilities,  # For calibration analysis
        'pattern_stats': pattern_stats,
        'classification_report': class_report,
        'confusion_matrix': confusion_matrix(all_targets, all_preds).tolist()
    }

def save_results(results: Dict, save_path: Path):
    """Save results to JSON file."""
    # Convert numpy arrays, torch tensors, and Path objects to JSON-serializable formats
    def convert_to_json_serializable(obj):
        """Recursively convert objects to JSON-serializable formats."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, Path):
            return str(obj)  # Convert Path to string
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    # Convert all non-serializable types to JSON-serializable formats
    results_copy = convert_to_json_serializable(results)
    
    with open(save_path, 'w') as f:
        json.dump(results_copy, f, indent=2)
    
    print(f"Results saved to: {save_path}")

def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False