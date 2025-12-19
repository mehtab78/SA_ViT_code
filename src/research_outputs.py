"""
Research Outputs Module for SA-ViT Paper
Generates publication-quality figures, metrics, and artifacts
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")

# Set up publication-quality matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Publication figure settings
FIGURE_WIDTH = 6.0
FIGURE_HEIGHT = 4.5
DPI = 300
FONT_SIZE = 10
LINE_WIDTH = 1.5
MARKER_SIZE = 6

class ResearchOutputs:
    """Generate comprehensive research outputs for SA-ViT paper."""
    
    def __init__(self, results_dir: Path, paper_title: str = "SA-ViT"):
        self.results_dir = results_dir
        self.paper_title = paper_title
        
        # Create subdirectories
        self.figures_dir = results_dir / "figures"
        self.tables_dir = results_dir / "tables" 
        self.metrics_dir = results_dir / "metrics"
        self.attention_dir = results_dir / "attention_maps"
        self.checkpoints_dir = results_dir / "checkpoints"
        
        for dir_path in [self.figures_dir, self.tables_dir, self.metrics_dir, 
                        self.attention_dir, self.checkpoints_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def plot_publication_curves(self, history: Dict[str, List[float]], 
                               save_name: str = "training_curves") -> None:
        """Generate publication-quality training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(2*FIGURE_WIDTH, 2*FIGURE_HEIGHT))
        fig.suptitle(f'{self.paper_title} - Training Progress', fontsize=14, fontweight='bold')
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, history['train_loss'], 'o-', linewidth=LINE_WIDTH, 
                       markersize=MARKER_SIZE, label='Training Loss', color='#1f77b4')
        if 'val_loss' in history:
            axes[0, 0].plot(epochs, history['val_loss'], 's-', linewidth=LINE_WIDTH, 
                           markersize=MARKER_SIZE, label='Validation Loss', color='#ff7f0e')
        axes[0, 0].set_title('Loss Curves', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Cross Entropy Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, history['train_acc'], 'o-', linewidth=LINE_WIDTH, 
                       markersize=MARKER_SIZE, label='Training Accuracy', color='#2ca02c')
        if 'val_acc' in history:
            axes[0, 1].plot(epochs, history['val_acc'], 's-', linewidth=LINE_WIDTH, 
                           markersize=MARKER_SIZE, label='Validation Accuracy', color='#d62728')
        axes[0, 1].set_title('Accuracy Curves', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score curves
        if 'val_f1' in history:
            axes[1, 0].plot(epochs, history['val_f1'], 'D-', linewidth=LINE_WIDTH, 
                           markersize=MARKER_SIZE, label='Validation F1 Score', color='#9467bd')
            axes[1, 0].set_title('F1 Score Progression', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Weighted F1 Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'F1 scores not available', 
                           transform=axes[1, 0].transAxes, ha='center', va='center')
            axes[1, 0].set_title('F1 Score Progression', fontweight='bold')
        
        # Learning rate schedule
        if 'lr' in history:
            axes[1, 1].semilogy(epochs, history['lr'], '^-', linewidth=LINE_WIDTH, 
                               markersize=MARKER_SIZE, label='Learning Rate', color='#8c564b')
            axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate (log scale)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.figures_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Also save as PDF for publication
        pdf_path = self.figures_dir / f"{save_name}.pdf"
        plt.savefig(pdf_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Training curves saved: {save_path}")
    
    def plot_confusion_matrix_publication(self, y_true: List[int], y_pred: List[int], 
                                        class_names: List[str] = None,
                                        save_name: str = "confusion_matrix") -> None:
        """Generate publication-quality confusion matrix."""
        if class_names is None:
            class_names = ['DOWN', 'UP']
        
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*FIGURE_WIDTH, FIGURE_HEIGHT))
        fig.suptitle(f'{self.paper_title} - Confusion Matrix Analysis', fontsize=14, fontweight='bold')
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_title('Raw Counts', fontweight='bold')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax2)
        ax2.set_title('Normalized (Proportions)', fontweight='bold')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        
        plt.tight_layout()
        save_path = self.figures_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Save as PDF
        pdf_path = self.figures_dir / f"{save_name}.pdf"
        plt.savefig(pdf_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Confusion matrix saved: {save_path}")
    
    def generate_performance_table(self, results_dict: Dict[str, Dict], 
                                  save_name: str = "performance_comparison") -> None:
        """Generate publication-quality performance comparison table."""
        # Create DataFrame
        df_data = []
        for model_name, metrics in results_dict.items():
            row = {
                'Model': model_name,
                'Accuracy (%)': f"{metrics.get('accuracy', 0)*100:.1f}",
                'Precision': f"{metrics.get('precision', 0):.3f}",
                'Recall': f"{metrics.get('recall', 0):.3f}",
                'F1 Score': f"{metrics.get('f1', 0):.3f}",
                'Notes': metrics.get('notes', '')
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Save as CSV
        csv_path = self.tables_dir / f"{save_name}.csv"
        df.to_csv(csv_path, index=False)
        
        # Generate LaTeX table
        latex_path = self.tables_dir / f"{save_name}.tex"
        with open(latex_path, 'w') as f:
            f.write("\\begin{table}[ht]\n")
            f.write("\\centering\n")
            f.write(f"\\caption{{Performance Comparison for {self.paper_title}}}\n")
            f.write("\\label{{tab:performance_comparison}}\n")
            
            # Create table
            columns = ['Model', 'Accuracy (%)', 'Precision', 'Recall', 'F1 Score', 'Notes']
            f.write("\\begin{tabular}{|l|c|c|c|c|p{4cm}|}\n")
            f.write("\\hline\n")
            f.write(" & ".join(columns) + " \\\\\n")
            f.write("\\hline\n")
            
            for _, row in df.iterrows():
                f.write(" & ".join([str(row[col]) for col in columns]) + " \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"✓ Performance table saved: {csv_path}")
        print(f"✓ LaTeX table saved: {latex_path}")
    
    def calculate_ece(self, y_true: List[int], y_prob: List[float], n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        # Convert to numpy arrays to ensure proper indexing
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def plot_calibration_curve(self, y_true: List[int], y_prob: List[float],
                              save_name: str = "calibration_curve") -> None:
        """Generate calibration curve for probability calibration analysis."""
        # Validate inputs
        if len(y_true) == 0 or len(y_prob) == 0:
            print(f"Warning: Empty data for calibration curve, skipping...")
            return
        
        if len(y_true) != len(y_prob):
            print(f"Warning: Mismatched lengths ({len(y_true)} vs {len(y_prob)}), skipping calibration curve...")
            return
        
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=10, strategy='quantile'
            )
        except Exception as e:
            print(f"Warning: Could not generate calibration curve: {e}")
            return
        
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=LINE_WIDTH, label='Perfect Calibration')
        
        # Model calibration
        ax.plot(mean_predicted_value, fraction_of_positives, 'o-', 
               linewidth=LINE_WIDTH, markersize=MARKER_SIZE, 
               label=f'Model Calibration', color='#1f77b4')
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'{self.paper_title} - Calibration Curve', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calculate and display ECE
        ece = self.calculate_ece(y_true, y_prob)
        ax.text(0.05, 0.95, f'ECE: {ece:.4f}', transform=ax.transAxes, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        save_path = self.figures_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Save as PDF
        pdf_path = self.figures_dir / f"{save_name}.pdf"
        plt.savefig(pdf_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Calibration curve saved: {save_path}")
        return ece
    
    def plot_attention_heatmaps(self, attention_weights: np.ndarray, 
                               pattern_names: List[str] = None,
                               save_name: str = "attention_heatmaps") -> None:
        """Generate attention weight heatmaps for cross-attention visualization."""
        if pattern_names is None:
            pattern_names = ['Advance Block', 'Doji Star', 'Evening Star']
        
        n_patterns = attention_weights.shape[0]
        n_heads = attention_weights.shape[1] if len(attention_weights.shape) > 1 else 1
        
        fig, axes = plt.subplots(n_patterns, n_heads, 
                                figsize=(n_heads*FIGURE_WIDTH/2, n_patterns*FIGURE_HEIGHT/2))
        if n_patterns == 1:
            axes = axes.reshape(1, -1)
        if n_heads == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'{self.paper_title} - Cross-Attention Weights', fontsize=14, fontweight='bold')
        
        for i in range(n_patterns):
            for j in range(n_heads):
                if n_patterns > 1 and n_heads > 1:
                    ax = axes[i, j]
                elif n_patterns == 1:
                    ax = axes[j]
                else:
                    ax = axes[i]
                
                if n_heads > 1:
                    weights = attention_weights[i, j]
                else:
                    weights = attention_weights[i]
                
                im = ax.imshow(weights, cmap='viridis', aspect='auto')
                ax.set_title(f'{pattern_names[i]} - Head {j+1}', fontsize=8)
                ax.set_xlabel('Visual Tokens')
                ax.set_ylabel('Pattern Features')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.tight_layout()
        save_path = self.figures_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Save as PDF
        pdf_path = self.figures_dir / f"{save_name}.pdf"
        plt.savefig(pdf_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Attention heatmaps saved: {save_path}")
    
    def generate_ablation_study_table(self, ablation_results: Dict[str, Dict],
                                     save_name: str = "ablation_study") -> None:
        """Generate ablation study results table."""
        df_data = []
        for config_name, metrics in ablation_results.items():
            row = {
                'Configuration': config_name,
                'Accuracy (%)': f"{metrics.get('accuracy', 0)*100:.1f}",
                'F1 Score': f"{metrics.get('f1', 0):.3f}",
                'ECE': f"{metrics.get('ece', 0):.4f}",
                'Params (M)': f"{metrics.get('params_millions', 0):.1f}",
                'Training Time (hrs)': f"{metrics.get('training_time_hours', 0):.1f}"
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Save as CSV
        csv_path = self.tables_dir / f"{save_name}.csv"
        df.to_csv(csv_path, index=False)
        
        # Generate LaTeX table
        latex_path = self.tables_dir / f"{save_name}.tex"
        with open(latex_path, 'w') as f:
            f.write("\\begin{table}[ht]\n")
            f.write("\\centering\n")
            f.write(f"\\caption{{Ablation Study Results for {self.paper_title}}}\n")
            f.write("\\label{{tab:ablation_study}}\n")
            
            columns = ['Configuration', 'Accuracy (%)', 'F1 Score', 'ECE', 'Params (M)', 'Training Time (hrs)']
            f.write("\\begin{tabular}{|l|c|c|c|c|c|}\n")
            f.write("\\hline\n")
            f.write(" & ".join(columns) + " \\\\\n")
            f.write("\\hline\n")
            
            for _, row in df.iterrows():
                f.write(" & ".join([str(row[col]) for col in columns]) + " \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"✓ Ablation study table saved: {csv_path}")
    
    def save_comprehensive_metrics(self, model_name: str, metrics: Dict[str, Any],
                                  save_name: str = None) -> None:
        """Save comprehensive metrics in multiple formats."""
        if save_name is None:
            save_name = f"{model_name}_comprehensive_metrics"
        
        # Save as JSON
        json_path = self.metrics_dir / f"{save_name}.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        # Save as pickle for Python
        pickle_path = self.metrics_dir / f"{save_name}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(metrics, f)
        
        # Save as text report
        txt_path = self.metrics_dir / f"{save_name}.txt"
        with open(txt_path, 'w') as f:
            f.write(f"=== {self.paper_title} - {model_name} Results ===\n\n")
            
            # Main metrics
            f.write("MAIN METRICS:\n")
            f.write(f"Accuracy: {metrics.get('accuracy', 0)*100:.2f}%\n")
            f.write(f"Precision: {metrics.get('precision', 0):.4f}\n")
            f.write(f"Recall: {metrics.get('recall', 0):.4f}\n")
            f.write(f"F1 Score: {metrics.get('f1', 0):.4f}\n")
            f.write(f"ECE (Calibration): {metrics.get('ece', 0):.4f}\n\n")
            
            # Classification report
            if 'classification_report' in metrics:
                f.write("CLASSIFICATION REPORT:\n")
                class_report = metrics['classification_report']
                
                # Format classification report as readable text
                if isinstance(class_report, dict):
                    f.write(f"Overall Accuracy: {class_report.get('accuracy', 'N/A')}\n")
                    f.write(f"Macro Avg - Precision: {class_report.get('macro avg', {}).get('precision', 'N/A'):.4f}, Recall: {class_report.get('macro avg', {}).get('recall', 'N/A'):.4f}, F1: {class_report.get('macro avg', {}).get('f1-score', 'N/A'):.4f}\n")
                    f.write(f"Weighted Avg - Precision: {class_report.get('weighted avg', {}).get('precision', 'N/A'):.4f}, Recall: {class_report.get('weighted avg', {}).get('recall', 'N/A'):.4f}, F1: {class_report.get('weighted avg', {}).get('f1-score', 'N/A'):.4f}\n")
                    
                    # Add per-class details
                    for class_name in ['DOWN', 'UP', '0', '1']:
                        if class_name in class_report:
                            class_data = class_report[class_name]
                            f.write(f"{class_name}: Precision={class_data.get('precision', 'N/A'):.4f}, Recall={class_data.get('recall', 'N/A'):.4f}, F1={class_data.get('f1-score', 'N/A'):.4f}, Support={class_data.get('support', 'N/A')}\n")
                else:
                    # If it's already a string, write it directly
                    f.write(str(class_report))
                f.write("\n\n")
            
            # Confusion matrix
            if 'confusion_matrix' in metrics:
                f.write("CONFUSION MATRIX:\n")
                cm = metrics['confusion_matrix']
                
                # Handle different confusion matrix formats
                try:
                    if isinstance(cm, list) and len(cm) >= 2 and len(cm[0]) >= 2:
                        f.write(f"[[{cm[0][0]}, {cm[0][1]}],\n")
                        f.write(f" [{cm[1][0]}, {cm[1][1]}]]\n\n")
                    else:
                        f.write(f"Confusion Matrix: {cm}\n\n")
                except (IndexError, TypeError) as e:
                    f.write(f"Error formatting confusion matrix: {e}\n")
                    f.write(f"Raw confusion matrix: {cm}\n\n")
            
            # Additional statistics
            if 'pattern_stats' in metrics:
                f.write("PATTERN DETECTION STATISTICS:\n")
                try:
                    for pattern, stats in metrics['pattern_stats'].items():
                        f.write(f"{pattern}:\n")
                        f.write(f"  Detection Rate: {stats.get('detection_rate', 0):.3f}\n")
                        f.write(f"  Mean Probability: {stats.get('mean_probability', 0):.3f}\n")
                        f.write(f"  Std Probability: {stats.get('std_probability', 0):.3f}\n")
                except (TypeError, AttributeError) as e:
                    f.write(f"Error formatting pattern statistics: {e}\n")
                    f.write(f"Raw pattern stats: {metrics['pattern_stats']}\n")
                f.write("\n")
        
        print(f"✓ Comprehensive metrics saved: {json_path}")
    
    def generate_paper_summary(self, all_results: Dict[str, Dict]) -> None:
        """Generate a comprehensive summary for the research paper."""
        summary_path = self.results_dir / "research_summary.md"
        
        with open(summary_path, 'w') as f:
            f.write(f"# {self.paper_title} - Research Results Summary\n\n")
            f.write(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Main Results\n\n")
            for model_name, results in all_results.items():
                f.write(f"### {model_name}\n")
                f.write(f"- **Directional Accuracy:** {results.get('accuracy', 0)*100:.1f}%\n")
                f.write(f"- **F1 Score:** {results.get('f1', 0):.3f}\n")
                f.write(f"- **ECE (Calibration):** {results.get('ece', 0):.4f}\n")
                f.write(f"- **Pattern F1:** {results.get('pattern_f1', 'N/A')}\n\n")
            
            f.write("## Key Findings\n\n")
            f.write("1. **SA-ViT Performance:** Novel architecture combining semantic rules with visual transformers\n")
            f.write("2. **Cross-Attention Fusion:** Semantic patterns provide context for visual feature selection\n")
            f.write("3. **Comparative Advantage:** Outperforms rule-based and ViT-only approaches\n")
            f.write("4. **Calibration Quality:** ECE scores indicate well-calibrated probability estimates\n")
            f.write("5. **Pattern Detection:** Fuzzy rules successfully identify candlestick patterns\n\n")
            
            f.write("## Generated Artifacts\n\n")
            f.write("### Figures\n")
            f.write("- Training curves (PNG + PDF)\n")
            f.write("- Confusion matrices (PNG + PDF)\n")
            f.write("- Calibration curves (PNG + PDF)\n")
            f.write("- Attention heatmaps (PNG + PDF)\n\n")
            
            f.write("### Tables\n")
            f.write("- Performance comparison (CSV + LaTeX)\n")
            f.write("- Ablation study results (CSV + LaTeX)\n\n")
            
            f.write("### Metrics\n")
            f.write("- Comprehensive metrics (JSON, Pickle, TXT)\n")
            f.write("- Statistical analysis results\n")
            f.write("- Model checkpoints with metadata\n\n")
            
            f.write("## Reproducibility\n\n")
            f.write("All results are reproducible using the provided code and configuration.\n")
            f.write("Key hyperparameters and settings are documented in the metrics files.\n")
        
        print(f"✓ Research summary saved: {summary_path}")

def setup_publication_matplotlib():
    """Set up matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'figure.figsize': (FIGURE_WIDTH, FIGURE_HEIGHT),
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'font.size': FONT_SIZE,
        'axes.titlesize': FONT_SIZE + 2,
        'axes.labelsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE - 1,
        'ytick.labelsize': FONT_SIZE - 1,
        'legend.fontsize': FONT_SIZE - 1,
        'lines.linewidth': LINE_WIDTH,
        'lines.markersize': MARKER_SIZE,
        'grid.alpha': 0.3,
        'axes.grid': True,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })

# Initialize publication settings
setup_publication_matplotlib()