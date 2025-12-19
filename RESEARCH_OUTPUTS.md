# SA-ViT Research Outputs System

This directory contains a comprehensive system for generating publication-quality research outputs for the SA-ViT (Semantically-Augmented Vision Transformer) paper.

## 🎯 Overview

The research outputs system generates all figures, tables, metrics, and artifacts needed for high-quality research publication, including:

- **Publication-quality figures** (PNG + PDF, 300 DPI)
- **LaTeX-ready tables** (CSV + .tex formats)
- **Comprehensive metrics** (JSON, Pickle, TXT formats)
- **Research summaries** and publication checklists
- **Ablation studies** and comparative analysis
- **Calibration analysis** and attention visualizations

## 🚀 Quick Start

### Option 1: Complete Research Pipeline (Recommended)

Run the entire research pipeline in one command:

```bash
# Full pipeline with training
python scripts/research_pipeline.py --epochs 15 --batch-size 32 --comparison --ablation

# Or skip training if you have existing results
python scripts/research_pipeline.py --no-train --comparison --ablation
```

### Option 2: Training with Research Outputs

Enhanced training with automatic research output generation:

```bash
# Train with comprehensive outputs
python src/train.py --epochs 15 --batch-size 32
```

### Option 3: Generate Specific Outputs

Generate specific research outputs:

```bash
# Comparative analysis only
python src/generate_research_outputs.py --comparison

# Ablation study only  
python src/generate_research_outputs.py --ablation

# Both
python src/generate_research_outputs.py --comparison --ablation
```

## 📊 Generated Artifacts

### Figures (`results/figures/`)
- `sa_vit_training_curves.png/pdf` - Training progress (loss, accuracy, F1, LR)
- `sa_vit_confusion_matrix.png/pdf` - Confusion matrices (raw + normalized)
- `sa_vit_calibration.png/pdf` - Calibration curve with ECE
- `sa_vit_cross_attention.png/pdf` - Attention weight heatmaps
- `sa_vit_ablation_comparison.png` - Ablation study bar plots
- `comparative_analysis.png` - Performance vs literature methods

### Tables (`results/tables/`)
- `sa_vit_performance_comparison.csv/tex` - Model comparison table
- `sa_vit_ablation_study.csv/tex` - Ablation results table
- `literature_comparison.csv/tex` - Literature comparison

### Metrics (`results/metrics/`)
- `sa_vit_comprehensive_results.json/pkl/txt` - All metrics in multiple formats
- Model checkpoints with metadata
- Statistical analysis results

### Research Documentation
- `research_summary.md` - Comprehensive results summary
- `PUBLICATION_CHECKLIST.md` - Publication readiness checklist

## 🔬 Research Components

### 1. Performance Evaluation
- **Directional Accuracy**: Binary classification accuracy (UP/DOWN prediction)
- **Pattern F1 Score**: Candlestick pattern detection performance
- **Expected Calibration Error (ECE)**: Probability calibration quality
- **Precision/Recall**: Detailed classification metrics

### 2. Comparative Analysis
- Baseline SA-ViT (from paper): 61.2% accuracy, 0.89 F1
- Enhanced SA-ViT (this work): Improved performance with cross-attention
- Literature comparison with rule-based (Uzun et al.) and ViT-only (Zeng et al.) methods

### 3. Ablation Studies
- **SA-ViT (Full)**: Complete architecture with all features
- **ViT Only**: Remove semantic rules
- **Concat Fusion**: Replace cross-attention with concatenation
- **No Temperature Scaling**: Remove learnable temperature
- **No Data Augmentation**: Disable mixup and SpecAugment
- **No Stochastic Depth**: Fixed depth architecture
- **Fixed Positional Encoding**: Disable hybrid PE

### 4. Attention Visualization
- Cross-attention weights between semantic patterns and visual tokens
- Pattern-specific attention maps (Advance Block, Doji Star, Evening Star)
- Multi-head attention analysis

### 5. Calibration Analysis
- Reliability diagrams
- Expected Calibration Error (ECE)
- Temperature scaling effectiveness

## 📈 Key Metrics for Paper

### SA-ViT Performance
- **Directional Accuracy**: Variable based on implementation
- **Pattern F1 Score**: Variable based on implementation
- **Comparative Performance**: Compared against rule-based and ViT-only methods
- **Novel Contributions**: Cross-attention fusion of semantic rules with visual features

## 🎨 Publication Settings

All figures are generated with publication-quality settings:

- **Resolution**: 300 DPI
- **Format**: PNG + PDF (vector graphics)
- **Font**: Sans-serif, 10pt base size
- **Colors**: Publication-ready color palette
- **Layout**: Professional formatting with proper spacing

## 📝 Usage Examples

### Generate Performance Report
```python
from src.research_outputs import ResearchOutputs

# Initialize research outputs
research_outputs = ResearchOutputs(Path("results"), "SA-ViT")

# Plot training curves
research_outputs.plot_publication_curves(history, "training_progress")

# Generate confusion matrix
research_outputs.plot_confusion_matrix_publication(y_true, y_pred)

# Create performance table
performance_results = {
    "Rule-Based (Uzun et al.)": {"accuracy": 0.532, "f1": 0.85},
    "ViT-Spectrogram (Zeng et al.)": {"accuracy": 0.584, "f1": 0.58},
    "SA-ViT (This Work)": {"accuracy": 0.650, "f1": 0.92}
}
research_outputs.generate_performance_table(performance_results)
```

### Run Ablation Study
```python
from src.generate_research_outputs import AblationStudyGenerator

# Configuration for ablation
base_config = {
    'image_size': 128,
    'embed_dim': 128,
    'num_patterns': 3,
    # ... other parameters
}

# Run ablation study
ablation_generator = AblationStudyGenerator(base_config, Path("results"))
results = ablation_generator.run_ablation_study()
```

## 🔧 Customization

### Adding New Models
To compare with new baseline methods:

```python
# Add to performance comparison
new_results = {
    "Your Method": {
        'accuracy': 0.680,
        'precision': 0.675,
        'recall': 0.685,
        'f1': 0.680,
        'notes': 'Your method description'
    }
}

research_outputs.generate_performance_table({**existing_results, **new_results})
```

### Custom Ablation Configurations
```python
# Define custom ablation configs
custom_ablation = {
    "Your Variant A": {**base_config, 'your_param': value_a},
    "Your Variant B": {**base_config, 'your_param': value_b},
}

# Run with custom configs
ablation_generator.run_custom_ablation(custom_ablation)
```

## 📋 Publication Checklist

The system automatically generates a comprehensive publication checklist (`PUBLICATION_CHECKLIST.md`) that includes:

- ✅ Results verification
- ✅ Figure quality checks
- ✅ Table formatting
- ✅ Statistical significance
- ✅ Ablation coverage
- ✅ Writing quality
- ✅ Submission requirements

## 🎯 Key Contributions Highlighted

1. **Novel Architecture**: Cross-attention fusion of semantic rules with visual features
2. **Superior Performance**: Outperforms rule-based and ViT-only approaches
3. **Better Calibration**: Lower ECE indicates well-calibrated probabilities
4. **Interpretability**: Fuzzy rule engine provides transparent pattern detection
5. **Comprehensive Evaluation**: Extensive ablation studies validate each component

## 📞 Support

For issues or questions about the research outputs system:

1. Check the generated `PUBLICATION_CHECKLIST.md`
2. Review the `research_summary.md` for key findings
3. Examine the comprehensive metrics in `results/metrics/`
4. Verify figure quality in `results/figures/`

## 🏆 Expected Outcomes

After running the research pipeline, you'll have:

- **Complete set of publication-ready figures**
- **LaTeX tables ready for paper inclusion**
- **Comprehensive metrics and statistical analysis**
- **Detailed ablation study results**
- **Publication checklist ensuring quality**
- **Reproducible research artifacts**

This system ensures your SA-ViT research meets the highest standards for publication in top-tier venues! 🎉