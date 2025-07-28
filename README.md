# MLCQ Comparative Benchmark

A comprehensive framework for evaluating different machine learning approaches for code smell detection, featuring advanced data balancing techniques, evolutionary algorithms, and multiple model architectures.

## 🚀 Features

- **Multi-Modal ML Models**: Metric-based, sequence-based, and transformer-based models
- **Advanced Data Balancing**: SMOTE, oversampling, undersampling, and class weights for imbalanced datasets
- **Evolutionary Algorithms**: Genetic Programming and evolutionary feature selection
- **Comprehensive Evaluation**: Standardized metrics, visualizations, and model comparison
- **Configurable Pipeline**: YAML-based configuration system for easy experimentation

##  Supported Models

### Metric-Based Models
- **Random Forest**: Ensemble tree-based classifier
- **Decision Tree**: Rule-based classifier
- **MLP**: Multi-layer perceptron neural network
- **Genetic Programming**: Evolved classification programs

### Sequence-Based Models
- **CNN**: Convolutional neural network for code sequences
- **LSTM**: Long Short-Term Memory for sequential patterns
- **GRU**: Gated Recurrent Unit for code analysis

### Transformer-Based Models
- **CodeBERT**: Pre-trained transformer for code understanding


### Code Smells Detected

The benchmark detects 4 types of code smells:
- **Blob** 
- **Feature Envy**
- **Data Class**
- **Long Method**

## 🏗️ Architecture

```
mlcq-comparative-benchmark/
├── src/
│   ├── config/
│   │   └── config.py                 # Configuration management
│   ├── data/
│   │   ├── balancing.py             # Data balancing utilities
│   │   ├── metrics_processor.py     # Designite metrics processing
│   │   ├── token_processor.py       # Code tokenization
│   ├── models/
│   │   ├── metric_based/
│   │   │   ├── sklearn_models.py    # RF, DT implementations
│   │   │   ├── mlp_model.py         # Neural network
│   │   │   └── evolutionary_algorithms.py # GP, feature selection
│   │   ├── sequence_based/
│   │   │   ├── cnn_model.py         # Convolutional NN
│   │   │   ├── lstm_model.py        # LSTM
│   │   │   └── gru_model.py         # GRU
│   │   ├── transformer_based/
│   │   │   └── codebert_model.py    # CodeBERT
│   ├── evaluation/
│   │   └── metrics.py               # Evaluation metrics
│   └── utils/                       # Utility functions
├── experiments/
│   └── run_benchmark.py             # Main benchmark runner
├── main.py                          # Entry point
```

## 🔧 Installation

### Prerequisites
- Python 3.10+.
- CUDA-compatible GPU (optional, for deep learning models).
- uv package and project manager. 

### Basic Installation
```bash
# Clone the repository
git clone https://github.com/your-username/mlcq-comparative-benchmark.git
cd mlcq-comparative-benchmark

uv python install 3.10
uv venv --python=3.10
source .venv/bin/activate
uv sync
```


### Required Data
- `temp/designite_output/` - Designite analysis results
- `dataset/MLCQCodeSmellSamples.json` - MLCQ dataset

## 🚀 Quick Start

### Basic Usage
```bash
# Run full benchmark with default configuration
python main.py

# Run with custom experiment name
python main.py --experiment-name my_experiment

# Run only specific models
python main.py --models rf dt mlp gp

# Data preparation only
python main.py --data-only
```

### Configuration File
```bash
# Run with custom configuration
python main.py --config config.yaml --experiment-name custom_run
```

### Command Line Options
Available options:
- `--config CONFIG`: Path to configuration YAML file
- `--experiment-name NAME`: Name of the experiment
- `--data-only`: Only prepare data without running models
- `--models MODEL1 MODEL2`: Run specific models only (rf, dt, mlp, gp, cnn, lstm, gru, codebert)
- `--output-dir DIR`: Output directory for results
- `--verbose`: Enable verbose logging

### Model-Specific Examples
```bash
# Traditional machine learning
python main.py --models rf dt mlp gp --experiment-name traditional_ml

# Deep learning models
python main.py --models cnn lstm gru --experiment-name deep_learning

# Transformer models
python main.py --models codebert --experiment-name transformer_models

# Mixed selection
python main.py --models rf gp cnn codebert
```

## 📝 Configuration

### Basic Configuration Structure
```yaml
# config.yaml
data:
  designite_output_path: "temp/designite_output"
  json_data_path: "dataset/MLCQCodeSmellSamples.json"
  test_size: 0.2
  random_state: 42
  
  balancing:
    enable_balancing: true
    balancing_method: "smote"  # smote, oversample, undersample, class_weight
    smote_k_neighbors: 5
    apply_to_validation: false

model:
  # Random Forest
  rf_n_estimators: 100
  rf_max_depth: null
  
  # Genetic Programming
  gp_population_size: 100
  gp_generations: 50
  gp_mutation_rate: 0.1
  

evaluation:
  metrics: ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
  average_method: 'weighted'

experiment_name: "mlcq_benchmark"
output_dir: "experiments"
```

## Data Balancing Techniques

This framework provides several data balancing techniques to address the imbalanced nature of code smell datasets:

### SMOTE (Synthetic Minority Oversampling Technique) - **Default Method**
- **Definition**: Generates synthetic samples for minority classes by interpolating between existing minority samples
- **How it works**: Creates new samples along the line segments connecting k-nearest neighbors of minority class instances
- **Advantages**: Reduces overfitting compared to simple oversampling, creates realistic synthetic samples
- **Configuration**:
```yaml
data:
  balancing:
    enable_balancing: true
    balancing_method: "smote"
    smote_k_neighbors: 5
    smote_random_state: 42
```

### Random Oversampling
- **Definition**: Randomly duplicates samples from minority classes until class distribution is balanced
- **How it works**: Randomly selects and replicates existing minority class samples
- **Advantages**: Simple to implement, preserves original data characteristics
- **Disadvantages**: Can lead to overfitting due to exact duplicates
- **Configuration**:
```yaml
data:
  balancing:
    enable_balancing: true
    balancing_method: "oversample"
    oversample_random_state: 42
```

### Random Undersampling
- **Definition**: Randomly removes samples from majority classes to balance the dataset
- **How it works**: Randomly discards majority class samples until balanced distribution is achieved
- **Advantages**: Reduces dataset size, faster training
- **Disadvantages**: Loss of potentially useful information
- **Configuration**:
```yaml
data:
  balancing:
    enable_balancing: true
    balancing_method: "undersample"
    undersample_random_state: 42
```

### Class Weights
- **Definition**: Assigns higher weights to minority classes during model training
- **How it works**: Modifies the loss function to penalize misclassification of minority classes more heavily
- **Advantages**: No data modification required, works with all algorithms that support sample weights
- **Disadvantages**: May not work as well with some algorithms
- **Configuration**:
```yaml
data:
  balancing:
    enable_balancing: true
    balancing_method: "class_weight"
    class_weight_method: "balanced"
```

## Data Formats

### Metric Data
- **Input**: CSV files from Designite tool
- **Features**: LOC, CC, PC, LCOM, CBO, etc.
- **Format**: One row per code sample with metrics
- **Processing**: Averages metrics per code snippet, extracts labels from folder names

### Token Data
- **Input**: JSON files with code snippets
- **Features**: Tokenized code sequences
- **Format**: `{"code": "...", "label": "blob"}`
- **Processing**: Tokenizes code, builds vocabulary, creates sequences


## 📚 Model Details

### Random Forest
- **Type**: Ensemble method
- **Best for**: Tabular data, feature importance
- **Hyperparameters**: n_estimators, max_depth, min_samples_split
- **Balancing**: Class weights, SMOTE

### Genetic Programming
- **Type**: Evolutionary algorithm
- **Best for**: Feature construction, interpretability
- **Hyperparameters**: population_size, generations, mutation_rate
- **Balancing**: Fitness function weighting


### Metrics-based Models
- **Features**: Averaged Designite metrics (LOC, CC, PC, NOF, etc.)
- **Task**: 5-class classification (4 smells + no smell)
- **Preprocessing**: Feature scaling, label encoding
- **Balancing**: All methods supported

### Sequence-based Models
- **Features**: Tokenized code sequences
- **Task**: classification
- **Architecture**: Embedding → Model → Classification
- **Balancing**: Sequence resampling

### Transformer Models
- **Features**: Raw code text
- **Task**: classification
- **Model**: Pre-trained CodeBERT with classification head
- **Balancing**: Text data resampling



## 📈 Results Structure

After running experiments, results are saved in the following structure:

```
experiments/
└── experiment_name/
    ├── config.yaml                  # Saved configuration
    ├── logs/                        # Training logs
    ├── models/                      # Saved model files
    │   ├── random_forest.joblib
    │   ├── decision_tree.joblib
    │   ├── mlp.pth
    │   ├── genetic_programming.joblib
    ├── plots/                       # Visualization plots
    │   ├── model_comparison_f1.png
    │   ├── model_comparison_accuracy.png
    │   └── confusion_matrices.png
    └── results/                     # Detailed results
        ├── model_comparison.csv     # Summary comparison
        ├── detailed_results.json    # Complete results
        ├── metrics_dataset.csv      # Processed metrics
        ├── token_dataset.csv        # Processed tokens
        └── vocabulary.json          # Token vocabulary
```

### Key Result Files


#### `detailed_results.json`
```json
{
  "Random Forest": {
    "accuracy": 0.85,
    "precision": 0.83,
    "recall": 0.87,
    "f1": 0.85,
    "roc_auc": 0.92,
    "confusion_matrix": [[450, 50], [30, 470]],
    "training_time": 12.5
  }
}
```

### Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value (weighted)
- **Recall**: Sensitivity (weighted)
- **F1-score**: Harmonic mean of precision and recall (weighted)
- **ROC-AUC**: Area under the receiver operating characteristic curve


##  Advanced Features

### Evolutionary Algorithms

#### Genetic Programming
```python
# Example GP configuration
gp_config = {
    'population_size': 100,
    'generations': 50,
    'tournament_size': 3,
    'mutation_rate': 0.1,
    'crossover_rate': 0.8,
    'max_depth': 10
}
```



### Adding New Models
1. Create model file in appropriate directory
2. Implement standard interface (fit, predict, predict_proba)
3. Add configuration parameters
4. Update benchmark runner

### Adding New Balancing Methods
1. Implement in `src/data/balancing.py`
2. Add configuration options
3. Update DataBalancer class

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
- **Email**: [djamel.mesbah@adservio.fr]

## 🙏 Acknowledgments

- MLCQ dataset contributors
- Designite tool developers
- Transformers library team
- scikit-learn contributors
- DEAP library maintainers


## 📚 Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{mlcq_benchmark,
  title={A Survey on Code Smells Detection using Machine Learning Techniques},
  author={Mesbah, Djamel and  El Madhoun, Nour and Al Agha, Khaldoun and Zouaoui, Anis},
  year={2025},
  url={https://github.com/Kheims/mlcq-comparative-benchmark}
}
```

---

## 🔮 Future Work

### Graph Neural Networks (In Progress)
Graph Neural Networks for AST-based code smell detection are currently under development. This will include:
- **GCN**: Graph Convolutional Networks
- **GAT**: Graph Attention Networks  
- **GraphSAGE**: Inductive representation learning

---

*This is the repository containing all the experiments relative to Code Smells Detection as part of the paper: "A Survey on Code Smells Detection using Machine Learning Techniques"*

