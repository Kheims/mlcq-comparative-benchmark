"""
Configuration system for the MLCQ benchmark experiments.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import torch


@dataclass
class DataBalancingConfig:
    """Configuration for data balancing techniques."""
    enable_balancing: bool = True
    
    balancing_method: str = 'smote'  # smote, oversample, undersample, class_weight

    smote_k_neighbors: int = 5
    smote_random_state: int = 42
    oversample_random_state: int = 42
    undersample_random_state: int = 42
    class_weight_method: str = 'balanced'  # balanced, balanced_subsample, or dict
    apply_to_validation: bool = False  # Whether to balance validation set


@dataclass
class DataConfig:
    """Configuration for data processing."""
    designite_output_path: str = "temp/designite_output"
    json_data_path: str = "dataset/MLCQCodeSmellSamples.json"
    results_path: str = "results"
    max_sequence_length: int = 512
    min_vocab_freq: int = 2
    test_size: float = 0.2
    random_state: int = 42
    balancing: DataBalancingConfig = field(default_factory=DataBalancingConfig)


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    # Random Forest
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: int = 2
    rf_random_state: int = 42
    
    # Decision Tree
    dt_max_depth: Optional[int] = None
    dt_min_samples_split: int = 2
    dt_random_state: int = 42
    
    # MLP
    mlp_hidden_sizes: list = field(default_factory=lambda: [256, 128, 64])
    mlp_dropout_rate: float = 0.3
    mlp_learning_rate: float = 0.001
    mlp_batch_size: int = 32
    mlp_epochs: int = 1
    
    # CNN
    cnn_embedding_dim: int = 128
    cnn_filter_sizes: list = field(default_factory=lambda: [3, 4, 5])
    cnn_num_filters: int = 100
    cnn_dropout_rate: float = 0.5
    cnn_learning_rate: float = 0.001
    cnn_batch_size: int = 32
    cnn_epochs: int = 2
    
    # LSTM
    lstm_embedding_dim: int = 128
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout_rate: float = 0.3
    lstm_learning_rate: float = 0.001
    lstm_batch_size: int = 32
    lstm_epochs: int = 50
    
    # GRU
    gru_embedding_dim: int = 128
    gru_hidden_size: int = 128
    gru_num_layers: int = 2
    gru_dropout_rate: float = 0.3
    gru_learning_rate: float = 0.001
    gru_batch_size: int = 32
    gru_epochs: int = 50
    
    # CodeBERT
    codebert_model_name: str = "microsoft/codebert-base"
    codebert_max_length: int = 512
    codebert_learning_rate: float = 2e-5
    codebert_batch_size: int = 16
    codebert_epochs: int = 10
    codebert_warmup_steps: int = 100
    
    # Genetic Programming
    gp_population_size: int = 100
    gp_generations: int = 50
    gp_tournament_size: int = 3
    gp_mutation_rate: float = 0.1
    gp_crossover_rate: float = 0.8
    gp_max_depth: int = 10
    gp_random_state: int = 42
    gp_n_jobs: int = -1


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    device: str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() \
        else 'cpu'
    num_workers: int = 4
    pin_memory: bool = True
    early_stopping_patience: int = 10
    save_best_model: bool = True
    log_interval: int = 100
    eval_interval: int = 1000
    gradient_clip_norm: float = 1.0


@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters."""
    cv_folds: int = 5
    cv_random_state: int = 42
    metrics: list = field(default_factory=lambda: ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
    average_method: str = 'weighted'  
    save_predictions: bool = True
    save_confusion_matrix: bool = True
    save_roc_curves: bool = True


@dataclass
class ExperimentConfig:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    experiment_name: str = "mlcq_benchmark"
    output_dir: str = "experiments"
    log_level: str = "INFO"
    reproducible: bool = True
    
    def save_config(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__,
            'experiment_name': self.experiment_name,
            'output_dir': self.output_dir,
            'log_level': self.log_level,
            'reproducible': self.reproducible
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @classmethod
    def load_config(cls, config_path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = cls()
        
        if 'data' in config_dict:
            for key, value in config_dict['data'].items():
                if key == 'balancing' and isinstance(value, dict):
                    for bal_key, bal_value in value.items():
                        setattr(config.data.balancing, bal_key, bal_value)
                else:
                    setattr(config.data, key, value)
        
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                setattr(config.model, key, value)
        
        if 'training' in config_dict:
            for key, value in config_dict['training'].items():
                setattr(config.training, key, value)
        
        if 'evaluation' in config_dict:
            for key, value in config_dict['evaluation'].items():
                setattr(config.evaluation, key, value)
        
        main_keys = ['experiment_name', 'output_dir', 'log_level', 'reproducible']
        for key in main_keys:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        return config
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        model_configs = {
            'random_forest': {
                'n_estimators': self.model.rf_n_estimators,
                'max_depth': self.model.rf_max_depth,
                'min_samples_split': self.model.rf_min_samples_split,
                'random_state': self.model.rf_random_state
            },
            'decision_tree': {
                'max_depth': self.model.dt_max_depth,
                'min_samples_split': self.model.dt_min_samples_split,
                'random_state': self.model.dt_random_state
            },
            'mlp': {
                'hidden_sizes': self.model.mlp_hidden_sizes,
                'dropout_rate': self.model.mlp_dropout_rate,
                'learning_rate': self.model.mlp_learning_rate,
                'batch_size': self.model.mlp_batch_size,
                'epochs': self.model.mlp_epochs
            },
            'cnn': {
                'embedding_dim': self.model.cnn_embedding_dim,
                'filter_sizes': self.model.cnn_filter_sizes,
                'num_filters': self.model.cnn_num_filters,
                'dropout_rate': self.model.cnn_dropout_rate,
                'learning_rate': self.model.cnn_learning_rate,
                'batch_size': self.model.cnn_batch_size,
                'epochs': self.model.cnn_epochs
            },
            'lstm': {
                'embedding_dim': self.model.lstm_embedding_dim,
                'hidden_size': self.model.lstm_hidden_size,
                'num_layers': self.model.lstm_num_layers,
                'dropout_rate': self.model.lstm_dropout_rate,
                'learning_rate': self.model.lstm_learning_rate,
                'batch_size': self.model.lstm_batch_size,
                'epochs': self.model.lstm_epochs
            },
            'gru': {
                'embedding_dim': self.model.gru_embedding_dim,
                'hidden_size': self.model.gru_hidden_size,
                'num_layers': self.model.gru_num_layers,
                'dropout_rate': self.model.gru_dropout_rate,
                'learning_rate': self.model.gru_learning_rate,
                'batch_size': self.model.gru_batch_size,
                'epochs': self.model.gru_epochs
            },
            'codebert': {
                'model_name': self.model.codebert_model_name,
                'max_length': self.model.codebert_max_length,
                'learning_rate': self.model.codebert_learning_rate,
                'batch_size': self.model.codebert_batch_size,
                'epochs': self.model.codebert_epochs,
                'warmup_steps': self.model.codebert_warmup_steps
            },
            'genetic_programming': {
                'population_size': self.model.gp_population_size,
                'generations': self.model.gp_generations,
                'tournament_size': self.model.gp_tournament_size,
                'mutation_rate': self.model.gp_mutation_rate,
                'crossover_rate': self.model.gp_crossover_rate,
                'max_depth': self.model.gp_max_depth,
                'random_state': self.model.gp_random_state,
                'n_jobs': self.model.gp_n_jobs
            }
        }
        
        return model_configs.get(model_name, {})


def get_default_config() -> ExperimentConfig:
    """Get default configuration."""
    return ExperimentConfig()


def setup_experiment_dir(config: ExperimentConfig) -> str:
    """Set up experiment directory structure."""
    exp_dir = Path(config.output_dir) / config.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    (exp_dir / "models").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)
    
    return str(exp_dir)

