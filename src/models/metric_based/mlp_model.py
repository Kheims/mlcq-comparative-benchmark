"""
Multi-Layer Perceptron (MLP) model for metric-based classification.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, Sampler
from typing import Dict, Any, Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
from collections import Counter
import random

logger = logging.getLogger(__name__)


class MLPDataset(Dataset):
    """Dataset class for MLP model."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            X: Features
            y: Labels
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BalancedBatchSampler(Sampler):
    """Balanced batch sampler for handling class imbalance during training."""
    
    def __init__(self, labels: np.ndarray, batch_size: int, random_state: int = 42):
        """
        Initialize balanced batch sampler.
        
        Args:
            labels: Array of labels
            batch_size: Size of each batch
            random_state: Random state for reproducibility
        """
        self.labels = labels
        self.batch_size = batch_size
        self.random_state = random_state
        super().__init__(data_source=None)
        
        # Group indices by class
        self.class_indices = {}
        for idx, label in enumerate(labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        
        self.classes = list(self.class_indices.keys())
        self.n_classes = len(self.classes)
        
        # Set random seed
        random.seed(random_state)
        np.random.seed(random_state)
    
    def __iter__(self):
        """Generate balanced batches."""
        # Calculate samples per class per batch
        samples_per_class = max(1, self.batch_size // self.n_classes)
        
        # Create copies of class indices for sampling
        class_indices_copy = {cls: indices.copy() for cls, indices in self.class_indices.items()}
        
        # Shuffle indices for each class
        for cls in class_indices_copy:
            random.shuffle(class_indices_copy[cls])
        
        batch = []
        class_pointers = {cls: 0 for cls in self.classes}
        
        while True:
            # Check if we have enough samples left
            total_remaining = sum(len(indices) - class_pointers[cls] 
                                for cls, indices in class_indices_copy.items())
            
            if total_remaining < self.batch_size:
                # Yield remaining batch if it's not empty
                if batch:
                    yield batch
                break
            
            # Sample from each class
            current_batch = []
            for cls in self.classes:
                indices = class_indices_copy[cls]
                pointer = class_pointers[cls]
                
                # If we've exhausted this class, reshuffle and reset
                if pointer >= len(indices):
                    random.shuffle(indices)
                    pointer = 0
                    class_pointers[cls] = 0
                
                # Sample up to samples_per_class from this class
                end_idx = min(pointer + samples_per_class, len(indices))
                current_batch.extend(indices[pointer:end_idx])
                class_pointers[cls] = end_idx
            
            # Add to batch
            batch.extend(current_batch)
            
            # Yield batch when it reaches desired size
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
    
    def __len__(self):
        """Estimate number of batches."""
        return len(self.labels) // self.batch_size


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Reference: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, 
                 reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for classes (optional)
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: True labels
            
        Returns:
            Focal loss value
        """
        ce_loss = self.cross_entropy(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MLPNetwork(nn.Module):
    """Multi-Layer Perceptron network."""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], 
                 num_classes: int, dropout_rate: float = 0.3):
        """
        Initialize MLP network.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes
            dropout_rate: Dropout rate
        """
        super(MLPNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class MLPModel:
    """MLP model for metric-based classification."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize MLP model.
        
        Args:
            model_config: Model configuration
        """
        self.model_config = model_config
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_trained = False
        self.training_history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    def _compute_focal_loss_alpha(self, y: np.ndarray) -> torch.Tensor:
        """
        Compute alpha weights for focal loss based on class frequencies.
        
        Args:
            y: Training labels
            
        Returns:
            Alpha tensor for focal loss
        """
        class_counts = Counter(y)
        classes = sorted(class_counts.keys())
        counts = np.array([class_counts[cls] for cls in classes])
        
        # Compute inverse frequency weights
        total_samples = len(y)
        alpha_weights = total_samples / (len(classes) * counts)
        
        # Normalize weights
        alpha_weights = alpha_weights / alpha_weights.sum() * len(classes)
        
        return torch.FloatTensor(alpha_weights)
    
    def _create_model(self, input_size: int, num_classes: int):
        """
        Create MLP model.
        
        Args:
            input_size: Number of input features
            num_classes: Number of output classes
        """
        self.model = MLPNetwork(
            input_size=input_size,
            hidden_sizes=self.model_config.get('hidden_sizes', [256, 128, 64]),
            num_classes=num_classes,
            dropout_rate=self.model_config.get('dropout_rate', 0.3)
        ).to(self.device)
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray = None, 
                       fit_preprocessing: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess features and labels.
        
        Args:
            X: Features
            y: Labels (optional)
            fit_preprocessing: Whether to fit preprocessing
            
        Returns:
            Tuple of (preprocessed_X, preprocessed_y)
        """
        if fit_preprocessing:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        y_encoded = None
        if y is not None:
            if fit_preprocessing:
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = self.label_encoder.transform(y)
        
        return X_scaled, y_encoded
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """
        Train the MLP model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        X_train_scaled, y_train_encoded = self.preprocess_data(X_train, y_train, fit_preprocessing=True)
        
        input_size = X_train_scaled.shape[1]
        num_classes = len(np.unique(y_train_encoded))
        self._create_model(input_size, num_classes)
        
        train_dataset = MLPDataset(X_train_scaled, y_train_encoded)
        
        # Use balanced batch sampling if enabled
        use_balanced_sampling = self.model_config.get('use_balanced_sampling', True)
        batch_size = self.model_config.get('batch_size', 32)
        
        if use_balanced_sampling:
            balanced_sampler = BalancedBatchSampler(
                y_train_encoded, 
                batch_size=batch_size,
                random_state=self.model_config.get('random_state', 42)
            )
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=balanced_sampler
            )
        else:
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size,
                shuffle=True
            )
        
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_scaled, y_val_encoded = self.preprocess_data(X_val, y_val, fit_preprocessing=False)
            val_dataset = MLPDataset(X_val_scaled, y_val_encoded)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.model_config.get('batch_size', 32),
                shuffle=False
            )
        
        # Choose loss function
        use_focal_loss = self.model_config.get('use_focal_loss', False)
        if use_focal_loss:
            alpha_weights = self._compute_focal_loss_alpha(y_train_encoded).to(self.device)
            gamma = self.model_config.get('focal_loss_gamma', 2.0)
            criterion = FocalLoss(alpha=alpha_weights, gamma=gamma)
            logger.info(f"Using Focal Loss with gamma={gamma} and alpha weights")
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.model_config.get('learning_rate', 0.001)
        )
        
        epochs = self.model_config.get('epochs', 100)
        best_val_accuracy = 0.0
        patience = 10
        patience_counter = 0
        
        logger.info(f"Training MLP model for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            self.training_history['train_loss'].append(avg_train_loss)
            
            # Eval phase
            if val_loader is not None:
                val_loss, val_accuracy = self._validate(val_loader, criterion)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_accuracy'].append(val_accuracy)
                
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
                              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
        
        self.is_trained = True
        logger.info("MLP model training completed")
    
    def _validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (validation_loss, validation_accuracy)
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        val_accuracy = correct / total
        avg_val_loss = val_loss / len(val_loader)
        
        return avg_val_loss, val_accuracy
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled, _ = self.preprocess_data(X, fit_preprocessing=False)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_numpy = predicted.cpu().numpy()
        
        y_pred = self.label_encoder.inverse_transform(predicted_numpy)
        
        return y_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features
            
        Returns:
            Predicted probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled, _ = self.preprocess_data(X, fit_preprocessing=False)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            probabilities_numpy = probabilities.cpu().numpy()
        
        return probabilities_numpy
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate model accuracy.
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Accuracy score
        """
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model_config,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'training_history': self.training_history
        }
        
        torch.save(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, input_size: int, num_classes: int) -> None:
        """
        Load model from file.
        
        Args:
            filepath: Path to load the model from
            input_size: Number of input features
            num_classes: Number of output classes
        """
        save_dict = torch.load(filepath, map_location=self.device)
        
        self.model_config = save_dict['model_config']
        self.scaler = save_dict['scaler']
        self.label_encoder = save_dict['label_encoder']
        self.training_history = save_dict['training_history']
        
        self._create_model(input_size, num_classes)
        self.model.load_state_dict(save_dict['model_state_dict'])
        
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """
        Get training history.
        
        Returns:
            Dictionary with training history
        """
        return self.training_history
