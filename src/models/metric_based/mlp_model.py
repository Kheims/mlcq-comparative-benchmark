"""
Multi-Layer Perceptron (MLP) model for metric-based classification.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Dict, Any, Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

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
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.model_config.get('batch_size', 32),
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
