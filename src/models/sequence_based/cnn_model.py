"""
Convolutional Neural Network (CNN) model for code smell detection.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Tuple, Optional, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CNNDataset(Dataset):
    """Dataset class for CNN model."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            X: Tokenized sequences
            y: Labels
        """
        self.X = torch.LongTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNNNetwork(nn.Module):
    """CNN network for code smell detection."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, filter_sizes: List[int],
                 num_filters: int, num_classes: int, dropout_rate: float = 0.5,
                 max_seq_len: int = 512):
        """
        Initialize CNN network.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            filter_sizes: List of filter sizes
            num_filters: Number of filters per size
            num_classes: Number of output classes
            dropout_rate: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super(CNNNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.num_classes = num_classes
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
       
        # TODO 
        # add 2d handling 
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=filter_size)
            for filter_size in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input sequences [batch_size, seq_len]
            
        Returns:
            Output logits [batch_size, num_classes]
        """
        # Embedding: [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(x)
        
        # Transpose for conv1d: [batch_size, embedding_dim, seq_len]
        embedded = embedded.transpose(1, 2)
        
        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            # Conv1d: [batch_size, num_filters, new_seq_len]
            conv_out = torch.relu(conv(embedded))
            # Max pooling: [batch_size, num_filters, 1]
            pooled = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pooled)
        
        # Concatenate all conv outputs: [batch_size, len(filter_sizes) * num_filters]
        concatenated = torch.cat(conv_outputs, dim=1)
        
        dropout_out = self.dropout(concatenated)
        
        output = self.fc(dropout_out)
        
        return output


class CNNModel:
    """CNN model for code smell detection."""
    
    def __init__(self, vocab_size: int, model_config: Dict[str, Any]):
        """
        Initialize CNN model.
        
        Args:
            vocab_size: Size of vocabulary
            model_config: Model configuration
        """
        self.vocab_size = vocab_size
        self.model_config = model_config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_trained = False
        self.training_history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    def _create_model(self, num_classes: int):
        """
        Create CNN model.
        
        Args:
            num_classes: Number of output classes
        """
        self.model = CNNNetwork(
            vocab_size=self.vocab_size,
            embedding_dim=self.model_config.get('embedding_dim', 128),
            filter_sizes=self.model_config.get('filter_sizes', [3, 4, 5]),
            num_filters=self.model_config.get('num_filters', 100),
            num_classes=num_classes,
            dropout_rate=self.model_config.get('dropout_rate', 0.5)
        ).to(self.device)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """
        Train the CNN model.
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences (optional)
            y_val: Validation labels (optional)
        """
        num_classes = len(np.unique(y_train))
        self._create_model(num_classes)
        
        train_dataset = CNNDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.model_config.get('batch_size', 32),
            shuffle=True
        )
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = CNNDataset(X_val, y_val)
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
        
        epochs = self.model_config.get('epochs', 50)
        best_val_accuracy = 0.0
        patience = 10
        patience_counter = 0
        
        logger.info(f"Training CNN model for {epochs} epochs...")
        
        for epoch in range(epochs):
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
               
                # early stop 
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                
                if (epoch + 1) % 5 == 0:
                    logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
                              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            else:
                if (epoch + 1) % 5 == 0:
                    logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
        
        self.is_trained = True
        logger.info("CNN model training completed")
    
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
            X: Input sequences
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        predictions = []
        
        dataset = CNNDataset(X, np.zeros(len(X)))  # Dummy labels
        loader = DataLoader(dataset, batch_size=self.model_config.get('batch_size', 32), shuffle=False)
        
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input sequences
            
        Returns:
            Predicted probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        probabilities = []
        
        dataset = CNNDataset(X, np.zeros(len(X))) 
        loader = DataLoader(dataset, batch_size=self.model_config.get('batch_size', 32), shuffle=False)
        
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                probs = torch.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate model accuracy.
        
        Args:
            X: Input sequences
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
            'vocab_size': self.vocab_size,
            'model_config': self.model_config,
            'training_history': self.training_history
        }
        
        torch.save(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, num_classes: int) -> None:
        """
        Load model from file.
        
        Args:
            filepath: Path to load the model from
            num_classes: Number of output classes
        """
        save_dict = torch.load(filepath, map_location=self.device)
        
        self.vocab_size = save_dict['vocab_size']
        self.model_config = save_dict['model_config']
        self.training_history = save_dict['training_history']
        
        self._create_model(num_classes)
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

