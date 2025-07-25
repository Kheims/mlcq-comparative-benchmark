"""
CodeBERT transformer model for code smell detection.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from typing import Dict, Any, Tuple, Optional, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CodeBERTDataset(Dataset):
    """Dataset class for CodeBERT model."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        """
        Initialize dataset.
        
        Args:
            texts: List of code snippets
            labels: List of labels
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class CodeBERTClassifier(nn.Module):
    """CodeBERT classifier for code smell detection."""
    
    def __init__(self, model_name: str, num_classes: int, dropout_rate: float = 0.1):
        """
        Initialize CodeBERT classifier.
        
        Args:
            model_name: Name of the pre-trained model
            num_classes: Number of output classes
            dropout_rate: Dropout rate
        """
        super(CodeBERTClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Output logits
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.pooler_output
        
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)
        
        return logits


class CodeBERTModel:
    """CodeBERT model for code smell detection."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize CodeBERT model.
        
        Args:
            model_config: Model configuration
        """
        self.model_config = model_config
        self.model_name = model_config.get('model_name', 'microsoft/codebert-base')
        self.max_length = model_config.get('max_length', 512)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_trained = False
        self.training_history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    def _create_model(self, num_classes: int):
        """
        Create CodeBERT model.
        
        Args:
            num_classes: Number of output classes
        """
        self.model = CodeBERTClassifier(
            model_name=self.model_name,
            num_classes=num_classes,
            dropout_rate=0.1
        ).to(self.device)
    
    def train(self, train_texts: List[str], train_labels: List[int],
              val_texts: Optional[List[str]] = None, val_labels: Optional[List[int]] = None) -> None:
        """
        Train the CodeBERT model.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts (optional)
            val_labels: Validation labels (optional)
        """
        num_classes = len(set(train_labels))
        self._create_model(num_classes)
        
        train_dataset = CodeBERTDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.model_config.get('batch_size', 16),
            shuffle=True
        )
        
        val_loader = None
        if val_texts is not None and val_labels is not None:
            val_dataset = CodeBERTDataset(val_texts, val_labels, self.tokenizer, self.max_length)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.model_config.get('batch_size', 16),
                shuffle=False
            )
        
        epochs = self.model_config.get('epochs', 10)
        learning_rate = self.model_config.get('learning_rate', 2e-5)
        warmup_steps = self.model_config.get('warmup_steps', 100)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        best_val_accuracy = 0.0
        patience = 3
        patience_counter = 0
        
        logger.info(f"Training CodeBERT model for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            self.training_history['train_loss'].append(avg_train_loss)
            
            # eval phase
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
                
                logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            else:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
        
        self.is_trained = True
        logger.info("CodeBERT model training completed")
    
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
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = correct / total
        avg_val_loss = val_loss / len(val_loader)
        
        return avg_val_loss, val_accuracy
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            texts: Input texts
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        predictions = []
        
        dummy_labels = [0] * len(texts)
        dataset = CodeBERTDataset(texts, dummy_labels, self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=self.model_config.get('batch_size', 16), shuffle=False)
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            texts: Input texts
            
        Returns:
            Predicted probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        probabilities = []
        
        dummy_labels = [0] * len(texts)
        dataset = CodeBERTDataset(texts, dummy_labels, self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=self.model_config.get('batch_size', 16), shuffle=False)
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)
    
    def evaluate(self, texts: List[str], labels: List[int]) -> float:
        """
        Evaluate model accuracy.
        
        Args:
            texts: Input texts
            labels: True labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(texts)
        accuracy = np.mean(predictions == np.array(labels))
        return accuracy
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        self.model.bert.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)
        
        save_dict = {
            'model_config': self.model_config,
            'classifier_state_dict': self.model.classifier.state_dict(),
            'training_history': self.training_history,
            'num_classes': self.model.num_classes
        }
        
        torch.save(save_dict, Path(filepath) / 'classifier_info.pth')
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load model from file.
        
        Args:
            filepath: Path to load the model from
        """
        filepath = Path(filepath)
        
        save_dict = torch.load(filepath / 'classifier_info.pth', map_location=self.device)
        
        self.model_config = save_dict['model_config']
        self.training_history = save_dict['training_history']
        num_classes = save_dict['num_classes']
        
        self.tokenizer = AutoTokenizer.from_pretrained(filepath)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = CodeBERTClassifier(
            model_name=str(filepath),
            num_classes=num_classes,
            dropout_rate=0.1
        ).to(self.device)
        
        self.model.classifier.load_state_dict(save_dict['classifier_state_dict'])
        
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """
        Get training history.
        
        Returns:
            Dictionary with training history
        """
        return self.training_history

