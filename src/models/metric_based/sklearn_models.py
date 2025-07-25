"""
Scikit-learn based models for metric-based classification.
Includes Random Forest and Decision Tree models.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import joblib
import logging
from typing import Union

logger = logging.getLogger(__name__)


class MetricBasedModel:
    """Base class for metric-based models."""
    
    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        """
        Initialize metric-based model.
        
        Args:
            model_name: Name of the model
            model_config: Model configuration
        """
        self.model_name = model_name
        self.model_config = model_config
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_trained = False
    
    def _create_model(self, class_weight=None):
        """Create the model instance (to be implemented by subclasses)."""
        raise NotImplementedError
    
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
        
        # encode labels
        y_encoded = None
        if y is not None:
            if fit_preprocessing:
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = self.label_encoder.transform(y)
        
        return X_scaled, y_encoded
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              class_weight: Optional[Union[str, Dict]] = None) -> None:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            class_weight: Class weights for handling imbalanced data (optional)
        """
        X_train_scaled, y_train_encoded = self.preprocess_data(X_train, y_train, fit_preprocessing=True)
        
        self.model = self._create_model(class_weight=class_weight)
        
        logger.info(f"Training {self.model_name} model...")
        if class_weight is not None:
            logger.info(f"Using class weights: {class_weight}")
        self.model.fit(X_train_scaled, y_train_encoded)
        
        self.is_trained = True
        logger.info(f"{self.model_name} model trained successfully")
        
        if X_val is not None and y_val is not None:
            val_accuracy = self.evaluate(X_val, y_val)
            logger.info(f"Validation accuracy: {val_accuracy:.4f}")
    
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
        y_pred_encoded = self.model.predict(X_scaled)
        
        # decode labels
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
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
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate model accuracy.
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Accuracy score
        """
        X_scaled, y_encoded = self.preprocess_data(X, y, fit_preprocessing=False)
        return self.model.score(X_scaled, y_encoded)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance (if available).
        
        Returns:
            Feature importance array or None
        """
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        
        return None
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'model_name': self.model_name,
            'model_config': self.model_config,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load model from file.
        
        Args:
            filepath: Path to load the model from
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.model_name = model_data['model_name']
        self.model_config = model_data['model_config']
        self.feature_names = model_data['feature_names']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")


class RandomForestModel(MetricBasedModel):
    """Random Forest model for metric-based classification."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize Random Forest model.
        
        Args:
            model_config: Model configuration
        """
        super().__init__("Random Forest", model_config)
    
    def _create_model(self, class_weight=None):
        """Create Random Forest model."""
        return RandomForestClassifier(
            n_estimators=self.model_config.get('n_estimators', 100),
            max_depth=self.model_config.get('max_depth', None),
            min_samples_split=self.model_config.get('min_samples_split', 2),
            random_state=self.model_config.get('random_state', 42),
            class_weight=class_weight,
            n_jobs=-1
        )
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                             param_grid: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for search
            
        Returns:
            Best parameters
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        
        X_train_scaled, y_train_encoded = self.preprocess_data(X_train, y_train, fit_preprocessing=True)
        
        base_model = RandomForestClassifier(random_state=self.model_config.get('random_state', 42))
        
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1
        )
        
        logger.info("Performing hyperparameter tuning for Random Forest...")
        grid_search.fit(X_train_scaled, y_train_encoded)
        
        self.model_config.update(grid_search.best_params_)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_


class DecisionTreeModel(MetricBasedModel):
    """Decision Tree model for metric-based classification."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize Decision Tree model.
        
        Args:
            model_config: Model configuration
        """
        super().__init__("Decision Tree", model_config)
    
    def _create_model(self, class_weight=None):
        """Create Decision Tree model."""
        return DecisionTreeClassifier(
            max_depth=self.model_config.get('max_depth', None),
            min_samples_split=self.model_config.get('min_samples_split', 2),
            random_state=self.model_config.get('random_state', 42),
            class_weight=class_weight
        )
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                             param_grid: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for search
            
        Returns:
            Best parameters
        """
        if param_grid is None:
            param_grid = {
                'max_depth': [None, 5, 10, 20, 30],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 5, 10]
            }
        
        X_train_scaled, y_train_encoded = self.preprocess_data(X_train, y_train, fit_preprocessing=True)
        
        base_model = DecisionTreeClassifier(random_state=self.model_config.get('random_state', 42))
        
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1
        )
        
        logger.info("Performing hyperparameter tuning for Decision Tree...")
        grid_search.fit(X_train_scaled, y_train_encoded)
        
        self.model_config.update(grid_search.best_params_)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_

