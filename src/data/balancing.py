"""
Data balancing utilities for handling imbalanced datasets.
Includes SMOTE, oversampling, undersampling, and class weight techniques.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, Union
from collections import Counter
import logging

try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler, BorderlineSMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN
    from sklearn.utils.class_weight import compute_class_weight
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataBalancer:
    """Class for handling data balancing techniques."""
    
    def __init__(self, balancing_config: Dict[str, Any]):
        """
        Initialize data balancer.
        
        Args:
            balancing_config: Configuration dictionary for balancing parameters
        """
        self.config = balancing_config
        self.method = balancing_config.get('balancing_method', 'smote')
        self.enabled = balancing_config.get('enable_balancing', True)
        
        advanced_methods = ['smote', 'borderline_smote', 'adasyn', 'smoteenn', 'oversample', 'undersample']
        if not IMBALANCED_LEARN_AVAILABLE and self.method in advanced_methods:
            logger.warning(
                "imbalanced-learn not available. Install with: pip install imbalanced-learn"
            )
            logger.warning("Falling back to class_weight method")
            self.method = 'class_weight'
    
    def balance_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance the dataset using the configured method.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Tuple of (balanced_X, balanced_y)
        """
        if not self.enabled:
            return X, y
        
        logger.info(f"Balancing data using method: {self.method}")
        logger.info(f"Original class distribution: {Counter(y)}")
        
        if self.method == 'smote':
            return self._apply_smote(X, y)
        elif self.method == 'borderline_smote':
            return self._apply_borderline_smote(X, y)
        elif self.method == 'adasyn':
            return self._apply_adasyn(X, y)
        elif self.method == 'smoteenn':
            return self._apply_smoteenn(X, y)
        elif self.method == 'oversample':
            return self._apply_oversample(X, y)
        elif self.method == 'undersample':
            return self._apply_undersample(X, y)
        elif self.method == 'class_weight':
            # For class_weight, we don't modify the data
            return X, y
        else:
            logger.warning(f"Unknown balancing method: {self.method}. Using original data.")
            return X, y
    
    def _apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE oversampling."""
        if not IMBALANCED_LEARN_AVAILABLE:
            return X, y
        
        try:
            smote = SMOTE(
                k_neighbors=self.config.get('smote_k_neighbors', 5),
                random_state=self.config.get('smote_random_state', 42)
            )
            X_balanced, y_balanced = smote.fit_resample(X, y)
            logger.info(f"SMOTE balanced class distribution: {Counter(y_balanced)}")
            return X_balanced, y_balanced
        except Exception as e:
            logger.error(f"SMOTE failed: {e}. Using original data.")
            return X, y
    
    def _apply_oversample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random oversampling."""
        if not IMBALANCED_LEARN_AVAILABLE:
            return X, y
        
        try:
            oversampler = RandomOverSampler(
                random_state=self.config.get('oversample_random_state', 42)
            )
            X_balanced, y_balanced = oversampler.fit_resample(X, y)
            logger.info(f"Oversampled class distribution: {Counter(y_balanced)}")
            return X_balanced, y_balanced
        except Exception as e:
            logger.error(f"Random oversampling failed: {e}. Using original data.")
            return X, y
    
    def _apply_undersample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random undersampling."""
        if not IMBALANCED_LEARN_AVAILABLE:
            return X, y
        
        try:
            undersampler = RandomUnderSampler(
                random_state=self.config.get('undersample_random_state', 42)
            )
            X_balanced, y_balanced = undersampler.fit_resample(X, y)
            logger.info(f"Undersampled class distribution: {Counter(y_balanced)}")
            return X_balanced, y_balanced
        except Exception as e:
            logger.error(f"Random undersampling failed: {e}. Using original data.")
            return X, y
    
    def _apply_borderline_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply BorderlineSMOTE oversampling."""
        if not IMBALANCED_LEARN_AVAILABLE:
            return X, y
        
        try:
            borderline_smote = BorderlineSMOTE(
                k_neighbors=self.config.get('smote_k_neighbors', 5),
                random_state=self.config.get('smote_random_state', 42)
            )
            X_balanced, y_balanced = borderline_smote.fit_resample(X, y)
            logger.info(f"BorderlineSMOTE balanced class distribution: {Counter(y_balanced)}")
            return X_balanced, y_balanced
        except Exception as e:
            logger.error(f"BorderlineSMOTE failed: {e}. Using original data.")
            return X, y
    
    def _apply_adasyn(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply ADASYN oversampling."""
        if not IMBALANCED_LEARN_AVAILABLE:
            return X, y
        
        try:
            adasyn = ADASYN(
                n_neighbors=self.config.get('smote_k_neighbors', 5),
                random_state=self.config.get('smote_random_state', 42)
            )
            X_balanced, y_balanced = adasyn.fit_resample(X, y)
            logger.info(f"ADASYN balanced class distribution: {Counter(y_balanced)}")
            return X_balanced, y_balanced
        except Exception as e:
            logger.error(f"ADASYN failed: {e}. Using original data.")
            return X, y
    
    def _apply_smoteenn(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTEENN (SMOTE + Edited Nearest Neighbours) hybrid method."""
        if not IMBALANCED_LEARN_AVAILABLE:
            return X, y
        
        try:
            smoteenn = SMOTEENN(
                smote=SMOTE(
                    k_neighbors=self.config.get('smote_k_neighbors', 5),
                    random_state=self.config.get('smote_random_state', 42)
                ),
                random_state=self.config.get('smote_random_state', 42)
            )
            X_balanced, y_balanced = smoteenn.fit_resample(X, y)
            logger.info(f"SMOTEENN balanced class distribution: {Counter(y_balanced)}")
            return X_balanced, y_balanced
        except Exception as e:
            logger.error(f"SMOTEENN failed: {e}. Using original data.")
            return X, y
    
    def compute_class_weights(self, y: np.ndarray) -> Dict[str, float]:
        """
        Compute class weights for imbalanced datasets.
        
        Args:
            y: Target labels
            
        Returns:
            Dictionary mapping class labels to weights
        """
        if not self.enabled or self.method != 'class_weight':
            return {}
        
        try:
            classes = np.unique(y)
            
            if self.config.get('class_weight_method') == 'balanced':
                weights = compute_class_weight('balanced', classes=classes, y=y)
            elif self.config.get('class_weight_method') == 'balanced_subsample':
                weights = compute_class_weight('balanced_subsample', classes=classes, y=y)
            else:
                # default to balanced
                weights = compute_class_weight('balanced', classes=classes, y=y)
            
            class_weights = {cls: weight for cls, weight in zip(classes, weights)}
            
            logger.info(f"Computed class weights: {class_weights}")
            return class_weights
            
        except Exception as e:
            logger.error(f"Class weight computation failed: {e}")
            return {}
    
    def get_sklearn_class_weight(self, y: np.ndarray) -> Union[str, Dict, None]:
        """
        Get class weight parameter for sklearn models.
        
        Args:
            y: Target labels
            
        Returns:
            Class weight parameter for sklearn models
        """
        if not self.enabled or self.method != 'class_weight':
            return None
        
        class_weight_method = self.config.get('class_weight_method', 'balanced')
        
        if class_weight_method in ['balanced', 'balanced_subsample']:
            return class_weight_method
        else:
            return self.compute_class_weights(y)
    
    def balance_sequence_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance sequence data (for neural networks).
        
        Args:
            X: Sequence features (2D or 3D array)
            y: Target labels
            
        Returns:
            Tuple of (balanced_X, balanced_y)
        """
        if not self.enabled:
            return X, y
        
        # For sequence data, we need to handle the shape carefully
        original_shape = X.shape
        
        # Reshape to 2D for balancing if needed
        if len(original_shape) > 2:
            X_reshaped = X.reshape(X.shape[0], -1)
        else:
            X_reshaped = X
        
        X_balanced, y_balanced = self.balance_data(X_reshaped, y)
        
        # Reshape back to original format
        if len(original_shape) > 2:
            new_samples = X_balanced.shape[0]
            X_balanced = X_balanced.reshape(new_samples, *original_shape[1:])
        
        return X_balanced, y_balanced
    
    def get_balancing_info(self) -> Dict[str, Any]:
        """
        Get information about the balancing configuration.
        
        Returns:
            Dictionary with balancing information
        """
        return {
            'enabled': self.enabled,
            'method': self.method,
            'config': self.config,
            'imbalanced_learn_available': IMBALANCED_LEARN_AVAILABLE
        }


def apply_balancing_to_dataframe(df: pd.DataFrame, target_column: str, 
                                balancing_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply balancing to a pandas DataFrame.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        balancing_config: Balancing configuration
        
    Returns:
        Balanced DataFrame
    """
    balancer = DataBalancer(balancing_config)
    
    if not balancer.enabled:
        return df
    
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values
    
    X_balanced, y_balanced = balancer.balance_data(X, y)
    
    feature_columns = [col for col in df.columns if col != target_column]
    balanced_df = pd.DataFrame(X_balanced, columns=feature_columns)
    balanced_df[target_column] = y_balanced
    
    return balanced_df

