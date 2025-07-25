"""
Token processor for handling raw code snippets from MLCQCodeSmellSamples.json.
Processes code snippets for sequence-based and transformer-based models using CodeT5 tokenizer.
"""
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import logging
import re
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenProcessor:
    """Process raw code snippets for sequence and transformer models using CodeT5 tokenizer."""
    
    def __init__(self, json_path: str, max_length: int = 512, tokenizer_name: str = "Salesforce/codet5-base"):
        """
        Initialize the token processor.
        
        Args:
            json_path: Path to MLCQCodeSmellSamples.json file
            max_length: Maximum sequence length for models
            tokenizer_name: HuggingFace tokenizer model name
        """
        self.json_path = Path(json_path)
        self.max_length = max_length
        self.tokenizer_name = tokenizer_name
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info(f"Loaded tokenizer: {tokenizer_name}")
            logger.info(f"Vocabulary size: {len(self.tokenizer.vocab)}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer {tokenizer_name}: {e}")
            raise
        
        self.vocabulary = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        
    def load_data(self) -> List[Dict]:
        """
        Load data from JSON file.
        
        Returns:
            List of sample dictionaries
        """
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded {len(data)} samples from {self.json_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from {self.json_path}: {e}")
            return []
    
    def extract_label(self, sample: Dict) -> str:
        """
        Extract multi-class label from sample.
        
        Args:
            sample: Sample dictionary with 'severity' and 'smell' fields
            
        Returns:
            Multi-class label: 'no_smell', 'blob', 'feature_envy', 'data_class', 'long_method'
        """
        severity = sample.get('severity', 'none').lower()
        
        # If severity is none, it's clean code
        if severity == 'none':
            return 'no_smell'
        
        smell = sample.get('smell', 'unknown').lower().replace(' ', '_')
        
        smell_mapping = {
            'blob': 'blob',
            'feature_envy': 'feature_envy', 
            'data_class': 'data_class',
            'long_method': 'long_method'
        }
        
        return smell_mapping.get(smell, 'unknown')
    
    def clean_code(self, code: str) -> str:
        """
        Clean and normalize code snippet.
        
        Args:
            code: Raw code snippet
            
        Returns:
            Cleaned code snippet
        """
        if not code:
            return ""
        
        code = re.sub(r'\s+', ' ', code)
        
        # Remove comments 
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # Normalize string literals
        code = re.sub(r'"[^"]*"', '"STRING"', code)
        code = re.sub(r"'[^']*'", "'STRING'", code)
        
        # Normalize numeric literals
        code = re.sub(r'\b\d+\b', 'NUMBER', code)
        
        return code.strip()
    
    def tokenize_code(self, code: str) -> List[str]:
        """
        Tokenize code snippet using CodeT5 tokenizer.
        
        Args:
            code: Cleaned code snippet
            
        Returns:
            List of tokens
        """
        if not code:
            return []
        
        # Use CodeT5 tokenizer
        tokens = self.tokenizer.tokenize(code)
        
        # Limit sequence length (reserve space for special tokens)
        if len(tokens) > self.max_length - 2:  # Reserve space for <s> and </s>
            tokens = tokens[:self.max_length - 2]
        
        return tokens
    
    def build_vocabulary(self) -> None:
        """
        Use pre-trained tokenizer vocabulary (no need to build from scratch).
        """
        # Use pre-trained tokenizer vocabulary
        self.vocabulary = self.tokenizer.vocab
        self.word_to_idx = self.tokenizer.vocab
        self.idx_to_word = {idx: token for token, idx in self.tokenizer.vocab.items()}
        
        logger.info(f"Using pre-trained vocabulary with {len(self.word_to_idx)} tokens")
    
    def tokens_to_indices(self, tokens: List[str]) -> List[int]:
        """
        Convert tokens to indices using pre-trained tokenizer.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of token indices
        """
        return self.tokenizer.convert_tokens_to_ids(tokens)
    
    def pad_sequence(self, indices: List[int], max_length: int) -> List[int]:
        """
        Pad sequence to fixed length using tokenizer's pad token.
        
        Args:
            indices: List of token indices
            max_length: Target sequence length
            
        Returns:
            Padded sequence
        """
        pad_idx = self.tokenizer.pad_token_id
        
        if len(indices) >= max_length:
            return indices[:max_length]
        else:
            return indices + [pad_idx] * (max_length - len(indices))
    
    def process_samples(self, samples: List[Dict]) -> pd.DataFrame:
        """
        Process all samples into a DataFrame.
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            DataFrame with processed samples
        """
        processed_samples = []
        
        for sample in samples:
            # Extract information
            code = sample.get('code_snippet', '')
            label = self.extract_label(sample)
            smell_type = sample.get('smell', 'unknown')
            
            # Clean and tokenize code
            cleaned_code = self.clean_code(code)
            tokens = self.tokenize_code(cleaned_code)
            
            processed_sample = {
                'original_code': code,
                'cleaned_code': cleaned_code,
                'tokens': tokens,
                'label': label,
                'smell_type': smell_type,
                'token_count': len(tokens)
            }
            
            processed_samples.append(processed_sample)
        
        df = pd.DataFrame(processed_samples)
        
        logger.info(f"Processed {len(df)} samples")
        logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
        logger.info(f"Smell type distribution: {df['smell_type'].value_counts().to_dict()}")
        logger.info(f"Average token count: {df['token_count'].mean():.2f}")
        
        return df
    
    def prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for sequence-based models using CodeT5 tokenizer.
        
        Args:
            df: DataFrame with processed samples
            
        Returns:
            Tuple of (X, y) where X is padded sequences and y is encoded labels
        """
        self.build_vocabulary()
        
        # Create label mapping
        unique_labels = df['label'].unique()
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Convert to sequences
        sequences = []
        labels = []
        
        for _, row in df.iterrows():
            code = row['cleaned_code']
            if not code:
                continue
                
            # Tokenize and convert to indices directly
            encoding = self.tokenizer.encode_plus(
                code,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='np'
            )
            
            sequences.append(encoding['input_ids'][0])
            labels.append(self.label_to_idx[row['label']])
        
        X = np.array(sequences)
        y = np.array(labels)
        
        logger.info(f"Prepared sequences: X.shape={X.shape}, y.shape={y.shape}")
        logger.info(f"Label mapping: {self.label_to_idx}")
        
        return X, y
    
    def prepare_text_data(self, df: pd.DataFrame) -> Tuple[List[str], List[int]]:
        """
        Prepare text data for transformer models.
        
        Args:
            df: DataFrame with processed samples
            
        Returns:
            Tuple of (texts, labels) for transformer models
        """
        # Use the same label mapping as sequences
        if not hasattr(self, 'label_to_idx'):
            unique_labels = df['label'].unique()
            self.label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
            self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        texts = []
        labels = []
        
        for _, row in df.iterrows():
            code = row['cleaned_code']
            if not code:
                continue
                
            texts.append(code)
            labels.append(self.label_to_idx[row['label']])
        
        logger.info(f"Prepared text data: {len(texts)} samples")
        logger.info(f"Label mapping: {self.label_to_idx}")
        
        return texts, labels
    
    def split_data(self, X, y, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Split data into train and test sets.
        
        Args:
            X: Features
            y: Labels
            test_size: Fraction of data for testing
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def save_vocabulary(self, output_path: str) -> None:
        """
        Save tokenizer configuration and label mappings.
        
        Args:
            output_path: Path to save vocabulary
        """
        vocab_data = {
            'tokenizer_name': self.tokenizer_name,
            'max_length': self.max_length,
            'vocab_size': len(self.tokenizer.vocab),
            'label_to_idx': getattr(self, 'label_to_idx', {}),
            'idx_to_label': getattr(self, 'idx_to_label', {})
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(vocab_data, f, indent=2)
            logger.info(f"Saved tokenizer config to {output_path}")
        except Exception as e:
            logger.error(f"Error saving vocabulary: {e}")
    
    def load_vocabulary(self, vocab_path: str) -> None:
        """
        Load tokenizer configuration and label mappings.
        
        Args:
            vocab_path: Path to vocabulary file
        """
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            # Load tokenizer if different from current
            tokenizer_name = vocab_data.get('tokenizer_name', self.tokenizer_name)
            if tokenizer_name != self.tokenizer_name:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                self.tokenizer_name = tokenizer_name
            
            # Load label mappings
            self.label_to_idx = vocab_data.get('label_to_idx', {})
            self.idx_to_label = {int(k): v for k, v in vocab_data.get('idx_to_label', {}).items()}
            
            logger.info(f"Loaded tokenizer config from {vocab_path}")
            logger.info(f"Tokenizer: {self.tokenizer_name}, vocab size: {len(self.tokenizer.vocab)}")
        except Exception as e:
            logger.error(f"Error loading vocabulary: {e}")

