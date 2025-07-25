"""
Metrics processor for handling Designite output files.
Processes methodMetrics.csv and typeMetrics.csv files to create a unified dataset.
"""
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsProcessor:
    """Process Designite metrics and create unified dataset."""
    
    def __init__(self, designite_output_path: str):
        """
        Initialize the metrics processor.
        
        Args:
            designite_output_path: Path to the designite output directory
        """
        self.designite_output_path = Path(designite_output_path)
        self.smell_mapping = {
            'blob': 'blob',
            'feature_envy': 'feature_envy', 
            'data_class': 'data_class',
            'long_method': 'long_method',
            'NoSmell': 'no_smell'
        }
        
    def extract_smell_label(self, folder_name: str) -> str:
        """
        Extract smell label from folder name.
        
        Args:
            folder_name: Name of the folder containing the metrics
            
        Returns:
            Smell label (blob, feature_envy, data_class, long_method, no_smell)
        """
        folder_lower = folder_name.lower()
        
        for smell_key, smell_label in self.smell_mapping.items():
            if smell_key.lower() in folder_lower:
                return smell_label
                
        # Default to no_smell if no pattern matches
        return 'no_smell'
    
    def read_method_metrics(self, metrics_path: Path) -> pd.DataFrame:
        """
        Read and process method metrics CSV file.
        
        Args:
            metrics_path: Path to methodMetrics.csv file
            
        Returns:
            DataFrame with method metrics
        """
        try:
            df = pd.read_csv(metrics_path)
            
            if df.empty:
                return pd.DataFrame()
                
            expected_cols = ['Project Name', 'Package Name', 'Type Name', 'MethodName', 'LOC', 'CC', 'PC']
            
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns in {metrics_path}: {missing_cols}")
                return pd.DataFrame()
                
            numeric_cols = ['LOC', 'CC', 'PC']
            df_numeric = df[numeric_cols].select_dtypes(include=[np.number])
            
            return df_numeric
            
        except Exception as e:
            logger.error(f"Error reading method metrics from {metrics_path}: {e}")
            return pd.DataFrame()
    
    def read_type_metrics(self, metrics_path: Path) -> pd.DataFrame:
        """
        Read and process type metrics CSV file.
        
        Args:
            metrics_path: Path to typeMetrics.csv file
            
        Returns:
            DataFrame with type metrics
        """
        try:
            df = pd.read_csv(metrics_path)
            
            # Handle empty files
            if df.empty:
                return pd.DataFrame()
                
            # Expected columns: Project Name, Package Name, Type Name, NOF, NOPF, NOM, NOPM, LOC, WMC, NC, DIT, LCOM, FANIN, FANOUT
            expected_cols = ['Project Name', 'Package Name', 'Type Name', 'NOF', 'NOPF', 'NOM', 'NOPM', 'LOC', 'WMC', 'NC', 'DIT', 'LCOM', 'FANIN', 'FANOUT']
            
            # Check if all expected columns exist
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns in {metrics_path}: {missing_cols}")
                return pd.DataFrame()
                
            # Select only numeric columns for averaging
            numeric_cols = ['NOF', 'NOPF', 'NOM', 'NOPM', 'LOC', 'WMC', 'NC', 'DIT', 'LCOM', 'FANIN', 'FANOUT']
            df_numeric = df[numeric_cols].select_dtypes(include=[np.number])
            
            return df_numeric
            
        except Exception as e:
            logger.error(f"Error reading type metrics from {metrics_path}: {e}")
            return pd.DataFrame()
    
    def process_sample_metrics(self, sample_dir: Path) -> Optional[Dict]:
        """
        Process metrics for a single code sample.
        
        Args:
            sample_dir: Directory containing methodMetrics.csv and typeMetrics.csv
            
        Returns:
            Dictionary with averaged metrics and label, or None if processing fails
        """
        method_metrics_path = sample_dir / 'methodMetrics.csv'
        type_metrics_path = sample_dir / 'typeMetrics.csv'
        
        method_df = self.read_method_metrics(method_metrics_path)
        type_df = self.read_type_metrics(type_metrics_path)
        
        smell_label = self.extract_smell_label(sample_dir.name)
        
        result = {'sample_id': sample_dir.name, 'smell_label': smell_label}
        
        if not method_df.empty:
            method_means = method_df.mean()
            for col in method_means.index:
                result[f'method_{col}'] = method_means[col]
        else:
            for col in ['LOC', 'CC', 'PC']:
                result[f'method_{col}'] = 0.0
        
        if not type_df.empty:
            type_means = type_df.mean()
            for col in type_means.index:
                result[f'type_{col}'] = type_means[col]
        else:
            for col in ['NOF', 'NOPF', 'NOM', 'NOPM', 'LOC', 'WMC', 'NC', 'DIT', 'LCOM', 'FANIN', 'FANOUT']:
                result[f'type_{col}'] = 0.0
        
        return result
    
    def process_all_samples(self) -> pd.DataFrame:
        """
        Process all samples in the designite output directory.
        
        Returns:
            DataFrame with all processed samples
        """
        all_samples = []
        
        for batch_dir in self.designite_output_path.glob('batch_*'):
            if not batch_dir.is_dir():
                continue
                
            logger.info(f"Processing batch: {batch_dir.name}")
            
            for sample_dir in batch_dir.iterdir():
                if not sample_dir.is_dir():
                    continue
                    
                sample_result = self.process_sample_metrics(sample_dir)
                if sample_result:
                    all_samples.append(sample_result)
        
        if not all_samples:
            logger.error("No samples processed successfully")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_samples)
        
        logger.info(f"Processed {len(df)} samples")
        logger.info(f"Smell distribution: {df['smell_label'].value_counts().to_dict()}")
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save processed data to CSV file.
        
        Args:
            df: Processed DataFrame
            output_path: Path to save the CSV file
        """
        try:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns (excluding sample_id and smell_label).
        
        Args:
            df: Processed DataFrame
            
        Returns:
            List of feature column names
        """
        return [col for col in df.columns if col not in ['sample_id', 'smell_label']]

