#!/usr/bin/env python3
"""
Main script for the MLCQ Code Smell Detection Benchmark.
This script provides a convenient way to run the benchmark with different options.
"""
import sys
import os
import argparse
from pathlib import Path

# Add src and experiments to path(to access the Benchmark Runner) 
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent / 'experiments'))

from run_benchmark import BenchmarkRunner
from src.config.config import ExperimentConfig, get_default_config


def main():
    """Main function to run the MLCQ benchmark."""
    parser = argparse.ArgumentParser(
        description='MLCQ Code Smell Detection Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run full benchmark with default config
  python main.py --experiment-name my_experiment    # Run with custom experiment name
  python main.py --config config.yaml              # Run with custom configuration
  python main.py --data-only                       # Only prepare data
  python main.py --models rf dt mlp                # Run only specific models
        """
    )
    
    parser.add_argument(
        '--config', type=str, 
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--experiment-name', type=str, default='mlcq_benchmark',
        help='Name of the experiment (default: mlcq_benchmark)'
    )
    
    parser.add_argument(
        '--data-only', action='store_true',
        help='Only prepare data without running models'
    )
    
    parser.add_argument(
        '--models', nargs='+', 
        choices=['rf', 'dt', 'mlp', 'gp', 'cnn', 'lstm', 'gru', 'codebert'],
        help='Specific models to run (default: all). gp = Genetic Programming'
    )
    
    parser.add_argument(
        '--output-dir', type=str, default='experiments',
        help='Output directory for results (default: experiments)'
    )
    
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.config:
        if not Path(args.config).exists():
            print(f"Error: Configuration file {args.config} not found")
            sys.exit(1)
        config = ExperimentConfig.load_config(args.config)
    else:
        config = get_default_config()
    
    config.experiment_name = args.experiment_name
    config.output_dir = args.output_dir
    
    if args.verbose:
        config.log_level = 'DEBUG'
    
    try:
        runner = BenchmarkRunner(config)
        
        if args.data_only:
            print("Preparing data only...")
            runner.prepare_data()
            print("Data preparation completed!")
        
        elif args.models:
            print(f"Running selected models: {args.models}")
            runner.prepare_data()
            
            if 'rf' in args.models or 'dt' in args.models or 'mlp' in args.models or 'gp' in args.models:
                runner.run_metric_based_models()
            
            if 'cnn' in args.models or 'lstm' in args.models or 'gru' in args.models:
                runner.run_sequence_based_models()
            
            if 'codebert' in args.models:
                runner.run_transformer_based_models()
            
            runner.generate_results()
            
        else:
            print("Running full benchmark...")
            runner.run_full_benchmark()
        
        print("Benchmark completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
