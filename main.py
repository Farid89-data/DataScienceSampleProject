#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataScienceSampleProject - Main Application Entry Point
Author: Farid N.
Date: February 2025

This module serves as the entry point for the image classification system,
orchestrating the entire workflow from data loading to evaluation and reporting.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from config import get_config
from data_loader import DataLoader
from model_builder import ModelBuilder
from trainer import ModelTrainer
from evaluator import ModelEvaluator
from visualizer import ResultVisualizer
from math_explainer import MathExplainer
from utils import setup_logging, timer


def parse_arguments():
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description='Image Classification System for Computer Vision & Data Science',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Output and configuration options
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save all outputs')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging verbosity level')

    # Model hyperparameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for model training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate for optimizer')

    # Data processing options
    parser.add_argument('--use_data_augmentation', action='store_true',
                        help='Enable data augmentation for training')
    parser.add_argument('--use_subset', action='store_true',
                        help='Use smaller dataset subset for faster execution')
    parser.add_argument('--subset_size', type=int, default=5000,
                        help='Number of samples to use if using subset')

    # Visualization and reporting options
    parser.add_argument('--visualize_model', action='store_true',
                        help='Generate model architecture visualization')
    parser.add_argument('--generate_report', action='store_true',
                        help='Generate comprehensive PDF report of results')

    return parser.parse_args()


@timer
def main():
    """Main execution function that orchestrates the entire workflow."""
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    logger = setup_logging(args.log_level, args.output_dir)
    logger.info("Starting DataScienceSampleProject")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Get configuration
    config = get_config(vars(args))
    logger.info(f"Configuration: {config}")

    try:
        # 1. Data Loading and Preprocessing
        logger.info("Phase 1/6: Data Loading and Analysis")
        data_loader = DataLoader(config)
        data = data_loader.load_data()
        data_loader.analyze_data()

        # 2. Model Building
        logger.info("Phase 2/6: Model Architecture Construction")
        model_builder = ModelBuilder(config, data['input_shape'])
        model = model_builder.build_model()

        # 3. Model Training
        logger.info("Phase 3/6: Model Training")
        trainer = ModelTrainer(config, model, data)
        history = trainer.train()

        # 4. Model Evaluation
        logger.info("Phase 4/6: Model Evaluation")
        evaluator = ModelEvaluator(config, model, data)
        evaluation_results = evaluator.evaluate()

        # 5. Results Visualization
        logger.info("Phase 5/6: Results Visualization")
        visualizer = ResultVisualizer(config, data, history, evaluation_results)
        visualizer.plot_training_history()
        visualizer.plot_confusion_matrix()
        visualizer.visualize_predictions()
        visualizer.visualize_feature_maps()

        # 6. Mathematical Explanations
        logger.info("Phase 6/6: Mathematical Concept Explanations")
        math_explainer = MathExplainer(config, data, model, evaluation_results)
        math_explainer.explain_cnn_mathematics()
        math_explainer.demonstrate_eigenvectors()
        math_explainer.demonstrate_bayes_theorem()

        if config.get('generate_report', False):
            logger.info("Generating comprehensive project report")
            from report_generator import generate_report
            generate_report(config, data, model, history, evaluation_results)

        logger.info(f"Project execution completed successfully. All results saved to {config['output_dir']}")

    except Exception as e:
        logger.error(f"Project execution failed: {str(e)}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())