"""
Configuration Management Module

This module handles all configuration settings for the image classification project,
providing a centralized location for parameter management and validation.
"""

import os
from pathlib import Path


def get_config(args_dict):
    """
    Create and validate comprehensive configuration dictionary from command line arguments.

    Args:
        args_dict: Dictionary of command line arguments

    Returns:
        Dictionary with complete configuration settings
    """
    # Start with the command line arguments
    config = args_dict.copy()

    # Create output subdirectories
    output_dir = Path(config['output_dir'])
    config.update({
        # Output directories
        'output_dir': str(output_dir),
        'models_dir': str(output_dir / 'models'),
        'plots_dir': str(output_dir / 'plots'),
        'results_dir': str(output_dir / 'results'),
        'logs_dir': str(output_dir / 'logs'),
        'explanations_dir': str(output_dir / 'explanations'),

        # Model parameters
        'model_name': 'cifar10_cnn_classifier',
        'num_classes': 10,
        'dropout_rates': [0.2, 0.3, 0.4, 0.5],
        'optimizer': 'adam',
        'loss_function': 'sparse_categorical_crossentropy',
        'metrics': ['accuracy'],

        # Training parameters
        'shuffle_data': True,
        'validation_split': 0.2,
        'early_stopping_patience': 10,
        'reduce_lr_patience': 3,
        'reduce_lr_factor': 0.5,
        'min_lr': 1e-6,

        # Data augmentation parameters
        'rotation_range': 15,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'horizontal_flip': True,
        'zoom_range': 0.1,

        # Paths for saved artifacts
        'model_save_path': str(output_dir / 'models' / 'best_model.h5'),
        'model_architecture_path': str(output_dir / 'models' / 'model_architecture.png'),
        'history_save_path': str(output_dir / 'models' / 'training_history.json'),
    })

    # Ensure all output directories exist
    for dir_key in ['models_dir', 'plots_dir', 'results_dir', 'logs_dir', 'explanations_dir']:
        os.makedirs(config[dir_key], exist_ok=True)

    # Validate configuration
    _validate_config(config)

    return config


def _validate_config(config):
    """
    Validate configuration parameters to ensure they are within acceptable ranges.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If any configuration parameter is invalid
    """
    # Validate numeric parameters
    if config['batch_size'] <= 0:
        raise ValueError(f"Batch size must be positive, got {config['batch_size']}")

    if config['epochs'] <= 0:
        raise ValueError(f"Number of epochs must be positive, got {config['epochs']}")

    if config['learning_rate'] <= 0:
        raise ValueError(f"Learning rate must be positive, got {config['learning_rate']}")

    if config['use_subset'] and config['subset_size'] <= 0:
        raise ValueError(f"Subset size must be positive, got {config['subset_size']}")