"""
Data Loading and Preprocessing Module

This module handles loading, preprocessing, and analyzing the CIFAR-10 dataset
for image classification tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import logging
from pathlib import Path

from utils import ensure_dir_exists, timer, ProgressLogger

logger = logging.getLogger("ImageClassifier.DataLoader")


class DataLoader:
    """Class for loading and preprocessing image data."""

    def __init__(self, config):
        """
        Initialize the data loader with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck']
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.plots_dir = Path(config['plots_dir'])
        ensure_dir_exists(self.plots_dir)

    @timer
    def load_data(self):
        """
        Load and preprocess the CIFAR-10 dataset.

        Returns:
            Dictionary containing train and test datasets
        """
        logger.info("Loading CIFAR-10 dataset...")

        # Load CIFAR-10 dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        # Normalize pixel values to be between 0 and 1
        x_train, x_test = x_train / 255.0, x_test / 255.0

        # If using a smaller subset for demonstration
        if self.config['use_subset']:
            subset_size = self.config['subset_size']
            logger.info(f"Using subset of {subset_size} samples for demonstration")
            x_train = x_train[:subset_size]
            y_train = y_train[:subset_size]
            x_test = x_test[:subset_size // 5]  # 1/5 of training size for testing
            y_test = y_test[:subset_size // 5]

        # Store the data
        self.X_train, self.X_test = x_train, x_test
        self.y_train, self.y_test = y_train, y_test

        logger.info(f"Data loaded - Training: {self.X_train.shape}, Testing: {self.X_test.shape}")

        # Return the data
        return {
            'X_train': self.X_train,
            'y_train': self.y_train,
            'X_test': self.X_test,
            'y_test': self.y_test,
            'class_names': self.class_names,
            'input_shape': self.X_train.shape[1:]
        }

    @timer
    def analyze_data(self):
        """
        Analyze and visualize dataset statistics.
        """
        logger.info("Analyzing data statistics...")

        if self.X_train is None or self.y_train is None:
            logger.error("Data not loaded. Call load_data() first.")
            return

        # Calculate mean and variance for each channel
        channel_means = [np.mean(self.X_train[:, :, :, i]) for i in range(3)]
        channel_vars = [np.var(self.X_train[:, :, :, i]) for i in range(3)]

        logger.info(f"Channel means (RGB): {channel_means}")
        logger.info(f"Channel variances (RGB): {channel_vars}")

        # Analyze class distribution
        self._plot_class_distribution()

        # Visualize sample images
        self._plot_sample_images()

        # Analyze pixel value distribution
        self._plot_pixel_distribution()

        # Analyze image statistics
        self._analyze_image_statistics()

        logger.info("Data analysis completed")

    def _plot_class_distribution(self):
        """Plot the distribution of classes in the training data."""
        class_counts = np.bincount(self.y_train.flatten())

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(self.class_names)), class_counts[:len(self.class_names)])
        plt.xlabel('Class')
        plt.ylabel('Number of samples')
        plt.title('Class Distribution in Training Data')
        plt.xticks(range(len(self.class_names)), self.class_names, rotation=45)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'class_distribution.png')
        plt.close()

        # Log class distribution
        for i, count in enumerate(class_counts[:len(self.class_names)]):
            logger.info(f"Class {self.class_names[i]}: {count} samples")

    def _plot_sample_images(self):
        """Plot a grid of sample images from the training data."""
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.imshow(self.X_train[i])
            plt.title(self.class_names[self.y_train[i][0]])
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'sample_images.png')
        plt.close()

    def _plot_pixel_distribution(self):
        """Plot distribution of pixel values."""
        # Flatten all images to analyze pixel distribution
        pixels = self.X_train.flatten()

        plt.figure(figsize=(10, 6))
        plt.hist(pixels, bins=50, alpha=0.7)
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of Pixel Values in Training Data')
        plt.grid(alpha=0.3)
        plt.savefig(self.plots_dir / 'pixel_distribution.png')
        plt.close()

    def _analyze_image_statistics(self):
        """Analyze and log additional image statistics."""
        # Calculate per-image statistics
        image_means = np.mean(self.X_train, axis=(1, 2, 3))
        image_stds = np.std(self.X_train, axis=(1, 2, 3))

        # Plot image statistics
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(image_means, bins=50, alpha=0.7)
        plt.xlabel('Mean Pixel Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of Image Mean Values')

        plt.subplot(1, 2, 2)
        plt.hist(image_stds, bins=50, alpha=0.7)
        plt.xlabel('Standard Deviation of Pixel Values')
        plt.ylabel('Frequency')
        plt.title('Distribution of Image Standard Deviations')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'image_statistics.png')
        plt.close()

        # Log summary statistics
        logger.info(
            f"Image mean values - min: {np.min(image_means):.4f}, max: {np.max(image_means):.4f}, avg: {np.mean(image_means):.4f}")
        logger.info(
            f"Image std values - min: {np.min(image_stds):.4f}, max: {np.max(image_stds):.4f}, avg: {np.mean(image_stds):.4f}")