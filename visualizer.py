"""
Results Visualization Module

This module handles the visualization of model training results, evaluation metrics,
and model predictions using matplotlib and seaborn.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import os
import logging
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc

from utils import ensure_dir_exists, timer

logger = logging.getLogger("ImageClassifier.ResultVisualizer")


class ResultVisualizer:
    """Class for visualizing model results and performance."""

    def __init__(self, config, data, history, evaluation_results):
        """
        Initialize the result visualizer.

        Args:
            config: Configuration dictionary
            data: Dictionary containing data
            history: Training history
            evaluation_results: Dictionary containing evaluation results
        """
        self.config = config
        self.data = data
        self.history = history
        self.results = evaluation_results

        # Set style for plots
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })

        # Ensure output directories exist
        self.plots_dir = Path(config['plots_dir'])
        ensure_dir_exists(self.plots_dir)

    @timer
    def plot_training_history(self):
        """Plot training and validation accuracy/loss."""
        plt.figure(figsize=(12, 5))

        # Plot training & validation accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], linewidth=2)
        plt.plot(self.history.history['val_accuracy'], linewidth=2)
        plt.title('Model Accuracy', fontweight='bold')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.ylim([0, 1.05])
        plt.legend(['Train', 'Validation'], loc='lower right')
        plt.grid(True, alpha=0.3)

        # Plot training & validation loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], linewidth=2)
        plt.plot(self.history.history['val_loss'], linewidth=2)
        plt.title('Model Loss', fontweight='bold')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("Training history plot saved")

    @timer
    def plot_confusion_matrix(self):
        """Visualize confusion matrix."""
        cm = np.array(self.results['confusion_matrix'])
        class_names = self.data['class_names']

        plt.figure(figsize=(10, 8))

        # Create normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plot confusion matrix with normalized values
        ax = sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            annot_kws={"size": 10}
        )

        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_title('Normalized Confusion Matrix', fontweight='bold', pad=20)

        # Add a colorbar legend
        cbar = ax.collections[0].colorbar
        cbar.set_label('Normalized Frequency')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'confusion_matrix_norm.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Also create a version with raw counts for absolute reference
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            annot_kws={"size": 10}
        )

        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_title('Confusion Matrix (Counts)', fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'confusion_matrix_counts.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("Confusion matrix plots saved")

    @timer
    def visualize_predictions(self, num_images=25):
        """
        Visualize model predictions on test images.

        Args:
            num_images: Number of images to visualize
        """
        plt.figure(figsize=(15, 15))

        # Get a sample of test images
        indices = np.random.choice(
            len(self.data['X_test']),
            min(num_images, len(self.data['X_test'])),
            replace=False
        )

        # Display a grid of images with their predictions
        for i, idx in enumerate(indices):
            plt.subplot(5, 5, i + 1)
            plt.imshow(self.data['X_test'][idx])

            pred_label = self.data['class_names'][self.results['y_pred_classes'][idx]]
            true_label = self.data['class_names'][self.results['y_true'][idx]]

            # Calculate confidence percentage
            confidence = self.results['y_pred'][idx][self.results['y_pred_classes'][idx]] * 100

            # Title in green if correct, red if wrong
            title_color = 'green' if pred_label == true_label else 'red'
            plt.title(
                f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%',
                color=title_color, fontsize=9
            )
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'prediction_results.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("Prediction visualization saved")

    @timer
    def visualize_feature_maps(self, layer_names=None):
        """
        Visualize feature maps from convolutional layers.

        Args:
            layer_names: List of layer names to visualize, or None to auto-select
        """
        # Get convolutional layers if not specified
        if layer_names is None:
            layer_names = [
                              layer.name for layer in self.model.layers
                              if isinstance(layer, tf.keras.layers.Conv2D)
                          ][:3]  # Limit to first 3 conv layers

        # Select a sample image
        img_idx = np.random.randint(0, len(self.data['X_test']))
        img = self.data['X_test'][img_idx]
        img_class = self.data['class_names'][self.data['y_test'][img_idx][0]]

        # Create a model that outputs feature maps
        for layer_name in layer_names:
            try:
                # Create a feature extraction model
                feature_model = tf.keras.Model(
                    inputs=self.model.input,
                    outputs=self.model.get_layer(layer_name).output
                )

                # Get feature maps for the sample image
                img_tensor = np.expand_dims(img, axis=0)
                feature_maps = feature_model.predict(img_tensor)

                # Plot the feature maps
                num_filters = min(16, feature_maps.shape[-1])
                fig, axes = plt.subplots(4, 4, figsize=(12, 12))
                fig.suptitle(f'Feature maps for layer: {layer_name}', fontweight='bold', fontsize=16)

                # Plot the original image in the first position
                axes[0, 0].imshow(img)
                axes[0, 0].set_title(f'Original Image\n(Class: {img_class})')
                axes[0, 0].axis('off')

                # Plot the feature maps
                map_index = 1
                for i in range(4):
                    for j in range(4):
                        if i == 0 and j == 0:
                            continue  # Skip the first position (original image)

                        if map_index < num_filters:
                            axes[i, j].imshow(feature_maps[0, :, :, map_index], cmap='viridis')
                            axes[i, j].set_title(f'Filter {map_index}')
                        axes[i, j].axis('off')
                        map_index += 1

                plt.tight_layout()
                plt.savefig(self.plots_dir / f'feature_maps_{layer_name}.png', dpi=300, bbox_inches='tight')
                plt.close()

                logger.info(f"Feature maps for layer {layer_name} saved")

            except Exception as e:
                logger.warning(f"Could not visualize feature maps for layer {layer_name}: {str(e)}")

    @timer
    def plot_roc_curves(self):
        """Plot ROC curves for all classes."""
        if 'roc_auc' not in self.results:
            logger.warning("ROC AUC data not found in evaluation results")
            return

        plt.figure(figsize=(10, 8))

        # Plot ROC curve for each class
        class_names = self.data['class_names']
        colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

        for i, class_name in enumerate(class_names):
            if class_name in self.results['roc_auc']:
                auc_score = self.results['roc_auc'][class_name]
                plt.plot(
                    self.results['fpr'][class_name],
                    self.results['tpr'][class_name],
                    color=colors[i],
                    linewidth=2,
                    label=f'{class_name} (AUC = {auc_score:.3f})'
                )

        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontweight='bold')
        plt.ylabel('True Positive Rate', fontweight='bold')
        plt.title('Receiver Operating Characteristic (ROC) Curves', fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("ROC curves saved")

    @timer
    def plot_precision_recall_bars(self):
        """Plot precision, recall, and F1 scores for each class as a bar chart."""
        if not all(k in self.results for k in ['precision', 'recall', 'f1_score']):
            logger.warning("Precision-recall data not found in evaluation results")
            return

        precision = np.array(self.results['precision'])
        recall = np.array(self.results['recall'])
        f1 = np.array(self.results['f1_score'])

        # Create plot
        plt.figure(figsize=(12, 6))
        bar_width = 0.25
        index = np.arange(len(self.data['class_names']))

        plt.bar(index, precision, bar_width, label='Precision', color='#3366cc')
        plt.bar(index + bar_width, recall, bar_width, label='Recall', color='#ff9900')
        plt.bar(index + 2 * bar_width, f1, bar_width, label='F1 Score', color='#109618')

        plt.xlabel('Class', fontweight='bold')
        plt.ylabel('Score', fontweight='bold')
        plt.title('Precision, Recall, and F1 Score by Class', fontweight='bold')
        plt.xticks(index + bar_width, self.data['class_names'], rotation=45)
        plt.legend()
        plt.ylim([0, 1.05])
        plt.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'precision_recall_f1.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("Precision-recall-F1 plot saved")