"""
Model Evaluation Module

This module handles the evaluation of trained models, computing various
metrics and generating comprehensive performance reports.
"""

import numpy as np
import json
import os
import logging
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, roc_curve, auc
import matplotlib.pyplot as plt
import tensorflow as tf

from utils import ensure_dir_exists, save_text_to_file, save_json, timer

logger = logging.getLogger("ImageClassifier.ModelEvaluator")


class ModelEvaluator:
    """Class for comprehensive model evaluation."""

    def __init__(self, config, model, data):
        """
        Initialize the model evaluator.

        Args:
            config: Configuration dictionary
            model: Trained Keras model
            data: Dictionary containing test data
        """
        self.config = config
        self.model = model
        self.data = data
        self.results_dir = Path(config['results_dir'])
        ensure_dir_exists(self.results_dir)

    @timer
    def evaluate(self):
        """
        Evaluate the model on test data.

        Returns:
            Dictionary containing comprehensive evaluation results
        """
        logger.info("Evaluating model performance...")

        # Check if test data is available
        if not self.data.get('X_test') or not self.data.get('y_test'):
            raise ValueError("Test data not found in the data dictionary")

        # Basic evaluation with Keras
        test_loss, test_acc = self._evaluate_with_keras()

        # Get predictions
        y_pred, y_pred_classes = self._get_predictions()

        # Calculate various metrics
        cm, class_report, precision, recall, f1, support = self._calculate_metrics(y_pred_classes)

        # Calculate ROC and AUC (one-vs-rest for multiclass)
        roc_auc_data = self._calculate_roc_auc(y_pred)

        # Save evaluation results
        evaluation_results = {
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_score': f1.tolist(),
            'support': support.tolist(),
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'roc_auc': roc_auc_data['auc_scores'],
            'y_pred': y_pred.tolist(),
            'y_pred_classes': y_pred_classes.tolist(),
            'y_true': self.data['y_test'].flatten().tolist()
        }

        # Save results to file
        self._save_evaluation_results(evaluation_results, class_report)

        return evaluation_results

    def _evaluate_with_keras(self):
        """
        Evaluate the model using Keras' evaluate method.

        Returns:
            Tuple of (loss, accuracy)
        """
        test_loss, test_acc = self.model.evaluate(
            self.data['X_test'],
            self.data['y_test'],
            verbose=1
        )
        logger.info(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")
        return test_loss, test_acc

    def _get_predictions(self):
        """
        Get model predictions on test data.

        Returns:
            Tuple of (prediction probabilities, predicted classes)
        """
        # Get prediction probabilities
        y_pred = self.model.predict(self.data['X_test'], verbose=1)

        # Get predicted class indices
        y_pred_classes = np.argmax(y_pred, axis=1)

        return y_pred, y_pred_classes

    def _calculate_metrics(self, y_pred_classes):
        """
        Calculate comprehensive classification metrics.

        Args:
            y_pred_classes: Predicted class indices

        Returns:
            Tuple of (confusion matrix, classification report, precision, recall, f1, support)
        """
        # Get true class indices
        y_true = self.data['y_test'].flatten()

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)

        # Get classification report as dictionary
        class_report_dict = classification_report(
            y_true,
            y_pred_classes,
            target_names=self.data['class_names'],
            output_dict=True
        )

        # Get classification report as text
        class_report_text = classification_report(
            y_true,
            y_pred_classes,
            target_names=self.data['class_names']
        )

        # Log classification report
        logger.info(f"Classification Report:\n{class_report_text}")

        # Calculate precision, recall, f1, support
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred_classes,
            average=None,
            labels=range(len(self.data['class_names']))
        )

        # Log summary metrics
        logger.info(f"Average precision: {np.mean(precision):.4f}")
        logger.info(f"Average recall: {np.mean(recall):.4f}")
        logger.info(f"Average F1 score: {np.mean(f1):.4f}")

        return cm, class_report_dict, precision, recall, f1, support

    def _calculate_roc_auc(self, y_pred):
        """
        Calculate ROC curves and AUC scores for each class (one-vs-rest approach).

        Args:
            y_pred: Prediction probabilities from the model

        Returns:
            Dictionary with ROC curve data and AUC scores
        """
        y_true = self.data['y_test'].flatten()
        n_classes = len(self.data['class_names'])

        # Initialize results dictionary
        results = {
            'fpr': {},
            'tpr': {},
            'auc_scores': {},
            'roc_data': {}
        }

        # Calculate ROC and AUC for each class
        for i in range(n_classes):
            # Create binary labels (current class vs rest)
            y_true_binary = (y_true == i).astype(int)
            y_score = y_pred[:, i]

            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)

            # Calculate AUC
            roc_auc = auc(fpr, tpr)

            # Store results
            class_name = self.data['class_names'][i]
            results['fpr'][class_name] = fpr.tolist()
            results['tpr'][class_name] = tpr.tolist()
            results['auc_scores'][class_name] = float(roc_auc)
            results['roc_data'][class_name] = {'thresholds': thresholds.tolist()}

        # Calculate macro-average AUC
        macro_auc = np.mean(list(results['auc_scores'].values()))
        results['auc_scores']['macro_avg'] = float(macro_auc)

        logger.info(f"Macro-average AUC: {macro_auc:.4f}")

        return results

    def _save_evaluation_results(self, evaluation_results, class_report):
        """
        Save evaluation results to files.

        Args:
            evaluation_results: Dictionary with evaluation metrics
            class_report: Classification report as text
        """
        # Save classification report as text
        save_text_to_file(
            class_report,
            self.results_dir / 'classification_report.txt'
        )

        # Save evaluation results as JSON
        save_json(
            evaluation_results,
            self.results_dir / 'evaluation_results.json'
        )

        logger.info(f"Evaluation results saved to {self.results_dir}")