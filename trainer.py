"""
Model Training Module

This module handles the training of CNN models with advanced functionality
including callbacks, data augmentation, and proper progress tracking.
"""

import tensorflow as tf
import numpy as np
import json
import os
import logging
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from pathlib import Path

from utils import ensure_dir_exists, save_json, timer

logger = logging.getLogger("ImageClassifier.ModelTrainer")


class ModelTrainer:
    """Class for training CNN models with advanced features."""

    def __init__(self, config, model, data):
        """
        Initialize the model trainer.

        Args:
            config: Configuration dictionary
            model: Keras model to train
            data: Dictionary containing training and validation data
        """
        self.config = config
        self.model = model
        self.data = data
        ensure_dir_exists(os.path.dirname(config['model_save_path']))

    @timer
    def train(self):
        """
        Train the model on the dataset.

        Implements best practices:
        - Learning rate scheduling
        - Early stopping
        - Model checkpointing
        - Optional data augmentation
        - TensorBoard logging

        Returns:
            History object containing training history
        """
        logger.info(f"Starting model training for {self.config['epochs']} epochs")

        # Create callbacks for training
        callbacks = self._create_callbacks()

        # Use data augmentation if specified
        if self.config['use_data_augmentation']:
            history = self._train_with_augmentation(callbacks)
        else:
            history = self._train_without_augmentation(callbacks)

        # Save training history to file
        self._save_history(history)

        logger.info("Model training completed")
        return history

    def _create_callbacks(self):
        """
        Create training callbacks for monitoring and optimization.

        Returns:
            List of Keras callbacks
        """
        callbacks = [
            # Save best model based on validation accuracy
            ModelCheckpoint(
                filepath=self.config['model_save_path'],
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),

            # Reduce learning rate when validation loss plateaus
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config['reduce_lr_factor'],
                patience=self.config['reduce_lr_patience'],
                min_lr=self.config['min_lr'],
                verbose=1
            ),

            # Stop training early if validation loss doesn't improve
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),

            # TensorBoard logging
            TensorBoard(
                log_dir=os.path.join(self.config['logs_dir'], 'tensorboard'),
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            ),

            # Custom callback for epoch-level logging
            self._create_logging_callback()
        ]

        return callbacks

    def _create_logging_callback(self):
        """
        Create a custom callback for detailed epoch-level logging.

        Returns:
            Keras callback for logging
        """

        class LoggingCallback(tf.keras.callbacks.Callback):
            def __init__(self, logger):
                super().__init__()
                self.logger = logger
                self.epoch_start_time = None

            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = time.time()
                self.logger.info(f"Starting epoch {epoch + 1}/{self.params['epochs']}")

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                epoch_time = time.time() - self.epoch_start_time
                log_str = f"Epoch {epoch + 1}/{self.params['epochs']} completed in {epoch_time:.2f}s - "
                log_str += " - ".join(f"{k}: {v:.4f}" for k, v in logs.items())
                self.logger.info(log_str)

        return LoggingCallback(logger)

    def _train_with_augmentation(self, callbacks):
        """
        Train the model with data augmentation.

        Args:
            callbacks: List of Keras callbacks

        Returns:
            History object containing training history
        """
        logger.info("Training with data augmentation enabled")

        # Create data generator with augmentation
        datagen = ImageDataGenerator(
            rotation_range=self.config['rotation_range'],
            width_shift_range=self.config['width_shift_range'],
            height_shift_range=self.config['height_shift_range'],
            horizontal_flip=self.config['horizontal_flip'],
            zoom_range=self.config['zoom_range'],
            validation_split=self.config['validation_split'] if not self.data.get('X_test') else 0
        )

        # Determine training and validation data
        if self.data.get('X_test') and self.data.get('y_test'):
            # If test data is provided, use it for validation
            validation_data = (self.data['X_test'], self.data['y_test'])
            datagen.fit(self.data['X_train'])

            # Create training data generator
            train_generator = datagen.flow(
                self.data['X_train'],
                self.data['y_train'],
                batch_size=self.config['batch_size'],
                shuffle=self.config.get('shuffle_data', True)
            )

            # Train the model
            history = self.model.fit(
                train_generator,
                epochs=self.config['epochs'],
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1
            )
        else:
            # If no test data, use validation split
            datagen.fit(self.data['X_train'])

            # Create training and validation generators
            train_generator = datagen.flow(
                self.data['X_train'],
                self.data['y_train'],
                batch_size=self.config['batch_size'],
                subset='training',
                shuffle=self.config.get('shuffle_data', True)
            )

            validation_generator = datagen.flow(
                self.data['X_train'],
                self.data['y_train'],
                batch_size=self.config['batch_size'],
                subset='validation'
            )

            # Train the model
            history = self.model.fit(
                train_generator,
                epochs=self.config['epochs'],
                validation_data=validation_generator,
                callbacks=callbacks,
                verbose=1
            )

        return history

    def _train_without_augmentation(self, callbacks):
        """
        Train the model without data augmentation.

        Args:
            callbacks: List of Keras callbacks

        Returns:
            History object containing training history
        """
        logger.info("Training without data augmentation")

        # Determine validation data or split
        if self.data.get('X_test') and self.data.get('y_test'):
            # If test data is provided, use it for validation
            validation_data = (self.data['X_test'], self.data['y_test'])
            history = self.model.fit(
                self.data['X_train'],
                self.data['y_train'],
                batch_size=self.config['batch_size'],
                epochs=self.config['epochs'],
                validation_data=validation_data,
                shuffle=self.config.get('shuffle_data', True),
                callbacks=callbacks,
                verbose=1
            )
        else:
            # If no test data, use validation split
            history = self.model.fit(
                self.data['X_train'],
                self.data['y_train'],
                batch_size=self.config['batch_size'],
                epochs=self.config['epochs'],
                validation_split=self.config['validation_split'],
                shuffle=self.config.get('shuffle_data', True),
                callbacks=callbacks,
                verbose=1
            )

        return history

    def _save_history(self, history):
        """
        Save training history to JSON file.

        Args:
            history: History object from model training
        """
        # Convert history object to serializable dictionary
        history_dict = {}
        for key, values in history.history.items():
            # Convert numpy values to Python native types
            history_dict[key] = [float(x) for x in values]

        # Add training metadata
        history_dict['metadata'] = {
            'epochs': self.config['epochs'],
            'batch_size': self.config['batch_size'],
            'learning_rate': float(self.config['learning_rate']),
            'data_augmentation': self.config['use_data_augmentation'],
            'completed_epochs': len(history.history['loss']),
            'training_samples': len(self.data['X_train']),
            'validation_samples': len(self.data.get('X_test', [])) or int(
                len(self.data['X_train']) * self.config['validation_split']),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save history to file
        save_json(history_dict, self.config['history_save_path'])
        logger.info(f"Training history saved to {self.config['history_save_path']}")