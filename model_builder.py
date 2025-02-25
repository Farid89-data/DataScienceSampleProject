"""
CNN Model Architecture Module

This module handles building and configuring CNN models for image classification.
Implements state-of-the-art architectural patterns with careful documentation.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
import logging
import os
from pathlib import Path

from utils import ensure_dir_exists, timer

logger = logging.getLogger("ImageClassifier.ModelBuilder")


class ModelBuilder:
    """Class for building CNN architecture for image classification."""

    def __init__(self, config, input_shape):
        """
        Initialize the model builder.

        Args:
            config: Configuration dictionary
            input_shape: Shape of input images (height, width, channels)
        """
        self.config = config
        self.input_shape = input_shape
        self.l2_lambda = 1e-4  # L2 regularization factor

    @timer
    def build_model(self):
        """
        Build a convolutional neural network for image classification.

        The architecture follows modern CNN design principles with:
        - Convolutional blocks (conv + batch norm + activation)
        - Residual connections where appropriate
        - Proper regularization (dropout, L2)
        - Global pooling before classification head

        Returns:
            Compiled Keras model
        """
        logger.info(f"Building CNN model for input shape {self.input_shape}")

        # Use Keras Functional API for more flexibility
        inputs = layers.Input(shape=self.input_shape)

        # Initial convolutional layer
        x = self._conv_block(inputs, filters=32, kernel_size=3, strides=1, name='conv1')

        # First residual block
        x = self._residual_block(x, filters=32, name='res1')
        x = layers.MaxPooling2D((2, 2), name='pool1')(x)
        x = layers.Dropout(self.config['dropout_rates'][0], name='drop1')(x)

        # Second residual block
        x = self._residual_block(x, filters=64, name='res2')
        x = layers.MaxPooling2D((2, 2), name='pool2')(x)
        x = layers.Dropout(self.config['dropout_rates'][1], name='drop2')(x)

        # Third residual block
        x = self._residual_block(x, filters=128, name='res3')
        x = layers.MaxPooling2D((2, 2), name='pool3')(x)
        x = layers.Dropout(self.config['dropout_rates'][2], name='drop3')(x)

        # Global pooling and classification head
        x = layers.GlobalAveragePooling2D(name='global_pool')(x)
        x = layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_lambda),
            name='dense1'
        )(x)
        x = layers.BatchNormalization(name='bn_dense')(x)
        x = layers.Dropout(self.config['dropout_rates'][3], name='drop4')(x)

        # Output layer
        outputs = layers.Dense(
            self.config['num_classes'],
            activation='softmax',
            name='output'
        )(x)

        # Create and compile the model
        model = models.Model(inputs=inputs, outputs=outputs, name=self.config['model_name'])

        model.compile(
            optimizer=self._get_optimizer(),
            loss=self.config['loss_function'],
            metrics=self.config['metrics']
        )

        # Print model summary
        model.summary(print_fn=logger.info)

        # Visualize model architecture
        if self.config.get('visualize_model', False):
            self._visualize_model_architecture(model)

        logger.info(f"Model built with {model.count_params():,} trainable parameters")

        return model

    def _conv_block(self, x, filters, kernel_size, strides, name):
        """
        Create a convolutional block with batch normalization and activation.

        Args:
            x: Input tensor
            filters: Number of filters in convolutional layer
            kernel_size: Size of convolutional kernel
            strides: Stride of convolution
            name: Base name for the block components

        Returns:
            Output tensor from the convolutional block
        """
        conv = layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            kernel_regularizer=regularizers.l2(self.l2_lambda),
            name=f'{name}_conv'
        )(x)
        bn = layers.BatchNormalization(name=f'{name}_bn')(conv)
        return layers.Activation('relu', name=f'{name}_relu')(bn)

    def _residual_block(self, x, filters, name):
        """
        Create a residual block with two convolutional layers and a skip connection.

        Args:
            x: Input tensor
            filters: Number of filters in convolutional layers
            name: Base name for the block components

        Returns:
            Output tensor from the residual block
        """
        # First convolutional layer
        conv1 = self._conv_block(x, filters, 3, 1, f'{name}_1')

        # Second convolutional layer (without activation)
        conv2 = layers.Conv2D(
            filters,
            kernel_size=3,
            padding='same',
            kernel_regularizer=regularizers.l2(self.l2_lambda),
            name=f'{name}_2_conv'
        )(conv1)
        bn2 = layers.BatchNormalization(name=f'{name}_2_bn')(conv2)

        # Skip connection
        if x.shape[-1] != filters:
            # If channel dimensions don't match, use 1x1 convolution
            x = layers.Conv2D(
                filters,
                kernel_size=1,
                padding='same',
                kernel_regularizer=regularizers.l2(self.l2_lambda),
                name=f'{name}_skip_conv'
            )(x)

        # Add skip connection and apply activation
        add = layers.Add(name=f'{name}_add')([bn2, x])
        return layers.Activation('relu', name=f'{name}_out')(add)

    def _get_optimizer(self):
        """
        Create optimizer based on configuration.

        Returns:
            Configured Keras optimizer
        """
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 0.001)

        if optimizer_name == 'adam':
            return optimizers.Adam(learning_rate=lr)
        elif optimizer_name == 'sgd':
            return optimizers.SGD(learning_rate=lr, momentum=0.9)
        elif optimizer_name == 'rmsprop':
            return optimizers.RMSprop(learning_rate=lr)
        else:
            logger.warning(f"Unknown optimizer '{optimizer_name}', using Adam")
            return optimizers.Adam(learning_rate=lr)

    def _visualize_model_architecture(self, model):
        """
        Visualize the model architecture using Keras utils.

        Args:
            model: Keras model to visualize
        """
        try:
            from tensorflow.keras.utils import plot_model

            # Create the output directory if it doesn't exist
            model_dir = Path(self.config['models_dir'])
            ensure_dir_exists(model_dir)

            # Save the model visualization
            plot_model(
                model,
                to_file=self.config['model_architecture_path'],
                show_shapes=True,
                show_layer_names=True,
                show_dtype=True,
                expand_nested=True
            )
            logger.info(f"Model architecture visualization saved to {self.config['model_architecture_path']}")
        except ImportError:
            logger.warning("Could not visualize model: required packages missing (pydot, graphviz)")
        except Exception as e:
            logger.warning(f"Failed to visualize model architecture: {str(e)}")