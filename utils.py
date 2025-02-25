"""
Utility Functions Module

This module provides utility functions used throughout the project,
including logging setup, timing, file operations, and other helper functions.
"""

import os
import json
import logging
import time
from pathlib import Path
from functools import wraps


def setup_logging(log_level, output_dir):
    """
    Set up logging configuration with file and console handlers.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        output_dir: Directory to save log file

    Returns:
        Logger object configured for the project
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "image_classifier.log"),
            logging.StreamHandler()
        ]
    )

    # Create and return logger
    logger = logging.getLogger("ImageClassifier")
    logger.info(f"Logging initialized at {log_level} level")

    return logger


def timer(func):
    """
    Decorator that measures and logs the execution time of the decorated function.

    Args:
        func: Function to time

    Returns:
        Wrapped function with timing
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(f"ImageClassifier.{func.__name__}")
        start_time = time.time()
        logger.info(f"Starting {func.__name__}")

        result = func(*args, **kwargs)

        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Completed {func.__name__} in {execution_time:.2f} seconds")

        return result

    return wrapper


def ensure_dir_exists(directory):
    """
    Ensure that a directory exists, create it if it doesn't.

    Args:
        directory: Path to the directory
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_text_to_file(text, file_path):
    """
    Save text content to a file.

    Args:
        text: Text content to save
        file_path: Path to the output file
    """
    # Ensure the directory exists
    ensure_dir_exists(os.path.dirname(file_path))

    # Write the text to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)


def save_json(data, file_path):
    """
    Save data as JSON file.

    Args:
        data: Data to save as JSON
        file_path: Path to the output file
    """
    # Ensure the directory exists
    ensure_dir_exists(os.path.dirname(file_path))

    # Write the data to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def load_json(file_path):
    """
    Load data from JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Loaded data from JSON file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class ProgressLogger:
    """Class for logging progress of iterative processes."""

    def __init__(self, total, name="Process", log_interval=5):
        """
        Initialize progress logger.

        Args:
            total: Total number of iterations
            name: Name of the process being tracked
            log_interval: How often to log progress (in percentage)
        """
        self.total = total
        self.name = name
        self.log_interval = log_interval
        self.current = 0
        self.last_log_percent = 0
        self.logger = logging.getLogger(f"ImageClassifier.{name}")
        self.start_time = time.time()

        self.logger.info(f"Starting {name} with {total} iterations")

    def update(self, increment=1):
        """
        Update progress counter and log if necessary.

        Args:
            increment: Amount to increment the counter by
        """
        self.current += increment
        percent_complete = int(100 * self.current / self.total)

        # Log progress at specified intervals
        if percent_complete >= self.last_log_percent + self.log_interval:
            elapsed = time.time() - self.start_time
            items_per_sec = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / items_per_sec if items_per_sec > 0 else 0

            self.logger.info(
                f"{self.name}: {percent_complete}% complete - "
                f"{self.current}/{self.total} iterations "
                f"({items_per_sec:.2f} it/s, ETA: {eta:.2f}s)"
            )
            self.last_log_percent = percent_complete

    def finish(self):
        """Log completion of the process."""
        total_time = time.time() - self.start_time
        self.logger.info(
            f"{self.name} completed in {total_time:.2f} seconds "
            f"({self.total / total_time:.2f} it/s average)"
        )