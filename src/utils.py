"""
Utility functions used across the project.

Provides logging setup, random seed management, directory creation,
and common helper functions.
"""
import logging
import os
import random
from typing import Optional

import numpy as np

from config.settings import (
    BASE_DIR,
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    DOCS_DIR,
    TESTS_DIR,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    RANDOM_STATE,
)


def setup_logging(
    name: str = "predict_success",
    level: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return a project logger.

    Parameters
    ----------
    name : str
        Logger name (typically the module ``__name__``).
    level : str, optional
        Logging level string (e.g. ``"DEBUG"``).  Defaults to the
        value in ``config.settings.LOG_LEVEL``.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    level = level or LOG_LEVEL
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        )
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def set_random_seeds(seed: int = RANDOM_STATE) -> None:
    """
    Set random seeds for full reproducibility.

    Sets seeds for Python's ``random``, NumPy, and (if available)
    scikit-learn's global random state.

    Parameters
    ----------
    seed : int
        The seed value.  Defaults to ``RANDOM_STATE`` from config.
    """
    random.seed(seed)
    np.random.seed(seed)

    logger = setup_logging(__name__)
    logger.info("Random seeds set to %d", seed)


def ensure_directories() -> None:
    """
    Create all required project directories if they don't exist.

    Directories created:
    - ``data/raw/``
    - ``data/processed/``
    - ``models/``
    - ``reports/figures/``
    - ``docs/``
    - ``tests/``
    """
    dirs = [
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        DOCS_DIR,
        TESTS_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    logger = setup_logging(__name__)
    logger.debug("All project directories verified.")


def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    return BASE_DIR
