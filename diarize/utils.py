"""
Utility functions for VRAM management, device detection, and logging.

Functions:
    clear_vram()    - Force garbage collection and clear CUDA cache.
    get_device()    - Detect and return the best available compute device.
    setup_logging() - Configure the root logger format and level.
"""

import gc
import logging

import torch

logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    """
    Configure the root logger with a timestamped format.

    Args:
        level (int): Logging level (default: logging.INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def clear_vram():
    """
    Clear GPU VRAM by running garbage collection and emptying the CUDA cache.

    Call this after deleting a model to reclaim GPU memory before loading
    the next model in the pipeline.
    """
    gc.collect()
    torch.cuda.empty_cache()
    logger.debug("VRAM cleared")


def get_device():
    """
    Determine the best available compute device.

    Returns:
        str: ``'cuda'`` if an NVIDIA GPU is available, otherwise ``'cpu'``.

    Side effects:
        Logs GPU name and available VRAM when CUDA is detected.
    """
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"Available GPU VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
        return "cuda"
    else:
        logger.warning(
            "CUDA is not available. Using CPU. This will be significantly slower."
        )
        return "cpu"
