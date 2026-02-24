"""
Utility functions for VRAM management, device detection, and logging.

Functions:
    clear_vram()           - Force garbage collection and clear CUDA cache.
    get_device()           - Detect and return the best available compute device.
    setup_logging()        - Configure the root logger format and level.
    get_total_vram()       - Return total GPU VRAM in GB.
    get_free_vram()        - Return currently free GPU VRAM in GB.
    log_vram_usage()       - Log a snapshot of used / free / total VRAM.
    optimal_batch_size()   - Calculate the largest safe batch size given free VRAM.
"""

import gc
import logging

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants for VRAM estimation (float16, large-v2)
# ---------------------------------------------------------------------------
# Approximate VRAM consumed per batch item during WhisperX transcription.
# Each item is a 30-second audio chunk processed through the encoder.
# Measured empirically on large-v2 float16: ~280-350 MB per item.
WHISPERX_PER_BATCH_ITEM_MB = 320

# Safety headroom to avoid OOM (MB).  Accounts for CUDA fragmentation,
# kernel overhead, and temporary allocations.
VRAM_SAFETY_MARGIN_MB = 1200

# Hard bounds on auto-calculated batch size.
MIN_BATCH_SIZE = 4
MAX_BATCH_SIZE = 128


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


# ---------------------------------------------------------------------------
# VRAM introspection helpers
# ---------------------------------------------------------------------------


def get_total_vram():
    """
    Return the total GPU VRAM in gigabytes.

    Returns:
        float: Total VRAM in GB, or 0.0 if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1e9


def get_free_vram():
    """
    Return the currently free GPU VRAM in gigabytes.

    Runs ``gc.collect()`` and ``torch.cuda.empty_cache()`` first to get an
    accurate reading.

    Returns:
        float: Free VRAM in GB, or 0.0 if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return 0.0
    gc.collect()
    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info(0)
    return free / 1e9


def log_vram_usage(label=""):
    """
    Log a one-line snapshot of GPU VRAM usage.

    Args:
        label (str): Optional label printed before the numbers (e.g.
            ``"After model load"``).

    Example log output::

        [VRAM] After model load: 10.24 GB used / 13.76 GB free / 24.00 GB total
    """
    if not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free, _ = torch.cuda.mem_get_info(0)
    free_gb = free / 1e9
    used_gb = total - free_gb
    prefix = f"[VRAM] {label}: " if label else "[VRAM] "
    logger.info(
        f"{prefix}{used_gb:.2f} GB used / {free_gb:.2f} GB free / {total:.2f} GB total"
    )


def optimal_batch_size(per_item_mb=WHISPERX_PER_BATCH_ITEM_MB,
                       safety_mb=VRAM_SAFETY_MARGIN_MB):
    """
    Calculate the largest safe batch size given current free VRAM.

    Call this **after** loading the model so that the model's own VRAM
    footprint is already accounted for.

    Args:
        per_item_mb (int): Estimated VRAM per batch item in MB.
        safety_mb (int): Safety headroom in MB to subtract from free VRAM.

    Returns:
        int: Recommended batch size, clamped to
        [``MIN_BATCH_SIZE``, ``MAX_BATCH_SIZE``].
    """
    free_gb = get_free_vram()
    free_mb = free_gb * 1024
    usable_mb = free_mb - safety_mb

    if usable_mb <= 0:
        logger.warning(
            f"Very low free VRAM ({free_gb:.2f} GB). Using minimum batch size {MIN_BATCH_SIZE}."
        )
        return MIN_BATCH_SIZE

    batch = int(usable_mb / per_item_mb)
    batch = max(MIN_BATCH_SIZE, min(MAX_BATCH_SIZE, batch))

    logger.info(
        f"[Auto batch] Free VRAM: {free_gb:.2f} GB | "
        f"Usable: {usable_mb:.0f} MB | "
        f"Per-item est: {per_item_mb} MB | "
        f"Optimal batch size: {batch}"
    )
    return batch
