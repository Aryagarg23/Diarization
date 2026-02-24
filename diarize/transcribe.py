"""
WhisperX transcription and word-level timestamp alignment.

Functions:
    transcribe_audio() - Transcribe audio with WhisperX large-v2.
    align_timestamps()  - Refine word-level timestamps with the alignment model.

Both functions support dynamic VRAM-aware batch sizing.  Pass ``batch_size=0``
(the default) to let the system auto-calculate the largest safe batch size
after the model is loaded.
"""

import logging
import sys

import whisperx

from diarize.utils import clear_vram, log_vram_usage, optimal_batch_size

logger = logging.getLogger(__name__)


def transcribe_audio(audio_path, device, batch_size=0, compute_type="float16"):
    """
    Transcribe an audio file using the WhisperX large-v2 model.

    The model is loaded, used, and immediately deleted so VRAM is available
    for subsequent pipeline stages.

    **Dynamic batch sizing:** When *batch_size* is ``0`` (default), the
    function probes free VRAM after loading the model and calculates the
    largest safe batch size automatically.  Any positive value is used as-is.

    Args:
        audio_path (str): Path to a WAV audio file (16 kHz mono recommended).
        device (str): ``'cuda'`` or ``'cpu'``.
        batch_size (int): Inference batch size.  ``0`` = auto (default).
        compute_type (str): Model precision (default: ``'float16'``).

    Returns:
        dict: WhisperX result dict containing ``segments`` with word-level
        timestamps and a ``language`` key.

    Raises:
        SystemExit: On any transcription error.
    """
    logger.info("Loading WhisperX model (large-v2)...")

    try:
        model = whisperx.load_model(
            "large-v2",
            device=device,
            compute_type=compute_type,
            language="en",
        )

        log_vram_usage("After WhisperX model load")

        # --- Dynamic batch sizing ---
        if batch_size <= 0 and device == "cuda":
            batch_size = optimal_batch_size()
            logger.info(f"Using auto batch size: {batch_size}")
        elif batch_size <= 0:
            batch_size = 16  # sensible CPU default
            logger.info(f"CPU mode: using default batch size {batch_size}")
        else:
            logger.info(f"Using user-specified batch size: {batch_size}")

        logger.info("Starting transcription...")
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=batch_size, language="en")
        logger.info("Transcription completed")
        log_vram_usage("After transcription")

    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        sys.exit(1)
    finally:
        if "model" in locals():
            del model
        clear_vram()
        log_vram_usage("After WhisperX cleanup")

    return result


def align_timestamps(audio_path, result, device):
    """
    Refine word-level timestamps using the WhisperX alignment model.

    The alignment model is language-specific and selected automatically from
    the ``language`` key in *result*.  The model is deleted after use.

    Args:
        audio_path (str): Path to the WAV audio file.
        result (dict): Transcription result from :func:`transcribe_audio`.
        device (str): ``'cuda'`` or ``'cpu'``.

    Returns:
        dict: Updated result dict with refined ``word_segments`` timestamps.
    """
    logger.info("Loading alignment model...")

    try:
        language = result.get("language", "en")
        model_a, metadata = whisperx.load_align_model(
            language_code=language, device=device
        )
        log_vram_usage("After alignment model load")

        logger.info("Alignment model loaded. Aligning timestamps...")
        result = whisperx.align(
            result["segments"], model_a, metadata, audio_path, device
        )
        logger.info("Alignment completed")
        log_vram_usage("After alignment")

    except Exception as e:
        logger.error(f"Error during alignment: {e}")
        logger.warning("Continuing without alignment...")
    finally:
        if "model_a" in locals():
            del model_a
        clear_vram()
        log_vram_usage("After alignment cleanup")

    return result
