"""
WhisperX transcription and word-level timestamp alignment.

Functions:
    transcribe_audio() - Transcribe audio with WhisperX large-v2.
    align_timestamps()  - Refine word-level timestamps with the alignment model.
"""

import logging
import sys

import whisperx

from diarize.utils import clear_vram

logger = logging.getLogger(__name__)


def transcribe_audio(audio_path, device, batch_size=16, compute_type="float16"):
    """
    Transcribe an audio file using the WhisperX large-v2 model.

    The model is loaded, used, and immediately deleted so VRAM is available
    for subsequent pipeline stages.

    Args:
        audio_path (str): Path to a WAV audio file (16 kHz mono recommended).
        device (str): ``'cuda'`` or ``'cpu'``.
        batch_size (int): Inference batch size (default: 16).
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

        logger.info("Model loaded. Starting transcription...")
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=batch_size, language="en")
        logger.info("Transcription completed")

    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        sys.exit(1)
    finally:
        if "model" in locals():
            del model
        clear_vram()

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
        logger.info("Alignment model loaded. Aligning timestamps...")
        result = whisperx.align(
            result["segments"], model_a, metadata, audio_path, device
        )
        logger.info("Alignment completed")

    except Exception as e:
        logger.error(f"Error during alignment: {e}")
        logger.warning("Continuing without alignment...")
    finally:
        if "model_a" in locals():
            del model_a
        clear_vram()

    return result
