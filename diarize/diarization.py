"""
Speaker diarization using pyannote.audio.

Functions:
    diarize_audio() - Identify who spoke when in an audio file.

Important:
    pyannote.audio 4.x returns a ``DiarizeOutput`` dataclass, **not** a raw
    ``Annotation``.  This module handles the unwrapping transparently.
"""

import logging
import time

import torch
import torchaudio
from pyannote.audio import Pipeline

from diarize.utils import clear_vram

logger = logging.getLogger(__name__)


def diarize_audio(audio_path, hf_token, device):
    """
    Run speaker diarization on an audio file.

    Uses the ``pyannote/speaker-diarization-3.0`` pipeline.  Audio is loaded
    via torchaudio (not torchcodec) and resampled to 16 kHz if necessary.

    Args:
        audio_path (str): Path to a WAV audio file.
        hf_token (str): Hugging Face API token with access to the pyannote
            gated models.
        device (str): ``'cuda'`` or ``'cpu'``.

    Returns:
        pyannote.core.Annotation | None: Speaker annotation with labelled
        speech turns, or ``None`` if the token is missing or diarization
        fails.

    Note:
        The pipeline is deleted and VRAM is cleared after diarization.
        pyannote 4.x returns a ``DiarizeOutput`` dataclass; this function
        extracts the ``.speaker_diarization`` ``Annotation`` automatically.
    """
    if not hf_token:
        logger.error(
            "Hugging Face token not provided. "
            "Diarization requires HF_TOKEN environment variable or --hf_token argument."
        )
        logger.warning(
            "To use diarization, set HF_TOKEN in .env or pass --hf_token <token>"
        )
        return None

    logger.info("Loading pyannote diarization pipeline...")

    diarization = None
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0",
            token=hf_token,
        )
        pipeline = pipeline.to(torch.device(device))
        logger.info("Pipeline loaded.")

        # Load audio via torchaudio (avoids torchcodec incompatibility)
        logger.info("Loading audio file...")
        waveform, sample_rate = torchaudio.load(audio_path)

        duration_seconds = waveform.shape[1] / sample_rate
        duration_minutes = duration_seconds / 60

        logger.info(f"Audio duration: {duration_minutes:.1f} minutes")
        logger.info(
            "Note: Diarization time varies based on speaker count and overlap "
            "(not linear)"
        )
        logger.info("Starting speaker diarization (this may take a while)...")

        # Resample to 16 kHz if needed
        if sample_rate != 16000:
            logger.debug(f"Resampling from {sample_rate}Hz to 16000Hz...")
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        audio_dict = {"waveform": waveform, "sample_rate": sample_rate}

        logger.info(
            "Processing speakers (this is real-time processing, monitor GPU usage)..."
        )
        start_time = time.time()
        diarize_output = pipeline(audio_dict)
        elapsed_time = time.time() - start_time

        # pyannote 4.x wraps Annotation in DiarizeOutput dataclass
        if hasattr(diarize_output, "speaker_diarization"):
            diarization = diarize_output.speaker_diarization
        else:
            diarization = diarize_output

        num_speakers = len(diarization.labels())
        realtime_factor = (
            elapsed_time / duration_seconds if duration_seconds > 0 else 0
        )

        logger.info("Diarization completed.")
        logger.info(
            f"  Duration: {duration_minutes:.1f} min | "
            f"Processed in {elapsed_time:.0f}s ({realtime_factor:.2f}x realtime)"
        )
        logger.info(
            f"  Speakers detected: {num_speakers} | "
            f"Labels: {list(diarization.labels())}"
        )

    except Exception as e:
        logger.error(f"Error during diarization: {type(e).__name__}: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        logger.warning("Continuing without diarization...")
        return None
    finally:
        if "pipeline" in locals():
            del pipeline
        clear_vram()

    return diarization
