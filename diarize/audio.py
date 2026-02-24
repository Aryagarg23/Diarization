"""
Audio extraction from media files using FFmpeg.

Functions:
    extract_audio() - Extract mono 16 kHz WAV audio from any media file.
"""

import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


def extract_audio(media_file, output_audio_path, sr=16000):
    """
    Extract audio from a media file using FFmpeg.

    Converts the input to a mono 16-bit PCM WAV at the given sample rate.
    FFmpeg must be installed and available on ``$PATH``.

    Args:
        media_file (str): Path to the input video or audio file.
        output_audio_path (str): Destination path for the extracted WAV file.
        sr (int): Target sample rate in Hz (default: 16000).

    Returns:
        str: The *output_audio_path* on success.

    Raises:
        SystemExit: If FFmpeg is missing or the extraction command fails.
    """
    logger.info(f"Extracting audio from {media_file}...")

    # Verify FFmpeg is installed
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("ffmpeg is not installed. Please install it and try again.")
        sys.exit(1)

    ffmpeg_cmd = [
        "ffmpeg",
        "-i", media_file,
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-ac", "1",
        "-y",
        output_audio_path,
    ]

    try:
        subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
        logger.info(f"Audio extracted successfully to {output_audio_path}")
        return output_audio_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract audio: {e}")
        sys.exit(1)
