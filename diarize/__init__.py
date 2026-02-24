"""
diarize - Modular audio transcription and speaker diarization toolkit.

Modules:
    audio       - Audio extraction from media files via FFmpeg
    transcribe  - WhisperX transcription and timestamp alignment
    diarize     - Pyannote speaker diarization
    speakers    - Speaker label assignment and experimental name detection
    export      - Text file export (diarized and simple modes)
    utils       - VRAM management, device detection, logging setup
    pipeline    - Orchestrates the full processing workflow
"""

from diarize.utils import clear_vram, get_device, setup_logging
from diarize.audio import extract_audio
from diarize.transcribe import transcribe_audio, align_timestamps
from diarize.diarization import diarize_audio
from diarize.speakers import assign_speakers_to_words, detect_speaker_names, apply_speaker_names
from diarize.export import export_to_txt
from diarize.pipeline import process_single_file, process_folder

__all__ = [
    "clear_vram",
    "get_device",
    "setup_logging",
    "extract_audio",
    "transcribe_audio",
    "align_timestamps",
    "diarize_audio",
    "assign_speakers_to_words",
    "detect_speaker_names",
    "apply_speaker_names",
    "export_to_txt",
    "process_single_file",
    "process_folder",
]
