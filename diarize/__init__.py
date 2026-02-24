"""
diarize - Modular audio transcription and speaker diarization toolkit.

Modules:
    audio       - Audio extraction from media files via FFmpeg
    transcribe  - WhisperX transcription and timestamp alignment
    diarize     - Pyannote speaker diarization
    speakers    - Speaker label assignment and experimental name detection
    emotions    - Speech emotion recognition (audio + text, experimental)
    visualize   - Emotion timeline graphs (matplotlib PNG output)
    export      - Text file export (diarized, simple, and emotion-annotated modes)
    utils       - VRAM management, device detection, logging setup
    pipeline    - Orchestrates the full processing workflow
"""

from diarize.utils import (
    clear_vram,
    get_device,
    setup_logging,
    get_total_vram,
    get_free_vram,
    log_vram_usage,
    optimal_batch_size,
)
from diarize.audio import extract_audio
from diarize.transcribe import transcribe_audio, align_timestamps
from diarize.diarization import diarize_audio
from diarize.speakers import assign_speakers_to_words, detect_speaker_names, apply_speaker_names
from diarize.export import export_to_txt, export_emotions_to_txt
from diarize.emotions import analyze_emotions, aggregate_emotions_by_speaker
from diarize.visualize import plot_emotion_timeline
from diarize.pipeline import process_single_file, process_folder

__all__ = [
    "clear_vram",
    "get_device",
    "setup_logging",
    "get_total_vram",
    "get_free_vram",
    "log_vram_usage",
    "optimal_batch_size",
    "extract_audio",
    "transcribe_audio",
    "align_timestamps",
    "diarize_audio",
    "assign_speakers_to_words",
    "detect_speaker_names",
    "apply_speaker_names",
    "export_to_txt",
    "export_emotions_to_txt",
    "analyze_emotions",
    "aggregate_emotions_by_speaker",
    "plot_emotion_timeline",
    "process_single_file",
    "process_folder",
]
