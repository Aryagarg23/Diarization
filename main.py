#!/usr/bin/env python3
"""
CLI entry point for the diarize transcription and speaker diarization toolkit.

Usage:
    python main.py <path_to_video_or_audio_or_folder> [options]

Examples:
    python main.py video.mov
    python main.py video.mov --simple
    python main.py video.mov --experimental
    python main.py /path/to/folder --output_dir outputs
    python main.py video.mov --hf_token hf_xxx --batch_size 32
"""

import os
import sys
import argparse
from pathlib import Path

from dotenv import load_dotenv

from diarize.utils import setup_logging, get_device
from diarize.pipeline import process_single_file, process_folder

# Load .env before anything else so HF_TOKEN is available
load_dotenv()


def build_parser():
    """
    Build and return the argument parser.

    Returns:
        argparse.ArgumentParser: Configured parser with all CLI flags.
    """
    parser = argparse.ArgumentParser(
        description="WhisperX Audio Transcription and Speaker Diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transcription + diarization
  python main.py video.mov

  # Simple transcript (no speaker labels, split by pauses)
  python main.py video.mov --simple

  # Experimental: detect speaker names from self-introductions
  python main.py video.mov --experimental

  # Batch-process a folder
  python main.py /path/to/folder --output_dir outputs

  # All options
  python main.py video.mov --hf_token hf_xxx --batch_size 32 --output_dir out --experimental
        """,
    )

    parser.add_argument(
        "media_file",
        type=str,
        help="Path to a video/audio file, or a folder containing media files.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face API token (or set HF_TOKEN in .env / environment).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help=(
            "Batch size for WhisperX transcription. "
            "0 = auto (probe free VRAM after model load and use as much as possible). "
            "Any positive value is used as-is. Default: 0 (auto)."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output text file path (single file only; default: <input>.txt).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: same directory as input).",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Simple mode: transcript split by speech pauses, no diarization.",
    )
    parser.add_argument(
        "--experimental",
        action="store_true",
        help=(
            "Experimental mode: detect speaker names from self-introductions "
            '(e.g. "I\'m John", "my name is Sarah") and replace SPEAKER_N labels.'
        ),
    )
    parser.add_argument(
        "--emotions",
        action="store_true",
        help=(
            "Experimental: run dual-model speech emotion analysis on every segment. "
            "Produces arousal / valence / dominance (from vocal tone) and categorical "
            "emotions (from text content) in a separate *_emotions.txt file. "
            "Ideal for HCD interview analysis."
        ),
    )

    return parser


def main():
    """
    Parse arguments, select device, and dispatch to the appropriate pipeline.

    Exit codes:
        0  Success.
        1  Input path not found or unexpected error.
    """
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()

    import logging
    log = logging.getLogger(__name__)

    media_path = Path(args.media_file)
    if not media_path.exists():
        log.error(f"File or folder not found: {media_path}")
        sys.exit(1)

    device = get_device()

    if media_path.is_dir():
        process_folder(media_path, args, device)
    else:
        # Determine output file path
        if args.output:
            output_file = args.output
        elif args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = str(output_dir / f"{media_path.stem}.txt")
        else:
            output_file = str(media_path.with_suffix(".txt"))

        try:
            process_single_file(media_path, output_file, args, device)

            log.info("\n" + "=" * 50)
            log.info("TRANSCRIPTION AND DIARIZATION COMPLETE!")
            log.info("=" * 50)
            log.info(f"Output saved to: {output_file}")

        except KeyboardInterrupt:
            log.info("Process interrupted by user")
            sys.exit(0)
        except Exception as e:
            log.error(f"Unexpected error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
