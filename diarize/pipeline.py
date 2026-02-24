"""
High-level pipeline orchestration for single files and batch folders.

Functions:
    process_single_file() - Run the full pipeline on one media file.
    process_folder()       - Batch-process every media file in a directory.
"""

import logging
import os
from pathlib import Path

from diarize.audio import extract_audio
from diarize.transcribe import transcribe_audio, align_timestamps
from diarize.diarization import diarize_audio
from diarize.speakers import assign_speakers_to_words, detect_speaker_names, apply_speaker_names
from diarize.export import export_to_txt

logger = logging.getLogger(__name__)

# Media formats recognised during batch processing
SUPPORTED_FORMATS = {
    ".mp4", ".mov", ".avi", ".mkv", ".flv",
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg",
}


def process_single_file(media_file, output_file, args, device):
    """
    Run the full transcription / diarization pipeline on a single media file.

    Pipeline stages (each model is loaded, used, then deleted):

    1. **Audio extraction** -- FFmpeg converts the media to 16 kHz mono WAV.
    2. **Transcription** -- WhisperX large-v2 produces word-level output.
    3. **Timestamp alignment** -- WhisperX alignment model refines word times.
    4. **Speaker diarization** -- pyannote identifies who spoke when.
       *(skipped in ``--simple`` mode)*
    5. **Speaker assignment** -- each word is labelled with a speaker ID.
       *(skipped in ``--simple`` mode)*
    6. **Name detection** -- *(experimental, ``--experimental`` flag)*
       scans for self-introductions and replaces SPEAKER_N with real names.
    7. **Export** -- writes the final ``.txt`` file.

    Args:
        media_file (Path): Input media file path.
        output_file (str): Destination text file path.
        args: Parsed CLI arguments (needs ``.batch_size``, ``.hf_token``,
            ``.simple``, ``.experimental``).
        device (str): ``'cuda'`` or ``'cpu'``.
    """
    logger.info(f"\nProcessing: {media_file.name}")

    temp_audio = f"temp_audio_{media_file.stem}.wav"

    try:
        # -- Step 1: Extract audio ----------------------------------------
        audio_path = extract_audio(str(media_file), temp_audio)

        # -- Step 2: Transcribe -------------------------------------------
        logger.info("\n" + "=" * 50)
        logger.info("STEP 1: TRANSCRIPTION")
        logger.info("=" * 50)
        result = transcribe_audio(
            audio_path, device, batch_size=args.batch_size,
        )

        # -- Step 3: Align timestamps -------------------------------------
        logger.info("\n" + "=" * 50)
        logger.info("STEP 2: TIMESTAMP ALIGNMENT")
        logger.info("=" * 50)
        result = align_timestamps(audio_path, result, device)

        # -- Step 4 & 5: Diarize + assign speakers ------------------------
        if args.simple:
            logger.info("\n" + "=" * 50)
            logger.info("SIMPLE MODE - Skipping diarization")
            logger.info("=" * 50)
            diarization = None
        else:
            logger.info("\n" + "=" * 50)
            logger.info("STEP 3: SPEAKER DIARIZATION")
            logger.info("=" * 50)
            diarization = diarize_audio(audio_path, args.hf_token, device)

            logger.info("\n" + "=" * 50)
            logger.info("STEP 4: ASSIGNING SPEAKERS")
            logger.info("=" * 50)
            result = assign_speakers_to_words(result, diarization)

            # -- Step 6 (optional): Experimental name detection ------------
            if getattr(args, "experimental", False):
                logger.info("\n" + "=" * 50)
                logger.info("STEP 5: EXPERIMENTAL NAME DETECTION")
                logger.info("=" * 50)
                name_map = detect_speaker_names(result)
                if name_map:
                    result = apply_speaker_names(result, name_map)

        # -- Step 7: Export ------------------------------------------------
        logger.info("\n" + "=" * 50)
        logger.info(
            "STEP 6: EXPORTING RESULTS"
            if not args.simple
            else "STEP 3: EXPORTING RESULTS"
        )
        logger.info("=" * 50)
        export_to_txt(result, output_file, simple=args.simple)

        logger.info(f"Completed: {output_file}")

    except Exception as e:
        logger.error(f"Failed to process {media_file.name}: {e}")
        raise
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
            logger.debug(f"Removed temporary audio file: {temp_audio}")


def process_folder(folder_path, args, device):
    """
    Batch-process every recognised media file in a directory.

    Iterates over files matching :data:`SUPPORTED_FORMATS`, running
    :func:`process_single_file` on each.  Processing continues even if an
    individual file fails.

    Args:
        folder_path (Path): Directory containing media files.
        args: Parsed CLI arguments.
        device (str): ``'cuda'`` or ``'cpu'``.
    """
    media_files = []
    for ext in SUPPORTED_FORMATS:
        media_files.extend(folder_path.glob(f"*{ext}"))
        media_files.extend(folder_path.glob(f"*{ext.upper()}"))

    if not media_files:
        logger.warning(f"No media files found in {folder_path}")
        return

    logger.info(f"Found {len(media_files)} media files to process")

    output_dir = Path(args.output_dir) if args.output_dir else folder_path
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, media_file in enumerate(sorted(media_files), 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"File {idx}/{len(media_files)}")
        logger.info(f"{'=' * 60}")

        output_file = str(output_dir / f"{media_file.stem}.txt")

        try:
            process_single_file(media_file, output_file, args, device)
        except KeyboardInterrupt:
            logger.info("Process interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error processing {media_file.name}: {e}")
            continue

    logger.info(f"\n{'=' * 60}")
    logger.info("BATCH PROCESSING COMPLETE!")
    logger.info(f"{'=' * 60}")
    logger.info(f"Output directory: {output_dir}")
