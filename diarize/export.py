"""
Export transcription results to text files.

Functions:
    export_to_txt() - Write diarized or simple transcript to a ``.txt`` file.
"""

import logging

logger = logging.getLogger(__name__)


def export_to_txt(result, output_file, simple=False):
    """
    Write the transcription result to a plain-text file.

    Two output modes are supported:

    **Diarized mode** (default):
        Groups consecutive words from the same speaker into a single line
        prefixed by the speaker label::

            SPEAKER_1: Welcome to the meeting.
            SPEAKER_2: Thank you for having me.

    **Simple mode** (``simple=True``):
        Writes one line per segment (natural speech pause), with no speaker
        labels::

            Welcome to the meeting.
            Thank you for having me.

    Args:
        result (dict): Final transcription result with ``segments`` containing
            ``words``.  Each word dict should have ``word`` and (in diarized
            mode) ``speaker`` keys.
        output_file (str): Destination file path.
        simple (bool): If ``True``, use simple mode.  Default ``False``.
    """
    logger.info(f"Exporting results to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        if simple:
            _write_simple(f, result)
        else:
            _write_diarized(f, result)

    logger.info(f"Results exported to {output_file}")


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _write_simple(f, result):
    """Write one line per segment (split by pauses)."""
    for segment in result.get("segments", []):
        words = [
            w.get("word", "").strip()
            for w in segment.get("words", [])
            if w.get("word", "").strip()
        ]
        if words:
            f.write(f"{' '.join(words)}\n")


def _write_diarized(f, result):
    """Group consecutive words by speaker and write labelled lines."""
    current_speaker = None
    current_text = []

    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            speaker = word.get("speaker", "UNKNOWN")
            text = word.get("word", "").strip()
            if not text:
                continue

            if speaker != current_speaker and current_text:
                f.write(f"{current_speaker}: {' '.join(current_text)}\n")
                current_text = []
                current_speaker = None

            if speaker != current_speaker:
                current_speaker = speaker
                current_text = [text]
            else:
                current_text.append(text)

    if current_text and current_speaker:
        f.write(f"{current_speaker}: {' '.join(current_text)}\n")
