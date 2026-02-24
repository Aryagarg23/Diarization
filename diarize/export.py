"""
Export transcription results to text files.

Functions:
    export_to_txt()          - Write diarized or simple transcript to a ``.txt`` file.
    export_emotions_to_txt() - Write emotion-annotated transcript with per-speaker profiles.
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


# ------------------------------------------------------------------
# Emotion-annotated export
# ------------------------------------------------------------------


def _fmt_timestamp(seconds):
    """Format seconds as MM:SS.ss."""
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:05.2f}"


def export_emotions_to_txt(emotion_results, speaker_summary, output_file):
    """
    Write an emotion-annotated transcript to a text file.

    The file contains:

    1. **Header** with model information and dimension descriptions.
    2. **Per-segment transcript** with both audio and text emotion scores.
    3. **Per-speaker emotion profiles** (averages across all segments).

    Args:
        emotion_results (list[dict]): Per-segment emotion data from
            :func:`diarize.emotions.analyze_emotions`.
        speaker_summary (dict): Per-speaker aggregation from
            :func:`diarize.emotions.aggregate_emotions_by_speaker`.
        output_file (str): Destination file path.
    """
    logger.info(f"Exporting emotion analysis to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        # -- Header --------------------------------------------------------
        f.write("=" * 80 + "\n")
        f.write("EMOTION-ANNOTATED TRANSCRIPT\n")
        f.write("=" * 80 + "\n\n")
        f.write("Models:\n")
        f.write(
            "  Audio: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim\n"
        )
        f.write(
            "    Arousal  (0-1): activation / excitement\n"
        )
        f.write(
            "    Dominance(0-1): control / confidence\n"
        )
        f.write(
            "    Valence  (0-1): positivity / pleasantness\n"
        )
        f.write("\n")
        f.write(
            "  Text:  j-hartmann/emotion-english-distilroberta-base\n"
        )
        f.write(
            "    anger | disgust | fear | joy | neutral | sadness | surprise\n"
        )
        f.write("\n" + "=" * 80 + "\n\n")

        # -- Per-segment ---------------------------------------------------
        for seg in emotion_results:
            ts = (
                f"[{_fmt_timestamp(seg['start'])} -> "
                f"{_fmt_timestamp(seg['end'])}]"
            )
            f.write(f"{ts} {seg['speaker']}:\n")
            f.write(f"  {seg['text']}\n")

            # Audio emotions
            ae = seg["audio_emotions"]
            f.write(
                f"  Voice:  "
                f"arousal={ae['arousal']:.2f}  "
                f"valence={ae['valence']:.2f}  "
                f"dominance={ae['dominance']:.2f}\n"
            )

            # Text emotions (sorted by score, show top 3)
            te = seg["text_emotions"]
            sorted_te = sorted(te.items(), key=lambda x: x[1], reverse=True)
            top_str = "  ".join(
                f"{label}={score:.2f}" for label, score in sorted_te[:3]
            )
            f.write(f"  Text:   {top_str}\n")
            f.write("\n")

        # -- Speaker profiles ----------------------------------------------
        f.write("=" * 80 + "\n")
        f.write("SPEAKER EMOTION PROFILES (averaged across all segments)\n")
        f.write("=" * 80 + "\n\n")

        for spk, profile in speaker_summary.items():
            count = profile["segment_count"]
            f.write(f"{spk} ({count} segments):\n")

            ae = profile["audio_emotions"]
            f.write(
                f"  Voice avg:  "
                f"arousal={ae['arousal']:.2f}  "
                f"valence={ae['valence']:.2f}  "
                f"dominance={ae['dominance']:.2f}\n"
            )

            te = profile["text_emotions"]
            sorted_te = sorted(
                te.items(), key=lambda x: x[1], reverse=True,
            )
            te_str = "  ".join(
                f"{label}={score:.2f}" for label, score in sorted_te
            )
            f.write(f"  Text avg:   {te_str}\n")

            if sorted_te:
                f.write(
                    f"  Dominant:   {sorted_te[0][0]} ({sorted_te[0][1]:.2f})\n"
                )
            f.write("\n")

    logger.info(f"Emotion analysis exported to {output_file}")
