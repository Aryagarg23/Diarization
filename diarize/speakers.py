"""
Speaker label assignment and experimental speaker name detection.

Functions:
    assign_speakers_to_words() - Map diarization labels onto transcribed words.
    detect_speaker_names()     - (Experimental) Scan transcript for self-introductions.
    apply_speaker_names()      - Replace generic labels with detected names.
"""

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patterns that capture self-introductions.
#
# Each pattern must have a named group ``name`` that captures the speaker's
# first name.  Patterns are tried in order; the first match wins for a given
# speaker.
# ---------------------------------------------------------------------------
_NAME_PATTERNS = [
    # "my name is John"  /  "my name's John"
    re.compile(
        r"\bmy\s+name(?:'s|\s+is)\s+(?P<name>[A-Z][a-z]{1,20})\b",
        re.IGNORECASE,
    ),
    # "I'm John"  /  "I am John"
    re.compile(
        r"\bI(?:'m|\s+am)\s+(?P<name>[A-Z][a-z]{1,20})\b",
    ),
    # "call me John"
    re.compile(
        r"\bcall\s+me\s+(?P<name>[A-Z][a-z]{1,20})\b",
        re.IGNORECASE,
    ),
    # "hello/hey/hi, I'm John"  (comma-separated variant)
    re.compile(
        r"\b(?:hello|hey|hi|yo)\s*,?\s*I(?:'m|\s+am)\s+(?P<name>[A-Z][a-z]{1,20})\b",
        re.IGNORECASE,
    ),
    # "this is John speaking"
    re.compile(
        r"\bthis\s+is\s+(?P<name>[A-Z][a-z]{1,20})\s+speaking\b",
        re.IGNORECASE,
    ),
    # "the name is John"
    re.compile(
        r"\bthe\s+name(?:'s|\s+is)\s+(?P<name>[A-Z][a-z]{1,20})\b",
        re.IGNORECASE,
    ),
]

# Words that commonly follow "I'm" but are NOT names.
_FALSE_POSITIVE_WORDS = {
    "a", "an", "the", "not", "so", "very", "just", "also", "really",
    "gonna", "going", "trying", "looking", "getting", "doing", "having",
    "here", "there", "back", "sure", "sorry", "happy", "glad", "excited",
    "tired", "done", "ready", "fine", "good", "great", "okay", "ok",
    "like", "about", "from", "with", "still", "pretty", "super",
    "actually", "literally", "basically", "honestly", "afraid",
    "interested", "impressed", "confused", "surprised", "worried",
    "thinking", "talking", "saying", "telling", "starting", "coming",
}


def assign_speakers_to_words(result, diarization):
    """
    Assign a speaker label to every transcribed word.

    For each word, the midpoint of its timestamp span is compared against the
    diarization speaker turns.  Words whose midpoint falls within a turn are
    assigned that turn's speaker.  Words that fall outside all turns are
    labelled ``UNKNOWN``.

    Speaker labels are normalised to ``SPEAKER_1``, ``SPEAKER_2``, ... in the
    order they first appear in the diarization timeline.

    Args:
        result (dict): Aligned transcription result containing ``segments``
            with ``words`` that have ``start`` / ``end`` timestamps.
        diarization: A ``pyannote.core.Annotation`` object, or ``None``.

    Returns:
        dict: The same *result* dict with a ``speaker`` key added to every
        word dict.
    """
    if diarization is None:
        logger.warning(
            "No diarization data available. Assigning all text to SPEAKER_1"
        )
        for segment in result.get("segments", []):
            for word in segment.get("words", []):
                word["speaker"] = "SPEAKER_1"
        return result

    logger.info("Assigning speakers to words...")

    # Build ordered label map: raw pyannote label -> SPEAKER_N
    speaker_map = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_map:
            speaker_map[speaker] = f"SPEAKER_{len(speaker_map) + 1}"

    # Assign each word to the speaker whose turn covers its midpoint
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            if "start" in word and "end" in word:
                word_mid = (word["start"] + word["end"]) / 2

                assigned = False
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if turn.start <= word_mid <= turn.end:
                        word["speaker"] = speaker_map.get(speaker, "UNKNOWN")
                        assigned = True
                        break

                if not assigned:
                    word["speaker"] = "UNKNOWN"
            else:
                word["speaker"] = "UNKNOWN"

    logger.info("Speaker assignment completed")
    return result


# ---------------------------------------------------------------------------
# Experimental: speaker name detection
# ---------------------------------------------------------------------------


def detect_speaker_names(result):
    """
    Scan the transcript for self-introductions and map speaker labels to names.

    **Experimental.**  This function looks for phrases such as:

    - "My name is Benj"
    - "I'm Toby"
    - "I am Sarah"
    - "Call me Dave"
    - "Hello, I'm Benj"

    It groups consecutive words by speaker, reconstructs each speaker's text
    blocks, and applies the patterns above.  Only the **first** name detected
    per speaker label is kept.

    Args:
        result (dict): Transcription result with ``speaker`` keys on words
            (as produced by :func:`assign_speakers_to_words`).

    Returns:
        dict: Mapping of speaker label to detected name, e.g.
        ``{"SPEAKER_1": "Benj", "SPEAKER_2": "Toby"}``.
        Empty if no introductions are found.
    """
    logger.info("[Experimental] Scanning transcript for speaker introductions...")

    # Gather text blocks grouped by speaker
    speaker_texts = {}  # {speaker_label: [text_block, ...]}
    current_speaker = None
    current_words = []

    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            speaker = word.get("speaker", "UNKNOWN")
            text = word.get("word", "").strip()
            if not text:
                continue

            if speaker != current_speaker:
                if current_speaker and current_words:
                    speaker_texts.setdefault(current_speaker, []).append(
                        " ".join(current_words)
                    )
                current_speaker = speaker
                current_words = [text]
            else:
                current_words.append(text)

    # Flush last block
    if current_speaker and current_words:
        speaker_texts.setdefault(current_speaker, []).append(
            " ".join(current_words)
        )

    # Search for name patterns
    name_map = {}
    for speaker_label, blocks in speaker_texts.items():
        for block in blocks:
            if speaker_label in name_map:
                break  # already found a name for this speaker
            for pattern in _NAME_PATTERNS:
                match = pattern.search(block)
                if match:
                    candidate = match.group("name")
                    # Filter false positives
                    if candidate.lower() not in _FALSE_POSITIVE_WORDS:
                        name_map[speaker_label] = candidate
                        logger.info(
                            f"  Detected: {speaker_label} -> {candidate} "
                            f'(from: "...{block[max(0, match.start()-15):match.end()+15]}...")'
                        )
                        break

    if name_map:
        logger.info(f"  Name mapping: {name_map}")
    else:
        logger.info("  No speaker introductions detected.")

    return name_map


def apply_speaker_names(result, name_map):
    """
    Replace generic speaker labels with detected names in the result dict.

    Args:
        result (dict): Transcription result with ``speaker`` keys on words.
        name_map (dict): Mapping from label to name, e.g.
            ``{"SPEAKER_1": "Benj"}``.

    Returns:
        dict: The modified *result* (same object, mutated in place).
    """
    if not name_map:
        return result

    logger.info(f"Applying speaker names: {name_map}")
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            speaker = word.get("speaker", "")
            if speaker in name_map:
                word["speaker"] = name_map[speaker]

    return result
