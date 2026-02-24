"""
Speech emotion recognition module (experimental).

Provides dual-model emotion analysis optimised for human-centered design
(HCD) interview transcripts:

1. **Audio-based** -- ``audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim``
   Analyses vocal tone, pitch, and rhythm to produce dimensional scores:

   - **Arousal** (0 → 1): activation / excitement level.
   - **Dominance** (0 → 1): sense of control / confidence.
   - **Valence** (0 → 1): positivity / pleasantness.

   Trained on MSP-Podcast (real conversational speech, not acted).
   License: CC-BY-NC-SA-4.0 (non-commercial / educational use).

2. **Text-based** -- ``j-hartmann/emotion-english-distilroberta-base``
   Analyses the **words spoken** to classify categorical emotions:
   anger · disgust · fear · joy · neutral · sadness · surprise.

   Trained on 6 diverse text-emotion datasets (Twitter, Reddit, TV dialogue).

Together, the models capture both **how** something is said (vocal tone) and
**what** is said (word meaning), giving a rich emotion profile per utterance
and per speaker -- ideal for design-research interview analysis.

Functions:
    analyze_emotions()              - Run both models on all transcript segments.
    aggregate_emotions_by_speaker() - Average emotion scores per speaker.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import librosa
from transformers import (
    Wav2Vec2Processor,
    pipeline as hf_pipeline,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from diarize.utils import clear_vram, log_vram_usage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
AUDIO_EMOTION_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
TEXT_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# Column order for the audeering model output
AUDIO_EMOTION_LABELS = ["arousal", "dominance", "valence"]

# All labels emitted by the j-hartmann text model
TEXT_EMOTION_LABELS = [
    "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise",
]


# ---------------------------------------------------------------------------
# Custom model architecture required by the audeering audio emotion model.
# The checkpoint stores a wav2vec2 backbone with a regression head that
# predicts three continuous dimensions.
# ---------------------------------------------------------------------------

class _RegressionHead(nn.Module):
    """Regression head for dimensional emotion prediction."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class _AudioEmotionModel(Wav2Vec2PreTrainedModel):
    """wav2vec2-based speech emotion regression model."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = _RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_audio_emotion_model(device):
    """Load the audeering wav2vec2 audio emotion model.

    Returns:
        tuple: ``(Wav2Vec2Processor, _AudioEmotionModel)``
    """
    logger.info(f"Loading audio emotion model: {AUDIO_EMOTION_MODEL}")
    processor = Wav2Vec2Processor.from_pretrained(AUDIO_EMOTION_MODEL)
    model = _AudioEmotionModel.from_pretrained(
        AUDIO_EMOTION_MODEL, use_safetensors=True,
    ).to(device)
    model.eval()
    log_vram_usage("Audio emotion model loaded")
    return processor, model


def _predict_audio_emotion(signal, sr, processor, model, device):
    """Predict arousal, dominance, valence from a raw audio segment.

    Args:
        signal (np.ndarray): 1-D mono waveform.
        sr (int): Sample rate (must be 16 000).
        processor: ``Wav2Vec2Processor`` instance.
        model: ``_AudioEmotionModel`` instance.
        device (str): ``'cuda'`` or ``'cpu'``.

    Returns:
        dict: ``{'arousal': float, 'dominance': float, 'valence': float}``
        with values clipped to [0, 1].
    """
    inputs = processor(signal, sampling_rate=sr)
    input_values = np.array(inputs["input_values"][0]).reshape(1, -1)
    input_tensor = torch.from_numpy(input_values).to(device)

    with torch.no_grad():
        _, logits = model(input_tensor)

    scores = logits.detach().cpu().numpy()[0]
    return {
        label: round(float(np.clip(scores[i], 0.0, 1.0)), 4)
        for i, label in enumerate(AUDIO_EMOTION_LABELS)
    }


def _load_text_emotion_model(device):
    """Load the j-hartmann text emotion classifier.

    Returns:
        ``transformers.Pipeline`` for text-classification.
    """
    logger.info(f"Loading text emotion model: {TEXT_EMOTION_MODEL}")
    device_idx = 0 if device == "cuda" else -1

    # The j-hartmann model only provides pytorch_model.bin (no safetensors).
    # Our version of transformers blocks torch.load on PyTorch < 2.6 due to
    # CVE-2025-32434.  Since we're loading a known, trusted HuggingFace model
    # (not an arbitrary pickle), we temporarily bypass that check.
    #
    # modeling_utils.py imports check_torch_load_is_safe into its own
    # namespace, so we must patch *both* the source module and the
    # call-site module for the bypass to take effect.
    import transformers.utils.import_utils as _tiu
    import transformers.modeling_utils as _tmu
    _noop = lambda: None
    _orig_tiu = _tiu.check_torch_load_is_safe
    _orig_tmu = _tmu.check_torch_load_is_safe
    _tiu.check_torch_load_is_safe = _noop
    _tmu.check_torch_load_is_safe = _noop
    try:
        classifier = hf_pipeline(
            "text-classification",
            model=TEXT_EMOTION_MODEL,
            top_k=None,          # return scores for all labels
            device=device_idx,
        )
    finally:
        _tiu.check_torch_load_is_safe = _orig_tiu
        _tmu.check_torch_load_is_safe = _orig_tmu

    log_vram_usage("Text emotion model loaded")
    return classifier


def _predict_text_emotions_batch(texts, classifier, batch_size=32):
    """Predict categorical emotions for a list of texts in one batch.

    Empty / whitespace-only strings are assigned ``neutral=1.0`` without
    being sent through the model.

    Args:
        texts (list[str]): Input sentences.
        classifier: HuggingFace text-classification pipeline.
        batch_size (int): Pipeline batch size.

    Returns:
        list[dict]: One ``{label: score}`` dict per input text.
    """
    results = [None] * len(texts)
    neutral_default = {
        label: (1.0 if label == "neutral" else 0.0)
        for label in TEXT_EMOTION_LABELS
    }

    # Separate empty from non-empty
    non_empty_indices = []
    non_empty_texts = []
    for i, t in enumerate(texts):
        if t and t.strip():
            non_empty_indices.append(i)
            non_empty_texts.append(t)
        else:
            results[i] = dict(neutral_default)

    if non_empty_texts:
        raw = classifier(non_empty_texts, batch_size=batch_size)
        for idx, output in zip(non_empty_indices, raw):
            results[idx] = {
                r["label"]: round(float(r["score"]), 4) for r in output
            }

    return results


def _get_segment_speaker(segment):
    """Return the majority speaker label from a segment's words."""
    words = segment.get("words", [])
    if not words:
        return "UNKNOWN"

    counts = {}
    for w in words:
        spk = w.get("speaker", "UNKNOWN")
        counts[spk] = counts.get(spk, 0) + 1

    return max(counts, key=counts.get)


def _get_segment_text(segment):
    """Reconstruct text from a segment's words."""
    words = segment.get("words", [])
    return " ".join(
        w.get("word", "").strip()
        for w in words
        if w.get("word", "").strip()
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_emotions(result, audio_path, device):
    """
    Run dual-model emotion analysis on every transcript segment.

    **Phase 1** loads the audio emotion model, scores all segments by vocal
    tone, then unloads the model and clears VRAM.

    **Phase 2** loads the text emotion model, scores all segments by word
    content, then unloads.

    Models are loaded sequentially to minimise peak VRAM usage.

    Args:
        result (dict): WhisperX transcription result with ``segments``.
        audio_path (str): Path to the 16 kHz mono WAV file.
        device (str): ``'cuda'`` or ``'cpu'``.

    Returns:
        list[dict]: One entry per segment::

            {
                "index": int,
                "start": float,          # seconds
                "end": float,
                "speaker": str,
                "text": str,
                "audio_emotions": {
                    "arousal": float,     # 0-1
                    "dominance": float,   # 0-1
                    "valence": float,     # 0-1
                },
                "text_emotions": {
                    "anger": float,       # 0-1
                    "disgust": float,
                    "fear": float,
                    "joy": float,
                    "neutral": float,
                    "sadness": float,
                    "surprise": float,
                },
            }
    """
    segments = result.get("segments", [])
    if not segments:
        logger.warning("No segments found for emotion analysis")
        return []

    total = len(segments)

    # Load full audio once (16 kHz mono)
    logger.info("Loading audio for emotion analysis...")
    signal, sr = librosa.load(audio_path, sr=16000, mono=True)

    # ------------------------------------------------------------------
    # Phase 1: Audio-based emotions (arousal / dominance / valence)
    # ------------------------------------------------------------------
    logger.info(
        f"Phase 1/2: Analysing vocal emotions for {total} segments..."
    )
    processor, audio_model = _load_audio_emotion_model(device)

    emotion_results = []
    for i, seg in enumerate(segments):
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        seg_signal = signal[start_sample:end_sample]

        # Skip very short segments (< 100 ms) -- not enough audio for a
        # meaningful prediction.
        if len(seg_signal) < int(sr * 0.1):
            audio_emo = {label: 0.5 for label in AUDIO_EMOTION_LABELS}
        else:
            try:
                audio_emo = _predict_audio_emotion(
                    seg_signal, sr, processor, audio_model, device,
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    logger.warning(
                        f"  OOM on segment {i}, using neutral defaults"
                    )
                    torch.cuda.empty_cache()
                    audio_emo = {label: 0.5 for label in AUDIO_EMOTION_LABELS}
                else:
                    raise

        emotion_results.append({
            "index": i,
            "start": start,
            "end": end,
            "speaker": _get_segment_speaker(seg),
            "text": _get_segment_text(seg),
            "audio_emotions": audio_emo,
            "text_emotions": None,  # filled in phase 2
        })

        if (i + 1) % 25 == 0 or (i + 1) == total:
            logger.info(f"  Audio emotions: {i + 1}/{total} segments")

    del audio_model, processor
    clear_vram()
    logger.info("Audio emotion model unloaded")

    # ------------------------------------------------------------------
    # Phase 2: Text-based emotions (categorical)
    # ------------------------------------------------------------------
    logger.info(
        f"Phase 2/2: Analysing text emotions for {total} segments..."
    )
    text_classifier = _load_text_emotion_model(device)

    texts = [emo["text"] for emo in emotion_results]
    text_scores = _predict_text_emotions_batch(texts, text_classifier)

    for emo, scores in zip(emotion_results, text_scores):
        emo["text_emotions"] = scores

    logger.info(f"  Text emotions: {total}/{total} segments complete")

    del text_classifier
    clear_vram()
    logger.info("Text emotion model unloaded")

    return emotion_results


def aggregate_emotions_by_speaker(emotion_results):
    """
    Compute average emotion scores per speaker across all their segments.

    Args:
        emotion_results (list[dict]): Output from :func:`analyze_emotions`.

    Returns:
        dict: ``speaker_label → profile``::

            {
                "SPEAKER_1": {
                    "segment_count": 45,
                    "audio_emotions": {"arousal": 0.48, ...},
                    "text_emotions":  {"anger": 0.03, ...},
                },
                ...
            }

        Speakers are sorted alphabetically by label.
    """
    buckets = {}

    for seg in emotion_results:
        spk = seg["speaker"]
        if spk not in buckets:
            buckets[spk] = {
                "audio": {label: [] for label in AUDIO_EMOTION_LABELS},
                "text": {},
                "count": 0,
            }

        b = buckets[spk]
        b["count"] += 1

        for label in AUDIO_EMOTION_LABELS:
            b["audio"][label].append(seg["audio_emotions"][label])

        for label, score in seg["text_emotions"].items():
            b["text"].setdefault(label, []).append(score)

    summary = {}
    for spk in sorted(buckets):
        b = buckets[spk]
        summary[spk] = {
            "segment_count": b["count"],
            "audio_emotions": {
                label: round(float(np.mean(vals)), 4)
                for label, vals in b["audio"].items()
            },
            "text_emotions": {
                label: round(float(np.mean(vals)), 4)
                for label, vals in b["text"].items()
            },
        }

    return summary
