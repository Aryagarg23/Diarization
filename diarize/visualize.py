"""
Emotion timeline visualisation module.

Generates a single PNG with one subplot per speaker, showing how both
vocal (audio) and textual emotion scores evolve over the course of a
conversation.  The x-axis is the segment mid-point timestamp, the y-axis
is the score magnitude (0 → 1), and each emotion dimension / category is
a separate coloured line.

Functions:
    plot_emotion_timeline() - Render and save the emotion timeline PNG.
"""

import logging
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")                       # headless backend for servers / CI
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
# Audio dimensions use solid, saturated colours.
# Text categories use a wider palette with slightly lower saturation.
AUDIO_COLOURS = {
    "arousal":   "#E63946",   # red
    "dominance": "#457B9D",   # steel blue
    "valence":   "#2A9D8F",   # teal
}

TEXT_COLOURS = {
    "anger":    "#D62828",    # dark red
    "disgust":  "#6A0572",    # purple
    "fear":     "#F77F00",    # orange
    "joy":      "#FCBF49",    # gold
    "neutral":  "#8D99AE",    # grey-blue
    "sadness":  "#264653",    # navy
    "surprise": "#E76F51",    # coral
}


def _fmt_time(seconds):
    """Convert seconds to *MM:SS* for tick labels."""
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def plot_emotion_timeline(emotion_results, output_path):
    """
    Render a multi-panel emotion timeline and save it as a PNG.

    Each speaker gets two vertically-stacked subplots:

    * **Voice** — arousal, dominance, valence (3 lines).
    * **Text**  — anger, disgust, fear, joy, neutral, sadness, surprise
      (7 lines).

    Args:
        emotion_results (list[dict]): Per-segment emotion data from
            :func:`diarize.emotions.analyze_emotions`.
        output_path (str): Destination file path (should end in ``.png``).
    """
    if not emotion_results:
        logger.warning("No emotion results to plot")
        return

    # -- Group data by speaker ---------------------------------------------
    speaker_data = defaultdict(lambda: {"times": [], "audio": defaultdict(list), "text": defaultdict(list)})

    for seg in emotion_results:
        spk = seg["speaker"]
        mid = (seg["start"] + seg["end"]) / 2.0
        speaker_data[spk]["times"].append(mid)
        for k, v in seg["audio_emotions"].items():
            speaker_data[spk]["audio"][k].append(v)
        for k, v in seg["text_emotions"].items():
            speaker_data[spk]["text"][k].append(v)

    speakers = sorted(speaker_data.keys())
    n_speakers = len(speakers)

    # -- Layout: 2 rows per speaker (voice + text) -------------------------
    n_rows = n_speakers * 2
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(14, 3.0 * n_rows),
        sharex=True,
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = [axes]

    fig.suptitle("Emotion Timeline by Speaker", fontsize=16, fontweight="bold")

    for spk_idx, spk in enumerate(speakers):
        data = speaker_data[spk]
        times = np.array(data["times"])
        sort_idx = np.argsort(times)
        times = times[sort_idx]

        # --- Voice subplot ------------------------------------------------
        ax_voice = axes[spk_idx * 2]
        for label, colour in AUDIO_COLOURS.items():
            vals = np.array(data["audio"][label])[sort_idx]
            ax_voice.plot(times, vals, colour, linewidth=1.2, alpha=0.85, label=label)

        ax_voice.set_ylabel("Score (0-1)", fontsize=9)
        ax_voice.set_ylim(-0.05, 1.05)
        ax_voice.set_title(f"{spk} — Voice (arousal / dominance / valence)", fontsize=11, loc="left")
        ax_voice.legend(loc="upper right", fontsize=8, ncol=3, framealpha=0.7)
        ax_voice.grid(True, alpha=0.25)

        # --- Text subplot -------------------------------------------------
        ax_text = axes[spk_idx * 2 + 1]
        for label, colour in TEXT_COLOURS.items():
            vals = np.array(data["text"][label])[sort_idx]
            ax_text.plot(times, vals, colour, linewidth=1.2, alpha=0.85, label=label)

        ax_text.set_ylabel("Score (0-1)", fontsize=9)
        ax_text.set_ylim(-0.05, 1.05)
        ax_text.set_title(f"{spk} — Text (categorical emotions)", fontsize=11, loc="left")
        ax_text.legend(loc="upper right", fontsize=8, ncol=4, framealpha=0.7)
        ax_text.grid(True, alpha=0.25)

    # -- Shared x-axis formatting ------------------------------------------
    axes[-1].set_xlabel("Time", fontsize=10)
    axes[-1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: _fmt_time(x)))

    # Save
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Emotion timeline saved to {output_path}")
