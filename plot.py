"""
plot.py
-------
Step 3 of the pipeline.

Reads the tokenised corpus and produces two figures:
  fig_vocab_sizes.png     — vocabulary size per system (Tailo vs POJ),
                            grouped by raw / normalised, y-axis from 800
  fig_fragmentation.png   — mBERT fragmentation ratio per variant,
                            colour-coded by system (Tailo=blue, POJ=orange)

Input:  hokkien_corpus_tokenized.csv  (output of tokenize.py)
Output: fig_vocab_sizes.png
        fig_fragmentation.png

Note: if the CSV already contains *_bert_token_count columns (produced by
tokenize.py), those are used directly. Otherwise, mBERT counts are computed
on the fly — this adds ~1 min but avoids a hard dependency on tokenize.py.
"""

import ast

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer


MBERT_MODEL = "bert-base-multilingual-cased"
VARIANTS    = ["tailo_raw", "poj_raw", "tailo_norm", "poj_norm"]

# ---------------------------------------------------------------------------
# Colour palette & shared style
# ---------------------------------------------------------------------------

BLUE   = "#2563EB"
ORANGE = "#F97316"
GREY   = "#6B7280"
BG     = "#FAFAFA"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.8,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_token_list(x) -> list:
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return []


def vocab_size(token_lists) -> int:
    return len({token for sent in token_lists for token in sent})


def get_bert_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Add *_bert_token_count columns if not already present."""
    needed = [f"{v}_bert_token_count" for v in VARIANTS]
    if all(c in df.columns for c in needed):
        return df

    print("bert_token_count columns not found — computing with mBERT (this may take a minute)...")
    tokenizer = AutoTokenizer.from_pretrained(MBERT_MODEL)

    def bert_count(text):
        return len(tokenizer.tokenize(str(text))) if not pd.isna(text) else 0

    for v in VARIANTS:
        df[f"{v}_bert_token_count"] = df[v].apply(bert_count)

    return df


def label_bars(ax, bars, values, fmt="{}", offset=1):
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            fmt.format(val),
            ha="center", va="bottom",
            fontsize=11, color=GREY,
        )


# ---------------------------------------------------------------------------
# Figure 1 — Vocabulary size
# ---------------------------------------------------------------------------

def plot_vocab(tailo_vals: list, poj_vals: list, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.array([0, 1])
    w = 0.35

    b1 = ax.bar(x - w / 2, tailo_vals, w, label="Tailo", color=BLUE,   zorder=3)
    b2 = ax.bar(x + w / 2, poj_vals,   w, label="POJ",   color=ORANGE, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(["Raw", "Normalised"], fontsize=12)
    ax.set_ylim(800, 970)
    ax.set_ylabel("Unique baseline tokens", fontsize=12)
    ax.set_title("Vocabulary size by system and normalisation", fontsize=13, pad=12)
    ax.legend(frameon=False, fontsize=11)

    label_bars(ax, b1, tailo_vals, offset=1)
    label_bars(ax, b2, poj_vals,   offset=1)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 2 — Fragmentation ratio
# ---------------------------------------------------------------------------

def plot_fragmentation(frag_vals: list, path: str) -> None:
    labels = ["Tailo\n(raw)", "POJ\n(raw)", "Tailo\n(norm)", "POJ\n(norm)"]
    colors = [BLUE, ORANGE, BLUE,  ORANGE]
    alphas = [1.0,  1.0,    0.55,  0.55  ]

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(labels, frag_vals, width=0.55, zorder=3)
    for bar, color, alpha in zip(bars, colors, alphas):
        bar.set_color(color)
        bar.set_alpha(alpha)

    ax.set_ylim(0, 4.5)
    ax.set_ylabel("mBERT subwords / baseline tokens", fontsize=12)
    ax.set_title("mBERT fragmentation ratio", fontsize=13, pad=12)

    label_bars(ax, bars, frag_vals, fmt="{:.2f}", offset=0.04)

    legend_els = [
        mpatches.Patch(color=BLUE,   label="Tailo"),
        mpatches.Patch(color=ORANGE, label="POJ"),
    ]
    ax.legend(handles=legend_els, frameon=False, fontsize=11)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = pd.read_csv("hokkien_corpus_tokenised.csv")

    for v in VARIANTS:
        df[f"{v}_tokens"] = df[f"{v}_tokens"].apply(parse_token_list)

    df = get_bert_counts(df)

    # Vocabulary sizes
    tailo_raw_vocab  = vocab_size(df["tailo_raw_tokens"])
    poj_raw_vocab    = vocab_size(df["poj_raw_tokens"])
    tailo_norm_vocab = vocab_size(df["tailo_norm_tokens"])
    poj_norm_vocab   = vocab_size(df["poj_norm_tokens"])

    plot_vocab(
        tailo_vals=[tailo_raw_vocab, tailo_norm_vocab],
        poj_vals=[poj_raw_vocab, poj_norm_vocab],
        path="fig_vocab_sizes.png",
    )

    # Fragmentation ratios
    frag_vals = [
        df[f"{v}_bert_token_count"].sum() / df[f"{v}_token_count"].sum()
        for v in VARIANTS
    ]
    plot_fragmentation(frag_vals, path="fig_fragmentation.png")

    print("\nVocabulary sizes:")
    print(f"  Tailo  raw={tailo_raw_vocab},  norm={tailo_norm_vocab}")
    print(f"  POJ    raw={poj_raw_vocab},    norm={poj_norm_vocab}")
    print("\nFragmentation ratios:", dict(zip(
        ["tailo_raw", "poj_raw", "tailo_norm", "poj_norm"],
        [round(f, 4) for f in frag_vals]
    )))


if __name__ == "__main__":
    main()