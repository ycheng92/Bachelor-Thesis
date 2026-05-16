"""
tokenise.py
-----------
Step 2 of the pipeline.

Applies two tokenisation strategies to the Hokkien corpus:
  - Baseline: simple whitespace tokenisation (lowercased)
  - mBERT:    bert-base-multilingual-cased subword tokenisation

Computes per-sentence token counts, vocabulary sizes, fragmentation ratios,
and word-level split rates for all four text variants:
  tailo_raw, poj_raw, tailo_norm, poj_norm

Input:  hokkien_corpus_final.csv   (output of prepare_corpus.py)
Output: hokkien_corpus_tokenised.csv
        summary statistics printed to stdout
"""

import csv
import ast

import pandas as pd
from transformers import AutoTokenizer


MBERT_MODEL = "bert-base-multilingual-cased"

VARIANTS = ["tailo_raw", "poj_raw", "tailo_norm", "poj_norm"]


# ---------------------------------------------------------------------------
# Baseline tokenisation
# ---------------------------------------------------------------------------

def baseline_tokenise(text: str) -> list[str]:
    if pd.isna(text):
        return []
    return [t for t in text.lower().split() if t]


def vocab_size(token_lists) -> int:
    return len({token for sent in token_lists for token in sent})


# ---------------------------------------------------------------------------
# mBERT tokenisation
# ---------------------------------------------------------------------------

def build_tokeniser():
    return AutoTokenizer.from_pretrained(MBERT_MODEL)


def split_rate(tokeniser, token_lists) -> float:
    """Fraction of baseline words that mBERT splits into >1 subword."""
    total = split = 0
    for sent in token_lists:
        for word in sent:
            total += 1
            if len(tokeniser.tokenize(word)) > 1:
                split += 1
    return split / total if total else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = pd.read_csv("hokkien_corpus_final.csv")
    tokeniser = build_tokeniser()

    for v in VARIANTS:
        # Baseline
        df[f"{v}_tokens"] = df[v].apply(baseline_tokenise)
        df[f"{v}_token_count"] = df[f"{v}_tokens"].apply(len)

        # mBERT
        df[f"{v}_bert_tokens"] = df[v].apply(
            lambda t: tokeniser.tokenize(str(t)) if not pd.isna(t) else []
        )
        df[f"{v}_bert_token_count"] = df[f"{v}_bert_tokens"].apply(len)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    print(f"\n{'Variant':<20} {'Vocab':>8} {'Avg base':>10} {'Avg mBERT':>10} {'Frag ratio':>12} {'Split %':>9}")
    print("-" * 75)

    for v in VARIANTS:
        v_size = vocab_size(df[f"{v}_tokens"])
        avg_base = df[f"{v}_token_count"].mean()
        avg_bert = df[f"{v}_bert_token_count"].mean()
        frag = df[f"{v}_bert_token_count"].sum() / df[f"{v}_token_count"].sum()
        split_pct = split_rate(tokeniser, df[f"{v}_tokens"]) * 100
        print(f"{v:<20} {v_size:>8} {avg_base:>10.2f} {avg_bert:>10.2f} {frag:>12.4f} {split_pct:>8.1f}%")

    print()

    # Vocabulary overlap (raw vs norm, within each system)
    for system in ["tailo", "poj"]:
        raw_vocab = {t for sent in df[f"{system}_raw_tokens"] for t in sent}
        norm_vocab = {t for sent in df[f"{system}_norm_tokens"] for t in sent}
        reduction = (len(raw_vocab) - len(norm_vocab)) / len(raw_vocab) * 100
        print(f"{system.upper()} vocab reduction after normalisation: {reduction:.1f}%")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    df.to_csv("hokkien_corpus_tokenised.csv", index=False, quoting=csv.QUOTE_ALL)
    print("\nDone. Saved hokkien_corpus_tokenised.csv")


if __name__ == "__main__":
    main()