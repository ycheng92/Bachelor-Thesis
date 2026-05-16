"""
prepare_corpus.py
-----------------
Step 1 of the pipeline.

Loads the raw Hokkien corpus TSV, fills missing Tailo / POJ romanisation
using the taibun library, strips diacritics to produce normalised variants,
and writes the result to hokkien_corpus_final.csv.

Input:  hokkien_corpus.tsv
Output: hokkien_corpus_final.csv  (columns: chinese, tailo_raw, poj_raw,
                                             tailo_norm, poj_norm)
"""

import csv
import unicodedata

import pandas as pd
from taibun import Converter


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------

_converter_tailo = Converter()
_converter_poj = Converter(system="POJ")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fill_romanisation(df: pd.DataFrame) -> pd.DataFrame:
    """Fill empty Tailo / POJ cells by converting from Chinese characters."""

    def to_tailo(row):
        val = row.get("tailo", "")
        return val if isinstance(val, str) and val.strip() else _converter_tailo.get(row["chinese"])

    def to_poj(row):
        val = row.get("poj", "")
        return val if isinstance(val, str) and val.strip() else _converter_poj.get(row["chinese"])

    df["tailo_raw"] = df.apply(to_tailo, axis=1)
    df["poj_raw"] = df.apply(to_poj, axis=1)
    return df


def _strip_diacritics(text: str) -> str:
    """Remove combining diacritical marks (tone marks) from romanised text."""
    if pd.isna(text):
        return text
    normalised = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalised if unicodedata.category(ch) != "Mn")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = pd.read_csv("hokkien_corpus.tsv", sep="\t", encoding="utf-8")
    df.columns = df.columns.str.strip()

    df = _fill_romanisation(df)

    df["tailo_norm"] = df["tailo_raw"].apply(_strip_diacritics)
    df["poj_norm"] = df["poj_raw"].apply(_strip_diacritics)

    # Keep only the columns we actually need downstream
    out_cols = ["chinese", "tailo_raw", "poj_raw", "tailo_norm", "poj_norm"]
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv(
        "hokkien_corpus_final.csv",
        index=False,
        encoding="utf-8",
        quotechar='"',
        quoting=csv.QUOTE_ALL,
    )
    print("Done. Saved hokkien_corpus_final.csv")


if __name__ == "__main__":
    main()