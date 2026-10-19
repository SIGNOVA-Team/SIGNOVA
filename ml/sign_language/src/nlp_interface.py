"""
SIGNOVA — NLP Bridge Interface (Gloss-to-Text Translation).

Provides seamless translation and post-processing between:
- Machine Learning predicted sign gloss sequences (ArSL / ASL)
- Natural Spoken Arabic (Modern Standard Arabic - MSA)
- English translations and LLM/Seq2Seq conditioning prompts.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

import pandas as pd


class GlossToTextTranslator:
    """Translates continuous sign gloss sequences into natural spoken sentences."""

    def __init__(self, mapping_csv_path: str | Path | None = None):
        root_dir = Path(__file__).resolve().parents[3]
        if mapping_csv_path is None:
            mapping_csv_path = root_dir / "data/arabsign/metadata/arabsign_sentences_mapping.csv"

        self.mapping_path = Path(mapping_csv_path)
        self.gloss_map: dict[str, dict[str, Any]] = {}
        self.known_gloss_keys: list[str] = []

        if self.mapping_path.exists():
            self._load_mappings()

    def _load_mappings(self) -> None:
        """Load sentence mapping from CSV into fast lookup tables."""
        df = pd.read_csv(self.mapping_path)
        for _, row in df.iterrows():
            gloss_key = " ".join(str(row["arsl_gloss_structure"]).strip().split())
            entry = {
                "sentence_id": int(row["sentence_id"]),
                "arsl_gloss_structure": gloss_key,
                "spoken_arabic_standard": str(row["spoken_arabic_standard"]).strip(),
                "english_translation": str(row["english_translation"]).strip(),
                "domain_category": str(row.get("domain_category", "General")),
            }
            self.gloss_map[gloss_key] = entry
            self.known_gloss_keys.append(gloss_key)

    def translate_arabic_glosses(
        self,
        gloss_tokens: list[str] | str,
        fuzzy_cutoff: float = 0.55,
    ) -> dict[str, Any]:
        """
        Translate ArSL continuous gloss sequence into standard spoken Arabic.

        Args:
            gloss_tokens: List of recognized gloss strings or a single space-separated string.
            fuzzy_cutoff: Minimum similarity ratio for fuzzy matching (0.0 to 1.0).

        Returns:
            Dictionary containing:
            - spoken_arabic: Grammatical Modern Standard Arabic text.
            - english_translation: Corresponding English sentence.
            - match_type: 'exact', 'fuzzy', or 'gloss_fallback'.
            - confidence_score: Confidence of the match.
            - domain_category: Topical category of the utterance.
        """
        if isinstance(gloss_tokens, list):
            gloss_str = " ".join(t.strip() for t in gloss_tokens if t.strip())
        else:
            gloss_str = " ".join(str(gloss_tokens).strip().split())

        if not gloss_str:
            return {
                "spoken_arabic": "",
                "english_translation": "",
                "match_type": "empty_input",
                "confidence_score": 0.0,
                "domain_category": "None",
            }

        # 1. Exact match
        if gloss_str in self.gloss_map:
            entry = self.gloss_map[gloss_str]
            return {
                "spoken_arabic": entry["spoken_arabic_standard"],
                "english_translation": entry["english_translation"],
                "match_type": "exact",
                "confidence_score": 1.0,
                "domain_category": entry["domain_category"],
                "sentence_id": entry["sentence_id"],
            }

        # 2. Fuzzy match via sequence similarity
        if self.known_gloss_keys:
            matches = difflib.get_close_matches(
                gloss_str, self.known_gloss_keys, n=1, cutoff=fuzzy_cutoff
            )
            if matches:
                best_key = matches[0]
                ratio = difflib.SequenceMatcher(None, gloss_str, best_key).ratio()
                entry = self.gloss_map[best_key]
                return {
                    "spoken_arabic": entry["spoken_arabic_standard"],
                    "english_translation": entry["english_translation"],
                    "match_type": "fuzzy",
                    "confidence_score": round(ratio, 4),
                    "domain_category": entry["domain_category"],
                    "matched_canonical_gloss": best_key,
                    "sentence_id": entry["sentence_id"],
                }

        # 3. Fallback: return raw gloss as surface text
        return {
            "spoken_arabic": gloss_str,
            "english_translation": "",
            "match_type": "gloss_fallback",
            "confidence_score": 0.5,
            "domain_category": "Unmapped",
        }

    def generate_nlp_prompt(self, gloss_tokens: list[str] | str, target_lang: str = "ar") -> str:
        """
        Generate structured conditioning prompts for generative NLP models (e.g. mT5, CAMeL, LLMs).

        Args:
            gloss_tokens: Sequence of sign glosses.
            target_lang: 'ar' for Modern Standard Arabic or 'en' for English.

        Returns:
            Formatted prompt string ready for LLM/Seq2Seq inference.
        """
        if isinstance(gloss_tokens, list):
            gloss_str = " ".join(gloss_tokens)
        else:
            gloss_str = str(gloss_tokens)

        if target_lang == "ar":
            return (
                f"حول متتالية إشارات لغة الإشارة العربية (ArSL Glosses) التالية "
                f"إلى جملة عربية فصيحة ومترابطة نحوياً ودلالياً:\n"
                f"الإشارات: {gloss_str}\n"
                f"الجملة الفصيحة:"
            )
        return (
            f"Translate the following sign language gloss sequence into natural, grammatical English:\n"
            f"Glosses: {gloss_str}\n"
            f"Translation:"
        )


_DEFAULT_TRANSLATOR: GlossToTextTranslator | None = None


def get_translator() -> GlossToTextTranslator:
    """Get or create singleton GlossToTextTranslator."""
    global _DEFAULT_TRANSLATOR
    if _DEFAULT_TRANSLATOR is None:
        _DEFAULT_TRANSLATOR = GlossToTextTranslator()
    return _DEFAULT_TRANSLATOR
