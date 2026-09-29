"""Build CoNLL BIO datasets from CIMA section 4.8 raw text files.

Replaces the templated-sentence approach used by `src/ner_training.py` with a
real-text approach: for every drug `CODIGO` referenced in the train/test CSVs
this module reads `data/medicamentos/txt/{codigo}.txt`, tokenises it with
spaCy and tags the spans matching the CSV mention strings (one or several of
PACTIVO / REACADV / FRECUENCIA / SISTEMA) using a BIO scheme.

The resulting documents are kept in memory only — no CoNLL files are written
to disk.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from typing import Iterable, Optional

import pandas as pd

# Map the user-facing entity tag to the CSV column it is read from.
# Keep this list authoritative; other modules import it.
ENTITY_TYPE_TO_COLUMN: dict[str, str] = {
    "REACT": "REACADV",
    "ACTIVE": "PACTIVO",
    "FREQ": "FRECUENCIA",
    "SYS": "SISTEMA",
}

# Mirror of `ENTITY_TYPE_TO_COLUMN` accepting the lowercase column names used
# by `SupervisedTrainer.get_structured_data` so callers can pass either form.
_COLUMN_ALIASES: dict[str, list[str]] = {
    "REACT": ["REACADV", "reaccion_adversa"],
    "ACTIVE": ["PACTIVO", "principio_activo"],
    "FREQ": ["FRECUENCIA", "frecuencia"],
    "SYS": ["SISTEMA", "sistema"],
}

_CODIGO_ALIASES = ["CODIGO", "codigo"]

_FREQUENCY_SURFACE_FORMS = (
    ("frecuente", "frecuentes"),
    ("muy frecuente", "muy frecuentes"),
    ("poco frecuente", "poco frecuentes"),
)

logger = logging.getLogger(__name__)


def _strip_accents(text: str) -> str:
    """Return `text` with combining diacritical marks stripped."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _normalise(text: str) -> str:
    """Lower-case + accent-fold a string for accent-insensitive matching."""
    return _strip_accents(text).lower()


def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(
        f"None of the expected columns {candidates} are present in the "
        f"dataframe (have: {list(df.columns)})"
    )


class ConllBuilder:
    """Build per-drug CoNLL BIO documents from raw section-4.8 text."""

    def __init__(
        self,
        txt_dir: str,
        entity_types: list[str],
        spacy_model: str = "es_core_news_sm",
    ) -> None:
        if not entity_types:
            raise ValueError("`entity_types` must contain at least one tag")

        unknown = [t for t in entity_types if t not in ENTITY_TYPE_TO_COLUMN]
        if unknown:
            raise ValueError(
                f"Unknown entity type(s) {unknown}. Allowed values: "
                f"{sorted(ENTITY_TYPE_TO_COLUMN)}"
            )

        # Preserve user order while removing duplicates.
        seen: set[str] = set()
        self.entity_types: list[str] = []
        for t in entity_types:
            if t not in seen:
                self.entity_types.append(t)
                seen.add(t)

        self.txt_dir = os.path.realpath(txt_dir)
        self.spacy_model_name = spacy_model
        self._nlp = None  # lazily loaded
        self.label_to_id, self.id_to_label = self._build_label_vocab()

        # Stats populated by `build_dataset`.
        self.missing_txt_files: list[str] = []
        self.docs_with_no_tags: list[str] = []

    # ------------------------------------------------------------------ #
    # Vocab + spaCy
    # ------------------------------------------------------------------ #

    def _build_label_vocab(self) -> tuple[dict[str, int], dict[int, str]]:
        labels: list[str] = ["O"]
        for ent in self.entity_types:
            labels.append(f"B-{ent}")
            labels.append(f"I-{ent}")
        label_to_id = {lab: i for i, lab in enumerate(labels)}
        id_to_label = {i: lab for lab, i in label_to_id.items()}
        return label_to_id, id_to_label

    def _ensure_nlp(self):
        if self._nlp is None:
            try:
                import spacy
            except ImportError as exc:  # pragma: no cover - import-time failure
                raise ImportError(
                    "spaCy is required by ConllBuilder. Install it with "
                    "`pip install spacy` and download the Spanish model with "
                    "`python -m spacy download es_core_news_sm`."
                ) from exc

            try:
                self._nlp = spacy.load(
                    self.spacy_model_name,
                    disable=["tagger", "parser", "ner", "lemmatizer"],
                )
            except OSError as exc:
                raise OSError(
                    f"spaCy model '{self.spacy_model_name}' is not installed. "
                    f"Run: python -m spacy download {self.spacy_model_name}"
                ) from exc

            # Sentence segmentation: rule-based sentencizer (we disabled the
            # parser above, so without this `doc.sents` would not be available).
            if "sentencizer" not in self._nlp.pipe_names:
                self._nlp.add_pipe("sentencizer")
        return self._nlp

    # ------------------------------------------------------------------ #
    # Filesystem
    # ------------------------------------------------------------------ #

    def load_drug_text(self, codigo: str) -> str | None:
        """Read the section-4.8 raw text for `codigo` (treated as string)."""
        codigo_str = str(codigo).strip()
        if not codigo_str:
            return None
        path = os.path.join(self.txt_dir, f"{codigo_str}.txt")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # ------------------------------------------------------------------ #
    # BIO tagging
    # ------------------------------------------------------------------ #

    def _collect_mentions(
        self, rows: pd.DataFrame
    ) -> dict[str, list[str]]:
        """Return a dict mapping each entity tag to the set of mention strings
        for the rows of a single drug, sorted by descending length so longer
        matches win during char-level overwrite.

        FREQ includes singular/plural forms of the supported frequency
        categories without modifying the source annotations.
        """
        mentions: dict[str, list[str]] = {}
        for ent in self.entity_types:
            col = _resolve_column(rows, _COLUMN_ALIASES[ent])
            values = (
                rows[col]
                .dropna()
                .astype(str)
                .map(str.strip)
            )
            unique = {value for value in values if value}
            if ent == "FREQ":
                normalised_mentions = {_normalise(value) for value in unique}
                for forms in _FREQUENCY_SURFACE_FORMS:
                    if normalised_mentions.intersection(forms):
                        unique.update(forms)
            mentions[ent] = sorted(unique, key=len, reverse=True)
        return mentions

    @staticmethod
    def _find_all_spans(
        normalised_text: str, normalised_mention: str
    ) -> Iterable[tuple[int, int]]:
        """Yield (start, end) char offsets of every occurrence of
        `normalised_mention` inside `normalised_text`. Word-boundary aware so
        that short tokens like "ECG" do not match inside longer words."""
        if not normalised_mention:
            return
        # `re.escape` so any punctuation in the mention is treated literally.
        pattern = re.compile(
            r"(?<![\w\u00c0-\u017f])"
            + re.escape(normalised_mention)
            + r"(?![\w\u00c0-\u017f])"
        )
        for match in pattern.finditer(normalised_text):
            yield match.start(), match.end()

    def _split_into_sentences(self, doc) -> list[list]:
        """Group spaCy `Token`s into sentences using `doc.sents` (sentencizer)
        and additionally hard-break on newlines so each line of the original
        text is its own sentence."""
        sentences: list[list] = []
        for sent in doc.sents:
            current: list = []
            for tok in sent:
                if tok.is_space:
                    # A whitespace token containing a newline forces a break.
                    if current and "\n" in tok.text:
                        sentences.append(current)
                        current = []
                    continue
                # If the previous token ended on a different line than this
                # one starts, treat that as a sentence break too.
                if current:
                    prev = current[-1]
                    gap = tok.doc.text[prev.idx + len(prev.text) : tok.idx]
                    if "\n" in gap:
                        sentences.append(current)
                        current = []
                current.append(tok)
            if current:
                sentences.append(current)
        return sentences

    def _tag_text(
        self,
        text: str,
        mentions_by_type: dict[str, list[str]],
    ) -> list[tuple[list[str], list[str]]]:
        """Tokenise `text` with spaCy, split it into sentences and return one
        ``(tokens, bio_labels)`` pair per sentence."""
        nlp = self._ensure_nlp()
        normalised_text = _normalise(text)

        # `char_labels[i]` holds the BIO tag for the character at index i.
        # Default O. Longest mentions are written first so a shorter mention
        # cannot overwrite a longer one that already contains it.
        char_labels: list[str] = ["O"] * len(text)
        for ent in self.entity_types:
            for mention in mentions_by_type.get(ent, []):
                norm_mention = _normalise(mention)
                for start, end in self._find_all_spans(normalised_text, norm_mention):
                    # Skip if any char in this span has already been tagged
                    # (longer-first ordering means it was tagged by a longer
                    # mention of the same or another type).
                    if any(char_labels[i] != "O" for i in range(start, end)):
                        continue
                    char_labels[start] = f"B-{ent}"
                    for i in range(start + 1, end):
                        char_labels[i] = f"I-{ent}"

        # Tokenise + sentence-segment with spaCy.
        doc = nlp(text)
        sentences: list[tuple[list[str], list[str]]] = []
        for sent_tokens in self._split_into_sentences(doc):
            tokens: list[str] = []
            bio_labels: list[str] = []
            for tok in sent_tokens:
                start = tok.idx
                tag = char_labels[start] if start < len(char_labels) else "O"
                # Promote mid-token "I-X" to "B-X" if the previous token in
                # this sentence was not tagged with the same entity.
                if tag.startswith("I-"):
                    ent = tag[2:]
                    if not bio_labels or bio_labels[-1] not in (f"B-{ent}", f"I-{ent}"):
                        tag = f"B-{ent}"
                tokens.append(tok.text)
                bio_labels.append(tag)
            if tokens:
                sentences.append((tokens, bio_labels))

        return sentences

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def build_dataset(self, df: pd.DataFrame) -> list[dict]:
        """Group `df` by `CODIGO`, tokenise each drug's section-4.8 text and
        emit **one example per sentence**.

        Returns a list of dicts:
            {"codigo": str, "sent_idx": int,
             "tokens": [...], "ner_tags": [int, ...]}
        Drugs whose .txt file is missing are skipped (and recorded in
        `self.missing_txt_files`).
        """
        codigo_col = _resolve_column(df, _CODIGO_ALIASES)
        # Reset stats for this run.
        self.missing_txt_files = []
        self.docs_with_no_tags = []

        examples: list[dict] = []
        for codigo, rows in df.groupby(codigo_col, sort=False):
            codigo_str = str(codigo).strip()
            text = self.load_drug_text(codigo_str)
            if text is None:
                self.missing_txt_files.append(codigo_str)
                continue
            mentions = self._collect_mentions(rows)
            sentences = self._tag_text(text, mentions)
            if not sentences:
                continue
            doc_has_tag = False
            for sent_idx, (tokens, bio_labels) in enumerate(sentences):
                if any(lab != "O" for lab in bio_labels):
                    doc_has_tag = True
                examples.append(
                    {
                        "codigo": codigo_str,
                        "sent_idx": sent_idx,
                        "tokens": tokens,
                        "ner_tags": [self.label_to_id[lab] for lab in bio_labels],
                    }
                )
            if not doc_has_tag:
                self.docs_with_no_tags.append(codigo_str)
        return examples

    def summarise(self, examples: list[dict]) -> dict:
        """Return a small statistics dict useful for logging."""
        from collections import Counter

        label_counts: Counter[str] = Counter()
        token_count = 0
        unique_codigos: set[str] = set()
        for ex in examples:
            token_count += len(ex["tokens"])
            label_counts.update(self.id_to_label[i] for i in ex["ner_tags"])
            unique_codigos.add(ex["codigo"])
        return {
            "documents": len(unique_codigos),
            "sentences": len(examples),
            "tokens": token_count,
            "label_counts": dict(label_counts),
            "missing_txt_files": list(self.missing_txt_files),
            "docs_with_no_tags": list(self.docs_with_no_tags),
        }


def sanitise_bio_predictions(labels: list[str]) -> list[str]:
    """Repair invalid IOB2 prediction sequences.

    Token-classification heads can emit ``I-X`` tags that are not preceded
    by a matching ``B-X``/``I-X`` (or by a different entity type). Such
    sequences are illegal under IOB2 and seqeval/nervaluate count them as
    extra entities, inflating the spurious bucket. This helper rewrites
    every orphan ``I-X`` to ``B-X`` so the prediction sequence is always a
    well-formed IOB2 string.

    Examples
    --------
    >>> sanitise_bio_predictions(["I-REACT", "O", "I-REACT", "I-REACT"])
    ['B-REACT', 'O', 'B-REACT', 'I-REACT']
    >>> sanitise_bio_predictions(["B-REACT", "I-FREQ"])
    ['B-REACT', 'B-FREQ']
    """
    repaired: list[str] = []
    prev_entity: Optional[str] = None
    for label in labels:
        if label.startswith("I-"):
            entity = label[2:]
            if prev_entity != entity:
                label = f"B-{entity}"
            prev_entity = entity
        elif label.startswith("B-"):
            prev_entity = label[2:]
        else:
            prev_entity = None
        repaired.append(label)
    return repaired
