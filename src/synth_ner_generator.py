"""Synthetic NER dataset generator (Variante B) for the CIMA pipeline.

This module orchestrates the LLM calls that produce, for every drug
``CODIGO`` in the input annotations table, one or more synthetic section-4.8
documents whose entities can be re-located deterministically by the existing
:class:`src.conll_ner_builder.ConllBuilder` matcher.

Architecture in one paragraph
-----------------------------
The LLM is asked (see :mod:`src.synth_prompt_templates`) for a fenced JSON
``{"text": ..., "entities": [...]}`` where each entity declares its ``type``,
``canonical`` value (taken verbatim from ``full_annotations.csv``) and
``surface`` — the substring as it actually appears in ``text``. We then
re-run the same word-boundary regex used by Variante A
(:func:`src.conll_ner_builder.ConllBuilder._find_all_spans`) over the
generated text to confirm that every declared ``surface`` actually matches.
Entities whose ``surface`` cannot be located are silently dropped; if too
many are dropped (> ``max_unmatched_ratio``) the whole document is
considered invalid and a regeneration is attempted (up to
``max_retries`` times).

Outputs
-------
For each successful (codigo, variant) pair we write:

* ``{out_dir}/txt/{codigo}_v{n}.txt``         — synthetic document (long doc
  + concatenated snippets), suitable as ``txt_dir`` for a fresh
  :class:`ConllBuilder` instantiation.
* ``{out_dir}/raw_responses/{codigo}_v{n}.json`` — raw LLM payload (text,
  entities, metadata) for traceability.
* ``{out_dir}/entities/{codigo}_v{n}.json`` — validated entities only.

When the full corpus is processed (CLI :mod:`run_synth_dataset`), a
companion ``full_annotations_synth.csv`` is consolidated downstream so the
existing trainer / tester can consume the synthetic corpus unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from src.conll_ner_builder import (
    ENTITY_TYPE_TO_COLUMN,
    ConllBuilder,
    _normalise,
)

# Re-export the word-boundary span finder used by Variante A's matcher so the
# synthetic validator and the trainer agree on which substrings count as a
# match (otherwise we could declare an entity here that the trainer's
# ConllBuilder would silently fail to tag).
_find_all_spans = ConllBuilder._find_all_spans
from src.synth_prompt_templates import (
    build_long_doc_messages,
    build_snippet_messages,
    select_snippet_tuples,
)

logger = logging.getLogger(__name__)


VALID_ENTITY_TYPES = set(ENTITY_TYPE_TO_COLUMN.keys())  # {"REACT","ACTIVE","FREQ","SYS"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ValidatedEntity:
    """A surface mention confirmed to be present in the synthetic text."""

    type: str
    canonical: str
    surface: str
    start: int
    end: int

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "canonical": self.canonical,
            "surface": self.surface,
            "start": self.start,
            "end": self.end,
        }


@dataclass
class GeneratedDocument:
    """Outcome of generating a single (codigo, variante) document."""

    codigo: str  # original CODIGO (not the variant-suffixed one)
    variante: int
    codigo_variant: str  # f"{codigo}_v{variante}"
    medicamento: str
    text: str
    entities: list[ValidatedEntity]
    raw_text: str  # text returned by the LLM (long doc only)
    raw_entities: list[dict]  # entities exactly as the LLM declared them
    snippets: list[dict] = field(default_factory=list)  # per-snippet info
    discard_ratio: float = 0.0
    attempts: int = 1
    style_variant_idx: int = 0
    model_name: str = ""

    def to_payload(self) -> dict:
        return {
            "codigo": self.codigo,
            "variante": self.variante,
            "codigo_variant": self.codigo_variant,
            "medicamento": self.medicamento,
            "model": self.model_name,
            "style_variant_idx": self.style_variant_idx,
            "attempts": self.attempts,
            "discard_ratio": self.discard_ratio,
            "text": self.text,
            "entities": [e.to_dict() for e in self.entities],
            "raw_long_doc_text": self.raw_text,
            "raw_long_doc_entities": self.raw_entities,
            "snippets": self.snippets,
        }


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json_object(response: str) -> Optional[dict]:
    """Best-effort extraction of a top-level JSON object from `response`.

    Tries (in order):
      1. ```json ... ``` fenced blocks.
      2. The widest substring spanning the first ``{`` and the last ``}``.
    Returns the parsed dict or ``None`` if nothing parseable is found.
    """
    if not response:
        return None
    # 1) Fenced block
    match = _FENCED_JSON_RE.search(response)
    candidates: list[str] = []
    if match:
        candidates.append(match.group(1))
    # 2) Outer braces fallback
    start = response.find("{")
    end = response.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(response[start : end + 1])

    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _coerce_entities(raw: Any) -> list[dict]:
    """Normalise the ``entities`` payload, accepting both the documented
    schema and a few common deviations the LLMs produce."""
    if not isinstance(raw, list):
        return []
    entities: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        ent_type = (
            item.get("type")
            or item.get("tipo")
            or item.get("category")
            or ""
        )
        ent_type = str(ent_type).strip().upper()
        if ent_type not in VALID_ENTITY_TYPES:
            continue
        canonical = (
            item.get("canonical")
            or item.get("canonica")
            or item.get("canónica")
            or item.get("canonico")
            or ""
        )
        surface = (
            item.get("surface")
            or item.get("mention")
            or item.get("text")
            or ""
        )
        canonical = str(canonical).strip()
        surface = str(surface).strip()
        if not surface:
            continue
        entities.append(
            {"type": ent_type, "canonical": canonical, "surface": surface}
        )
    return entities


# ---------------------------------------------------------------------------
# Validation: re-locate every declared surface inside `text`
# ---------------------------------------------------------------------------
def _locate_surface(text_norm: str, surface: str) -> Optional[tuple[int, int]]:
    """Return the first (start, end) char span of `surface` in `text_norm`,
    using the same word-boundary semantics as :class:`ConllBuilder`."""
    norm = _normalise(surface)
    if not norm:
        return None
    for span in _find_all_spans(text_norm, norm):
        return span
    return None


def validate_response_text_and_entities(
    text: str, declared_entities: list[dict]
) -> tuple[list[ValidatedEntity], int, int]:
    """Re-locate each declared surface inside `text`.

    Returns
    -------
    validated:
        Entities whose surface was found, **deduplicated** by
        ``(type, canonical, surface)``. The matcher in :class:`ConllBuilder`
        already de-duplicates by value internally, so emitting each unique
        surface once is sufficient for downstream tagging.
    declared_count:
        Total declared entities (no deduplication). Used to compute the
        per-document discard ratio honestly: an LLM that repeats the same
        valid surface twice should not be penalised.
    matched_count:
        Number of declared entities whose surface was located in `text`,
        counted **before** deduplication so the ratio matched/declared
        reflects compliance with the JSON contract rather than artefacts of
        deduplication.
    """
    declared_count = len(declared_entities)
    if declared_count == 0:
        return [], 0, 0

    text_norm = _normalise(text)
    seen: set[tuple[str, str, str]] = set()
    validated: list[ValidatedEntity] = []
    matched_count = 0
    for ent in declared_entities:
        span = _locate_surface(text_norm, ent["surface"])
        if span is None:
            continue
        matched_count += 1
        key = (ent["type"], ent["canonical"], ent["surface"])
        if key in seen:
            continue
        seen.add(key)
        start, end = span
        validated.append(
            ValidatedEntity(
                type=ent["type"],
                canonical=ent["canonical"],
                surface=ent["surface"],
                start=start,
                end=end,
            )
        )
    return validated, declared_count, matched_count


# ---------------------------------------------------------------------------
# LLM call wrapper with retry
# ---------------------------------------------------------------------------
def _call_with_retry(
    fn: Callable[[], str],
    *,
    max_attempts: int = 5,
    base_sleep: float = 1.5,
    description: str = "LLM call",
) -> str:
    """Invoke `fn` retrying on any exception with exponential backoff.

    The Azure / OpenAI clients in this repo do not implement retries, so we
    add a thin wrapper here. Any non-empty string returned is considered a
    success.
    """
    last_err: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = fn()
            if response and response.strip():
                return response
            raise RuntimeError("Empty response from LLM")
        except BaseException as exc:  # pragma: no cover - external service
            last_err = exc
            wait = base_sleep * (2 ** (attempt - 1))
            logger.warning(
                "%s failed on attempt %d/%d (%s). Sleeping %.1fs.",
                description,
                attempt,
                max_attempts,
                exc.__class__.__name__,
                wait,
            )
            time.sleep(wait)
    assert last_err is not None
    raise RuntimeError(
        f"{description} exhausted {max_attempts} attempts: {last_err!r}"
    ) from last_err


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------
class SynthNerGenerator:
    """Generate synthetic NER training material for one drug at a time."""

    DEFAULT_MAX_UNMATCHED_RATIO = 0.30  # ≤30% of declared surfaces may miss

    def __init__(
        self,
        llm_client: Any,
        out_dir: str,
        *,
        n_variants: int = 2,
        n_snippets_per_doc: int = 3,
        target_min_words: int = 400,
        target_max_words: int = 900,
        max_unmatched_ratio: float = DEFAULT_MAX_UNMATCHED_RATIO,
        max_doc_retries: int = 2,
        llm_max_attempts: int = 5,
        llm_base_sleep: float = 1.5,
        seed: int = 42,
        model_name: str = "",
    ) -> None:
        if not hasattr(llm_client, "generate_response_messages"):
            raise TypeError(
                "llm_client must expose a `generate_response_messages(messages)`"
                " method (see src/llm/azure_foundry.py)."
            )
        if n_variants < 1:
            raise ValueError("n_variants must be ≥ 1")
        if n_snippets_per_doc < 0:
            raise ValueError("n_snippets_per_doc must be ≥ 0")
        if not 0.0 <= max_unmatched_ratio <= 1.0:
            raise ValueError("max_unmatched_ratio must be between 0 and 1")

        self.llm_client = llm_client
        self.out_dir = os.path.realpath(out_dir)
        self.txt_dir = os.path.join(self.out_dir, "txt")
        self.raw_dir = os.path.join(self.out_dir, "raw_responses")
        self.ent_dir = os.path.join(self.out_dir, "entities")
        self.failed_dir = os.path.join(self.raw_dir, "_failed")
        for d in (self.txt_dir, self.raw_dir, self.ent_dir, self.failed_dir):
            os.makedirs(d, exist_ok=True)

        self.n_variants = n_variants
        self.n_snippets_per_doc = n_snippets_per_doc
        self.target_min_words = target_min_words
        self.target_max_words = target_max_words
        self.max_unmatched_ratio = max_unmatched_ratio
        self.max_doc_retries = max_doc_retries
        self.llm_max_attempts = llm_max_attempts
        self.llm_base_sleep = llm_base_sleep
        self.seed = seed
        self.model_name = model_name or getattr(llm_client, "model_name", "")

    # ------------------------------------------------------------------ #
    # Filesystem helpers
    # ------------------------------------------------------------------ #
    def codigo_variant(self, codigo: str, variante: int) -> str:
        return f"{str(codigo).strip()}_v{variante}"

    def txt_path(self, codigo: str, variante: int) -> str:
        return os.path.join(self.txt_dir, f"{self.codigo_variant(codigo, variante)}.txt")

    def raw_path(self, codigo: str, variante: int) -> str:
        return os.path.join(self.raw_dir, f"{self.codigo_variant(codigo, variante)}.json")

    def ent_path(self, codigo: str, variante: int) -> str:
        return os.path.join(self.ent_dir, f"{self.codigo_variant(codigo, variante)}.json")

    def already_done(self, codigo: str, variante: int) -> bool:
        return (
            os.path.exists(self.txt_path(codigo, variante))
            and os.path.exists(self.raw_path(codigo, variante))
            and os.path.exists(self.ent_path(codigo, variante))
        )

    # ------------------------------------------------------------------ #
    # LLM stages
    # ------------------------------------------------------------------ #
    def _generate_long_doc(
        self,
        medicamento: str,
        pactivos: list[str],
        rows: pd.DataFrame,
        variante_idx: int,
    ) -> tuple[str, list[dict], dict]:
        """Return (text, declared_entities, raw_payload_dict)."""
        messages = build_long_doc_messages(
            medicamento=medicamento,
            pactivos=pactivos,
            rows=rows,
            variante_idx=variante_idx,
            target_min_words=self.target_min_words,
            target_max_words=self.target_max_words,
        )
        response = _call_with_retry(
            lambda: self.llm_client.generate_response_messages(messages),
            max_attempts=self.llm_max_attempts,
            base_sleep=self.llm_base_sleep,
            description=f"long_doc[{medicamento}/v{variante_idx}]",
        )
        payload = _extract_json_object(response) or {}
        text = str(payload.get("text") or "").strip()
        entities = _coerce_entities(payload.get("entities"))
        return text, entities, {"raw_response": response, "parsed": payload}

    def _generate_snippet(
        self, snippet_seed: dict, variante_idx: int
    ) -> tuple[str, list[dict], dict]:
        messages = build_snippet_messages(
            medicamento=snippet_seed["medicamento"],
            pactivo=snippet_seed["pactivo"],
            reaccion=snippet_seed["reaccion"],
            frecuencia=snippet_seed["frecuencia"],
            sistema=snippet_seed["sistema"],
            variante_idx=variante_idx,
        )
        response = _call_with_retry(
            lambda: self.llm_client.generate_response_messages(messages),
            max_attempts=self.llm_max_attempts,
            base_sleep=self.llm_base_sleep,
            description=(
                f"snippet[{snippet_seed['medicamento']}/{snippet_seed['reaccion']}"
                f"/v{variante_idx}]"
            ),
        )
        payload = _extract_json_object(response) or {}
        text = str(payload.get("text") or "").strip()
        entities = _coerce_entities(payload.get("entities"))
        return text, entities, {"raw_response": response, "parsed": payload, "seed": snippet_seed}

    # ------------------------------------------------------------------ #
    # Per-drug orchestration
    # ------------------------------------------------------------------ #
    def generate_for_codigo(
        self,
        codigo: str,
        rows: pd.DataFrame,
        *,
        force: bool = False,
    ) -> list[GeneratedDocument]:
        """Produce up to ``n_variants`` synthetic docs for `codigo`.

        Variants already present on disk are skipped unless ``force=True``.
        """
        codigo_str = str(codigo).strip()
        if rows.empty:
            logger.warning("No rows for codigo %s; skipping.", codigo_str)
            return []

        medicamento = ""
        if "MEDICAMENTO" in rows.columns:
            medicamento_series = rows["MEDICAMENTO"].dropna().astype(str)
            if not medicamento_series.empty:
                medicamento = medicamento_series.iloc[0].strip()
        pactivos = sorted(
            {
                str(p).strip()
                for p in rows.get("PACTIVO", pd.Series(dtype=str)).dropna().tolist()
                if str(p).strip()
            }
        )

        results: list[GeneratedDocument] = []
        for variante in range(1, self.n_variants + 1):
            if not force and self.already_done(codigo_str, variante):
                logger.info(
                    "[%s v%d] already on disk, skipping.", codigo_str, variante
                )
                continue

            # Each variante uses a deterministic but distinct sub-seed so
            # snippet selection differs across variants.
            rng = random.Random(f"{self.seed}|{codigo_str}|{variante}")
            style_idx = (variante - 1)  # 0,1,2,...

            doc = self._generate_one(
                codigo_str=codigo_str,
                variante=variante,
                style_idx=style_idx,
                medicamento=medicamento,
                pactivos=pactivos,
                rows=rows,
                rng=rng,
            )
            if doc is None:
                continue
            self._persist_document(doc)
            results.append(doc)

        return results

    def _generate_one(
        self,
        *,
        codigo_str: str,
        variante: int,
        style_idx: int,
        medicamento: str,
        pactivos: list[str],
        rows: pd.DataFrame,
        rng: random.Random,
    ) -> Optional[GeneratedDocument]:
        last_err: Optional[BaseException] = None
        for attempt in range(1, self.max_doc_retries + 2):  # +1 because retries are *additional* attempts
            try:
                text, declared, raw_payload = self._generate_long_doc(
                    medicamento=medicamento or codigo_str,
                    pactivos=pactivos,
                    rows=rows,
                    variante_idx=style_idx,
                )
            except BaseException as exc:
                last_err = exc
                logger.error(
                    "[%s v%d] long-doc generation failed (attempt %d): %s",
                    codigo_str,
                    variante,
                    attempt,
                    exc,
                )
                self._dump_failure(codigo_str, variante, attempt, str(exc), kind="long_doc")
                continue

            if not text:
                logger.warning(
                    "[%s v%d] LLM returned empty text (attempt %d).",
                    codigo_str,
                    variante,
                    attempt,
                )
                self._dump_failure(
                    codigo_str,
                    variante,
                    attempt,
                    "empty_text",
                    kind="long_doc",
                    payload=raw_payload,
                )
                continue

            validated, declared_n, matched_n = validate_response_text_and_entities(
                text, declared
            )
            discard_ratio = (
                0.0 if declared_n == 0 else 1.0 - (matched_n / declared_n)
            )
            if declared_n == 0 or discard_ratio > self.max_unmatched_ratio:
                logger.warning(
                    "[%s v%d] long-doc invalid (declared=%d matched=%d "
                    "discard_ratio=%.2f > %.2f) attempt %d/%d",
                    codigo_str,
                    variante,
                    declared_n,
                    matched_n,
                    discard_ratio,
                    self.max_unmatched_ratio,
                    attempt,
                    self.max_doc_retries + 1,
                )
                self._dump_failure(
                    codigo_str,
                    variante,
                    attempt,
                    f"discard_ratio={discard_ratio:.2f}",
                    kind="long_doc",
                    payload=raw_payload,
                )
                continue

            # Long doc accepted. Now generate snippets (failures here are
            # tolerated — we still keep the long doc).
            snippets_text_parts: list[str] = []
            snippets_meta: list[dict] = []
            seeds = select_snippet_tuples(
                rows=rows,
                pactivo=(pactivos[0] if pactivos else ""),
                n=self.n_snippets_per_doc,
                rng=rng,
            )
            for seed in seeds:
                try:
                    s_text, s_declared, s_raw = self._generate_snippet(
                        snippet_seed=seed, variante_idx=style_idx
                    )
                except BaseException as exc:
                    logger.warning(
                        "[%s v%d] snippet failed for %r: %s",
                        codigo_str,
                        variante,
                        seed.get("reaccion"),
                        exc,
                    )
                    continue
                if not s_text:
                    continue
                s_validated, s_decl_n, s_match_n = (
                    validate_response_text_and_entities(s_text, s_declared)
                )
                if s_decl_n == 0:
                    continue
                s_discard = 1.0 - (s_match_n / s_decl_n)
                if s_discard > self.max_unmatched_ratio:
                    # Skip this snippet but keep the doc.
                    self._dump_failure(
                        codigo_str,
                        variante,
                        attempt,
                        f"snippet_discard_ratio={s_discard:.2f}",
                        kind="snippet",
                        payload={"seed": seed, **s_raw},
                    )
                    continue
                snippets_text_parts.append(s_text)
                snippets_meta.append(
                    {
                        "seed": seed,
                        "text": s_text,
                        "declared": s_declared,
                        "validated": [v.to_dict() for v in s_validated],
                        "discard_ratio": s_discard,
                    }
                )

            full_text = self._compose_full_text(text, snippets_text_parts)
            # Re-validate over the concatenated text to recompute char offsets
            # (the long-doc surfaces may have shifted; snippets append at end
            # so the long-doc offsets stay valid, but we redo it for safety
            # and to also accept snippet entities under unified spans).
            combined_declared = list(declared) + [
                e
                for s in snippets_meta
                for e in s["declared"]
            ]
            full_validated, full_decl_n, full_match_n = (
                validate_response_text_and_entities(full_text, combined_declared)
            )
            full_discard_ratio = (
                0.0 if full_decl_n == 0 else 1.0 - (full_match_n / full_decl_n)
            )

            return GeneratedDocument(
                codigo=codigo_str,
                variante=variante,
                codigo_variant=self.codigo_variant(codigo_str, variante),
                medicamento=medicamento,
                text=full_text,
                entities=full_validated,
                raw_text=text,
                raw_entities=declared,
                snippets=snippets_meta,
                discard_ratio=full_discard_ratio,
                attempts=attempt,
                style_variant_idx=style_idx,
                model_name=self.model_name,
            )

        logger.error(
            "[%s v%d] gave up after %d attempts (last error: %r)",
            codigo_str,
            variante,
            self.max_doc_retries + 1,
            last_err,
        )
        return None

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compose_full_text(long_doc: str, snippets: list[str]) -> str:
        parts: list[str] = [long_doc.strip()]
        if snippets:
            parts.append("")
            parts.append("Descripción de reacciones adversas seleccionadas (extensión):")
            parts.append("")
            for i, s in enumerate(snippets, 1):
                parts.append(f"-- Caso {i} --")
                parts.append(s.strip())
                parts.append("")
        return "\n".join(parts).strip() + "\n"

    def _persist_document(self, doc: GeneratedDocument) -> None:
        with open(self.txt_path(doc.codigo, doc.variante), "w", encoding="utf-8") as f:
            f.write(doc.text)
        payload = doc.to_payload()
        with open(self.raw_path(doc.codigo, doc.variante), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with open(self.ent_path(doc.codigo, doc.variante), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "codigo": doc.codigo,
                    "variante": doc.variante,
                    "codigo_variant": doc.codigo_variant,
                    "medicamento": doc.medicamento,
                    "discard_ratio": doc.discard_ratio,
                    "entities": [e.to_dict() for e in doc.entities],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _dump_failure(
        self,
        codigo: str,
        variante: int,
        attempt: int,
        reason: str,
        *,
        kind: str,
        payload: Optional[dict] = None,
    ) -> None:
        path = os.path.join(
            self.failed_dir,
            f"{self.codigo_variant(codigo, variante)}__{kind}__a{attempt}.json",
        )
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "codigo": codigo,
                        "variante": variante,
                        "attempt": attempt,
                        "kind": kind,
                        "reason": reason,
                        "payload": payload,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:  # pragma: no cover - logging best-effort only
            logger.exception("Could not dump failure file at %s", path)


# ---------------------------------------------------------------------------
# Helpers used by the CLI to consolidate the generated entities into a
# Variante-A-compatible CSV (so SupervisedTrainerConll / ConllBuilder can
# consume the synthetic corpus without modification).
# ---------------------------------------------------------------------------
def consolidate_full_annotations(
    out_dir: str,
    annotations_df: pd.DataFrame,
) -> pd.DataFrame:
    """Walk ``out_dir/entities`` and return a CSV-ready DataFrame.

    The output mirrors the schema of ``data/splits/full_annotations.csv``
    (CODIGO, MEDICAMENTO, PACTIVO, REACADV, FRECUENCIA, SISTEMA) but the
    values in the entity columns are the **surface strings** the LLM used
    in the synthetic text. ``CODIGO`` is the variant-suffixed identifier
    (``{codigo}_v{n}``), so a fresh ``ConllBuilder`` reading from the
    synthetic ``txt_dir`` finds the matching file and tags every variant
    independently.

    A second column per entity type with the ``_canonical`` suffix is added
    for traceability (these are not consumed by the trainer).
    """
    ent_dir = os.path.join(out_dir, "entities")
    if not os.path.isdir(ent_dir):
        raise FileNotFoundError(f"entities directory not found: {ent_dir}")

    # Lookup: original CODIGO → MEDICAMENTO (first non-empty)
    annotations_df = annotations_df.copy()
    annotations_df["CODIGO"] = annotations_df["CODIGO"].astype(str).str.strip()
    medicamento_lookup = (
        annotations_df.dropna(subset=["MEDICAMENTO"]) 
        .drop_duplicates("CODIGO")
        .set_index("CODIGO")["MEDICAMENTO"]
        .to_dict()
    )

    rows: list[dict] = []
    for fname in sorted(os.listdir(ent_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(ent_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            logger.exception("Could not read entities file %s", path)
            continue

        codigo_orig = str(payload.get("codigo", "")).strip()
        codigo_variant = str(payload.get("codigo_variant", "")).strip()
        medicamento = (
            payload.get("medicamento")
            or medicamento_lookup.get(codigo_orig)
            or ""
        )
        if not codigo_variant:
            continue

        # Collect surfaces grouped by type. Store every (canonical, surface)
        # pair so the matcher can cover both the canonical mention if the
        # LLM happened to use it AND any synonyms it introduced.
        per_type: dict[str, list[tuple[str, str]]] = {t: [] for t in VALID_ENTITY_TYPES}
        for ent in payload.get("entities", []):
            t = str(ent.get("type", "")).strip().upper()
            if t not in VALID_ENTITY_TYPES:
                continue
            surface = str(ent.get("surface", "")).strip()
            canonical = str(ent.get("canonical", "")).strip() or surface
            if not surface:
                continue
            per_type[t].append((canonical, surface))

        # The CSV is fundamentally per-mention. To stay compatible with the
        # original layout we emit one row per REACT entity (the densest
        # category). For drugs without REACT entities we fall back to one
        # row per ACTIVE / FREQ / SYS surface so the variant is still seen
        # by the matcher.
        anchor_type = "REACT" if per_type["REACT"] else next(
            (t for t in ("ACTIVE", "FREQ", "SYS") if per_type[t]), None
        )
        if anchor_type is None:
            continue

        # Pre-compute a single PACTIVO surface (any), FREQ surface (any),
        # SYS surface (any) per variant for backward-compat: the matcher
        # iterates over unique surfaces per type, so collapsing N rows with
        # the same auxiliary surface to one is fine.
        any_pactivo_canonical = (
            per_type["ACTIVE"][0][0] if per_type["ACTIVE"] else ""
        )
        any_pactivo_surface = (
            per_type["ACTIVE"][0][1] if per_type["ACTIVE"] else ""
        )
        any_freq_canonical = per_type["FREQ"][0][0] if per_type["FREQ"] else ""
        any_freq_surface = per_type["FREQ"][0][1] if per_type["FREQ"] else ""
        any_sys_canonical = per_type["SYS"][0][0] if per_type["SYS"] else ""
        any_sys_surface = per_type["SYS"][0][1] if per_type["SYS"] else ""

        for canonical, surface in per_type[anchor_type]:
            row = {
                "CODIGO": codigo_variant,
                "MEDICAMENTO": medicamento,
                "PACTIVO": any_pactivo_surface,
                "REACADV": surface if anchor_type == "REACT" else "",
                "FRECUENCIA": any_freq_surface,
                "SISTEMA": any_sys_surface,
                "PACTIVO_canonical": any_pactivo_canonical,
                "REACADV_canonical": canonical if anchor_type == "REACT" else "",
                "FRECUENCIA_canonical": any_freq_canonical,
                "SISTEMA_canonical": any_sys_canonical,
                "ORIG_CODIGO": codigo_orig,
                "VARIANTE": payload.get("variante"),
            }
            if anchor_type != "REACT":
                # Use the surface in its native column.
                col_map = {
                    "ACTIVE": "PACTIVO",
                    "FREQ": "FRECUENCIA",
                    "SYS": "SISTEMA",
                }
                col = col_map[anchor_type]
                row[col] = surface
                row[f"{col}_canonical"] = canonical
            rows.append(row)

        # Add extra rows so EVERY surface of the auxiliary categories is
        # registered in at least one CSV row (the matcher de-duplicates by
        # value internally, so the only requirement is that each surface
        # appears at least once for this CODIGO_variant).
        for type_, csv_col in (
            ("ACTIVE", "PACTIVO"),
            ("FREQ", "FRECUENCIA"),
            ("SYS", "SISTEMA"),
        ):
            seen_surfaces: set[str] = set()
            for canonical, surface in per_type[type_]:
                if surface in seen_surfaces:
                    continue
                seen_surfaces.add(surface)
                rows.append(
                    {
                        "CODIGO": codigo_variant,
                        "MEDICAMENTO": medicamento,
                        "PACTIVO": surface if type_ == "ACTIVE" else "",
                        "REACADV": "",
                        "FRECUENCIA": surface if type_ == "FREQ" else "",
                        "SISTEMA": surface if type_ == "SYS" else "",
                        "PACTIVO_canonical": canonical if type_ == "ACTIVE" else "",
                        "REACADV_canonical": "",
                        "FRECUENCIA_canonical": canonical if type_ == "FREQ" else "",
                        "SISTEMA_canonical": canonical if type_ == "SYS" else "",
                        "ORIG_CODIGO": codigo_orig,
                        "VARIANTE": payload.get("variante"),
                    }
                )

    columns = [
        "CODIGO",
        "MEDICAMENTO",
        "PACTIVO",
        "REACADV",
        "FRECUENCIA",
        "SISTEMA",
        "PACTIVO_canonical",
        "REACADV_canonical",
        "FRECUENCIA_canonical",
        "SISTEMA_canonical",
        "ORIG_CODIGO",
        "VARIANTE",
    ]
    return pd.DataFrame(rows, columns=columns)


def derive_synth_splits(
    full_annotations_synth: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive train/test splits over the synthetic corpus from the original
    splits over real CODIGOs.

    A variant-suffixed CODIGO inherits the split of its parent CODIGO, which
    preserves the *no-leakage by drug* property of Variante A.
    """

    def _norm(value: object) -> str:
        text = str(value).replace("\xa0", "").strip()
        # CSVs sometimes round-trip integer CODIGOs through float64 (e.g.
        # "89413.0"). Strip a trailing ".0" so the keys match the
        # canonical integer form used elsewhere.
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
        return text

    train_codigos = {_norm(v) for v in train_df["CODIGO"].tolist()}
    test_codigos = {_norm(v) for v in test_df["CODIGO"].tolist()}

    fa = full_annotations_synth.copy()
    fa["ORIG_CODIGO"] = fa["ORIG_CODIGO"].map(_norm)
    train_synth = fa[fa["ORIG_CODIGO"].isin(train_codigos)].reset_index(drop=True)
    test_synth = fa[fa["ORIG_CODIGO"].isin(test_codigos)].reset_index(drop=True)
    return train_synth, test_synth


__all__ = [
    "GeneratedDocument",
    "ValidatedEntity",
    "SynthNerGenerator",
    "consolidate_full_annotations",
    "derive_synth_splits",
    "validate_response_text_and_entities",
]
