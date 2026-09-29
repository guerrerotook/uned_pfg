"""
NER Evaluation for LLM adverse reaction extraction using nervaluate.

Compares LLM-generated JSON files against ground truth CSV to compute
precision, recall, and F1 for three entity types:
  - REACADV  (adverse reaction name)
  - FRECUENCIA (frequency classification)
  - SISTEMA  (body system / organ class)

Usage:
    python -m src.tests.ner_evaluation \
        --test-csv data/splits/test.csv \
        --llm-dir data/llm_outputs/gpt-5-4-pro-azure/zero-shot
"""

import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from nervaluate import Evaluator
from sentence_transformers import SentenceTransformer
import torch

from src.tests.test_shared import normalize

# Entity type labels used for nervaluate
ENTITY_TAGS = ["REACADV", "FRECUENCIA", "SISTEMA"]

# Minimum cosine similarity to consider two SISTEMA strings equivalent
SISTEMA_SIMILARITY_THRESHOLD = 0.8


class NerEvaluation:
    """Standalone NER evaluation of LLM outputs against a ground-truth CSV.

    No trained model dependency — uses synthetic fixed-position entity spans
    so that nervaluate can compute standard NER metrics (strict, exact,
    partial, type) across the three entity categories.
    """

    def __init__(
        self,
        embeddings_model_path: str,
        test_csv_path: str,
        train_csv_path: Optional[str] = None,
    ) -> None:
        """
        Args:
            test_csv_path: Path to the ground-truth CSV
                           (columns: CODIGO, MEDICAMENTO, PACTIVO, REACADV, FRECUENCIA, SISTEMA).
        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.embeddings_model = SentenceTransformer(embeddings_model_path, device=self.device)
        self.test_csv_path = test_csv_path
        # Force CODIGO to string so leading-zero codes (e.g. 0119600) are
        # preserved instead of being silently coerced to int.
        self.test_data: pd.DataFrame = pd.read_csv(
            test_csv_path, dtype={"CODIGO": str}
        )
        if train_csv_path is not None:
            self.train_csv_path = train_csv_path
            self.train_data: pd.DataFrame = pd.read_csv(
                train_csv_path, dtype={"CODIGO": str}
            )
            # merge both dataframes to have all possible values
            self.test_data: pd.DataFrame = pd.concat(
                [self.test_data, self.train_data]
            ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, output_directory: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Run the full evaluation for one LLM output directory.

        Args:
            output_directory: Path containing ``{CODIGO}.json`` files produced
                              by an LLM.
            model_name: Display name for the model.  When ``None`` the
                        directory basename is used (backward-compatible).

        Returns:
            The nervaluate results dict keyed by entity type.
        """
        if model_name is None:
            model_name = os.path.basename(os.path.normpath(output_directory))
        unique_codes: List[str] = self._get_unique_codes()

        all_true_entities: List[List[Dict[str, Any]]] = []
        all_pred_entities: List[List[Dict[str, Any]]] = []

        total_csv_reactions = 0
        total_json_reactions = 0
        total_matched = 0
        total_missing_files = 0
        all_hallucinations: List[Dict[str, Any]] = []

        for code in unique_codes:
            filtered_data: pd.DataFrame = self.test_data[
                self.test_data["CODIGO"].astype(str).str.strip() == str(code).strip()
            ]

            code_file_path = os.path.join(output_directory, f"{code}.json")
            if not os.path.exists(code_file_path):
                total_missing_files += 1
                # Emit one sample per CSV row so that unmatched rows count as FN
                true_ents, pred_ents = self._build_entities_missing(filtered_data)
                all_true_entities.append(true_ents)
                all_pred_entities.append(pred_ents)
                total_csv_reactions += len(filtered_data)
                continue

            with open(code_file_path, "r", encoding="utf-8") as fh:
                json_content: List[Dict[str, str]] = json.load(fh)

            csv_rows = list(filtered_data.iterrows())
            true_ents, pred_ents, matched, hallucinated = self._build_entities(
                csv_rows, json_content
            )

            all_true_entities.append(true_ents)
            all_pred_entities.append(pred_ents)

            # Store hallucinations with their drug code
            for h in hallucinated:
                all_hallucinations.append({"codigo": code, **h})

            total_csv_reactions += len(csv_rows)
            total_json_reactions += len(json_content)
            total_matched += matched

        # ---- nervaluate evaluation ----
        if not all_true_entities:
            print(f"[{model_name}] No data to evaluate.")
            return {}

        evaluator = Evaluator(all_true_entities, all_pred_entities, tags=ENTITY_TAGS)
        eval_results = evaluator.evaluate()

        overall = eval_results["overall"]  # schema -> EvaluationResult
        entities = eval_results["entities"]  # tag -> schema -> EvaluationResult
        print(
            evaluator.summary_report()
        )  # Optional: print nervaluate's built-in summary report
        # ---- Console report ----
        self._print_report(
            model_name,
            overall,
            entities,
            total_csv_reactions,
            total_json_reactions,
            total_matched,
            total_missing_files,
            len(unique_codes),
        )

        # ---- Hallucination summary ----
        # self._print_hallucinations(model_name, all_hallucinations)

        # ---- Build exportable result dict ----
        detection_rate = (
            total_matched / total_csv_reactions * 100 if total_csv_reactions else 0.0
        )
        export_data: Dict[str, Any] = {
            "model_name": model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_codes": len(unique_codes),
                "missing_files": total_missing_files,
                "total_csv_reactions": total_csv_reactions,
                "total_json_reactions": total_json_reactions,
                "total_matched": total_matched,
                "detection_rate": round(detection_rate, 2),
            },
            "overall": {
                schema: self._serialize_eval_result(overall[schema])
                for schema in ["ent_type", "exact", "partial", "strict"]
                if schema in overall
            },
            "entities": {
                tag: {
                    schema: self._serialize_eval_result(entities[tag][schema])
                    for schema in ["ent_type", "exact", "partial", "strict"]
                    if schema in entities.get(tag, {})
                }
                for tag in ENTITY_TAGS
                if tag in entities
            },
            "hallucinations": self._build_hallucinations_export(all_hallucinations),
        }

        return export_data

    @staticmethod
    def _sanitize_for_json(obj: Any) -> Any:
        """Recursively replace NaN / Inf floats with ``None`` for valid JSON."""
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: NerEvaluation._sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [NerEvaluation._sanitize_for_json(v) for v in obj]
        return obj

    def export_json(
        self, export_data: Dict[str, Any], output_path: str = "data/reports"
    ) -> str:
        """Write evaluation results to a JSON file.

        Args:
            export_data: The dict returned by :meth:`evaluate`.
            output_path: Directory where the JSON file will be saved.

        Returns:
            The path to the written JSON file.
        """
        os.makedirs(output_path, exist_ok=True)
        model_name = export_data.get("model_name", "unknown")
        file_path = os.path.join(output_path, f"{model_name}_ner_results.json")
        safe_data = self._sanitize_for_json(export_data)
        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(safe_data, fh, ensure_ascii=False, indent=2)
        print(f"  Exported results to {file_path}")
        return file_path

    @staticmethod
    def export_models_index(
        model_files: List[str], output_path: str = "data/reports"
    ) -> str:
        """Write a models_index.json manifest listing all result files.

        Args:
            model_files: List of JSON file paths (relative or absolute).
            output_path: Directory where models_index.json will be saved.

        Returns:
            The path to the written index file.
        """
        os.makedirs(output_path, exist_ok=True)
        # Store only basenames so the HTML can load via relative paths
        index = {"models": [os.path.basename(f) for f in model_files]}
        file_path = os.path.join(output_path, "models_index.json")
        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(index, fh, ensure_ascii=False, indent=2)
        print(f"  Exported models index to {file_path}")
        return file_path

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_eval_result(result: Any) -> Dict[str, Any]:
        """Convert a nervaluate EvaluationResult to a plain dict."""
        return {
            "correct": result.correct,
            "incorrect": result.incorrect,
            "partial": result.partial,
            "missed": result.missed,
            "spurious": result.spurious,
            "precision": round(result.precision, 4),
            "recall": round(result.recall, 4),
            "f1": round(result.f1, 4),
        }

    @staticmethod
    def _build_hallucinations_export(
        hallucinations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Structure hallucinations for JSON export."""
        by_code: Dict[str, List[Dict[str, str]]] = {}
        for h in hallucinations:
            code = str(h.get("codigo", "?"))
            entry = {
                "principio_activo": h.get("principio activo", ""),
                "reaccion_adversa": h.get("reacción adversa", ""),
                "frecuencia": h.get("frecuencia", ""),
                "trastorno": h.get("trastorno", ""),
            }
            by_code.setdefault(code, []).append(entry)

        return {
            "total": len(hallucinations),
            "by_code": {
                code: {"count": len(entries), "entries": entries}
                for code, entries in sorted(by_code.items())
            },
        }

    # ------------------------------------------------------------------
    # Entity-span construction
    # ------------------------------------------------------------------

    def _build_entities(
        self,
        csv_rows: List[Tuple[int, pd.Series]],
        json_entries: List[Dict[str, str]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, List[Dict[str, str]]]:
        """Build true / predicted entity lists for a single drug code.

        Each adverse reaction occupies 6 positional units:
          - [i*6, i*6+1]   → REACADV
          - [i*6+2, i*6+3] → FRECUENCIA
          - [i*6+4, i*6+5] → SISTEMA

        Returns:
            (true_entities, pred_entities, matched_count, hallucinated_entries)
        """
        true_entities: List[Dict[str, Any]] = []
        pred_entities: List[Dict[str, Any]] = []
        matched = 0

        # Index JSON entries by normalized (principio_activo, reacción_adversa)
        json_index = self._index_json_entries(json_entries)

        # Track which JSON entries were consumed (for FP detection)
        consumed_json_keys: set = set()

        # --- Process CSV rows (ground truth) ---
        for i, (_, row) in enumerate(csv_rows):
            offset = i * 6

            # True entities always exist for every CSV row
            true_entities.append(
                {"label": "REACADV", "start": offset, "end": offset + 1}
            )
            true_entities.append(
                {"label": "FRECUENCIA", "start": offset + 2, "end": offset + 3}
            )
            true_entities.append(
                {"label": "SISTEMA", "start": offset + 4, "end": offset + 5}
            )

            # Try to find a matching JSON entry
            match_entry = self._search_json_entry(row, json_index)
            if match_entry is not None:
                matched += 1

                key = self._make_key(row)
                consumed_json_keys.add(key)

                # REACADV — always a predicted span (reaction was detected)
                pred_entities.append(
                    {"label": "REACADV", "start": offset, "end": offset + 1}
                )

                # FRECUENCIA — only if it matches
                if self._check_frequency(row, match_entry):
                    pred_entities.append(
                        {"label": "FRECUENCIA", "start": offset + 2, "end": offset + 3}
                    )

                # SISTEMA — only if it matches
                if self._check_system(row, match_entry):
                    pred_entities.append(
                        {"label": "SISTEMA", "start": offset + 4, "end": offset + 5}
                    )
            # else: no predicted spans → all three count as FN

        # --- False positives (hallucinations): JSON entries not in CSV ---
        hallucinated: List[Dict[str, str]] = []
        fp_offset = len(csv_rows) * 6
        fp_index = 0
        for key, entry in json_index.items():
            if key not in consumed_json_keys:
                offset = fp_offset + fp_index * 6
                pred_entities.append(
                    {"label": "REACADV", "start": offset, "end": offset + 1}
                )
                hallucinated.append(entry)
                fp_index += 1

        return true_entities, pred_entities, matched, hallucinated

    def _build_entities_missing(
        self, filtered_data: pd.DataFrame
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build entities when the JSON file is entirely missing.

        All CSV rows become FN — no predicted entities.
        """
        true_entities: List[Dict[str, Any]] = []
        for i in range(len(filtered_data)):
            offset = i * 6
            true_entities.append(
                {"label": "REACADV", "start": offset, "end": offset + 1}
            )
            true_entities.append(
                {"label": "FRECUENCIA", "start": offset + 2, "end": offset + 3}
            )
            true_entities.append(
                {"label": "SISTEMA", "start": offset + 4, "end": offset + 5}
            )
        return true_entities, []

    # ------------------------------------------------------------------
    # JSON look-up helpers
    # ------------------------------------------------------------------

    def _index_json_entries(
        self, json_entries: List[Dict[str, str]]
    ) -> Dict[str, Dict[str, str]]:
        """Index JSON entries by normalized (principio activo, reacción adversa).

        If duplicates exist, the first occurrence wins (matches existing
        ``LlmTest.search_llm_output`` behaviour).
        """
        index: Dict[str, Dict[str, str]] = {}
        for entry in json_entries:
            pa = normalize(str(entry.get("principio activo", "")))
            ra = normalize(str(entry.get("reacción adversa", "")))
            key = (pa, ra)
            if key not in index:
                index[key] = entry
        return index

    def _search_json_entry(
        self, row: pd.Series, json_index: Dict[str, Dict[str, str]]
    ) -> Optional[Dict[str, str]]:
        """Find matching JSON entry for a CSV row."""
        key = self._make_key(row)
        return json_index.get(key)

    def _make_key(self, row: pd.Series) -> Tuple[str, str]:
        pa = normalize(str(row["PACTIVO"]))
        ra = normalize(str(row["REACADV"]))
        return (pa, ra)

    # ------------------------------------------------------------------
    # Field-level comparisons
    # ------------------------------------------------------------------

    @staticmethod
    def _check_frequency(row: pd.Series, entry: Dict[str, str]) -> bool:
        csv_val = str(row.get("FRECUENCIA", ""))
        json_val = str(entry.get("frecuencia", ""))
        return normalize(csv_val) in normalize(json_val) or normalize(
            json_val
        ) in normalize(csv_val)

    def _check_system(self, row: pd.Series, entry: Dict[str, str]) -> bool:
        csv_val = normalize(str(row.get("SISTEMA", "")))
        json_val = normalize(str(entry.get("trastorno", "")))
        if csv_val == json_val:
            return True
        if not csv_val or not json_val:
            return False
        embeddings = self.embeddings_model.encode(
            [csv_val, json_val], convert_to_numpy=True
        )
        vec_a, vec_b = embeddings[0], embeddings[1]
        cosine_sim = float(
            np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        )
        return cosine_sim >= SISTEMA_SIMILARITY_THRESHOLD

    # ------------------------------------------------------------------
    # Unique codes
    # ------------------------------------------------------------------

    def _get_unique_codes(self) -> List[str]:
        return self.test_data["CODIGO"].astype(str).str.strip().unique().tolist()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _print_report(
        model_name: str,
        overall: Dict[str, Any],
        entities: Dict[str, Any],
        total_csv: int,
        total_json: int,
        total_matched: int,
        missing_files: int,
        total_codes: int,
    ) -> None:
        detection_rate = total_matched / total_csv * 100 if total_csv else 0.0

        print()
        print("=" * 100)
        print(f"  NER Evaluation — {model_name}")
        print("=" * 100)
        print(
            f"  Drug codes evaluated : {total_codes}  "
            f"(missing JSON files: {missing_files})"
        )
        print(f"  Ground-truth reactions (CSV) : {total_csv}")
        print(f"  LLM-predicted reactions (JSON) : {total_json}")
        print(
            f"  Matched reactions              : {total_matched}  "
            f"({detection_rate:.1f}%)"
        )
        print("-" * 100)

        # Overall results across all entity types
        header_fmt = "  {:>12} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}"
        row_fmt = (
            "  {:>12} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10.4f} {:>10.4f} {:>10.4f}"
        )

        print()
        print("  Overall results (all entity types combined):")
        print(
            header_fmt.format(
                "",
                "correct",
                "incorrect",
                "partial",
                "missed",
                "spurious",
                "precision",
                "recall",
                "f1-score",
            )
        )
        print()
        for schema in ["ent_type", "exact", "partial", "strict"]:
            if schema in overall:
                r = overall[schema]
                print(
                    row_fmt.format(
                        schema,
                        r.correct,
                        r.incorrect,
                        r.partial,
                        r.missed,
                        r.spurious,
                        r.precision,
                        r.recall,
                        r.f1,
                    )
                )

        # Per-tag results
        for tag in ENTITY_TAGS:
            if tag in entities:
                print()
                print(f"  Results for entity type: {tag}")
                print(
                    header_fmt.format(
                        "",
                        "correct",
                        "incorrect",
                        "partial",
                        "missed",
                        "spurious",
                        "precision",
                        "recall",
                        "f1-score",
                    )
                )
                print()
                tag_results = entities[tag]
                for schema in ["ent_type", "exact", "partial", "strict"]:
                    if schema in tag_results:
                        r = tag_results[schema]
                        print(
                            row_fmt.format(
                                schema,
                                r.correct,
                                r.incorrect,
                                r.partial,
                                r.missed,
                                r.spurious,
                                r.precision,
                                r.recall,
                                r.f1,
                            )
                        )

        print()
        print("=" * 100)
        print()

    @staticmethod
    def _print_hallucinations(
        model_name: str,
        hallucinations: List[Dict[str, Any]],
    ) -> None:
        """Print a summary of hallucinated entities not present in the ground truth."""
        if not hallucinations:
            print(f"  [{model_name}] No hallucinations detected.")
            print()
            return

        print("=" * 100)
        print(f"  Hallucination Summary — {model_name}")
        print(f"  Total hallucinated adverse reactions: {len(hallucinations)}")
        print("=" * 100)

        # Group by drug code
        by_code: Dict[str, List[Dict[str, Any]]] = {}
        for h in hallucinations:
            code = str(h.get("codigo", "?"))
            by_code.setdefault(code, []).append(h)

        header_fmt = "  {:<40} {:<25} {:<30}"
        print(header_fmt.format("Reacción adversa", "Frecuencia", "Trastorno"))
        print("  " + "-" * 95)

        for code in sorted(by_code.keys()):
            entries = by_code[code]
            print(f"\n  Drug code: {code}  ({len(entries)} hallucinations)")
            for entry in entries:
                ra = entry.get("reacción adversa", "—")
                freq = entry.get("frecuencia", "—")
                sist = entry.get("trastorno", "—")
                # Truncate long values for readability
                ra_display = (ra[:37] + "...") if len(ra) > 40 else ra
                freq_display = (freq[:22] + "...") if len(freq) > 25 else freq
                sist_display = (sist[:27] + "...") if len(sist) > 30 else sist
                print(header_fmt.format(ra_display, freq_display, sist_display))

        print()
        print("=" * 100)
        print()
