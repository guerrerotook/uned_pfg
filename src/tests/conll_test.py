"""Evaluate CoNLL-trained NER models against the test split.

* Loads a token-classification model from ``<model_dir>/ner_conll_model``.
* Reads the gold mentions from ``data/splits/test.csv``.
* Re-builds the BIO test corpus on the fly with :class:`ConllBuilder` so
  the tokens / sentence boundaries match the way the model was trained.
* Runs inference with a HF ``Trainer.predict`` and aligns predictions back
  to word-level BIO labels (mirrors ``other/pubmedbert_ner_sider.ipynb``).
* Reports both seqeval (entity-level micro P/R/F1) and nervaluate (strict /
    exact / partial / ent_type) metrics, then exports a JSON report.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from nervaluate import Evaluator
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from src.conll_ner_builder import ConllBuilder, sanitise_bio_predictions


class ConllNerTest:
    """Run NER evaluation on a CoNLL-trained token-classification model.

    Parameters
    ----------
    model_dir:
        Parent directory of the trained model. The token-classification
        weights are expected at ``<model_dir>/ner_conll_model`` and the
        label mapping at ``<model_dir>/ner_conll_label_mapping.json``.
    test_csv_path:
        CSV with the test split (columns CODIGO, MEDICAMENTO, PACTIVO,
        REACADV, FRECUENCIA, SISTEMA).
    train_csv_path:
        Optional second CSV concatenated with ``test_csv_path`` to mirror
        ``EmbeddingsTest`` behaviour. Defaults to ``None`` (test rows only).
    full_annotations_csv_path:
        Optional CSV with **every known mention** for every CODIGO
        (typically ``data/splits/full_annotations.csv`` produced by
        ``python -m src.training_data``). When provided, the BIO test
        corpus is tagged using the union of all mentions for the test
        códigos, so a prediction is only counted as ``spurious`` when it
        truly is not in any annotation. When ``None``, the gold is built
        from ``test_csv_path`` (+ ``train_csv_path`` if given).
    txt_dir:
        Directory containing the section-4.8 raw text files.
    spacy_model:
        spaCy pipeline used for word-level tokenisation. Must match training.
    max_length:
        Sequence length passed to the NER tokeniser. Should match training.
    batch_size:
        Per-device evaluation batch size for ``Trainer.predict``.
    """

    def __init__(
        self,
        model_dir: str,
        test_csv_path: str = "data/splits/test.csv",
        train_csv_path: Optional[str] = None,
        full_annotations_csv_path: Optional[str] = None,
        txt_dir: str = "data/medicamentos/txt",
        spacy_model: str = "es_core_news_sm",
        max_length: int = 256,
        batch_size: int = 32,
    ) -> None:
        self.model_dir = os.path.normpath(model_dir)
        self.ner_model_dir = os.path.join(self.model_dir, "ner_conll_model")
        self.label_mapping_path = os.path.join(
            self.model_dir, "ner_conll_label_mapping.json"
        )
        if not os.path.isdir(self.ner_model_dir):
            raise FileNotFoundError(
                f"Expected NER model at {self.ner_model_dir!r} (directory missing)."
            )
        if not os.path.isfile(self.label_mapping_path):
            raise FileNotFoundError(
                f"Expected label mapping at {self.label_mapping_path!r}."
            )

        with open(self.label_mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        self.entity_types: List[str] = list(mapping["entity_types"])
        self.label_to_id: Dict[str, int] = {
            k: int(v) for k, v in mapping["label_to_id"].items()
        }
        self.id_to_label: Dict[int, str] = {
            int(k): v for k, v in mapping["id_to_label"].items()
        }

        # Load test data (and optionally train, mirroring EmbeddingsTest).
        self.test_csv_path = test_csv_path
        self.test_data: pd.DataFrame = pd.read_csv(
            test_csv_path, dtype={"CODIGO": str}
        )
        if train_csv_path is not None:
            self.train_csv_path = train_csv_path
            self.train_data: pd.DataFrame = pd.read_csv(
                train_csv_path, dtype={"CODIGO": str}
            )
            self.test_data = pd.concat(
                [self.test_data, self.train_data]
            ).reset_index(drop=True)

        # CODIGOs that define the test gold partition (kept BEFORE we load
        # the full-annotations CSV so the partition is not polluted with
        # unrelated drugs).
        self.test_codigos: set[str] = set(
            self.test_data["CODIGO"].astype(str).str.strip().tolist()
        )

        # Optional full-annotations CSV used as the source of truth for
        # mention strings. We replace ``self.test_data`` with the rows of
        # the full file whose CODIGO is in the test partition so the BIO
        # tagger sees every known adverse reaction for that drug.
        self.full_annotations_csv_path = full_annotations_csv_path
        if full_annotations_csv_path is not None:
            full_df = pd.read_csv(
                full_annotations_csv_path, dtype={"CODIGO": str}
            )
            full_df["CODIGO"] = full_df["CODIGO"].astype(str).str.strip()
            kept = full_df[full_df["CODIGO"].isin(self.test_codigos)].reset_index(
                drop=True
            )
            if kept.empty:
                print(
                    "full_annotations_csv_path provided but none of its "
                    "CODIGOs match the test split — keeping the original "
                    "test rows."
                )
            else:
                print(
                    f"Using {len(kept):,} mention rows from "
                    f"{os.path.basename(full_annotations_csv_path)} as gold "
                    f"({kept['CODIGO'].nunique()} CODIGOs)."
                )
                self.test_data = kept

        # ConllBuilder uses the same convention as training (O first, then
        # B-X / I-X for each entity type, in user-supplied order).
        self.txt_dir = txt_dir
        self.spacy_model = spacy_model
        self.max_length = max_length
        self.batch_size = batch_size
        self.builder = ConllBuilder(
            txt_dir=txt_dir,
            entity_types=self.entity_types,
            spacy_model=spacy_model,
        )
        if self.builder.label_to_id != self.label_to_id:
            print(
                "ConllBuilder built a different label vocabulary than the "
                "model. Falling back to the model mapping for evaluation."
            )

        # Device + model loading.
        self.device = self._select_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.ner_model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.ner_model_dir
        ).to(self.device)
        self.model.eval()

        print(
            f"ConllNerTest ready | model={os.path.basename(self.model_dir)} "
            f"| entity_types={self.entity_types} | device={self.device}"
        )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if (
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")

    # ------------------------------------------------------------------
    # Dataset construction (mirrors notebook)
    # ------------------------------------------------------------------

    def _build_test_dataset(self) -> Tuple[Dataset, Dict[str, Any]]:
        examples = self.builder.build_dataset(self.test_data)
        summary = self.builder.summarise(examples)
        if not examples:
            raise RuntimeError(
                "ConllBuilder produced no examples. Check that the .txt files "
                f"exist under {self.txt_dir!r}."
            )
        return Dataset.from_list(examples), summary

    def _tokenize_and_align(self, ds: Dataset) -> Dataset:
        """Subword-align BIO labels (first subtoken keeps the label, rest -100)."""
        tokenizer = self.tokenizer
        max_length = self.max_length

        def _map(examples):
            tokenized = tokenizer(
                examples["tokens"],
                truncation=True,
                max_length=max_length,
                is_split_into_words=True,
            )
            aligned: List[List[int]] = []
            for i, label_seq in enumerate(examples["ner_tags"]):
                word_ids = tokenized.word_ids(batch_index=i)
                previous = None
                ids: List[int] = []
                for w in word_ids:
                    if w is None:
                        ids.append(-100)
                    elif w != previous:
                        ids.append(int(label_seq[w]))
                    else:
                        ids.append(-100)
                    previous = w
                aligned.append(ids)
            tokenized["labels"] = aligned
            return tokenized

        return ds.map(_map, batched=True, remove_columns=ds.column_names)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, output_directory: Optional[str] = None) -> Dict[str, Any]:
        """Run inference + metric computation and return an export-ready dict."""
        model_name = os.path.basename(self.model_dir)
        print(f"\nBuilding CoNLL test data for {model_name}...")
        ds, summary = self._build_test_dataset()
        tokenized = self._tokenize_and_align(ds)

        label_counts = summary["label_counts"]
        non_o_labels = sum(
            count for label, count in label_counts.items() if label != "O"
        )
        print(
            f"   {summary['documents']} drugs / {summary['sentences']} sentences "
            f"/ {summary['tokens']} tokens (non-O={non_o_labels:,})"
        )
        print("   Label distribution:")
        for label, count in sorted(
            label_counts.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            print(f"      {label:<12} : {count:>8,}")
        if summary["missing_txt_files"]:
            print(
                f"   Missing .txt for {len(summary['missing_txt_files'])} "
                f"drug code(s) — they were skipped."
            )

        # Run prediction with a Trainer (handles batching + device transfer).
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True,
            pad_to_multiple_of=8,
        )
        bf16_eval = bool(
            self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        )
        fp16_eval = bool(self.device.type == "cuda" and not bf16_eval)
        with tempfile.TemporaryDirectory(prefix=f"conll_eval_{model_name}_") as tmp_dir:
            args = TrainingArguments(
                output_dir=tmp_dir,
                per_device_eval_batch_size=self.batch_size,
                bf16_full_eval=bf16_eval,
                fp16_full_eval=fp16_eval,
                report_to="none",
                logging_strategy="no",
            )
            trainer = Trainer(
                model=self.model,
                args=args,
                processing_class=self.tokenizer,
                data_collator=data_collator,
            )
            prediction_output = trainer.predict(tokenized)

        logits = prediction_output.predictions
        label_ids = prediction_output.label_ids
        preds = np.argmax(logits, axis=-1)

        true_labels: List[List[str]] = []
        true_predictions: List[List[str]] = []
        for pred_seq, label_seq in zip(preds, label_ids):
            row_true: List[str] = []
            row_pred: List[str] = []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                row_true.append(self.id_to_label[int(l)])
                row_pred.append(self.id_to_label[int(p)])
            true_labels.append(row_true)
            # Repair orphan I-X tags so seqeval/nervaluate don't count
            # malformed IOB sequences as extra spurious entities.
            true_predictions.append(sanitise_bio_predictions(row_pred))

        # ---- seqeval (entity-level, micro-averaged) ----
        seqeval_metrics = {
            "precision": float(
                precision_score(true_labels, true_predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(true_labels, true_predictions, zero_division=0)
            ),
            "f1": float(
                f1_score(true_labels, true_predictions, zero_division=0)
            ),
        }
        per_entity_seqeval = classification_report(
            true_labels,
            true_predictions,
            digits=4,
            output_dict=True,
            zero_division=0,
        )

        # ---- nervaluate (span-level, multi-schema) ----
        # nervaluate expects entity *types* (without B-/I-) in `tags`.
        unique_tags = sorted(self.entity_types)
        evaluator = Evaluator(true_labels, true_predictions, tags=unique_tags)
        eval_results = evaluator.evaluate()
        overall_dict = eval_results.get("overall", {})
        entities_dict = eval_results.get("entities", {})

        if hasattr(evaluator, "summary_report"):
            print(evaluator.summary_report())

        # ---- loose overlap metric ----
        # When the gold labels are partial (a real risk on weakly
        # supervised data), strict precision under-reports the true
        # quality. The loose metric counts a predicted span as a hit if it
        # *overlaps* any gold span of the same entity type. It is reported
        # alongside the strict numbers for a more complete error analysis.
        loose_metrics = self._loose_overlap_metrics(
            true_labels, true_predictions
        )

        # ---- console reporting ----
        self._print_report(
            model_name, seqeval_metrics, overall_dict, entities_dict, summary
        )
        if loose_metrics:
            print(
                f"   Loose overlap (per-entity, predicted span overlaps any gold of same type):"
            )
            for tag, m in loose_metrics.items():
                print(
                    f"      {tag:<10s}  precision={m['precision']:.4f}  "
                    f"recall={m['recall']:.4f}  f1={m['f1']:.4f}  "
                    f"(matched_pred={m['matched_pred']}, gold={m['gold']}, pred={m['pred']})"
                )

        # ---- build export payload ----
        export_data: Dict[str, Any] = {
            "model_name": model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "entity_types": list(self.entity_types),
            "labels": list(self.id_to_label.values()),
            "summary": {
                "total_codes": summary["documents"],
                "total_sentences": summary["sentences"],
                "total_tokens": summary["tokens"],
                "label_counts": summary["label_counts"],
                "missing_txt_files": summary["missing_txt_files"],
                "docs_with_no_tags": summary["docs_with_no_tags"],
            },
            "seqeval": {
                **seqeval_metrics,
                "per_entity": per_entity_seqeval,
            },
            "overall": {
                schema: self._serialize_eval_result(overall_dict[schema])
                for schema in ("ent_type", "exact", "partial", "strict")
                if schema in overall_dict
            },
            "entities": {
                tag: {
                    schema: self._serialize_eval_result(entities_dict[tag][schema])
                    for schema in ("ent_type", "exact", "partial", "strict")
                    if schema in entities_dict.get(tag, {})
                }
                for tag in unique_tags
                if tag in entities_dict
            },
            "loose_overlap": loose_metrics,
        }

        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
            preds_path = os.path.join(output_directory, "predictions.json")
            safe_predictions = self._sanitize_for_json(
                {"true": true_labels, "pred": true_predictions}
            )
            with open(preds_path, "w", encoding="utf-8") as fh:
                json.dump(safe_predictions, fh, ensure_ascii=False)

        return export_data

    # ------------------------------------------------------------------
    # JSON helpers (mirrors EmbeddingsTest)
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_eval_result(result: Any) -> Dict[str, Any]:
        """Convert a nervaluate ``EvaluationResult`` (or dict) into a dict."""
        if isinstance(result, dict):
            getter = result.__getitem__
        else:
            def getter(key: str) -> Any:
                return getattr(result, key)
        return {
            "correct": int(getter("correct")),
            "incorrect": int(getter("incorrect")),
            "partial": int(getter("partial")),
            "missed": int(getter("missed")),
            "spurious": int(getter("spurious")),
            "precision": round(float(getter("precision")), 4),
            "recall": round(float(getter("recall")), 4),
            "f1": round(float(getter("f1")), 4),
        }

    @staticmethod
    def _extract_spans(
        labels_per_sentence: List[List[str]],
    ) -> List[List[Tuple[str, int, int]]]:
        """Convert BIO label sequences into per-sentence ``(type, start, end)`` spans.

        ``end`` is the exclusive token index. Orphan ``I-X`` tags are
        treated as ``B-X`` for symmetry with :func:`sanitise_bio_predictions`.
        """
        out: List[List[Tuple[str, int, int]]] = []
        for labels in labels_per_sentence:
            spans: List[Tuple[str, int, int]] = []
            current_type: Optional[str] = None
            start = 0
            for idx, label in enumerate(labels):
                if label == "O":
                    if current_type is not None:
                        spans.append((current_type, start, idx))
                        current_type = None
                    continue
                tag = label[2:]
                is_begin = label.startswith("B-") or current_type != tag
                if is_begin:
                    if current_type is not None:
                        spans.append((current_type, start, idx))
                    current_type = tag
                    start = idx
            if current_type is not None:
                spans.append((current_type, start, len(labels)))
            out.append(spans)
        return out

    @classmethod
    def _loose_overlap_metrics(
        cls,
        true_labels: List[List[str]],
        true_predictions: List[List[str]],
    ) -> Dict[str, Dict[str, float]]:
        """Per-entity loose-overlap precision / recall / F1.

        A predicted span is a *true positive* if it overlaps any gold span
        of the **same** entity type. The metric is symmetric: a gold span
        counts as recalled if any predicted span of the same type overlaps
        it. Useful as a sanity-check ceiling when the gold annotations are
        partial.
        """
        gold_spans = cls._extract_spans(true_labels)
        pred_spans = cls._extract_spans(true_predictions)

        tags: set[str] = set()
        for sent in gold_spans:
            tags.update(s[0] for s in sent)
        for sent in pred_spans:
            tags.update(s[0] for s in sent)

        results: Dict[str, Dict[str, float]] = {}
        for tag in sorted(tags):
            tp_pred = 0
            tp_gold = 0
            total_pred = 0
            total_gold = 0
            for golds, preds in zip(gold_spans, pred_spans):
                gold_for_tag = [g for g in golds if g[0] == tag]
                pred_for_tag = [p for p in preds if p[0] == tag]
                total_gold += len(gold_for_tag)
                total_pred += len(pred_for_tag)
                for p in pred_for_tag:
                    if any(
                        not (p[2] <= g[1] or g[2] <= p[1]) for g in gold_for_tag
                    ):
                        tp_pred += 1
                for g in gold_for_tag:
                    if any(
                        not (p[2] <= g[1] or g[2] <= p[1]) for p in pred_for_tag
                    ):
                        tp_gold += 1
            precision = tp_pred / total_pred if total_pred else 0.0
            recall = tp_gold / total_gold if total_gold else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            results[tag] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "matched_pred": tp_pred,
                "gold": total_gold,
                "pred": total_pred,
            }
        return results

    @staticmethod
    def _sanitize_for_json(obj: Any) -> Any:
        """Recursively normalise NaN/Inf and numpy scalars for JSON output."""
        if isinstance(obj, np.generic):
            obj = obj.item()
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: ConllNerTest._sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [ConllNerTest._sanitize_for_json(v) for v in obj]
        return obj

    def export_json(
        self,
        export_data: Dict[str, Any],
        output_path: str = "data/reports/conll",
    ) -> str:
        """Write evaluation results to ``<output_path>/<model_name>_ner_results.json``."""
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
        model_files: List[str], output_path: str = "data/reports/conll"
    ) -> str:
        """Write a manifest listing every result file in ``output_path``."""
        os.makedirs(output_path, exist_ok=True)
        index = {"models": [os.path.basename(f) for f in model_files]}
        file_path = os.path.join(output_path, "models_index.json")
        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(index, fh, ensure_ascii=False, indent=2)
        print(f"  Exported models index to {file_path}")
        return file_path

    # ------------------------------------------------------------------
    # Console reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _print_report(
        model_name: str,
        seqeval_metrics: Dict[str, float],
        overall: Dict[str, Any],
        entities: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> None:
        """Print a formatted CoNLL NER evaluation report to the console."""

        def _get(result: Any, key: str) -> Any:
            return result[key] if isinstance(result, dict) else getattr(result, key)

        sep = "=" * 80
        print(f"\n{sep}")
        print(f"  CoNLL NER Evaluation — {model_name}")
        print(sep)
        print(f"  Drugs evaluated      : {summary['documents']}")
        print(f"  Sentences            : {summary['sentences']}")
        print(f"  Tokens               : {summary['tokens']}")
        print(sep)
        print("  seqeval (entity-level, micro-averaged):")
        print(
            f"    precision={seqeval_metrics['precision']:.4f}  "
            f"recall={seqeval_metrics['recall']:.4f}  "
            f"f1={seqeval_metrics['f1']:.4f}"
        )
        print(sep)

        schemas = ("ent_type", "exact", "partial", "strict")
        header = (
            f"{'Schema':<12} {'Correct':>8} {'Incorrect':>10} {'Partial':>8} "
            f"{'Missed':>8} {'Spurious':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}"
        )
        print("  Overall (nervaluate):")
        print(f"  {header}")
        print(f"  {'-' * len(header)}")
        for schema in schemas:
            if schema not in overall:
                continue
            r = overall[schema]
            print(
                f"  {schema:<12} {int(_get(r, 'correct')):>8} "
                f"{int(_get(r, 'incorrect')):>10} {int(_get(r, 'partial')):>8} "
                f"{int(_get(r, 'missed')):>8} {int(_get(r, 'spurious')):>9} "
                f"{float(_get(r, 'precision')):>10.4f} "
                f"{float(_get(r, 'recall')):>8.4f} "
                f"{float(_get(r, 'f1')):>8.4f}"
            )

        for tag in sorted(entities):
            print(f"\n  Entity: {tag}")
            print(f"  {header}")
            print(f"  {'-' * len(header)}")
            for schema in schemas:
                if schema not in entities[tag]:
                    continue
                r = entities[tag][schema]
                print(
                    f"  {schema:<12} {int(_get(r, 'correct')):>8} "
                    f"{int(_get(r, 'incorrect')):>10} {int(_get(r, 'partial')):>8} "
                    f"{int(_get(r, 'missed')):>8} {int(_get(r, 'spurious')):>9} "
                    f"{float(_get(r, 'precision')):>10.4f} "
                    f"{float(_get(r, 'recall')):>8.4f} "
                    f"{float(_get(r, 'f1')):>8.4f}"
                )

        print(f"\n{sep}\n")
