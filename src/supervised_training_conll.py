"""Supervised trainer that uses real section-4.8 text for NER (CoNLL BIO).

The trainer pulls the original Spanish text for each ``CODIGO`` from
``data/medicamentos/txt/{codigo}.txt`` and tags the mentions referenced in the
train/test CSVs with a configurable subset of the four entity types
(REACT / ACTIVE / FREQ / SYS).
"""

from __future__ import annotations

import json
import math
import os
import random
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.conll_ner_builder import (
    ENTITY_TYPE_TO_COLUMN,
    ConllBuilder,
    sanitise_bio_predictions,
)

warnings.filterwarnings("ignore")


class SupervisedTrainerConll:
    """Train CoNLL-style NER on CIMA data.

    Parameters
    ----------
    pretrained_model_path:
        Path or HF hub id of the base encoder (BERT/RoBERTa).
    output_dir:
        Directory where the trained model, label mapping and metrics are saved.
    ner_entity_types:
        Subset of ``["REACT", "ACTIVE", "FREQ", "SYS"]`` to be tagged. A list
        of length 1 trains a single-tag NER model; longer lists train a
        multi-tag model. Order is preserved when building the label vocab.
    txt_dir:
        Directory containing the raw section-4.8 text files. Defaults to
        ``data/medicamentos/txt`` relative to the current working directory.
    spacy_model:
        Name of the spaCy pipeline used for word-level tokenisation.
    max_length:
        Sequence length used by the NER tokeniser.
    """

    def __init__(
        self,
        pretrained_model_path: str = "./cima-bert-model",
        output_dir: str = "./cima-bert-supervised",
        ner_entity_types: Optional[list[str]] = None,
        txt_dir: str = "../data/medicamentos/txt",
        spacy_model: str = "es_core_news_sm",
        max_length: int = 256,
    ) -> None:
        if ner_entity_types is None:
            ner_entity_types = ["REACT"]
        self._validate_entity_types(ner_entity_types)

        self.pretrained_model_path = pretrained_model_path
        self.output_dir = output_dir
        self.ner_entity_types = list(ner_entity_types)
        self.txt_dir = txt_dir
        self.spacy_model = spacy_model
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_path,
            add_prefix_space=True,
        )

        # CoNLL builder (entity_types validated above).
        self.conll_builder = ConllBuilder(
            txt_dir=self.txt_dir,
            entity_types=self.ner_entity_types,
            spacy_model=self.spacy_model,
        )
        self.label_to_id = self.conll_builder.label_to_id
        self.id_to_label = self.conll_builder.id_to_label

        self.train_dataframe: pd.DataFrame = pd.DataFrame()
        self.test_dataframe: pd.DataFrame = pd.DataFrame()
        self.full_annotations: pd.DataFrame = pd.DataFrame()
        self.tokenized_ner_datasets: Optional[DatasetDict] = None
        self.ner_results: dict = {}

        os.makedirs(self.output_dir, exist_ok=True)
        print(
            f"SupervisedTrainerConll ready | tags={self.ner_entity_types} "
            f"| labels={list(self.label_to_id.keys())}"
        )

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_entity_types(entity_types: list[str]) -> None:
        if not entity_types:
            raise ValueError("`ner_entity_types` must contain at least one tag")
        unknown = [t for t in entity_types if t not in ENTITY_TYPE_TO_COLUMN]
        if unknown:
            raise ValueError(
                f"Unknown entity type(s) {unknown}. Allowed values: "
                f"{sorted(ENTITY_TYPE_TO_COLUMN)}"
            )

    @staticmethod
    def _validate_split_codigos(
        train_codigos: set[str], test_codigos: set[str]
    ) -> None:
        overlap = train_codigos & test_codigos
        if overlap:
            raise ValueError(
                "Data leakage: CODIGO(s) appear in BOTH train and test: "
                f"{', '.join(sorted(overlap))}. Re-generate the split with "
                "`python -m src.training_data --split-mode drug`."
            )

    # ------------------------------------------------------------------ #
    # Data loading helpers
    # ------------------------------------------------------------------ #

    def load_csv_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load a CSV preserving CODIGO as a string."""
        try:
            df: pd.DataFrame = pd.read_csv(file_path, dtype={"CODIGO": str})
            print(f"Loaded {file_path} ({len(df)} rows)")
            return df
        except Exception as exc:
            print(f"Could not load {file_path}: {exc}")
            return None

    def import_split_data(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        full_annotations: Optional[pd.DataFrame] = None,
    ) -> None:
        """Import the train/test split used by the NER pipeline.

        Parameters
        ----------
        train, test:
            Per-código train/test split defining which códigos belong to each
            CoNLL partition.
        full_annotations:
            Optional dataframe with **every known mention** for every
            código. The CoNLL builder uses it to tag the section-4.8 text
            so that adverse reactions whose row landed in the *other* split
            are still labelled gold (otherwise the model is punished for
            correctly detecting them as ``spurious``).  When ``None``, this
            defaults to ``pd.concat([train, test])`` — fine when train and
            test together cover every annotation, but ideally callers
            should pass the rows loaded from
            ``data/splits/full_annotations.csv``.
        """
        # Make sure the CODIGO column is a string so .txt lookups work.
        train = train.copy()
        test = test.copy()
        if "CODIGO" in train.columns:
            train["CODIGO"] = train["CODIGO"].astype(str).str.strip()
        if "CODIGO" in test.columns:
            test["CODIGO"] = test["CODIGO"].astype(str).str.strip()

        self._validate_split_codigos(set(train["CODIGO"]), set(test["CODIGO"]))

        self.train_dataframe = train
        self.test_dataframe = test

        if full_annotations is None:
            full = pd.concat([train, test], ignore_index=True)
        else:
            full = full_annotations.copy()
            if "CODIGO" in full.columns:
                full["CODIGO"] = full["CODIGO"].astype(str).str.strip()
        self.full_annotations = full

    # ------------------------------------------------------------------ #
    # NER (CoNLL-based pipeline)
    # ------------------------------------------------------------------ #

    def prepare_ner_conll_data(self) -> None:
        """Build train/test CoNLL datasets from the raw section-4.8 text.

        The código partition comes from ``self.train_dataframe`` /
        ``self.test_dataframe`` (set by :meth:`import_split_data`), but the
        BIO tagger is fed the rows of ``self.full_annotations`` for those
        códigos so every known mention contributes to the gold labels.
        """
        if self.train_dataframe.empty or self.test_dataframe.empty:
            raise RuntimeError(
                "Train/test dataframes are empty. Call import_split_data() first."
            )

        train_codigos = set(self.train_dataframe["CODIGO"].astype(str).str.strip())
        test_codigos = set(self.test_dataframe["CODIGO"].astype(str).str.strip())
        self._validate_split_codigos(train_codigos, test_codigos)

        full = self.full_annotations
        if "CODIGO" in full.columns:
            train_full = full[full["CODIGO"].astype(str).str.strip().isin(train_codigos)]
            test_full = full[full["CODIGO"].astype(str).str.strip().isin(test_codigos)]
        else:
            train_full = self.train_dataframe
            test_full = self.test_dataframe

        print(
            f"\nBuilding CoNLL BIO documents from raw text "
            f"(train mentions: {len(train_full):,} rows, "
            f"test mentions: {len(test_full):,} rows)..."
        )
        train_docs = self.conll_builder.build_dataset(train_full)
        train_summary = self.conll_builder.summarise(train_docs)
        test_docs = self.conll_builder.build_dataset(test_full)
        test_summary = self.conll_builder.summarise(test_docs)

        if not train_docs:
            raise RuntimeError(
                "No training documents could be built. Check that the .txt "
                f"files exist under {self.conll_builder.txt_dir}"
            )

        def _bio_counts(summary: dict) -> tuple[int, int, float]:
            counts = summary.get("label_counts", {})
            o_count = int(counts.get("O", 0))
            non_o_count = int(sum(v for k, v in counts.items() if k != "O"))
            total = o_count + non_o_count
            pct = (non_o_count / total * 100.0) if total else 0.0
            return o_count, non_o_count, pct

        train_o, train_non_o, train_pct = _bio_counts(train_summary)
        test_o, test_non_o, test_pct = _bio_counts(test_summary)

        print(
            f"   Train: {train_summary['documents']} docs / "
            f"{train_summary['sentences']} sentences "
            f"({train_summary['tokens']} tokens) | "
            f"missing .txt: {len(train_summary['missing_txt_files'])} | "
            f"docs without entities: {len(train_summary['docs_with_no_tags'])}"
        )
        print(
            f"      BIO tags: non-O={train_non_o:,} | O={train_o:,} "
            f"| non-O share={train_pct:.2f}%"
        )
        print(
            f"   Test:  {test_summary['documents']} docs / "
            f"{test_summary['sentences']} sentences "
            f"({test_summary['tokens']} tokens) | "
            f"missing .txt: {len(test_summary['missing_txt_files'])} | "
            f"docs without entities: {len(test_summary['docs_with_no_tags'])}"
        )
        print(
            f"      BIO tags: non-O={test_non_o:,} | O={test_o:,} "
            f"| non-O share={test_pct:.2f}%"
        )
        print("   Train label distribution:")
        for label, count in sorted(
            train_summary["label_counts"].items(), key=lambda kv: (-kv[1], kv[0])
        ):
            print(f"      {label:<12s} {count:>8,}")

        train_dataset = Dataset.from_list(train_docs)
        test_dataset = Dataset.from_list(test_docs)
        self.ner_dataset = DatasetDict(
            {"train": train_dataset, "test": test_dataset}
        )
        self.ner_train_summary = train_summary
        self.ner_test_summary = test_summary

        # Persist the BIO-tagged corpora next to the (future) NER model so the
        # same files can be re-loaded by external trainers (e.g. parse_conll
        # in `other/pubmedbert_ner_sider.ipynb`). Example provided by Salva.
        self.export_conll_files()

    def export_conll_files(self, output_dir: Optional[str] = None) -> dict:
        """Write `train.conll` and `test.conll` to disk in standard BIO format.

        The files are saved to the same directory where the NER model is
        stored (``{self.output_dir}/ner_conll_model``) unless a custom
        ``output_dir`` is provided. Format: ``token<TAB>label`` per line, one
        blank line between sentences and a ``# {codigo}`` header on document
        boundaries — fully compatible with the ``parse_conll`` helper used in
        ``other/pubmedbert_ner_sider.ipynb`` (which skips ``#`` lines).
        """
        if not hasattr(self, "ner_dataset"):
            raise RuntimeError("Call prepare_ner_conll_data() first.")

        target_dir = output_dir or f"{self.output_dir}/ner_conll_model"
        os.makedirs(target_dir, exist_ok=True)

        written: dict = {}
        for split_name, file_name in (("train", "train.conll"), ("test", "test.conll")):
            split = self.ner_dataset[split_name]
            file_path = os.path.join(target_dir, file_name)
            self._write_conll_file(split, file_path)
            written[split_name] = file_path
            print(
                f"Wrote {file_path} ({len(split)} sentences)"
            )
        return written

    def _write_conll_file(self, examples, path: str) -> None:
        """Serialise an iterable of ``{codigo, tokens, ner_tags}`` dicts."""
        id_to_label = self.id_to_label
        last_codigo: Optional[str] = None
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                codigo = ex.get("codigo")
                if codigo and codigo != last_codigo:
                    f.write(f"# {codigo}\n")
                    last_codigo = codigo
                for token, tag_id in zip(ex["tokens"], ex["ner_tags"]):
                    label = id_to_label[int(tag_id)]
                    # Replace any stray whitespace inside a token so the
                    # ``token<TAB>label`` contract is never broken.
                    safe_token = token.replace("\t", " ").replace("\n", " ")
                    f.write(f"{safe_token}\t{label}\n")
                f.write("\n")

    def tokenize_and_align_ner_labels(
        self,
        drop_empty_sentence_prob: float = 0.0,
        drop_seed: Optional[int] = None,
    ) -> None:
        """Subword-align BIO labels (first subtoken keeps the label, rest -100).

        Parameters
        ----------
        drop_empty_sentence_prob:
            If > 0, drop that fraction of *training* sentences whose labels
            are entirely ``O`` before tokenisation. Helps with the heavy
            class imbalance once the gold labels are complete (most
            sentences in section 4.8 contain no adverse-reaction span).
            The test split is always kept intact so evaluation numbers stay
            comparable across runs.
        drop_seed:
            Optional seed for the empty-sentence sampler.
        """
        if not hasattr(self, "ner_dataset"):
            raise RuntimeError("Call prepare_ner_conll_data() first.")

        if not 0.0 <= drop_empty_sentence_prob < 1.0:
            raise ValueError(
                "drop_empty_sentence_prob must be in [0, 1)."
            )

        if drop_empty_sentence_prob > 0.0:
            o_id = self.label_to_id["O"]
            rng = random.Random(drop_seed)
            train_split = self.ner_dataset["train"]

            def _keep(example) -> bool:
                if any(int(t) != o_id for t in example["ner_tags"]):
                    return True
                return rng.random() >= drop_empty_sentence_prob

            kept = train_split.filter(_keep)
            dropped = len(train_split) - len(kept)
            print(
                f"Dropped {dropped:,}/{len(train_split):,} all-O training "
                f"sentences (p={drop_empty_sentence_prob})."
            )
            self.ner_dataset = DatasetDict(
                {"train": kept, "test": self.ner_dataset["test"]}
            )

        max_length = self.max_length
        tokenizer = self.tokenizer

        def tokenize_and_align(examples):
            tokenized = tokenizer(
                examples["tokens"],
                truncation=True,
                is_split_into_words=True,
                max_length=max_length,
            )

            aligned_labels = []
            for i, label_seq in enumerate(examples["ner_tags"]):
                word_ids = tokenized.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(label_seq[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                aligned_labels.append(label_ids)

            tokenized["labels"] = aligned_labels
            return tokenized

        self.tokenized_ner_datasets = self.ner_dataset.map(
            tokenize_and_align,
            batched=True,
            remove_columns=self.ner_dataset["train"].column_names,
        )
        print(
            f"Tokenised NER dataset | train={len(self.tokenized_ner_datasets['train'])} "
            f"| test={len(self.tokenized_ner_datasets['test'])}"
        )

    def train_ner_conll_model(
        self,
        num_train_epochs: int = 10,
        per_device_train_batch_size: int = 16,
        per_device_eval_batch_size: int = 16,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        early_stopping_patience: int = 3,
    ) -> None:
        """Fine-tune a token-classification model on the CoNLL dataset."""
        if self.tokenized_ner_datasets is None:
            raise RuntimeError("Call tokenize_and_align_ner_labels() first.")

        ner_model = AutoModelForTokenClassification.from_pretrained(
            self.pretrained_model_path,
            num_labels=len(self.label_to_id),
            id2label=self.id_to_label,
            label2id=self.label_to_id,
            ignore_mismatched_sizes=True,
        )

        # Pick a precision compatible with the available device.
        bf16 = bool(
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        )
        fp16 = bool(torch.cuda.is_available() and not bf16)

        steps_per_epoch = math.ceil(
            len(self.tokenized_ner_datasets["train"]) / per_device_train_batch_size
        )
        warmup_steps = math.ceil(steps_per_epoch * num_train_epochs * warmup_ratio)

        output_dir = f"{self.output_dir}/ner_conll_model"
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            lr_scheduler_type="linear",
            bf16=bf16,
            fp16=fp16,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            seed=42,
            data_seed=42,
            report_to="none",
        )

        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True,
            pad_to_multiple_of=8,
        )

        compute_metrics = self._build_compute_metrics()

        trainer = Trainer(
            model=ner_model,
            args=training_args,
            train_dataset=self.tokenized_ner_datasets["train"],
            eval_dataset=self.tokenized_ner_datasets["test"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
            ],
        )

        print(
            f"Starting CoNLL NER training | tags={self.ner_entity_types} "
            f"| epochs={num_train_epochs} | bf16={bf16} | fp16={fp16}"
        )
        trainer.train()

        print("Saving NER model...")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Re-emit the CoNLL files inside the final model directory so anyone
        # downloading the model also gets the exact training data layout.
        self.export_conll_files(output_dir)

        self._disable_notebook_progress_callback(trainer)

        # Final evaluation + per-entity report.
        eval_results = trainer.evaluate()
        per_entity = self._per_entity_report()
        self.ner_results = {
            "eval": {
                k: float(v) if isinstance(v, (int, float, np.floating)) else v
                for k, v in eval_results.items()
            },
            "per_entity": per_entity,
            "labels": list(self.label_to_id.keys()),
            "entity_types": list(self.ner_entity_types),
            "train_summary": self.ner_train_summary,
            "test_summary": self.ner_test_summary,
        }
        with open(f"{output_dir}/ner_results.json", "w", encoding="utf-8") as f:
            json.dump(self.ner_results, f, ensure_ascii=False, indent=2)

        # Keep a copy of the trainer/model accessible for the notebook.
        self.ner_trainer = trainer
        print("CoNLL NER training completed.")

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #

    def _build_compute_metrics(self):
        """Return a `compute_metrics` callable using seqeval (entity-level)."""
        try:
            from seqeval.metrics import (
                accuracy_score,
                classification_report,
                f1_score,
                precision_score,
                recall_score,
            )
            from seqeval.scheme import IOB2
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "seqeval is required for NER metrics. Install with `pip install seqeval`."
            ) from exc

        id_to_label = self.id_to_label
        entity_types = list(self.ner_entity_types)

        def compute_metrics(eval_preds):
            logits, labels = eval_preds
            predictions = np.argmax(logits, axis=-1)

            true_labels: list[list[str]] = []
            true_predictions: list[list[str]] = []
            for pred_seq, label_seq in zip(predictions, labels):
                cur_labels: list[str] = []
                cur_preds: list[str] = []
                for pred, label in zip(pred_seq, label_seq):
                    if label != -100:
                        cur_labels.append(id_to_label[int(label)])
                        cur_preds.append(id_to_label[int(pred)])
                true_labels.append(cur_labels)
                # Repair orphan I-X tags so the spurious bucket only counts
                # genuinely wrong entities, not malformed IOB sequences.
                true_predictions.append(sanitise_bio_predictions(cur_preds))

            # Strict, entity-level P/R/F1 under the IOB2 scheme: a predicted
            # entity counts as a true positive only if its type AND its full
            # span exactly match the gold entity (token-level accuracy is
            # reported separately for context).
            metrics = {
                "precision": precision_score(
                    true_labels, true_predictions, mode="strict", scheme=IOB2
                ),
                "recall": recall_score(
                    true_labels, true_predictions, mode="strict", scheme=IOB2
                ),
                "f1": f1_score(
                    true_labels, true_predictions, mode="strict", scheme=IOB2
                ),
                "token_accuracy": accuracy_score(true_labels, true_predictions),
            }

            # Per-entity F1 (also entity-level) so multi-tag training surfaces
            # each tag individually in the training logs.
            if len(entity_types) > 1:
                report = classification_report(
                    true_labels,
                    true_predictions,
                    output_dict=True,
                    mode="strict",
                    scheme=IOB2,
                    zero_division=0,
                )
                for ent in entity_types:
                    block = report.get(ent, {})
                    if isinstance(block, dict):
                        metrics[f"f1_{ent}"] = float(block.get("f1-score", 0.0))

            return metrics

        return compute_metrics

    def _per_entity_report(self) -> dict:
        """Return the seqeval classification_report as a dict (entity-level)."""
        try:
            from seqeval.metrics import classification_report
            from seqeval.scheme import IOB2
        except ImportError:  # pragma: no cover
            return {}

        if self.tokenized_ner_datasets is None or not hasattr(self, "ner_trainer"):
            return {}

        predictions = self.ner_trainer.predict(self.tokenized_ner_datasets["test"])
        preds = np.argmax(predictions.predictions, axis=-1)
        label_ids = predictions.label_ids

        true_labels: list[list[str]] = []
        true_predictions: list[list[str]] = []
        for pred_seq, label_seq in zip(preds, label_ids):
            cur_labels: list[str] = []
            cur_preds: list[str] = []
            for pred, label in zip(pred_seq, label_seq):
                if label != -100:
                    cur_labels.append(self.id_to_label[int(label)])
                    cur_preds.append(self.id_to_label[int(pred)])
            true_labels.append(cur_labels)
            true_predictions.append(sanitise_bio_predictions(cur_preds))

        report_str = classification_report(
            true_labels,
            true_predictions,
            digits=4,
            mode="strict",
            scheme=IOB2,
            zero_division=0,
        )
        report_dict = classification_report(
            true_labels,
            true_predictions,
            output_dict=True,
            mode="strict",
            scheme=IOB2,
            zero_division=0,
        )
        # Cast numpy floats to plain floats so the dict is JSON-serialisable.
        report_dict_clean: dict = {}
        for entity, metrics in report_dict.items():
            if isinstance(metrics, dict):
                report_dict_clean[entity] = {
                    k: float(v) for k, v in metrics.items()
                }
            else:
                report_dict_clean[entity] = float(metrics)

        print("\nPer-entity classification report (strict / IOB2):\n" + str(report_str))
        return report_dict_clean

    @staticmethod
    def _disable_notebook_progress_callback(trainer: Trainer) -> None:
        """Remove the Jupyter-only progress callback before standalone eval.

        Transformers installs NotebookProgressCallback automatically inside a
        notebook. After ``train()`` finishes, that callback clears its internal
        training tracker, so a later manual ``trainer.evaluate()`` raises
        ``RuntimeError: on_train_begin must be called before on_evaluate``.
        """
        for callback in list(trainer.callback_handler.callbacks):
            if callback.__class__.__name__ == "NotebookProgressCallback":
                trainer.remove_callback(callback)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save_ner_label_mapping(self) -> None:
        """Persist the NER label mapping."""
        os.makedirs(self.output_dir, exist_ok=True)

        with open(
            f"{self.output_dir}/ner_conll_label_mapping.json", "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "entity_types": list(self.ner_entity_types),
                    "label_to_id": self.label_to_id,
                    "id_to_label": {str(k): v for k, v in self.id_to_label.items()},
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"Saved NER label mapping under {self.output_dir}")

    # ------------------------------------------------------------------ #
    # Convenience orchestrator
    # ------------------------------------------------------------------ #

    def run(
        self,
        train_csv: str,
        test_csv: str,
        full_annotations_csv: Optional[str] = None,
        skip_ner: bool = False,
        ner_epochs: int = 5,
    ) -> None:
        """End-to-end pipeline: load CSVs, train NER and save artefacts.

        Parameters
        ----------
        full_annotations_csv:
            Optional path to ``data/splits/full_annotations.csv`` produced
            by ``python -m src.training_data``. When provided, the BIO
            tagger uses every mention in that CSV for the códigos in each
            split (recommended). When ``None``, falls back to the union of
            ``train_csv`` + ``test_csv``.
        """
        train_df = self.load_csv_data(train_csv)
        test_df = self.load_csv_data(test_csv)
        if train_df is None or test_df is None:
            raise RuntimeError("Could not load train/test CSVs.")

        full_df: Optional[pd.DataFrame] = None
        if full_annotations_csv is not None:
            full_df = self.load_csv_data(full_annotations_csv)
            if full_df is None:
                raise RuntimeError(
                    f"Could not load full annotations CSV: {full_annotations_csv}"
                )

        self.import_split_data(train_df, test_df, full_annotations=full_df)

        if not skip_ner:
            self.prepare_ner_conll_data()
            self.tokenize_and_align_ner_labels()
            self.train_ner_conll_model(num_train_epochs=ner_epochs)

        self.save_ner_label_mapping()
