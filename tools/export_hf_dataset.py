"""Export the CIMA section-4.8 NER dataset for publishing on Hugging Face.

Generates one HF *config* per tag combination (``react``, ``active``,
``freq``, ``sys``, ``react-active-freq-sys``) using only the **real**
corpus (``data/medicamentos/txt`` + ``data/splits/{train,test}.csv``
+ ``data/splits/full_annotations.csv``).

Each config is written twice under
``huggingface/cima-section48-ner/``:

- ``data/{config}/{split}.parquet``  → canonical HF format with
  ``Sequence(ClassLabel)`` for ``ner_tags``.
- ``conll/{config}/{split}.conll``   → plain ``token\\tlabel`` BIO file,
  blank line between sentences, ``# {codigo}`` document headers.

Also dumps ``stats/{config}.json`` with per-split label counts so the
dataset card can quote them directly.

Run from the repo root::

    python tools/export_hf_dataset.py

The script is intentionally self-contained: it does **not** touch the
training pipeline (``SupervisedTrainerConll``) and re-builds everything
from the same `ConllBuilder` the trainer uses, so the published artefacts
are byte-identical to what the training notebook would produce internally.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

import pandas as pd
from datasets import ClassLabel, Dataset, Features, Sequence, Value

# Make the repo root importable when invoking the script directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.conll_ner_builder import ConllBuilder  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TXT_DIR = os.path.join(_REPO_ROOT, "data/medicamentos/txt")
TRAIN_CSV = os.path.join(_REPO_ROOT, "data/splits/train.csv")
TEST_CSV = os.path.join(_REPO_ROOT, "data/splits/test.csv")
FULL_ANNOTATIONS_CSV = os.path.join(_REPO_ROOT, "data/splits/full_annotations.csv")

OUTPUT_DIR = os.path.join(_REPO_ROOT, "huggingface/CIMA-4.8-ADR-NER")

# Each entry becomes an HF config. The name is the lowercase, dash-joined
# tag combination (``react-active-freq-sys`` for multi-tag).
TAG_COMBINATIONS: list[list[str]] = [
    ["REACT"],
    ["ACTIVE"],
    ["FREQ"],
    ["SYS"],
    ["REACT", "ACTIVE", "FREQ", "SYS"],
]


def _config_name(tags: list[str]) -> str:
    return "-".join(t.lower() for t in tags)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"CODIGO": str})
    df["CODIGO"] = df["CODIGO"].astype(str).str.strip()
    return df


def _build_examples(
    builder: ConllBuilder,
    split_codigos: set[str],
    full_annotations: pd.DataFrame,
) -> list[dict]:
    """Return a list of CoNLL examples for the códigos in ``split_codigos``.

    Mirrors what ``SupervisedTrainerConll.prepare_ner_conll_data`` does
    internally: subsets ``full_annotations`` to the split's códigos so
    every known mention contributes to the gold BIO labels.
    """
    subset = full_annotations[full_annotations["CODIGO"].isin(split_codigos)]
    if subset.empty:
        return []
    return builder.build_dataset(subset)


def _to_hf_dataset(examples: list[dict], label_names: list[str]) -> Dataset:
    """Wrap ``examples`` in a typed ``datasets.Dataset``.

    ``ner_tags`` is declared as ``Sequence(ClassLabel(...))`` so the HF
    hub viewer renders human-readable labels and downstream consumers can
    call ``features["ner_tags"].feature.int2str(...)`` directly.
    """
    features = Features(
        {
            "codigo": Value("string"),
            "sent_idx": Value("int32"),
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=label_names)),
        }
    )
    return Dataset.from_list(examples, features=features)


def _write_conll(examples: list[dict], path: str, id_to_label: dict[int, str]) -> None:
    """Serialise ``examples`` to a plain CoNLL BIO file.

    Format matches ``SupervisedTrainerConll._write_conll_file`` so the
    published files use the same representation as the training artifacts.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    last_codigo: Optional[str] = None
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            codigo = ex.get("codigo")
            if codigo and codigo != last_codigo:
                f.write(f"# {codigo}\n")
                last_codigo = codigo
            for token, tag_id in zip(ex["tokens"], ex["ner_tags"]):
                label = id_to_label[int(tag_id)]
                safe_token = token.replace("\t", " ").replace("\n", " ")
                f.write(f"{safe_token}\t{label}\n")
            f.write("\n")


def _summary(examples: list[dict], id_to_label: dict[int, str]) -> dict:
    from collections import Counter

    label_counts: Counter[str] = Counter()
    token_count = 0
    unique_codigos: set[str] = set()
    for ex in examples:
        token_count += len(ex["tokens"])
        label_counts.update(id_to_label[int(i)] for i in ex["ner_tags"])
        unique_codigos.add(ex["codigo"])
    return {
        "documents": len(unique_codigos),
        "sentences": len(examples),
        "tokens": token_count,
        "label_counts": dict(label_counts),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def export_all() -> None:
    if not os.path.isdir(TXT_DIR):
        raise FileNotFoundError(
            f"Raw section-4.8 text directory not found: {TXT_DIR}. "
            "Run the CIMA download pipeline first."
        )

    train_df = _load_csv(TRAIN_CSV)
    test_df = _load_csv(TEST_CSV)
    full_df = _load_csv(FULL_ANNOTATIONS_CSV)

    train_codigos = set(train_df["CODIGO"])
    test_codigos = set(test_df["CODIGO"])
    overlap = train_codigos & test_codigos
    if overlap:
        # Mirror the trainer behaviour: warn but continue. The user can
        # regenerate the splits without leakage by running
        # ``python -m src.training_data --split-mode drug --random-state 42``.
        print(
            f"{len(overlap)} CODIGO(s) aparecen en train y test: {sorted(overlap)}. "
            "El dataset publicado heredará la fuga; considera regenerar los splits."
        )

    print(
        f"Exportando dataset NER para Hugging Face\n"
        f"   train.csv: {len(train_df):,} filas, {len(train_codigos)} fármacos\n"
        f"   test.csv:  {len(test_df):,} filas, {len(test_codigos)} fármacos\n"
        f"   full_annotations.csv: {len(full_df):,} filas\n"
        f"   destino: {OUTPUT_DIR}\n"
    )

    global_summary: dict[str, dict] = {}
    for tags in TAG_COMBINATIONS:
        cfg = _config_name(tags)
        print(f"\n=== Config: {cfg} (tags={tags}) ===")

        builder = ConllBuilder(txt_dir=TXT_DIR, entity_types=tags)
        label_names = list(builder.label_to_id.keys())
        id_to_label = builder.id_to_label

        train_examples = _build_examples(builder, train_codigos, full_df)
        train_missing = list(builder.missing_txt_files)
        train_no_tags = list(builder.docs_with_no_tags)
        test_examples = _build_examples(builder, test_codigos, full_df)
        test_missing = list(builder.missing_txt_files)
        test_no_tags = list(builder.docs_with_no_tags)

        train_summary = _summary(train_examples, id_to_label)
        test_summary = _summary(test_examples, id_to_label)
        train_summary.update(missing_txt_files=train_missing, docs_with_no_tags=train_no_tags)
        test_summary.update(missing_txt_files=test_missing, docs_with_no_tags=test_no_tags)

        parquet_dir = os.path.join(OUTPUT_DIR, "data", cfg)
        conll_dir = os.path.join(OUTPUT_DIR, "conll", cfg)
        stats_dir = os.path.join(OUTPUT_DIR, "stats")
        os.makedirs(parquet_dir, exist_ok=True)
        os.makedirs(conll_dir, exist_ok=True)
        os.makedirs(stats_dir, exist_ok=True)

        # Parquet
        train_ds = _to_hf_dataset(train_examples, label_names)
        test_ds = _to_hf_dataset(test_examples, label_names)
        train_parquet = os.path.join(parquet_dir, "train.parquet")
        test_parquet = os.path.join(parquet_dir, "test.parquet")
        train_ds.to_parquet(train_parquet)
        test_ds.to_parquet(test_parquet)

        # CoNLL plano
        train_conll = os.path.join(conll_dir, "train.conll")
        test_conll = os.path.join(conll_dir, "test.conll")
        _write_conll(train_examples, train_conll, id_to_label)
        _write_conll(test_examples, test_conll, id_to_label)

        # Stats
        stats_path = os.path.join(stats_dir, f"{cfg}.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": cfg,
                    "entity_types": tags,
                    "labels": label_names,
                    "train": train_summary,
                    "test": test_summary,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"  train: {train_summary['documents']} docs / "
            f"{train_summary['sentences']:,} sentences / "
            f"{train_summary['tokens']:,} tokens"
        )
        print(
            f"  test:  {test_summary['documents']} docs / "
            f"{test_summary['sentences']:,} sentences / "
            f"{test_summary['tokens']:,} tokens"
        )
        print(f"  {train_parquet}  ({len(train_ds)} rows)")
        print(f"  {test_parquet}   ({len(test_ds)} rows)")
        print(f"  {train_conll}")
        print(f"  {test_conll}")
        print(f"  {stats_path}")

        global_summary[cfg] = {
            "labels": label_names,
            "train_sentences": train_summary["sentences"],
            "test_sentences": test_summary["sentences"],
            "train_tokens": train_summary["tokens"],
            "test_tokens": test_summary["tokens"],
        }

    summary_path = os.path.join(OUTPUT_DIR, "stats", "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, ensure_ascii=False, indent=2)
    print(f"\nExportación completa. Resumen global: {summary_path}")


if __name__ == "__main__":
    export_all()
