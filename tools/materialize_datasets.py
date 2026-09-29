"""Materialize ignored training corpora from the public dataset releases."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

DEFAULT_REAL_DATASET = "guerrerotook/CIMA-4.8-ADR"
DEFAULT_SYNTHETIC_EXPORT = Path("huggingface/CIMA-4.8-ADR-NER-EXTENDED")
DEFAULT_SPLITS_DIRECTORY = Path("data/splits")

REAL_SPLITS = ("train.csv", "test.csv", "full_annotations.csv")
SYNTHETIC_SPLITS = (
    "train_synth.csv",
    "test_synth.csv",
    "full_annotations_synth.csv",
)


def load_split_codes(
    splits_directory: str | Path,
    filenames: Iterable[str],
) -> list[str]:
    codes: list[str] = []
    for filename in filenames:
        path = Path(splits_directory) / filename
        dataframe = pd.read_csv(path, usecols=["CODIGO"], dtype={"CODIGO": str})
        values = dataframe["CODIGO"].fillna("").astype(str).str.strip()
        codes.extend(value for value in values if value)
    return list(dict.fromkeys(codes))


def materialize_real_corpus(
    dataset_id: str,
    output_directory: str | Path,
    required_codes: set[str],
    scope: str,
    overwrite: bool,
) -> int:
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split="train")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    written = 0
    for row in dataset:
        codigo = str(row["file_id"]).strip()
        if not codigo or (scope == "splits" and codigo not in required_codes):
            continue
        path = output / f"{codigo}.txt"
        if path.exists() and not overwrite:
            continue
        path.write_text(str(row["text"]), encoding="utf-8")
        written += 1
    _validate_texts(required_codes, output, "real")
    return written


def materialize_synthetic_corpus(
    export_directory: str | Path,
    text_output_directory: str | Path,
    entities_output_directory: str | Path,
    expected_codes: Iterable[str],
    overwrite: bool,
) -> tuple[int, int]:
    export = Path(export_directory)
    source_texts = export / "txt"
    documents_directory = export / "data" / "documents"
    text_output = Path(text_output_directory)
    entities_output = Path(entities_output_directory)
    text_output.mkdir(parents=True, exist_ok=True)
    entities_output.mkdir(parents=True, exist_ok=True)

    expected_by_key = {_variant_key(code): code for code in expected_codes}
    written_texts = 0
    for source in sorted(source_texts.glob("*.txt")):
        target_code = expected_by_key.get(_variant_key(source.stem), source.stem)
        target = text_output / f"{target_code}.txt"
        if target.exists() and not overwrite:
            continue
        shutil.copyfile(source, target)
        written_texts += 1

    frames = [pd.read_parquet(path) for path in sorted(documents_directory.glob("*.parquet"))]
    if not frames:
        raise FileNotFoundError(f"No synthetic document Parquet files found in {documents_directory}")

    written_entities = 0
    for row in pd.concat(frames, ignore_index=True).to_dict(orient="records"):
        source_code = str(row["codigo"]).strip()
        target_code = expected_by_key.get(_variant_key(source_code), source_code)
        target = entities_output / f"{target_code}.json"
        if target.exists() and not overwrite:
            continue
        payload = {
            "codigo": target_code.rsplit("_v", 1)[0],
            "variante": int(row["variante"]),
            "codigo_variant": target_code,
            "medicamento": str(row["medicamento"]),
            "discard_ratio": float(row["discard_ratio"]),
            "entities": _json_compatible(row["entities"]),
        }
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written_entities += 1

    expected = set(expected_codes)
    _validate_texts(expected, text_output, "synthetic")
    _validate_entities(expected, entities_output)
    return written_texts, written_entities


def _variant_key(codigo: str) -> str:
    value = str(codigo).strip()
    if "_v" not in value:
        return value.lstrip("0") or "0"
    base, variant = value.rsplit("_v", 1)
    return f"{base.lstrip('0') or '0'}_v{variant}"


def _json_compatible(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _json_compatible(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _validate_texts(codes: Iterable[str], directory: Path, label: str) -> None:
    missing = sorted(code for code in codes if not (directory / f"{code}.txt").is_file())
    if missing:
        raise RuntimeError(f"Missing {label} text files for CODIGO values: {missing}")


def _validate_entities(codes: Iterable[str], directory: Path) -> None:
    missing = sorted(code for code in codes if not (directory / f"{code}.json").is_file())
    if missing:
        raise RuntimeError(f"Missing synthetic entity files for CODIGO values: {missing}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize ignored training data from public releases."
    )
    parser.add_argument("--real-dataset", default=DEFAULT_REAL_DATASET)
    parser.add_argument("--real-output", default="data/medicamentos/txt")
    parser.add_argument("--real-scope", choices=["all", "splits"], default="all")
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--synthetic-export", default=str(DEFAULT_SYNTHETIC_EXPORT))
    parser.add_argument("--synthetic-text-output", default="data/medicamentos_synth/txt")
    parser.add_argument(
        "--synthetic-entities-output",
        default="data/medicamentos_synth/entities",
    )
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--splits-dir", default=str(DEFAULT_SPLITS_DIRECTORY))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    real_codes = set(load_split_codes(args.splits_dir, REAL_SPLITS))
    synthetic_codes = load_split_codes(args.splits_dir, SYNTHETIC_SPLITS)

    if not args.skip_real:
        written = materialize_real_corpus(
            dataset_id=args.real_dataset,
            output_directory=args.real_output,
            required_codes=real_codes,
            scope=args.real_scope,
            overwrite=args.overwrite,
        )
        print(f"Materialized {written} real text files.")

    if not args.skip_synthetic:
        text_count, entity_count = materialize_synthetic_corpus(
            export_directory=args.synthetic_export,
            text_output_directory=args.synthetic_text_output,
            entities_output_directory=args.synthetic_entities_output,
            expected_codes=synthetic_codes,
            overwrite=args.overwrite,
        )
        print(
            f"Materialized {text_count} synthetic text files and "
            f"{entity_count} entity files."
        )


if __name__ == "__main__":
    main()