"""Utilities for loading and stratifying the salva_todos.xlsx workbook."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

# Default location relative to the repo root. Users can override with --excel-path
# or by passing an absolute path to the loader functions.
DEFAULT_EXCEL_PATH = Path("data/salva_todos.xlsx")
DEFAULT_FULL_ANNOTATIONS_FILENAME = "full_annotations.csv"

SplitMode = Literal["drug", "row"]


@dataclass(frozen=True)
class TrainTestSplit:
    """Container for the stratified DataFrame split."""
    train: pd.DataFrame
    test: pd.DataFrame


def load_salva_dataset(excel_path: Path | str = DEFAULT_EXCEL_PATH) -> pd.DataFrame:
    """
    Read the salva_todos.xlsx workbook into a DataFrame.

    Parameters
    ----------
    excel_path:
        Absolute or relative path to the Excel workbook.

    Returns
    -------
    pd.DataFrame
        Workbook contents including headers.

    Raises
    ------
    FileNotFoundError
        If the workbook cannot be located.
    ValueError
        If the workbook is empty.
    """
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"Workbook not found at {path.resolve()}")
    dataframe = pd.read_excel(path, dtype={"CODIGO": str})
    if dataframe.empty:
        raise ValueError(f"No rows found in {path.resolve()}")
    return dataframe


def _resolve_codigo_column(columns: Iterable[str]) -> str:
    """Find the column that represents 'codigo' (case-insensitive)."""
    for column in columns:
        if column.lower() == "codigo":
            return column
    return next(iter(columns))


def split_dataframe_by_codigo(
    dataframe: pd.DataFrame,
    train_ratio: float = 0.7,
    random_state: int | None = None,
) -> TrainTestSplit:
    """
    Split a DataFrame into train/test sets while preserving per-código proportions.

    .. deprecated::
        This splitter assigns *rows* of the same drug to both splits, which
        causes data leakage for any task that consumes the raw section-4.8
        text per drug (NER, document classification, etc.). Prefer
        :func:`split_dataframe_by_drug` unless you specifically need a
        per-row stratification.

    Parameters
    ----------
    dataframe:
        Source DataFrame containing a código column.
    train_ratio:
        Fraction of each código group to keep in the train set.
    random_state:
        Optional seed for deterministic shuffling.

    Returns
    -------
    TrainTestSplit
        Stratified train/test DataFrames.
    """
    if dataframe.empty:
        raise ValueError("Cannot split an empty DataFrame.")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")

    codigo_column = _resolve_codigo_column(dataframe.columns)
    rng = random.Random(random_state)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for _, group in dataframe.groupby(codigo_column, sort=False):
        indices = group.index.to_list()
        rng.shuffle(indices)

        train_count = int(len(indices) * train_ratio)
        if train_count == 0 and indices:
            train_count = 1
        if train_count == len(indices) and len(indices) > 1:
            train_count -= 1

        train_indices.extend(indices[:train_count])
        test_indices.extend(indices[train_count:])

    train_df = dataframe.loc[train_indices].reset_index(drop=True)
    test_df = dataframe.loc[test_indices].reset_index(drop=True)
    return TrainTestSplit(train=train_df, test=test_df)


def split_dataframe_by_drug(
    dataframe: pd.DataFrame,
    train_ratio: float = 0.7,
    random_state: int | None = None,
) -> TrainTestSplit:
    """
    Split a DataFrame so that *every código belongs to exactly one split*.

    This is the correct splitter for downstream tasks that consume the raw
    section-4.8 text of each drug (NER, document classification, …) because
    it eliminates the train/test leakage produced by
    :func:`split_dataframe_by_codigo`.

    Parameters
    ----------
    dataframe:
        Source DataFrame containing a código column.
    train_ratio:
        Approximate fraction of códigos assigned to the train split. The
        actual ratio may differ slightly because whole drugs are assigned
        atomically.
    random_state:
        Optional seed for deterministic shuffling.

    Returns
    -------
    TrainTestSplit
        Train/test DataFrames whose código sets are disjoint.
    """
    if dataframe.empty:
        raise ValueError("Cannot split an empty DataFrame.")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")

    codigo_column = _resolve_codigo_column(dataframe.columns)
    # Normalize the código column so that mixed-dtype values (e.g. int 77045
    # vs the string "\xa077045") collapse to a single key. Without this,
    # `dict.fromkeys` treats them as distinct drugs and may scatter rows of
    # the same código across both splits, producing apparent leakage after
    # downstream string normalization.
    dataframe = dataframe.copy()
    dataframe[codigo_column] = (
        dataframe[codigo_column]
        .astype(str)
        .str.replace("\xa0", "", regex=False)
        .str.strip()
    )
    unique_codigos = list(dict.fromkeys(dataframe[codigo_column].tolist()))
    if len(unique_codigos) < 2:
        raise ValueError(
            "Need at least 2 distinct códigos to perform a drug-level split."
        )

    rng = random.Random(random_state)
    rng.shuffle(unique_codigos)

    train_count = int(round(len(unique_codigos) * train_ratio))
    train_count = max(1, min(len(unique_codigos) - 1, train_count))

    train_codigos = set(unique_codigos[:train_count])
    test_codigos = set(unique_codigos[train_count:])

    train_df = (
        dataframe[dataframe[codigo_column].isin(train_codigos)]
        .reset_index(drop=True)
    )
    test_df = (
        dataframe[dataframe[codigo_column].isin(test_codigos)]
        .reset_index(drop=True)
    )
    return TrainTestSplit(train=train_df, test=test_df)


def split_salva_dataset(
    excel_path: Path | str = DEFAULT_EXCEL_PATH,
    train_ratio: float = 0.7,
    random_state: int | None = None,
    split_mode: SplitMode = "drug",
) -> TrainTestSplit:
    """
    Convenience wrapper that loads the salva_todos.xlsx dataset and
    returns a stratified train/test split.

    Parameters
    ----------
    split_mode:
        ``"drug"`` (default) keeps each código entirely in one split — the
        correct choice for NER / document tasks.
        ``"row"`` reproduces the legacy per-código row stratification kept
        for backward compatibility (see :func:`split_dataframe_by_codigo`).
    """
    dataframe = load_salva_dataset(excel_path)
    if split_mode == "drug":
        return split_dataframe_by_drug(
            dataframe,
            train_ratio=train_ratio,
            random_state=random_state,
        )
    if split_mode == "row":
        return split_dataframe_by_codigo(
            dataframe,
            train_ratio=train_ratio,
            random_state=random_state,
        )
    raise ValueError(
        f"Unknown split_mode {split_mode!r}. Expected 'drug' or 'row'."
    )


def export_full_annotations(
    excel_path: Path | str = DEFAULT_EXCEL_PATH,
    output_dir: Path | str = Path("data") / "splits",
    filename: str = DEFAULT_FULL_ANNOTATIONS_FILENAME,
) -> Path:
    """Dump every annotation row of the workbook into a single CSV.

    Downstream NER pipelines use this file as the **single source of truth**
    for mention strings per código: the train/test CSVs only define which
    códigos belong to which split, while the BIO tagger pulls the *union of
    all known mentions* for each código from the file written here. That
    way, every adverse reaction that actually appears in the raw text gets
    a gold label, regardless of whether its row landed in train or test.
    """
    dataframe = load_salva_dataset(excel_path)
    codigo_column = _resolve_codigo_column(dataframe.columns)
    # Normalize CODIGO so downstream consumers (BIO tagger, split derivers)
    # always see the same key regardless of the workbook's mixed dtypes.
    dataframe[codigo_column] = (
        dataframe[codigo_column]
        .astype(str)
        .str.replace("\xa0", "", regex=False)
        .str.strip()
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target = output_path / filename
    dataframe.to_csv(target, index=False)
    return target


def save_split_to_csv(
    split: TrainTestSplit,
    output_dir: Path | str,
    train_filename: str = "train.csv",
    test_filename: str = "test.csv",
) -> tuple[Path, Path]:
    """
    Persist the stratified split into CSV files.

    Returns
    -------
    tuple[Path, Path]
        Paths to the generated train and test CSV files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / train_filename
    test_path = output_path / test_filename
    split.train.to_csv(train_path, index=False)
    split.test.to_csv(test_path, index=False)

    return train_path, test_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a train/test split from salva_todos.xlsx and (by "
            "default) export the full-annotations CSV used by the CoNLL "
            "pipeline."
        )
    )
    parser.add_argument(
        "--excel-path",
        type=Path,
        default=DEFAULT_EXCEL_PATH,
        help="Path to the salva_todos.xlsx workbook (default: data/salva_todos.xlsx).",
    )
    parser.add_argument(
        "--split-mode",
        choices=("drug", "row"),
        default="drug",
        help=(
            "How to partition the workbook. 'drug' (default) keeps every "
            "código in a single split (recommended for NER); 'row' "
            "reproduces the legacy per-código row stratification."
        ),
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of códigos (or rows, in 'row' mode) assigned to train (default: 0.7).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Optional random seed for deterministic shuffling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "splits",
        help="Directory where train/test CSVs will be written (default: data/splits).",
    )
    parser.add_argument(
        "--train-filename",
        type=str,
        default="train.csv",
        help="Filename for the train CSV (default: train.csv).",
    )
    parser.add_argument(
        "--test-filename",
        type=str,
        default="test.csv",
        help="Filename for the test CSV (default: test.csv).",
    )
    parser.add_argument(
        "--full-annotations-filename",
        type=str,
        default=DEFAULT_FULL_ANNOTATIONS_FILENAME,
        help=(
            "Filename for the full-annotations CSV exported alongside the "
            f"split (default: {DEFAULT_FULL_ANNOTATIONS_FILENAME})."
        ),
    )
    parser.add_argument(
        "--no-full-annotations",
        action="store_true",
        help="Skip writing the full-annotations CSV.",
    )
    return parser


def main(args: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    parsed = parser.parse_args(args=args)

    split = split_salva_dataset(
        excel_path=parsed.excel_path,
        train_ratio=parsed.train_ratio,
        random_state=parsed.random_state,
        split_mode=parsed.split_mode,
    )
    train_path, test_path = save_split_to_csv(
        split,
        output_dir=parsed.output_dir,
        train_filename=parsed.train_filename,
        test_filename=parsed.test_filename,
    )

    print(f"Split mode: {parsed.split_mode}")
    print(f"Train rows: {len(split.train)} → {train_path}")
    print(f"Test rows:  {len(split.test)} → {test_path}")

    if not parsed.no_full_annotations:
        full_path = export_full_annotations(
            excel_path=parsed.excel_path,
            output_dir=parsed.output_dir,
            filename=parsed.full_annotations_filename,
        )
        print(f"Full annotations exported → {full_path}")


if __name__ == "__main__":
    main()