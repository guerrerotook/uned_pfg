"""Data access helpers shared by the public LLM runners."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class DrugContext:
    codigo: str
    medicamento: str
    active_ingredients: tuple[str, ...]
    text: str


def load_codes(csv_path: str | Path) -> list[str]:
    dataframe = pd.read_csv(csv_path, usecols=["CODIGO"], dtype={"CODIGO": str})
    values = dataframe["CODIGO"].fillna("").astype(str).str.strip()
    return list(dict.fromkeys(value for value in values if value))


class DrugRepository:
    def __init__(
        self,
        annotations_path: str | Path,
        text_directory: str | Path,
    ) -> None:
        self.annotations = pd.read_csv(
            annotations_path,
            dtype={"CODIGO": str},
        )
        self.annotations.columns = self.annotations.columns.str.strip()
        required = {"CODIGO", "MEDICAMENTO", "PACTIVO"}
        missing = required - set(self.annotations.columns)
        if missing:
            raise ValueError(
                f"Missing required annotation columns: {sorted(missing)}"
            )

        self.annotations["CODIGO"] = (
            self.annotations["CODIGO"].fillna("").astype(str).str.strip()
        )
        self.text_directory = Path(text_directory)

    def get(self, codigo: str) -> DrugContext:
        normalized_codigo = str(codigo).strip()
        rows = self.annotations[self.annotations["CODIGO"] == normalized_codigo]
        if rows.empty:
            raise FileNotFoundError(
                f"No annotations found for CODIGO {normalized_codigo}."
            )

        text_path = self.text_directory / f"{normalized_codigo}.txt"
        if not text_path.is_file():
            raise FileNotFoundError(f"Section 4.8 text not found: {text_path}")

        medicamento_values = _unique_strings(rows["MEDICAMENTO"])
        active_ingredients = tuple(_unique_strings(rows["PACTIVO"]))
        if not medicamento_values or not active_ingredients:
            raise ValueError(
                f"Incomplete medication metadata for CODIGO {normalized_codigo}."
            )

        return DrugContext(
            codigo=normalized_codigo,
            medicamento=medicamento_values[0],
            active_ingredients=active_ingredients,
            text=text_path.read_text(encoding="utf-8").replace("|", ""),
        )


def _unique_strings(values: Iterable[Any]) -> list[str]:
    normalized = [str(value).strip() for value in values if pd.notna(value)]
    return list(dict.fromkeys(value for value in normalized if value))


def output_exists(codigo: str, output_directory: str | Path) -> bool:
    return (Path(output_directory) / f"{codigo}.json").is_file()


def save_json_output(
    result: Any,
    codigo: str,
    output_directory: str | Path,
) -> Path:
    output_path = Path(output_directory) / f"{codigo}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path