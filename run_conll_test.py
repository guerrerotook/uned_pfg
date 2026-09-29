"""Run CoNLL-trained NER evaluation against the held-out test splits.

The runner discovers the final real and synthetic model families under an
explicit artifact root and evaluates them with
:class:`src.tests.conll_test.ConllNerTest`.

Two orthogonal axes are exposed via the CLI:

* ``--models {normal,synth,all}`` — which family of trained checkpoints
  to load.
* ``--test-set {real,synth,both}`` — which test corpus to evaluate on.
  ``real`` uses ``data/splits/test.csv`` + ``data/medicamentos/txt/`` +
  ``data/splits/full_annotations.csv``. ``synth`` uses the LLM-generated
  counterparts under ``data/splits/test_synth.csv`` +
  ``data/medicamentos_synth/txt/`` +
  ``data/splits/full_annotations_synth.csv``.

Synthetic models evaluated on real text provide the cross-domain comparison;
the other combinations remain available as diagnostics.
"""

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.tests.conll_test import ConllNerTest

# ---------------------------------------------------------------------------
# Test-set descriptors (independent of the model family)
# ---------------------------------------------------------------------------

DEFAULT_FULL_ANNOTATIONS_PATH = "data/splits/full_annotations.csv"
DEFAULT_FULL_ANNOTATIONS_SYNTH_PATH = "data/splits/full_annotations_synth.csv"


@dataclass(frozen=True)
class TestSet:
    """Describes one evaluation corpus (real or synthetic)."""

    name: str  # "real" | "synth"
    test_csv: str
    txt_dir: str
    default_full_annotations: str


TEST_SETS: dict = {
    "real": TestSet(
        name="real",
        test_csv="data/splits/test.csv",
        txt_dir="data/medicamentos/txt",
        default_full_annotations=DEFAULT_FULL_ANNOTATIONS_PATH,
    ),
    "synth": TestSet(
        name="synth",
        test_csv="data/splits/test_synth.csv",
        txt_dir="data/medicamentos_synth/txt",
        default_full_annotations=DEFAULT_FULL_ANNOTATIONS_SYNTH_PATH,
    ),
}

# ---------------------------------------------------------------------------
# Model families
# ---------------------------------------------------------------------------

NORMAL_MODEL_NAMES: List[str] = [
    "CIMA-BERTIN-4.8-NER-ADR-CONLL-active",
    "CIMA-BERTIN-4.8-NER-ADR-CONLL-freq",
    "CIMA-BERTIN-4.8-NER-ADR-CONLL-react",
    "CIMA-BERTIN-4.8-NER-ADR-CONLL-sys",
    "CIMA-BERTIN-4.8-NER-ADR-CONLL-react-active-freq-sys",
    "CIMA-RoBERTa-4.8-NER-ADR-CONLL-active",
    "CIMA-RoBERTa-4.8-NER-ADR-CONLL-freq",
    "CIMA-RoBERTa-4.8-NER-ADR-CONLL-react",
    "CIMA-RoBERTa-4.8-NER-ADR-CONLL-sys",
    "CIMA-RoBERTa-4.8-NER-ADR-CONLL-react-active-freq-sys",
    "RoBERTa-baseline-NER-ADR-CONLL-active",
    "RoBERTa-baseline-NER-ADR-CONLL-freq",
    "RoBERTa-baseline-NER-ADR-CONLL-react",
    "RoBERTa-baseline-NER-ADR-CONLL-sys",
    "RoBERTa-baseline-NER-ADR-CONLL-react-active-freq-sys",
]

SYNTH_MODEL_NAMES: List[str] = [
    name.replace("-CONLL-", "-CONLL-SYNTH-") for name in NORMAL_MODEL_NAMES
]

MODEL_FAMILIES: dict = {
    "normal": NORMAL_MODEL_NAMES,
    "synth": SYNTH_MODEL_NAMES,
}

# Reports directory for each (model_family, test_set) combination.
# - normal x real      : the original baseline (kept at data/reports/conll)
# - synth  x synth     : in-distribution diagnostic for the synth family
# - synth  x real      : THE comparison for "does synth training hurt real
#                        detection?" — reported alongside normal x real.
# - normal x synth     : optional cross-eval, kept for completeness.
REPORTS_SUBDIRS: dict = {
    ("normal", "real"): "conll",
    ("normal", "synth"): "conll_normal_on_synth",
    ("synth", "real"): "conll_synth_on_real",
    ("synth", "synth"): "conll_synth",
}


def process_result(
    model_dir: str,
    output_dir: str,
    test_set: TestSet,
    full_annotations_csv: Optional[str],
) -> Tuple[dict, ConllNerTest]:
    """Build a tester for ``model_dir`` and run the full evaluation."""
    tester = ConllNerTest(
        model_dir=model_dir,
        test_csv_path=test_set.test_csv,
        train_csv_path=None,  # evaluate on the held-out test rows only
        full_annotations_csv_path=full_annotations_csv,
        txt_dir=test_set.txt_dir,
    )
    result = tester.evaluate(output_directory=output_dir)
    return result, tester


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every CoNLL-trained NER model under an artifact root against the "
            "real and/or synthetic test split. Model family and test set are "
            "selected independently."
        )
    )
    parser.add_argument(
        "--models",
        choices=["normal", "synth", "all"],
        default="all",
        help="Which family of trained checkpoints to load (default: all).",
    )
    parser.add_argument(
        "--test-set",
        choices=["real", "synth", "both"],
        default="both",
        help=(
            "Which test corpus to evaluate against (default: both). 'real' "
            "uses data/splits/test.csv + data/medicamentos/txt; 'synth' uses "
            "data/splits/test_synth.csv + data/medicamentos_synth/txt."
        ),
    )
    parser.add_argument(
        "--full-annotations",
        type=str,
        default=DEFAULT_FULL_ANNOTATIONS_PATH,
        help=(
            "Full-annotations CSV used as gold for the REAL test set. "
            "Pass an empty string to disable and fall back to test.csv rows."
        ),
    )
    parser.add_argument(
        "--full-annotations-synth",
        type=str,
        default=DEFAULT_FULL_ANNOTATIONS_SYNTH_PATH,
        help=(
            "Full-annotations CSV used as gold for the SYNTH test set. "
            "Pass an empty string to disable and fall back to test_synth.csv "
            "rows."
        ),
    )
    parser.add_argument(
        "--no-full-annotations",
        action="store_true",
        help=(
            "Force gold to be derived from the test CSV only (applies to "
            "every selected test set)."
        ),
    )
    parser.add_argument("--model-root", default="artifacts/models")
    parser.add_argument(
        "--predictions-root",
        default="artifacts/predictions/conll",
    )
    parser.add_argument("--reports-root", default="artifacts/reports")
    parser.add_argument("--real-test-csv", default="data/splits/test.csv")
    parser.add_argument("--real-txt-dir", default="data/medicamentos/txt")
    parser.add_argument("--synth-test-csv", default="data/splits/test_synth.csv")
    parser.add_argument("--synth-txt-dir", default="data/medicamentos_synth/txt")
    return parser


def _resolve_full_annotations(
    test_set: TestSet,
    requested_path: str,
    disabled: bool,
) -> Optional[str]:
    if disabled or not requested_path:
        return None
    if not os.path.isfile(requested_path):
        print(
            f"Full annotations CSV not found at {requested_path!r}; "
            f"falling back to {test_set.test_csv}. Generate it with "
            "`python -m src.training_data --split-mode drug`."
        )
        return None
    return requested_path


def _run_pass(
    model_family: str,
    test_set: TestSet,
    full_annotations_csv: Optional[str],
    model_root: str,
    predictions_root: str,
    reports_root: str,
) -> None:
    model_names = MODEL_FAMILIES[model_family]
    reports_dir = os.path.join(
        reports_root,
        REPORTS_SUBDIRS[(model_family, test_set.name)],
    )

    print(
        f"\nCoNLL NER evaluation — models={model_family} | "
        f"test_set={test_set.name}"
    )
    print(f"   Test CSV: {test_set.test_csv}")
    print(f"   Txt dir : {test_set.txt_dir}")
    print(f"   Reports : {reports_dir}")
    if full_annotations_csv is None:
        print(
            f"   Using {os.path.basename(test_set.test_csv)}-only gold "
            "(partial labels expected)."
        )
    else:
        print(f"   Gold    : {full_annotations_csv}")

    exported_files: List[str] = []
    last_tester: Optional[ConllNerTest] = None

    for name in model_names:
        model_dir = os.path.join(model_root, name)
        # Keep predictions for each (model, test_set) combo separated so
        # cross-evaluations don't overwrite the in-distribution outputs.
        output_dir = os.path.join(
            predictions_root,
            name,
            f"on_{test_set.name}",
        )
        try:
            result, tester = process_result(
                model_dir=model_dir,
                output_dir=output_dir,
                test_set=test_set,
                full_annotations_csv=full_annotations_csv,
            )
        except FileNotFoundError as exc:
            print(f"Skipping {model_dir}: {exc}")
            continue

        last_tester = tester
        if result:
            path = tester.export_json(result, output_path=reports_dir)
            exported_files.append(path)

    if exported_files and last_tester is not None:
        last_tester.export_models_index(exported_files, output_path=reports_dir)


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    selected_models = (
        ["normal", "synth"] if args.models == "all" else [args.models]
    )
    selected_tests = (
        ["real", "synth"] if args.test_set == "both" else [args.test_set]
    )

    test_sets = {
        "real": TestSet(
            name="real",
            test_csv=args.real_test_csv,
            txt_dir=args.real_txt_dir,
            default_full_annotations=DEFAULT_FULL_ANNOTATIONS_PATH,
        ),
        "synth": TestSet(
            name="synth",
            test_csv=args.synth_test_csv,
            txt_dir=args.synth_txt_dir,
            default_full_annotations=DEFAULT_FULL_ANNOTATIONS_SYNTH_PATH,
        ),
    }
    requested_paths = {
        "real": args.full_annotations,
        "synth": args.full_annotations_synth,
    }

    # Resolve the full-annotations path once per test set so we don't print
    # the same warning twice when both model families are evaluated.
    resolved_gold: dict = {}
    for test_name in selected_tests:
        ts = test_sets[test_name]
        resolved_gold[test_name] = _resolve_full_annotations(
            test_set=ts,
            requested_path=requested_paths[test_name],
            disabled=args.no_full_annotations,
        )

    for model_family in selected_models:
        for test_name in selected_tests:
            _run_pass(
                model_family=model_family,
                test_set=test_sets[test_name],
                full_annotations_csv=resolved_gold[test_name],
                model_root=args.model_root,
                predictions_root=args.predictions_root,
                reports_root=args.reports_root,
            )


if __name__ == "__main__":
    main()
