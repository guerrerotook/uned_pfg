"""Run intrinsic MLM evaluation for adapted and upstream checkpoints.

The adapted checkpoints default to their public Hugging Face IDs and may be
overridden with local paths. Reports are generated under ``artifacts/`` by
default so committed reference reports remain immutable.

The masked corpus is built **once per tokenizer family** (BERTIN BPE and
roberta-base-bne BPE) and reused across the fine-tuned model and its
upstream baseline so the comparison is exactly apples-to-apples.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.tests.mlm_test import MlmEvaluator

REPORTS_DIR = "artifacts/reports/mlm"
TXT_DIR = "data/medicamentos/txt"
DEFAULT_TEST_PATH = "data/splits/test.csv"
DEFAULT_FULL_ANNOTATIONS_PATH = "data/splits/full_annotations.csv"


@dataclass(frozen=True)
class ModelEntry:
    """One checkpoint to evaluate."""

    path: str  # local dir or HF hub id
    slug: str  # used for the JSON filename and console report
    family: str  # tokenizer family key ("bertin" / "roberta-bne")
    group: str  # CLI filter ("bertin" / "roberta" / "baselines")


MODEL_REGISTRY: Tuple[ModelEntry, ...] = (
    ModelEntry(
        path="guerrerotook/CIMA-BERTIN-4.8",
        slug="CIMA-BERTIN-4.8",
        family="bertin",
        group="bertin",
    ),
    ModelEntry(
        path="bertin-project/bertin-base-gaussian-exp-512seqlen",
        slug="bertin-base-upstream",
        family="bertin",
        group="baselines",
    ),
    ModelEntry(
        path="guerrerotook/CIMA-RoBERTa-4.8",
        slug="CIMA-RoBERTa-4.8",
        family="roberta-bne",
        group="roberta",
    ),
    ModelEntry(
        path="PlanTL-GOB-ES/roberta-base-biomedical-es",
        slug="roberta-base-biomedical-es-upstream",
        family="roberta-bne",
        group="baselines",
    ),
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the base MLM checkpoints (CIMA-BERTIN-4.8, "
            "CIMA-RoBERTa-4.8) plus their upstream baselines on the "
            "held-out section-4.8 corpus using whole-word masking."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the whole-word masking RNG (default: 42).",
    )
    parser.add_argument(
        "--mlm-probability",
        type=float,
        default=0.18,
        help="Fraction of subtokens to mask (default: 0.18, matches training).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Window length in subtokens (default: 128, matches training).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Per-device batch size for the no-grad forward pass.",
    )
    parser.add_argument(
        "--full-annotations",
        type=str,
        default=DEFAULT_FULL_ANNOTATIONS_PATH,
        help=(
            "Path to data/splits/full_annotations.csv. Used to build the "
            "medical-vocabulary set that splits per-category metrics "
            "(medical vs generic). Pass --no-full-annotations to skip."
        ),
    )
    parser.add_argument(
        "--no-full-annotations",
        action="store_true",
        help="Disable the medical-vocabulary categoriser.",
    )
    parser.add_argument(
        "--limit-codes",
        type=int,
        default=None,
        help="Smoke-test mode: evaluate only the first N CODIGOs.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "Comma-separated list of groups to run (any of: bertin, "
            "roberta, baselines). Defaults to all four checkpoints."
        ),
    )
    parser.add_argument(
        "--bertin-model",
        default="guerrerotook/CIMA-BERTIN-4.8",
        help="Local path or Hugging Face ID for the adapted BERTIN model.",
    )
    parser.add_argument(
        "--roberta-model",
        default="guerrerotook/CIMA-RoBERTa-4.8",
        help="Local path or Hugging Face ID for the adapted RoBERTa model.",
    )
    parser.add_argument("--test-csv", default=DEFAULT_TEST_PATH)
    parser.add_argument("--txt-dir", default=TXT_DIR)
    parser.add_argument("--reports-dir", default=REPORTS_DIR)
    return parser


def _select_models(
    filter_arg: Optional[str],
    bertin_model: str,
    roberta_model: str,
) -> List[ModelEntry]:
    registry = [
        ModelEntry(
            path=(
                bertin_model
                if entry.slug == "CIMA-BERTIN-4.8"
                else roberta_model
                if entry.slug == "CIMA-RoBERTa-4.8"
                else entry.path
            ),
            slug=entry.slug,
            family=entry.family,
            group=entry.group,
        )
        for entry in MODEL_REGISTRY
    ]
    if not filter_arg:
        return registry
    wanted = {token.strip().lower() for token in filter_arg.split(",") if token.strip()}
    valid = {entry.group for entry in registry}
    unknown = wanted - valid
    if unknown:
        raise SystemExit(
            f"Unknown --models value(s): {sorted(unknown)}. "
            f"Valid groups are: {sorted(valid)}."
        )
    selected = [entry for entry in registry if entry.group in wanted]
    if not selected:
        raise SystemExit("No models matched the --models filter.")
    return selected


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.no_full_annotations or not args.full_annotations:
        full_annotations_path = None
    elif not os.path.isfile(args.full_annotations):
        print(
            f"Full annotations CSV not found at {args.full_annotations!r}; "
            "per-category metrics will fall back to 'generic' for every "
            "masked token."
        )
        full_annotations_path = None
    else:
        full_annotations_path = args.full_annotations

    print("Intrinsic MLM evaluation")
    print(f"   Section 4.8 corpus  : {args.txt_dir}")
    print(f"   Medical vocabulary  : {full_annotations_path or '(disabled)'}")
    print(
        f"   Mask settings        : mlm_probability={args.mlm_probability}, "
        f"max_length={args.max_length}, seed={args.seed}, "
        "whole_word_masking=True"
    )

    medical_vocab: set[str] = (
        MlmEvaluator.build_medical_vocab(full_annotations_path)
        if full_annotations_path is not None
        else set()
    )

    codigos = MlmEvaluator.load_test_codigos(args.test_csv)
    if args.limit_codes is not None and args.limit_codes > 0:
        codigos = codigos[: args.limit_codes]
        print(
            f"   --limit-codes={args.limit_codes}: evaluating only "
            f"{len(codigos)} CODIGO(s) (smoke-test mode)."
        )
    else:
        print(f"   Held-out CODIGOs    : {len(codigos)}")

    models = _select_models(
        args.models,
        bertin_model=args.bertin_model,
        roberta_model=args.roberta_model,
    )
    print(
        f"   Models to evaluate  : "
        f"{', '.join(entry.slug for entry in models)}"
    )

    # Cache: family -> (vocab_size, masked_corpus, summary). The vocab_size
    # is used as a sanity-check signature so that, if a fine-tuned model
    # was somehow shipped with an extended vocab, we rebuild the corpus.
    corpus_cache: Dict[str, Tuple[int, List[Dict[str, Any]], Dict[str, Any]]] = {}

    exported_files: List[str] = []
    last_evaluator: Optional[MlmEvaluator] = None

    for entry in models:
        try:
            evaluator = MlmEvaluator(
                model_id_or_path=entry.path,
                model_name=entry.slug,
                batch_size=args.batch_size,
            )
        except Exception as exc:  # noqa: BLE001 - we want to keep going
            print(f"Skipping {entry.slug}: failed to load — {exc!r}")
            continue
        last_evaluator = evaluator

        cached = corpus_cache.get(entry.family)
        if cached is None or cached[0] != evaluator.tokenizer.vocab_size:
            print(
                f"\nBuilding masked corpus for tokenizer family "
                f"'{entry.family}' (vocab_size={evaluator.tokenizer.vocab_size})…"
            )
            corpus, summary = MlmEvaluator.build_masked_corpus(
                tokenizer=evaluator.tokenizer,
                codigos=codigos,
                txt_dir=args.txt_dir,
                max_length=args.max_length,
                mlm_probability=args.mlm_probability,
                seed=args.seed,
                medical_vocab=medical_vocab,
            )
            print(
                f"   Built {summary['total_windows']:,} windows / "
                f"{summary['total_tokens']:,} tokens / "
                f"{summary['total_masked_tokens']:,} masked subtokens."
            )
            if summary["missing_txt_files"]:
                print(
                    f"   {len(summary['missing_txt_files'])} CODIGO(s) "
                    "had no .txt file and were skipped."
                )
            corpus_cache[entry.family] = (
                evaluator.tokenizer.vocab_size,
                corpus,
                summary,
            )
        else:
            corpus, summary = cached[1], cached[2]
            print(
                f"\nReusing cached masked corpus for family "
                f"'{entry.family}' ({len(corpus):,} windows)."
            )

        report = evaluator.evaluate(
            masked_corpus=corpus,
            corpus_summary=summary,
            tokenizer_family=entry.family,
        )
        evaluator.print_report(report)
        path = evaluator.export_json(report, output_path=args.reports_dir)
        exported_files.append(path)

    if exported_files and last_evaluator is not None:
        last_evaluator.export_models_index(
            exported_files, output_path=args.reports_dir
        )


if __name__ == "__main__":
    main()
