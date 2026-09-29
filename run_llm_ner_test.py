import argparse
import os
from typing import List, Tuple

from src.tests.ner_evaluation import NerEvaluation

REPORTS_DIR = "artifacts/reports/llm"
DEFAULT_EMBEDDINGS_MODEL = "PlanTL-GOB-ES/roberta-base-biomedical-es"

DEFAULT_RESULTS_DIRS = [
    "data/llm_outputs/gpt-5-4-pro-azure",
    "data/llm_outputs/deep-seek-v4-pro",
]

PROMPT_STRATEGIES: list[str] = ["few-shot", "zero-shot", "chain-of-thought"]


def _resolve_strategy_dirs(
    base_dir: str,
) -> List[Tuple[str, str, str]]:
    """Return a list of (strategy_dir, model_name, strategy) tuples for
    every prompt strategy that has at least one JSON file."""
    base_model = os.path.basename(os.path.normpath(base_dir))
    entries: List[Tuple[str, str, str]] = []
    for strategy in PROMPT_STRATEGIES:
        strategy_dir = os.path.join(base_dir, strategy)
        model_name = f"{base_model}_{strategy}"

        if not os.path.isdir(strategy_dir):
            continue
        # Skip directories with no JSON files (e.g. empty chain-of-thought)
        if not any(f.endswith(".json") for f in os.listdir(strategy_dir)):
            continue
        entries.append((strategy_dir, model_name, strategy))
    return entries


def execute_evaluator(
    evaluator: NerEvaluation, output_path: str, llm_results: list[str] | None = None
) -> None:
    if llm_results is None:
        llm_results = DEFAULT_RESULTS_DIRS

    exported_files: list[str] = []
    for base_dir in llm_results:
        if not os.path.isdir(base_dir):
            print(f"WARNING: {base_dir} is not a valid directory — skipping.")
            continue

        for strategy_dir, model_name, strategy in _resolve_strategy_dirs(base_dir):
            print(f"\n>>> Evaluating {model_name} (strategy: {strategy}) ...")
            result = evaluator.evaluate(strategy_dir, model_name=model_name)
            if result:
                path = evaluator.export_json(result, output_path=output_path)
                exported_files.append(path)

    # Write an index alongside the generated reports.
    if exported_files:
        evaluator.export_models_index(exported_files, output_path=output_path)


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluar JSON de LLM con metricas NER.")
    parser.add_argument(
        "--results-dirs", nargs="+", default=None,
        help="Carpetas de modelos que contienen JSON y subcarpetas de estrategias. "
        "Por defecto: GPT-5.4 Pro y DeepSeek V4 Pro en data/llm_outputs.",
    )
    parser.add_argument(
        "--output-dir", default=REPORTS_DIR,
        help="Raiz de informes; se crean subcarpetas test y/o full.",
    )
    parser.add_argument("--split", choices=["test", "full", "both"], default="both")
    parser.add_argument(
        "--embeddings-model",
        default=DEFAULT_EMBEDDINGS_MODEL,
        help="Modelo local o de Hugging Face usado para similitud semántica.",
    )
    parser.add_argument("--test-csv", default="data/splits/test.csv")
    parser.add_argument("--train-csv", default="data/splits/train.csv")
    args = parser.parse_args()

    splits = ["test", "full"] if args.split == "both" else [args.split]
    for split in splits:
        evaluator = NerEvaluation(
            embeddings_model_path=args.embeddings_model,
            test_csv_path=args.test_csv,
            train_csv_path=args.train_csv if split == "full" else None,
        )
        execute_evaluator(
            evaluator,
            output_path=os.path.join(args.output_dir, split),
            llm_results=args.results_dirs,
        )


if __name__ == "__main__":
    main()
