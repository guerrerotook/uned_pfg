"""Run Azure ADR extraction with the final prompting strategies."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from src.llm.azure_foundry import DeepSeekAzureOpenAI, Gpt5ProAzureOpenAI
from src.llm.data_io import (
    DrugRepository,
    load_codes,
    output_exists,
    save_json_output,
)
from src.llm.prompt_strategies import ExampleSelector, PromptStrategy, build_messages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

STRATEGIES = ("zero_shot", "few_shot", "chain_of_thought")


@dataclass(frozen=True)
class ModelConfig:
    slug: str
    display_name: str


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "gpt-5.4-pro": ModelConfig(
        slug="gpt-5-4-pro-azure",
        display_name="GPT-5.4 Pro",
    ),
    "deepseek-v4-pro": ModelConfig(
        slug="deep-seek-v4-pro",
        display_name="DeepSeek V4 Pro",
    ),
}


def _parse_json_response(response: str) -> list[dict]:
    start = response.find("[")
    end = response.rfind("]") + 1
    if start < 0 or end <= start:
        raise ValueError("The response does not contain a JSON array.")
    parsed = json.loads(response[start:end])
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        raise ValueError("The response must be a JSON array of objects.")
    return parsed


def _instantiate_model(model_name: str, args: argparse.Namespace):
    if model_name == "gpt-5.4-pro":
        return Gpt5ProAzureOpenAI(
            model_name="gpt-5.4-pro",
            azure_endpoint=args.azure_openai_endpoint,
        )
    if model_name == "deepseek-v4-pro":
        return DeepSeekAzureOpenAI(
            model_name="DeepSeek-V4-Pro",
            deployment_name="DeepSeek-V4-Pro",
            endpoint=args.azure_deepseek_endpoint,
        )
    raise ValueError(f"Unknown Azure model: {model_name}")


def _output_directory(
    output_root: str | Path,
    model_name: str,
    strategy: str,
) -> Path:
    return Path(output_root) / MODEL_REGISTRY[model_name].slug / strategy.replace("_", "-")


def _selected_codes(args: argparse.Namespace) -> list[str]:
    codes = load_codes(args.test_csv)
    if args.split == "full":
        codes = list(dict.fromkeys(codes + load_codes(args.train_csv)))
    if args.limit is not None:
        codes = codes[: args.limit]
    return codes


def run(args: argparse.Namespace) -> dict[str, int]:
    strategy = PromptStrategy(args.strategy)
    repository = DrugRepository(args.annotations, args.txt_dir)
    codes = _selected_codes(args)

    example_selector = None
    if strategy in (PromptStrategy.FEW_SHOT, PromptStrategy.CHAIN_OF_THOUGHT):
        example_selector = ExampleSelector(
            train_csv_path=args.train_csv,
            txt_dir=args.txt_dir,
        )

    model = None if args.dry_run else _instantiate_model(args.model, args)
    output_directory = Path(args.output_directory)
    summary = {"processed": 0, "skipped": 0, "errors": 0}

    for codigo in tqdm(codes, desc=f"{args.model}:{args.strategy}"):
        if (
            not args.dry_run
            and not args.overwrite
            and output_exists(codigo, output_directory)
        ):
            summary["skipped"] += 1
            continue

        try:
            drug = repository.get(codigo)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Skipping CODIGO %s: %s", codigo, exc)
            summary["errors"] += 1
            continue

        all_reactions: list[dict] = []
        drug_failed = False
        for active_ingredient in drug.active_ingredients:
            messages = build_messages(
                strategy=strategy,
                medicamento=drug.medicamento,
                p_activo=active_ingredient,
                texto=drug.text,
                example_selector=example_selector,
                n_examples=args.n_examples,
                max_excerpt_chars=args.max_excerpt_chars,
                seed=args.seed,
                exclude_codigo=codigo,
            )

            if args.dry_run:
                print(
                    f"\nCODIGO={codigo} MODEL={args.model} "
                    f"STRATEGY={args.strategy} ACTIVE={active_ingredient}"
                )
                for message in messages:
                    print(f"\n[{message['role'].upper()}]\n{message['content']}")
                continue

            try:
                response = model.generate_response_messages(messages)
                parsed = _parse_json_response(response)
                for item in parsed:
                    item.setdefault("principio activo", active_ingredient)
                all_reactions.extend(parsed)
            except Exception as exc:  # noqa: BLE001 - continue with remaining drugs
                logger.error(
                    "Failed CODIGO %s (%s): %s",
                    codigo,
                    active_ingredient,
                    exc,
                )
                summary["errors"] += 1
                drug_failed = True

        if drug_failed:
            continue
        if not args.dry_run:
            save_json_output(all_reactions, codigo, output_directory)
        summary["processed"] += 1

    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract adverse reactions with the final Azure LLM models."
    )
    parser.add_argument(
        "--model",
        choices=[*MODEL_REGISTRY, "all"],
        default="all",
    )
    parser.add_argument(
        "--strategy",
        choices=[*STRATEGIES, "all"],
        required=True,
    )
    parser.add_argument("--split", choices=["test", "full"], default="test")
    parser.add_argument("--test-csv", default="data/splits/test.csv")
    parser.add_argument("--train-csv", default="data/splits/train.csv")
    parser.add_argument("--annotations", default="data/splits/full_annotations.csv")
    parser.add_argument("--txt-dir", default="data/medicamentos/txt")
    parser.add_argument("--output-dir", default="artifacts/llm_outputs")
    parser.add_argument("--azure-openai-endpoint", default=None)
    parser.add_argument("--azure-deepseek-endpoint", default=None)
    parser.add_argument("--n-examples", type=int, default=2)
    parser.add_argument("--max-excerpt-chars", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    models = list(MODEL_REGISTRY) if args.model == "all" else [args.model]
    strategies = list(STRATEGIES) if args.strategy == "all" else [args.strategy]

    errors = 0
    for model_name in models:
        for strategy in strategies:
            run_args = argparse.Namespace(**vars(args))
            run_args.model = model_name
            run_args.strategy = strategy
            run_args.output_directory = _output_directory(
                args.output_dir,
                model_name,
                strategy,
            )
            summary = run(run_args)
            errors += summary["errors"]
            logger.info(
                "%s / %s: processed=%d skipped=%d errors=%d",
                MODEL_REGISTRY[model_name].display_name,
                strategy,
                summary["processed"],
                summary["skipped"],
                summary["errors"],
            )

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()