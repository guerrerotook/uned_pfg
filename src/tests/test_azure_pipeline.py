import argparse
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from src.llm.data_io import DrugContext, DrugRepository, load_codes


class DataIoTests(unittest.TestCase):
    def test_codes_and_context_preserve_leading_zero(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "annotations.csv"
            csv_path.write_text(
                "CODIGO,MEDICAMENTO,PACTIVO\n"
                "06960,Medicamento,Activo A\n"
                "06960,Medicamento,Activo B\n",
                encoding="utf-8",
            )
            text_directory = root / "txt"
            text_directory.mkdir()
            (text_directory / "06960.txt").write_text("Texto | limpio", encoding="utf-8")

            self.assertEqual(load_codes(csv_path), ["06960"])
            context = DrugRepository(csv_path, text_directory).get("06960")
            self.assertEqual(context.codigo, "06960")
            self.assertEqual(context.active_ingredients, ("Activo A", "Activo B"))
            self.assertEqual(context.text, "Texto  limpio")


class AzurePromptRunnerTests(unittest.TestCase):
    @staticmethod
    def _args(output_directory: str, dry_run: bool = True) -> argparse.Namespace:
        return argparse.Namespace(
            model="gpt-5.4-pro",
            strategy="zero_shot",
            split="test",
            test_csv="test.csv",
            train_csv="train.csv",
            annotations="annotations.csv",
            txt_dir="txt",
            output_directory=Path(output_directory),
            azure_openai_endpoint=None,
            azure_deepseek_endpoint=None,
            n_examples=2,
            max_excerpt_chars=1500,
            seed=42,
            limit=None,
            dry_run=dry_run,
            overwrite=False,
        )

    def test_dry_run_never_constructs_client_or_writes_output(self):
        from run_prompt_strategies import run

        with TemporaryDirectory() as temporary_directory:
            output_directory = str(Path(temporary_directory) / "not-created")
            repository = SimpleNamespace(
                get=lambda codigo: DrugContext(
                    codigo=codigo,
                    medicamento="Medicamento",
                    active_ingredients=("Activo",),
                    text="Texto",
                )
            )
            with patch("run_prompt_strategies.DrugRepository", return_value=repository), patch(
                "run_prompt_strategies.load_codes", return_value=["06960"]
            ), patch("run_prompt_strategies._instantiate_model") as factory, patch(
                "builtins.print"
            ):
                summary = run(self._args(output_directory))

            self.assertEqual(summary, {"processed": 1, "skipped": 0, "errors": 0})
            factory.assert_not_called()
            self.assertFalse(Path(output_directory).exists())

    def test_output_layout_separates_models_and_strategies(self):
        from run_prompt_strategies import _output_directory

        root = Path("artifacts/llm_outputs")
        self.assertEqual(
            _output_directory(root, "gpt-5.4-pro", "few_shot"),
            root / "gpt-5-4-pro-azure" / "few-shot",
        )
        self.assertEqual(
            _output_directory(root, "deepseek-v4-pro", "chain_of_thought"),
            root / "deep-seek-v4-pro" / "chain-of-thought",
        )

    def test_invalid_json_is_not_an_empty_prediction(self):
        from run_prompt_strategies import _parse_json_response

        self.assertEqual(_parse_json_response("```json\n[]\n```"), [])
        for response in ("No JSON", "[invalid]", '["not an object"]'):
            with self.subTest(response=response), self.assertRaises((ValueError, json.JSONDecodeError)):
                _parse_json_response(response)

    def test_all_runs_two_models_and_three_strategies(self):
        from run_prompt_strategies import main

        with patch(
            "sys.argv",
            ["runner", "--model", "all", "--strategy", "all", "--dry-run"],
        ), patch(
            "run_prompt_strategies.run",
            return_value={"processed": 0, "skipped": 0, "errors": 0},
        ) as run:
            main()

        self.assertEqual(run.call_count, 6)
        self.assertEqual(
            {call.args[0].model for call in run.call_args_list},
            {"gpt-5.4-pro", "deepseek-v4-pro"},
        )
        self.assertEqual(
            len({str(call.args[0].output_directory) for call in run.call_args_list}),
            6,
        )


class AzureEvaluationTests(unittest.TestCase):
    def test_default_embeddings_model_is_loadable_upstream_checkpoint(self):
        import run_llm_ner_test as runner

        self.assertEqual(
            runner.DEFAULT_EMBEDDINGS_MODEL,
            "PlanTL-GOB-ES/roberta-base-biomedical-es",
        )

    def test_default_results_are_the_two_azure_models(self):
        import run_llm_ner_test as runner

        self.assertEqual(
            runner.DEFAULT_RESULTS_DIRS,
            [
                "data/llm_outputs/gpt-5-4-pro-azure",
                "data/llm_outputs/deep-seek-v4-pro",
            ],
        )

    def test_evaluator_discovers_three_strategies(self):
        from run_llm_ner_test import _resolve_strategy_dirs

        with TemporaryDirectory() as temporary_directory:
            model_directory = Path(temporary_directory) / "model"
            for strategy in ("zero-shot", "few-shot", "chain-of-thought"):
                directory = model_directory / strategy
                directory.mkdir(parents=True)
                (directory / "06960.json").write_text("[]", encoding="utf-8")

            entries = _resolve_strategy_dirs(str(model_directory))
            self.assertEqual(len(entries), 3)
            self.assertEqual(
                {entry[2] for entry in entries},
                {"zero-shot", "few-shot", "chain-of-thought"},
            )


if __name__ == "__main__":
    unittest.main()