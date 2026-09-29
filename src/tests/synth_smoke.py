"""Quick end-to-end smoke test with a mocked LLM. Does NOT call any real
LLM. Run from repo root:

    python -m src.tests.synth_smoke
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import pandas as pd

# Make sure repo root is on path when run as a script
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.conll_ner_builder import ConllBuilder
from src.synth_ner_generator import (
    SynthNerGenerator,
    consolidate_full_annotations,
    derive_synth_splits,
    validate_response_text_and_entities,
)


class MockLLM:
    model_name = "mock-llm"

    def __init__(self):
        self.calls = 0

    def generate_response_messages(self, messages):
        self.calls += 1
        user_content = messages[-1]["content"]
        # Snippet prompts contain the word "BREVE" in the instructions; long
        # docs do not.
        if "BREVE" in user_content:
            payload = {
                "text": (
                    "Se ha descrito que la alopecia, una reacción adversa "
                    "frecuente, puede asociarse al uso de SOTALOL "
                    "HIDROCLORURO en pacientes susceptibles. Se enmarca "
                    "dentro de TRASTORNOS DE PIEL Y TEJIDOS."
                ),
                "entities": [
                    {"type": "ACTIVE", "canonical": "SOTALOL HIDROCLORURO", "surface": "SOTALOL HIDROCLORURO"},
                    {"type": "REACT", "canonical": "Alopecia", "surface": "alopecia"},
                    {"type": "FREQ", "canonical": "FRECUENTE", "surface": "frecuente"},
                    {"type": "SYS", "canonical": "TRASTORNOS DE PIEL Y TEJIDOS", "surface": "TRASTORNOS DE PIEL Y TEJIDOS"},
                ],
            }
        else:
            payload = {
                "text": (
                    "Resumen del perfil de seguridad:\n\n"
                    "SOTALOL HIDROCLORURO se ha asociado en general a un perfil "
                    "de seguridad bien caracterizado en pacientes adultos. La "
                    "cefalea es muy frecuente y suele aparecer durante las "
                    "primeras semanas. La alopecia es frecuente y reversible. "
                    "La bradicardia es poco frecuente.\n\n"
                    "Lista tabulada de reacciones adversas:\n\n"
                    "TRASTORNOS DEL SISTEMA NERVIOSO:\n"
                    "  - Muy frecuentes: cefalea.\n\n"
                    "TRASTORNOS DE PIEL Y TEJIDOS:\n"
                    "  - Frecuentes: alopecia.\n\n"
                    "TRASTORNOS CARDIACOS:\n"
                    "  - Poco frecuentes: bradicardia.\n\n"
                    "Descripción de reacciones adversas seleccionadas:\n"
                    "La cefalea suele resolverse espontáneamente; la alopecia "
                    "se ha notificado en pacientes que usan SOTALOL "
                    "HIDROCLORURO durante periodos prolongados."
                ),
                "entities": [
                    {"type": "ACTIVE", "canonical": "SOTALOL HIDROCLORURO", "surface": "SOTALOL HIDROCLORURO"},
                    {"type": "REACT", "canonical": "Cefalea", "surface": "cefalea"},
                    {"type": "FREQ", "canonical": "MUY FRECUENTE", "surface": "Muy frecuentes"},
                    {"type": "SYS", "canonical": "TRASTORNOS DEL SISTEMA NERVIOSO", "surface": "TRASTORNOS DEL SISTEMA NERVIOSO"},
                    {"type": "REACT", "canonical": "Alopecia", "surface": "alopecia"},
                    {"type": "FREQ", "canonical": "FRECUENTE", "surface": "Frecuentes"},
                    {"type": "SYS", "canonical": "TRASTORNOS DE PIEL Y TEJIDOS", "surface": "TRASTORNOS DE PIEL Y TEJIDOS"},
                    {"type": "REACT", "canonical": "Bradicardia", "surface": "bradicardia"},
                    {"type": "FREQ", "canonical": "POCO FRECUENTE", "surface": "Poco frecuentes"},
                    {"type": "SYS", "canonical": "TRASTORNOS CARDIACOS", "surface": "TRASTORNOS CARDIACOS"},
                ],
            }
        return "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"


def main() -> None:
    df = pd.DataFrame(
        [
            {"CODIGO": "89413", "MEDICAMENTO": "SOTALOL 80 mg comp", "PACTIVO": "SOTALOL HIDROCLORURO", "REACADV": "Alopecia", "FRECUENCIA": "FRECUENTE", "SISTEMA": "TRASTORNOS DE PIEL Y TEJIDOS"},
            {"CODIGO": "89413", "MEDICAMENTO": "SOTALOL 80 mg comp", "PACTIVO": "SOTALOL HIDROCLORURO", "REACADV": "Bradicardia", "FRECUENCIA": "POCO FRECUENTE", "SISTEMA": "TRASTORNOS CARDIACOS"},
            {"CODIGO": "89413", "MEDICAMENTO": "SOTALOL 80 mg comp", "PACTIVO": "SOTALOL HIDROCLORURO", "REACADV": "Cefalea", "FRECUENCIA": "MUY FRECUENTE", "SISTEMA": "TRASTORNOS DEL SISTEMA NERVIOSO"},
        ]
    )
    train_df = df[df["CODIGO"] == "89413"]  # everything in train for the smoke
    test_df = df.iloc[0:0]

    tmp = tempfile.mkdtemp(prefix="synth_smoke_")
    try:
        print(f"out_dir: {tmp}")
        gen = SynthNerGenerator(
            MockLLM(),
            tmp,
            n_variants=2,
            n_snippets_per_doc=2,
            llm_max_attempts=1,
            llm_base_sleep=0.0,
        )
        docs = gen.generate_for_codigo("89413", df)
        for d in docs:
            print(
                f"Generated {d.codigo_variant}: entities={len(d.entities)} "
                f"discard={d.discard_ratio:.2f} attempts={d.attempts}"
            )
            print(
                "  txt size:", os.path.getsize(gen.txt_path(d.codigo, d.variante)),
                "bytes",
            )
        assert len(docs) == 2, "expected 2 variants"

        fa = consolidate_full_annotations(tmp, df)
        print(f"\nfull_annotations_synth: {len(fa)} rows")
        print(fa.to_string(index=False))

        train_synth, test_synth = derive_synth_splits(fa, train_df, test_df)
        print(
            f"\ntrain_synth={len(train_synth)} rows ({train_synth['CODIGO'].nunique()} codigos), "
            f"test_synth={len(test_synth)} rows ({test_synth['CODIGO'].nunique()} codigos)"
        )

        builder = ConllBuilder(
            txt_dir=os.path.join(tmp, "txt"),
            entity_types=["REACT", "ACTIVE", "FREQ", "SYS"],
        )
        examples = builder.build_dataset(fa)
        stats = builder.summarise(examples)
        print(
            "\nbuilder.summarise:\n"
            + json.dumps(stats, ensure_ascii=False, indent=2)
        )

        # Spot-check a few sentences with non-O labels.
        non_empty = [
            ex
            for ex in examples
            if any(builder.id_to_label[i] != "O" for i in ex["ner_tags"])
        ]
        print(f"\nSentences with at least one tag: {len(non_empty)}/{len(examples)}")
        for ex in non_empty[:5]:
            labels = [builder.id_to_label[i] for i in ex["ner_tags"]]
            print(f"  [{ex['codigo']}#{ex['sent_idx']}]")
            print(f"    tokens: {ex['tokens']}")
            print(f"    labels: {labels}")
    finally:
        shutil.rmtree(tmp)
        print(f"\nCleaned up {tmp}")


if __name__ == "__main__":
    main()
