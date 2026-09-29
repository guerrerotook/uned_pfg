"""
Módulo de estrategias de prompts para la extracción de reacciones adversas.

Implementa tres técnicas de prompting:
- Zero-shot: solo instrucciones, sin ejemplos
- Few-shot: instrucciones + ejemplos dinámicos del conjunto de entrenamiento
- Chain-of-Thought: instrucciones + razonamiento paso a paso con ejemplo trabajado
"""

import json
import logging
import random
from enum import Enum
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Constantes del sistema de mensajes (idénticas a las de azure_foundry.py)
# ---------------------------------------------------------------------------
SYSTEM_MESSAGE = (
    "Eres un médico español y estas leyendo la sección 4.8 referida a "
    "reacciones adversas de un medicamento. Necesitamos tu conocimiento "
    "para clasificar adecuadamente las reacciones adversas de un medicamento."
)

ASSISTANT_MESSAGE = (
    "La respuesta debe ser una lista de JSONs con la lista de reacciones "
    "adversas clasificadas por frecuencia y trastorno. Cada reacción adversa "
    "debe tener los siguientes campos: 'reacción adversa', 'frecuencia' y "
    "'trastorno'."
)

FREQUENCY_REVIEW_MESSAGE = (
    "¿Puedes revisar las frecuencias de los JSONs para asignar correctamente "
    "a cada reacción adversa de acuerdo al texto?"
)

# Categorías de frecuencia estándar (EMA/ICH)
FRECUENCIAS_VALIDAS = [
    "Muy frecuentes (≥1/10)",
    "Frecuentes (≥1/100 a <1/10)",
    "Poco frecuentes (≥1/1.000 a <1/100)",
    "Raras (≥1/10.000 a <1/1.000)",
    "Muy raras (<1/10.000)",
    "Frecuencia no conocida (no puede estimarse a partir de los datos disponibles)",
]

# Códigos preferidos para ejemplos CoT (cortos, limpios, buena cobertura de frecuencias)
_PREFERRED_COT_CODIGOS = ["06960", "39411", "40537"]


# ---------------------------------------------------------------------------
# Enumeración de estrategias
# ---------------------------------------------------------------------------
class PromptStrategy(Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"


# ---------------------------------------------------------------------------
# Builders de prompts
# ---------------------------------------------------------------------------

def _base_task_description(medicamento: str, p_activo: str) -> str:
    """Descripción base de la tarea compartida por todas las estrategias."""
    frecuencias_str = "\n".join(f"  - {f}" for f in FRECUENCIAS_VALIDAS)
    return (
        f"Tu tarea es identificar TODAS las reacciones adversas del medicamento "
        f"{medicamento} (principio activo: {p_activo}) a partir del texto de la "
        f"sección 4.8 que se proporciona a continuación.\n\n"
        f"Para cada reacción adversa debes determinar:\n"
        f"1. La **frecuencia** según las categorías estándar:\n{frecuencias_str}\n"
        f"2. El **trastorno** o sistema de clasificación de órganos al que pertenece "
        f"(por ejemplo: Trastornos gastrointestinales, Trastornos del sistema nervioso, etc.)\n\n"
        f"Devuelve el resultado como un array JSON con el siguiente formato por cada "
        f"reacción adversa:\n"
        f'{{\n'
        f'    "principio activo": "{p_activo}",\n'
        f'    "reacción adversa": "nombre de la reacción adversa",\n'
        f'    "frecuencia": "frecuencia según las categorías indicadas",\n'
        f'    "trastorno": "sistema/trastorno al que pertenece"\n'
        f'}}\n'
    )


def build_zero_shot_prompt(medicamento: str, p_activo: str, texto: str) -> str:
    """Genera un prompt zero-shot: solo instrucciones, sin ejemplos."""
    task = _base_task_description(medicamento, p_activo)
    return (
        f"{task}\n"
        f"---\n"
        f"Texto de la sección 4.8 a analizar:\n\n"
        f"{texto}\n\n"
        f"---\n"
        f"Fin del texto. Devuelve ÚNICAMENTE el array JSON con todas las "
        f"reacciones adversas encontradas."
    )


def build_few_shot_prompt(
    medicamento: str,
    p_activo: str,
    texto: str,
    examples: List[dict],
) -> str:
    """Genera un prompt few-shot: instrucciones + ejemplos reales del training set."""
    task = _base_task_description(medicamento, p_activo)

    examples_block = ""
    for i, ex in enumerate(examples, 1):
        examples_block += (
            f"\n--- Ejemplo {i} ---\n"
            f"Texto de la sección 4.8:\n{ex['texto_excerpt']}\n\n"
            f"Resultado esperado (JSON):\n{ex['resultado_json']}\n"
            f"--- Fin ejemplo {i} ---\n"
        )

    return (
        f"{task}\n"
        f"A continuación se muestran algunos ejemplos ilustrativos de cómo extraer las "
        f"reacciones adversas de un texto de la sección 4.8 y devolver el "
        f"resultado en formato JSON. Estos ejemplos corresponden a otros "
        f"medicamentos y son solo de referencia; NO incluyas sus datos en "
        f"tu respuesta:\n"
        f"{examples_block}\n"
        f"Ahora analiza el siguiente texto (que es el único que debes procesar) "
        f"y extrae todas las reacciones adversas de la misma forma:\n\n"
        f"---\n"
        f"Texto de la sección 4.8 a analizar:\n\n"
        f"{texto}\n\n"
        f"---\n"
        f"Fin del texto. Devuelve ÚNICAMENTE el array JSON con todas las "
        f"reacciones adversas encontradas."
    )


def build_chain_of_thought_prompt(
    medicamento: str,
    p_activo: str,
    texto: str,
    cot_example: Optional[dict] = None,
) -> str:
    """Genera un prompt Chain-of-Thought: razonamiento paso a paso."""
    task = _base_task_description(medicamento, p_activo)

    reasoning_scaffold = (
        "Para resolver esta tarea, sigue estos pasos de razonamiento:\n\n"
        "Paso 1: Lee el texto e identifica todas las secciones correspondientes "
        "a trastornos o sistemas de clasificación de órganos (por ejemplo: "
        "'Trastornos gastrointestinales', 'Trastornos del sistema nervioso', etc.).\n\n"
        "Paso 2: Para cada trastorno identificado, determina qué categorías de "
        "frecuencia están presentes (Muy frecuentes, Frecuentes, Poco frecuentes, "
        "Raras, Muy raras, Frecuencia no conocida).\n\n"
        "Paso 3: Para cada combinación de trastorno y frecuencia, extrae todas "
        "las reacciones adversas individuales mencionadas.\n\n"
        "Paso 4: Compila todas las reacciones adversas en el array JSON final "
        "con los campos: 'principio activo', 'reacción adversa', 'frecuencia' "
        "y 'trastorno'.\n"
    )

    worked_example = ""
    if cot_example:
        worked_example = (
            "\n--- Ejemplo con razonamiento paso a paso ---\n"
            f"Texto de la sección 4.8:\n{cot_example['texto_excerpt']}\n\n"
            f"Razonamiento:\n{cot_example['razonamiento']}\n\n"
            f"Resultado JSON:\n{cot_example['resultado_json']}\n"
            "--- Fin del ejemplo ---\n\n"
        )

    return (
        f"{task}\n"
        f"{reasoning_scaffold}\n"
        f"{worked_example}"
        f"Ahora aplica estos mismos pasos al siguiente texto:\n\n"
        f"---\n"
        f"Texto de la sección 4.8 a analizar:\n\n"
        f"{texto}\n\n"
        f"---\n"
        f"Fin del texto. Muestra tu razonamiento paso a paso y después devuelve "
        f"el array JSON final con todas las reacciones adversas encontradas."
    )


# ---------------------------------------------------------------------------
# ExampleSelector: selección dinámica de ejemplos desde el training set
# ---------------------------------------------------------------------------

class ExampleSelector:
    """Selecciona ejemplos del conjunto de entrenamiento para few-shot y CoT.

    Usa ÚNICAMENTE datos de data/splits/train.csv como verdad de referencia.
    Los textos de la sección 4.8 se cargan de data/medicamentos/txt/.
    """

    def __init__(
        self,
        train_csv_path: Optional[str] = None,
        txt_dir: Optional[str] = None,
    ):
        self.train_csv_path = Path(
            train_csv_path or PROJECT_ROOT / "data" / "splits" / "train.csv"
        )
        self.txt_dir = Path(
            txt_dir or PROJECT_ROOT / "data" / "medicamentos" / "txt"
        )

        # Cargar datos de entrenamiento
        self.train_df = pd.read_csv(str(self.train_csv_path), dtype=str)
        self.train_df.columns = self.train_df.columns.str.strip()
        for col in ["CODIGO", "PACTIVO", "REACADV", "FRECUENCIA", "SISTEMA"]:
            # fillna antes de .str.strip() porque ese accesor preserva NaN como
            # float y deja la columna con tipos mixtos (str/float), lo cual
            # rompe operaciones como sorted() más adelante.
            self.train_df[col] = (
                self.train_df[col].fillna("").astype(str).str.strip()
            )

        # Construir índice de candidatos
        self._build_candidate_index()

    def _build_candidate_index(self):
        """Construye un índice de CODIGOs candidatos para ejemplos."""
        train_codigos = set(self.train_df["CODIGO"].unique())

        self.candidates: dict[str, dict] = {}
        for codigo in train_codigos:
            txt_path = self.txt_dir / f"{codigo}.txt"
            if not txt_path.exists():
                continue

            rows = self.train_df[self.train_df["CODIGO"] == codigo]
            frecuencias = set(rows["FRECUENCIA"].unique())
            n_reacciones = len(rows)

            self.candidates[codigo] = {
                "frecuencias": frecuencias,
                "n_frecuencias": len(frecuencias),
                "n_reacciones": n_reacciones,
                "txt_path": txt_path,
            }

        logger.info(
            f"ExampleSelector: {len(self.candidates)} candidatos disponibles "
            f"de {len(train_codigos)} códigos en train.csv"
        )

    def _build_json_from_train(self, codigo: str) -> List[dict]:
        """Construye la lista de reacciones adversas en formato JSON
        a partir de las filas de train.csv para un CODIGO dado."""
        rows = self.train_df[self.train_df["CODIGO"] == codigo]
        return [
            {
                "principio activo": row["PACTIVO"],
                "reacción adversa": row["REACADV"],
                "frecuencia": row["FRECUENCIA"],
                "trastorno": row["SISTEMA"],
            }
            for _, row in rows.iterrows()
        ]

    def _load_example_data(
        self, codigo: str, max_excerpt_chars: int = 1500
    ) -> dict:
        """Carga el texto (truncado) y construye el JSON desde train.csv."""
        info = self.candidates[codigo]

        with open(info["txt_path"], "r", encoding="utf-8") as f:
            texto_completo = f.read()

        # Truncar en "Notificación de sospechas" si existe
        marker = "Notificación de sospechas de reacciones adversas"
        idx = texto_completo.find(marker)
        if idx > 0:
            texto_excerpt = texto_completo[:idx].strip()
        else:
            texto_excerpt = texto_completo

        if len(texto_excerpt) > max_excerpt_chars:
            texto_excerpt = texto_excerpt[:max_excerpt_chars] + "\n[...]"

        resultado = self._build_json_from_train(codigo)
        resultado_json = json.dumps(resultado, ensure_ascii=False, indent=2)

        return {
            "codigo": codigo,
            "texto_excerpt": texto_excerpt,
            "resultado_json": resultado_json,
        }

    def select_examples(
        self,
        exclude_codigo: Optional[str] = None,
        n: int = 2,
        max_excerpt_chars: int = 1500,
        seed: Optional[int] = 42,
    ) -> List[dict]:
        """Selecciona n ejemplos diversos del training set para few-shot."""
        available = {
            k: v
            for k, v in self.candidates.items()
            if k != str(exclude_codigo)
        }

        if not available:
            logger.warning("No hay candidatos disponibles para few-shot.")
            return []

        # Ordenar por: más diversidad de frecuencias (desc), menos reacciones (asc)
        scored = sorted(
            available.items(),
            key=lambda x: (-x[1]["n_frecuencias"], x[1]["n_reacciones"]),
        )

        selected: List[dict] = []
        used_codigos: set[str] = set()
        covered_frecuencias: set[str] = set()

        # Selección greedy: en cada paso, elegir el CODIGO que más nuevas
        # categorías de frecuencia aporta
        remaining = list(scored)
        rng = random.Random(seed)

        while len(selected) < n and remaining:
            # Calcular contribución marginal de cada candidato
            best_score = -1
            best_indices: List[int] = []

            for i, (codigo, info) in enumerate(remaining):
                if codigo in used_codigos:
                    continue
                new_freqs = len(info["frecuencias"] - covered_frecuencias)
                score = new_freqs * 1000 - info["n_reacciones"]
                if score > best_score:
                    best_score = score
                    best_indices = [i]
                elif score == best_score:
                    best_indices.append(i)

            if not best_indices:
                break

            # Desempate aleatorio entre candidatos con el mismo score
            pick_idx = rng.choice(best_indices)
            codigo, info = remaining.pop(pick_idx)
            used_codigos.add(codigo)
            covered_frecuencias.update(info["frecuencias"])

            example = self._load_example_data(codigo, max_excerpt_chars)
            selected.append(example)

        return selected

    def get_cot_example(
        self,
        exclude_codigo: Optional[str] = None,
        max_excerpt_chars: int = 1500,
    ) -> Optional[dict]:
        """Selecciona un ejemplo y genera la anotación CoT dinámica."""
        # Intentar usar un CODIGO preferido
        chosen = None
        for pref in _PREFERRED_COT_CODIGOS:
            if pref != str(exclude_codigo) and pref in self.candidates:
                chosen = pref
                break

        # Si no hay preferido disponible, tomar el mejor candidato
        if chosen is None:
            available = {
                k: v
                for k, v in self.candidates.items()
                if k != str(exclude_codigo)
            }
            if not available:
                logger.warning("No hay candidatos disponibles para CoT.")
                return None
            # Elegir el más corto con buena diversidad de frecuencias
            chosen = min(
                available,
                key=lambda k: (-available[k]["n_frecuencias"], available[k]["n_reacciones"]),
            )

        example_data = self._load_example_data(chosen, max_excerpt_chars)

        # Generar razonamiento CoT dinámico desde train.csv
        resultado = self._build_json_from_train(chosen)
        razonamiento = self._generate_cot_reasoning(resultado)

        return {
            "texto_excerpt": example_data["texto_excerpt"],
            "razonamiento": razonamiento,
            "resultado_json": example_data["resultado_json"],
        }

    @staticmethod
    def _generate_cot_reasoning(resultado: List[dict]) -> str:
        """Genera la cadena de razonamiento paso a paso a partir del JSON gold."""
        # Paso 1: Identificar trastornos
        trastornos = list(dict.fromkeys(r["trastorno"] for r in resultado))
        paso1 = (
            "Paso 1: En el texto identifico los siguientes trastornos/sistemas "
            "de clasificación de órganos:\n"
        )
        for t in trastornos:
            paso1 += f"  - {t}\n"

        # Paso 2: Para cada trastorno, identificar frecuencias
        trastorno_freq: dict[str, set[str]] = {}
        for r in resultado:
            trastorno_freq.setdefault(r["trastorno"], set()).add(r["frecuencia"])

        paso2 = (
            "\nPaso 2: Para cada trastorno, las categorías de frecuencia "
            "presentes son:\n"
        )
        for t, freqs in trastorno_freq.items():
            freqs_str = ", ".join(sorted(freqs))
            paso2 += f"  - {t}: {freqs_str}\n"

        # Paso 3: Extraer reacciones adversas por trastorno y frecuencia
        paso3 = (
            "\nPaso 3: Las reacciones adversas individuales para cada "
            "combinación trastorno-frecuencia son:\n"
        )
        for t in trastornos:
            reacciones_t = [r for r in resultado if r["trastorno"] == t]
            freq_groups: dict[str, List[str]] = {}
            for r in reacciones_t:
                freq_groups.setdefault(r["frecuencia"], []).append(
                    r["reacción adversa"]
                )
            for freq, reacs in freq_groups.items():
                reacs_str = ", ".join(reacs)
                paso3 += f"  - {t} ({freq}): {reacs_str}\n"

        # Paso 4: Compilar
        paso4 = (
            "\nPaso 4: Compilo todas las reacciones adversas en el array "
            "JSON final con los campos requeridos."
        )

        return paso1 + paso2 + paso3 + paso4


# ---------------------------------------------------------------------------
# Factory: construye la lista de mensajes completa
# ---------------------------------------------------------------------------

def build_prompt(
    strategy: PromptStrategy,
    medicamento: str,
    p_activo: str,
    texto: str,
    example_selector: Optional[ExampleSelector] = None,
    n_examples: int = 2,
    max_excerpt_chars: int = 1500,
    seed: Optional[int] = 42,
    exclude_codigo: Optional[str] = None,
) -> str:
    """Construye el prompt textual según la estrategia elegida.

    Returns:
        El texto del prompt (contenido del mensaje user principal).
    """
    if strategy == PromptStrategy.ZERO_SHOT:
        return build_zero_shot_prompt(medicamento, p_activo, texto)

    elif strategy == PromptStrategy.FEW_SHOT:
        if example_selector is None:
            raise ValueError(
                "Se requiere un ExampleSelector para la estrategia few-shot."
            )
        examples = example_selector.select_examples(
            exclude_codigo=exclude_codigo,
            n=n_examples,
            max_excerpt_chars=max_excerpt_chars,
            seed=seed,
        )
        return build_few_shot_prompt(medicamento, p_activo, texto, examples)

    elif strategy == PromptStrategy.CHAIN_OF_THOUGHT:
        cot_example = None
        if example_selector is not None:
            cot_example = example_selector.get_cot_example(
                exclude_codigo=exclude_codigo,
                max_excerpt_chars=max_excerpt_chars,
            )
        return build_chain_of_thought_prompt(
            medicamento, p_activo, texto, cot_example
        )

    else:
        raise ValueError(f"Estrategia desconocida: {strategy}")


def build_messages(
    strategy: PromptStrategy,
    medicamento: str,
    p_activo: str,
    texto: str,
    example_selector: Optional[ExampleSelector] = None,
    n_examples: int = 2,
    max_excerpt_chars: int = 1500,
    seed: Optional[int] = 42,
    exclude_codigo: Optional[str] = None,
) -> List[dict]:
    """Construye la lista de mensajes completa para enviar al LLM.

    Returns:
        Lista de dicts con 'role' y 'content', compatible con el formato
        de azure_foundry.py y openai API.
    """
    prompt_text = build_prompt(
        strategy=strategy,
        medicamento=medicamento,
        p_activo=p_activo,
        texto=texto,
        example_selector=example_selector,
        n_examples=n_examples,
        max_excerpt_chars=max_excerpt_chars,
        seed=seed,
        exclude_codigo=exclude_codigo,
    )

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "assistant", "content": ASSISTANT_MESSAGE},
        {"role": "user", "content": prompt_text},
        {"role": "user", "content": FREQUENCY_REVIEW_MESSAGE},
    ]
