"""Prompt templates for the synthetic NER dataset generator (Variante B).

These builders produce the messages sent to an LLM (currently any of the
clients in :mod:`src.llm.azure_foundry`) so it generates realistic Spanish
SmPC section-4.8-style text whose entities can be deterministically located
in the resulting text by :class:`src.conll_ner_builder.ConllBuilder`.

Two builders are exposed:

* :func:`build_long_doc_messages` — one rich, multi-paragraph synthetic
  document covering *all* the adverse reactions of a single drug.
* :func:`build_snippet_messages`  — a short paragraph centred on a single
  ``(reaction, frequency, system)`` tuple.

Both prompts pin the response format to a fenced JSON block containing the
synthetic ``text`` and an explicit ``entities`` list with ``type``,
``canonical`` and ``surface`` fields, which is what
:class:`src.synth_ner_generator.SynthNerGenerator` validates and then feeds
back into the existing CoNLL/BIO matcher.

The frequency vocabulary is mapped from the upper-case strings of
``data/splits/full_annotations.csv`` to the EMA/MedDRA textual form expected
by Salvador in the meeting notes (``muy frecuente``, ``frecuente``…).
"""

from __future__ import annotations

import json
import textwrap
from typing import Iterable, List

import pandas as pd


# ---------------------------------------------------------------------------
# Frequency mapping (CSV upper-case → EMA/MedDRA textual form)
# ---------------------------------------------------------------------------
# Used to phrase the LLM instructions; the *surface* string the LLM writes
# is the one we ultimately tag, not these canonical labels.
FRECUENCIA_TEXTUAL: dict[str, str] = {
    "MUY FRECUENTE": "muy frecuentes",
    "FRECUENTE": "frecuentes",
    "POCO FRECUENTE": "poco frecuentes",
    "RARA": "raras",
    "MUY RARA": "muy raras",
    "FRECUENCIA NO CONOCIDA": "frecuencia no conocida",
    "NO CONOCIDA": "frecuencia no conocida",
    "SIN FRECUENCIA": "frecuencia no conocida",
}


# Variante 0 = más fiel al CSV; variantes superiores van introduciendo más
# sinónimos / reformulaciones para favorecer la diversidad léxica.
_STYLE_VARIANTS: list[str] = [
    "Estilo neutro y técnico, calcado del original SmPC. Usa el principio "
    "activo tal cual aparece en la tabla y reacciones adversas en su forma "
    "estándar. Pocas perífrasis.",
    "Estilo todavía técnico pero introduce una o dos denominaciones comunes "
    "del principio activo (sinónimos, marca o nombre comercial breve) y, en "
    "alguna reacción puntual, una variante coloquial entre paréntesis. "
    "Mantén las reacciones de la tabla literalmente al menos en la lista "
    "tabulada.",
    "Estilo redactado con mayor naturalidad. Puedes usar varios sinónimos "
    "del principio activo y reescribir alguna reacción adversa con un "
    "término clínico equivalente, siempre incluyendo también la forma "
    "literal de la tabla en la lista tabulada.",
]


def _pick_style(variante_idx: int) -> str:
    if variante_idx < 0:
        variante_idx = 0
    if variante_idx >= len(_STYLE_VARIANTS):
        return _STYLE_VARIANTS[-1]
    return _STYLE_VARIANTS[variante_idx]


def _normalise_frequency(value: str) -> str:
    """Map an upper-case CSV frequency to its EMA textual form."""
    if not isinstance(value, str):
        return "frecuencia no conocida"
    key = value.strip().upper()
    return FRECUENCIA_TEXTUAL.get(key, value.strip().lower())


def _group_rows_by_system_and_frequency(
    rows: pd.DataFrame,
) -> list[dict]:
    """Return a tabular view sistema → [(frecuencia_textual, [reacciones])]."""
    grouped: list[dict] = []
    for sistema, sys_rows in rows.groupby("SISTEMA", sort=False):
        bloques = []
        for frec, freq_rows in sys_rows.groupby("FRECUENCIA", sort=False):
            reacciones = sorted(
                {
                    str(r).strip()
                    for r in freq_rows["REACADV"].dropna().tolist()
                    if str(r).strip()
                }
            )
            bloques.append(
                {
                    "frecuencia_csv": str(frec).strip(),
                    "frecuencia_textual": _normalise_frequency(str(frec)),
                    "reacciones": reacciones,
                }
            )
        grouped.append({"sistema": str(sistema).strip(), "bloques": bloques})
    return grouped


# ---------------------------------------------------------------------------
# System prompt (shared by long-doc and snippet builders)
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "Eres un redactor médico farmacéutico español, especializado en preparar "
    "la sección 4.8 «Reacciones adversas» de fichas técnicas (SmPC) "
    "registradas en la AEMPS y más concretamente en CIMA. "
    "Tu objetivo es generar texto SINTÉTICO en español neutro,"
    "fiel al estilo de las fichas técnicas reales (EMA / "
    "MedDRA), a partir de una tabla estructurada que se te proporciona. "
    "Debes devolver SIEMPRE un único bloque JSON estricto que contenga el "
    "texto sintético y la lista exacta de entidades que aparecen en él, sin "
    "añadir ningún otro contenido fuera del bloque JSON."
)


# ---------------------------------------------------------------------------
# JSON contract description shared by both prompts
# ---------------------------------------------------------------------------
_JSON_CONTRACT = textwrap.dedent(
    """
    Devuelve EXCLUSIVAMENTE un bloque JSON con la siguiente forma exacta
    (sin texto adicional antes ni después, sin Markdown, sin explicación):

    {
      "text": "<texto sintético en español, multilínea permitido>",
      "entities": [
        {
          "type":      "ACTIVE | REACT | FREQ | SYS",
          "canonical": "<valor original de la tabla>",
          "surface":   "<substring EXACTO tal y como aparece en `text`>"
        }
      ]
    }

    Reglas estrictas e innegociables:
    1. `surface` debe ser una **subcadena LITERAL** que aparezca tal cual en
       `text` (mismas mayúsculas, mismos acentos, misma puntuación interna).
       Si decides reformular una mención, vuelve a escribirla idéntica en
       `surface`. Si una entidad de la tabla NO aparece en el texto, NO la
       incluyas en `entities`.
    2. Cada mención del texto que pertenezca a una de las cuatro categorías
       debe estar declarada en `entities`. No inventes entidades que no
       estén respaldadas por la tabla original (no añadas reacciones
       adversas, frecuencias o sistemas que no figuren en la tabla).
    3. `type` debe ser uno de: "ACTIVE" (principio activo), "REACT"
       (reacción adversa), "FREQ" (frecuencia EMA: «muy frecuentes»,
       «frecuentes», «poco frecuentes», «raras», «muy raras», «frecuencia
       no conocida»), "SYS" (clasificación de órganos / sistema).
    4. `canonical` para ACTIVE/REACT/SYS debe ser el valor exacto de la
       tabla. Para FREQ usa la cadena en MAYÚSCULAS de la tabla
       («MUY FRECUENTE», «FRECUENTE», «POCO FRECUENTE», «RARA»,
       «MUY RARA», «FRECUENCIA NO CONOCIDA»).
    5. Está permitido (y deseable) que el texto incluya frases neutras de
       contexto sin entidades (resumen del perfil de seguridad, advertencias
       generales, etc.). No fuerces que cada frase contenga una entidad.
    6. El JSON debe ser parseable por `json.loads` sin modificaciones. No
       uses comillas tipográficas («…»), comas finales sobrantes ni
       comentarios. Las comillas de los valores deben ir escapadas si el
       texto las contiene.
    """
).strip()


# ---------------------------------------------------------------------------
# Long-document prompt: one rich SmPC-style document per drug
# ---------------------------------------------------------------------------
def build_long_doc_messages(
    medicamento: str,
    pactivos: list[str],
    rows: pd.DataFrame,
    variante_idx: int = 0,
    target_min_words: int = 400,
    target_max_words: int = 900,
) -> list[dict]:
    """Build the chat messages that ask the LLM for a full synthetic 4.8 doc.

    Parameters
    ----------
    medicamento:
        Commercial name as it appears in the ``MEDICAMENTO`` column.
    pactivos:
        Unique active ingredients of the drug (``PACTIVO`` column dedup'd).
    rows:
        All annotation rows belonging to a single ``CODIGO``. The DataFrame
        must contain ``REACADV``, ``FRECUENCIA`` and ``SISTEMA``.
    variante_idx:
        Style variant (0 = closest to the table, ≥1 progressively freer).
    target_min_words / target_max_words:
        Soft length budget for the LLM. Kept conservative to bound cost.
    """
    style = _pick_style(variante_idx)
    grouped = _group_rows_by_system_and_frequency(rows)

    # Table view that we paste into the prompt.
    tabla_lineas: list[str] = []
    for entry in grouped:
        tabla_lineas.append(f"- Sistema: {entry['sistema']}")
        for bloque in entry["bloques"]:
            tabla_lineas.append(
                f"    · Frecuencia: {bloque['frecuencia_csv']} "
                f"(forma textual: «{bloque['frecuencia_textual']}»)"
            )
            for reac in bloque["reacciones"]:
                tabla_lineas.append(f"        * {reac}")
    tabla_str = "\n".join(tabla_lineas) if tabla_lineas else "(sin reacciones registradas)"

    pactivos_str = ", ".join(pactivos) if pactivos else "(no especificado)"

    user = textwrap.dedent(
        f"""
        Genera un documento SINTÉTICO en español, original y plausible, que
        imite la sección 4.8 «Reacciones adversas» de la ficha técnica del
        siguiente medicamento. El documento se usará para entrenar un modelo
        de NER y no debe copiarse de ninguna ficha real.

        Datos de partida (tabla original anotada por humanos):

        - Medicamento: {medicamento}
        - Principio(s) activo(s) (forma canónica): {pactivos_str}
        - Tabla de reacciones por sistema y frecuencia:
        {tabla_str}

        Estilo de redacción solicitado para esta variante (idx {variante_idx}):
        {style}

        Estructura recomendada del documento (puedes adaptarla):
        1. Un breve «Resumen del perfil de seguridad» (3-6 frases).
        2. Una «Lista tabulada de reacciones adversas» agrupada por sistema
           de clasificación de órganos y, dentro de cada sistema, por
           frecuencia (en forma textual: «muy frecuentes», «frecuentes»,
           «poco frecuentes», «raras», «muy raras», «frecuencia no
           conocida»). Lista las reacciones separadas por comas o saltos de
           línea, manteniendo la denominación original de la tabla.
        3. Opcional: una «Descripción de reacciones adversas seleccionadas»
           con 1-3 párrafos en prosa que mencionen algunas de las reacciones
           contextualizándolas (sin inventar nuevas).

        Restricciones de contenido:
        - El texto debe contener entre {target_min_words} y {target_max_words}
          palabras aproximadamente.
        - Aproximadamente la mitad del texto debe ser prosa de contexto sin
          menciones de las cuatro categorías (la otra mitad puede contener
          entidades). Esto es importante: no satures el texto sólo de
          reacciones.
        - Toda reacción adversa, frecuencia o sistema mencionado debe
          provenir de la tabla anterior. No inventes nuevas reacciones,
          sistemas ni categorías de frecuencia.
        - El principio activo puede aparecer con su forma canónica y, si la
          variante lo permite, con uno o dos sinónimos breves (denominación
          común, abreviatura aceptada, nombre comercial). Cualquier sinónimo
          adicional debe registrarse como entidad ACTIVE con su `canonical`
          igual al principio activo de la tabla.

        {_JSON_CONTRACT}
        """
    ).strip()

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Snippet prompt: short paragraph for a single reaction
# ---------------------------------------------------------------------------
def build_snippet_messages(
    medicamento: str,
    pactivo: str,
    reaccion: str,
    frecuencia: str,
    sistema: str,
    variante_idx: int = 0,
) -> list[dict]:
    """Build messages for a short synthetic paragraph (~80–180 words)."""
    style = _pick_style(variante_idx)
    frecuencia_textual = _normalise_frequency(frecuencia)

    user = textwrap.dedent(
        f"""
        Genera un fragmento BREVE en español (entre 80 y 180 palabras) que
        podría aparecer dentro de la sección 4.8 de una ficha técnica
        sintética para describir UNA reacción adversa concreta del
        medicamento.

        Datos:
        - Medicamento: {medicamento}
        - Principio activo (forma canónica): {pactivo}
        - Reacción adversa: {reaccion}
        - Frecuencia (CSV): {frecuencia}
          forma textual EMA: «{frecuencia_textual}»
        - Sistema de clasificación de órganos: {sistema}

        Estilo (variante idx {variante_idx}): {style}

        Restricciones:
        - El fragmento debe mencionar la reacción adversa, su frecuencia
          (en forma textual EMA), el sistema al que pertenece, y al menos
          una vez el principio activo (puede ser por su forma canónica o
          por un sinónimo, en cuyo caso regístralo igualmente).
        - Incluye 1-2 frases neutras de contexto (mecanismo plausible,
          recomendación de seguimiento, etc.) sin inventar nuevas
          reacciones, frecuencias ni sistemas.

        {_JSON_CONTRACT}
        """
    ).strip()

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Convenience: select up to N (reaction, frequency, system) tuples for the
# snippet phase, prioritising under-represented frequencies for diversity.
# ---------------------------------------------------------------------------
def select_snippet_tuples(
    rows: pd.DataFrame,
    pactivo: str,
    n: int,
    rng,
) -> list[dict]:
    """Pick `n` (reaccion, frecuencia, sistema) rows with stratified sampling.

    The same `rows` DataFrame used for the long doc is reused here. Each
    returned dict contains the exact strings to pass to
    :func:`build_snippet_messages`.
    """
    if n <= 0 or rows.empty:
        return []

    # Group by frequency to round-robin across categories so we don't end up
    # with all snippets of the same frequency for drugs heavily skewed.
    by_freq: dict[str, list[dict]] = {}
    for _, r in rows.iterrows():
        reaccion = str(r.get("REACADV", "")).strip()
        sistema = str(r.get("SISTEMA", "")).strip()
        frec = str(r.get("FRECUENCIA", "")).strip()
        if not (reaccion and sistema and frec):
            continue
        by_freq.setdefault(frec, []).append(
            {
                "medicamento": str(r.get("MEDICAMENTO", "")).strip(),
                "pactivo": pactivo,
                "reaccion": reaccion,
                "frecuencia": frec,
                "sistema": sistema,
            }
        )

    if not by_freq:
        return []

    # Shuffle within each bucket for variety across runs with same seed.
    for bucket in by_freq.values():
        rng.shuffle(bucket)

    # Round-robin pick.
    chosen: list[dict] = []
    freq_keys = list(by_freq.keys())
    rng.shuffle(freq_keys)
    while len(chosen) < n:
        progressed = False
        for k in freq_keys:
            if by_freq[k]:
                chosen.append(by_freq[k].pop())
                progressed = True
                if len(chosen) >= n:
                    break
        if not progressed:
            break
    return chosen


__all__ = [
    "FRECUENCIA_TEXTUAL",
    "build_long_doc_messages",
    "build_snippet_messages",
    "select_snippet_tuples",
]
