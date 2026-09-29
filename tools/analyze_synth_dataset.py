"""One-shot analysis of the synthetic NER corpus (Variante B).

Compares ``data/medicamentos_synth/`` and ``full_annotations_synth.csv``
against the original human-annotated ``full_annotations.csv`` and emits a
JSON report at ``data/reports/synth_dataset_quality.json`` plus a short
human-readable summary on stdout.

Usage
-----
Run from the repo root::

    python tools/analyze_synth_dataset.py

The script is intentionally self-contained (no CLI flags, no dependencies
beyond what the project already uses): pandas and the project's
:class:`src.conll_ner_builder.ConllBuilder` for the BIO statistics over
the same set of drugs in real and synthetic form.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any

import pandas as pd

# Make the repo root importable when invoking the script directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.conll_ner_builder import ConllBuilder  # noqa: E402
from src.synth_prompt_templates import FRECUENCIA_TEXTUAL  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ANNOTATIONS_REAL = os.path.join(_REPO_ROOT, "data/splits/full_annotations.csv")
ANNOTATIONS_SYNTH = os.path.join(_REPO_ROOT, "data/splits/full_annotations_synth.csv")
TRAIN_SYNTH = os.path.join(_REPO_ROOT, "data/splits/train_synth.csv")
TEST_SYNTH = os.path.join(_REPO_ROOT, "data/splits/test_synth.csv")

SYNTH_OUT_DIR = os.path.join(_REPO_ROOT, "data/medicamentos_synth")
SYNTH_TXT_DIR = os.path.join(SYNTH_OUT_DIR, "txt")
SYNTH_RAW_DIR = os.path.join(SYNTH_OUT_DIR, "raw_responses")
SYNTH_FAILED_DIR = os.path.join(SYNTH_RAW_DIR, "_failed")

REAL_TXT_DIR = os.path.join(_REPO_ROOT, "data/medicamentos/txt")
REPORT_DIR = os.path.join(_REPO_ROOT, "data/reports")
REPORT_PATH = os.path.join(REPORT_DIR, "synth_dataset_quality.json")

ENTITY_TYPES = ["REACT", "ACTIVE", "FREQ", "SYS"]
TYPE_TO_COL = {
    "REACT": "REACADV",
    "ACTIVE": "PACTIVO",
    "FREQ": "FRECUENCIA",
    "SYS": "SISTEMA",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _norm(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip().lower()


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0, "mean": 0.0}
    s = sorted(values)
    n = len(s)
    return {
        "min": float(s[0]),
        "p25": float(s[max(0, int(n * 0.25) - 1)]),
        "median": float(statistics.median(s)),
        "p75": float(s[min(n - 1, int(n * 0.75))]),
        "max": float(s[-1]),
        "mean": float(statistics.mean(s)),
    }


# ---------------------------------------------------------------------------
# 1. Coverage
# ---------------------------------------------------------------------------
def coverage_section(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    real_codigos = set(real_df["CODIGO"].astype(str).str.strip().unique())
    synth_orig = set(synth_df["ORIG_CODIGO"].astype(str).str.strip().unique())
    synth_variants = set(synth_df["CODIGO"].astype(str).str.strip().unique())

    failed_files: list[str] = []
    if os.path.isdir(SYNTH_FAILED_DIR):
        failed_files = sorted(os.listdir(SYNTH_FAILED_DIR))

    return {
        "n_real_codigos_total": len(real_codigos),
        "n_real_codigos_attempted": len(synth_orig),
        "n_synth_documents": len(synth_variants),
        "real_codigos_attempted_pct": (
            round(100.0 * len(synth_orig) / max(1, len(real_codigos)), 2)
        ),
        "n_failed_attempts_files": len(failed_files),
        "failed_attempts_examples": failed_files[:10],
    }


# ---------------------------------------------------------------------------
# 2. Tuple coverage (REACADV, FRECUENCIA, SISTEMA) per drug
# ---------------------------------------------------------------------------
def tuple_coverage_section(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    """For each ORIG_CODIGO with synth output, fraction of real (R, F, S)
    tuples that appear at least once across its synthetic variants
    (matched on the `_canonical` columns of synth)."""

    def _tuples(df: pd.DataFrame, react_col: str, freq_col: str, sys_col: str) -> set:
        out: set = set()
        for _, r in df.iterrows():
            t = (_norm(r[react_col]), _norm(r[freq_col]), _norm(r[sys_col]))
            if all(t):
                out.add(t)
        return out

    ratios: list[float] = []
    perfect = 0
    zero = 0
    per_drug: list[dict] = []

    synth_orig = set(synth_df["ORIG_CODIGO"].astype(str).str.strip().unique())
    for codigo in sorted(synth_orig):
        real_rows = real_df[real_df["CODIGO"].astype(str).str.strip() == codigo]
        synth_rows = synth_df[synth_df["ORIG_CODIGO"].astype(str).str.strip() == codigo]
        real_tuples = _tuples(real_rows, "REACADV", "FRECUENCIA", "SISTEMA")
        synth_tuples = _tuples(
            synth_rows, "REACADV_canonical", "FRECUENCIA_canonical", "SISTEMA_canonical"
        )
        if not real_tuples:
            continue
        recovered = len(real_tuples & synth_tuples)
        ratio = recovered / len(real_tuples)
        ratios.append(ratio)
        if ratio == 1.0:
            perfect += 1
        if ratio == 0.0:
            zero += 1
        per_drug.append(
            {
                "codigo": codigo,
                "n_real_tuples": len(real_tuples),
                "n_recovered": recovered,
                "ratio": round(ratio, 4),
            }
        )

    per_drug_sorted = sorted(per_drug, key=lambda d: d["ratio"])
    return {
        "n_drugs_evaluated": len(ratios),
        "mean_recovery_ratio": round(statistics.mean(ratios), 4) if ratios else 0.0,
        "median_recovery_ratio": round(statistics.median(ratios), 4) if ratios else 0.0,
        "drugs_with_perfect_recovery": perfect,
        "drugs_with_zero_recovery": zero,
        "worst_5_drugs": per_drug_sorted[:5],
        "best_5_drugs": per_drug_sorted[-5:][::-1],
    }


# ---------------------------------------------------------------------------
# 3. Hallucinations
# ---------------------------------------------------------------------------
def hallucinations_section(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    """Synthetic entity whose `_canonical` value is not present in the real
    table for the same ORIG_CODIGO."""
    real_lookup: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for _, r in real_df.iterrows():
        codigo = _norm(r["CODIGO"])
        for ent_type, col in TYPE_TO_COL.items():
            v = _norm(r[col])
            if v:
                real_lookup[codigo][ent_type].add(v)

    counts = {t: {"total": 0, "hallucinated": 0} for t in ENTITY_TYPES}
    examples: dict[str, list[dict]] = {t: [] for t in ENTITY_TYPES}

    for _, r in synth_df.iterrows():
        codigo = _norm(r["ORIG_CODIGO"])
        for ent_type in ENTITY_TYPES:
            canon_col = f"{TYPE_TO_COL[ent_type]}_canonical"
            v = _norm(r[canon_col])
            if not v:
                continue
            counts[ent_type]["total"] += 1
            if v not in real_lookup[codigo][ent_type]:
                counts[ent_type]["hallucinated"] += 1
                if len(examples[ent_type]) < 10:
                    examples[ent_type].append(
                        {
                            "codigo_variant": str(r["CODIGO"]),
                            "orig_codigo": codigo,
                            "canonical": v,
                        }
                    )

    summary: dict[str, dict] = {}
    for t in ENTITY_TYPES:
        total = counts[t]["total"]
        hallu = counts[t]["hallucinated"]
        summary[t] = {
            "total_mentions": total,
            "hallucinated": hallu,
            "hallucination_rate_pct": round(100.0 * hallu / total, 2) if total else 0.0,
            "examples": examples[t],
        }
    return summary


# ---------------------------------------------------------------------------
# 4. BIO label distribution
# ---------------------------------------------------------------------------
def bio_distribution_section(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    real_stats_path = os.path.join(REPORT_DIR, "synth_dataset_stats.json")
    synth_stats: dict[str, Any] = {}
    if os.path.isfile(real_stats_path):
        with open(real_stats_path, "r", encoding="utf-8") as f:
            synth_stats = json.load(f)

    drugs = sorted(synth_df["ORIG_CODIGO"].astype(str).str.strip().unique())
    real_subset = real_df[real_df["CODIGO"].astype(str).str.strip().isin(drugs)]

    real_stats: dict[str, Any] = {}
    try:
        builder = ConllBuilder(txt_dir=REAL_TXT_DIR, entity_types=ENTITY_TYPES)
        examples = builder.build_dataset(real_subset)
        real_stats = builder.summarise(examples)
    except Exception as exc:  # pragma: no cover
        real_stats = {"error": f"{type(exc).__name__}: {exc}"}

    def _label_pct(stats: dict[str, Any]) -> dict[str, float]:
        lc = stats.get("label_counts", {}) or {}
        total = sum(lc.values()) or 1
        return {k: round(100.0 * v / total, 3) for k, v in lc.items()}

    return {
        "real_subset_same_drugs": {
            "documents": real_stats.get("documents"),
            "tokens": real_stats.get("tokens"),
            "label_counts": real_stats.get("label_counts"),
            "label_pct": _label_pct(real_stats),
            "missing_txt_files": real_stats.get("missing_txt_files", []),
        },
        "synth_full": {
            "documents": synth_stats.get("documents"),
            "tokens": synth_stats.get("tokens"),
            "label_counts": synth_stats.get("label_counts"),
            "label_pct": _label_pct(synth_stats),
        },
    }


# ---------------------------------------------------------------------------
# 5. Lexical diversity of surfaces
# ---------------------------------------------------------------------------
def diversity_section(synth_df: pd.DataFrame) -> dict:
    out: dict[str, dict] = {}
    for ent_type in ENTITY_TYPES:
        col = TYPE_TO_COL[ent_type]
        canon_col = f"{col}_canonical"
        sub = synth_df[[col, canon_col]].dropna()
        sub = sub[sub[canon_col].astype(str).str.strip() != ""]
        canon_to_surfaces: dict[str, set[str]] = defaultdict(set)
        for _, r in sub.iterrows():
            canon = str(r[canon_col]).strip()
            surf = str(r[col]).strip()
            if canon and surf:
                canon_to_surfaces[canon].add(surf)
        if not canon_to_surfaces:
            out[ent_type] = {"n_canonicals": 0}
            continue
        surfaces_per_canon = [len(v) for v in canon_to_surfaces.values()]
        with_synonyms = sum(1 for n in surfaces_per_canon if n > 1)
        top = sorted(canon_to_surfaces.items(), key=lambda kv: len(kv[1]), reverse=True)[:10]
        out[ent_type] = {
            "n_canonicals": len(canon_to_surfaces),
            "n_canonicals_with_synonyms": with_synonyms,
            "pct_canonicals_with_synonyms": round(
                100.0 * with_synonyms / len(canon_to_surfaces), 2
            ),
            "mean_surfaces_per_canonical": round(statistics.mean(surfaces_per_canon), 3),
            "max_surfaces_per_canonical": max(surfaces_per_canon),
            "top10_most_synonyms": [
                {"canonical": k, "n_surfaces": len(v), "surfaces": sorted(v)[:8]}
                for k, v in top
            ],
        }
    return out


# ---------------------------------------------------------------------------
# 6. Frequency normalisation (CSV upper -> EMA textual form)
# ---------------------------------------------------------------------------
def frequency_normalisation_section(synth_df: pd.DataFrame) -> dict:
    expected = {k: v.lower() for k, v in FRECUENCIA_TEXTUAL.items()}
    rows = synth_df[["FRECUENCIA", "FRECUENCIA_canonical"]].dropna()
    total = len(rows)
    if total == 0:
        return {"total": 0}

    by_canon: dict[str, Counter] = defaultdict(Counter)
    for _, r in rows.iterrows():
        canon = str(r["FRECUENCIA_canonical"]).strip().upper()
        surf = str(r["FRECUENCIA"]).strip().lower()
        by_canon[canon][surf] += 1

    conformant = 0
    deviations: list[dict] = []
    for canon, surf_counts in by_canon.items():
        target = expected.get(canon)
        for surf, n in surf_counts.items():
            if target is None:
                deviations.append(
                    {"canonical": canon, "surface": surf, "n": n, "reason": "unknown_canonical"}
                )
                continue
            # Accept singular/plural and prefix variants of the EMA form.
            target_singular = target.rstrip("s")
            if surf == target or surf == target_singular or target.startswith(surf):
                conformant += n
            else:
                deviations.append(
                    {"canonical": canon, "surface": surf, "n": n, "expected": target}
                )

    deviations.sort(key=lambda d: -d["n"])
    return {
        "total": total,
        "conformant": conformant,
        "conformant_pct": round(100.0 * conformant / total, 2),
        "n_deviating_surface_forms": len(deviations),
        "top_deviations": deviations[:15],
        "by_canonical": {
            canon: dict(counts.most_common(5)) for canon, counts in by_canon.items()
        },
    }


# ---------------------------------------------------------------------------
# 7. Document length (synth vs real subset)
# ---------------------------------------------------------------------------
def length_section(synth_df: pd.DataFrame) -> dict:
    def _stats_for(dir_path: str, codigos: set[str] | None) -> dict:
        word_counts: list[int] = []
        char_counts: list[int] = []
        if not os.path.isdir(dir_path):
            return {"error": f"missing {dir_path}"}
        for fname in os.listdir(dir_path):
            if not fname.endswith(".txt"):
                continue
            stem = fname[:-4]
            if codigos is not None and stem not in codigos:
                continue
            path = os.path.join(dir_path, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    txt = f.read()
            except OSError:
                continue
            word_counts.append(len(txt.split()))
            char_counts.append(len(txt))
        return {
            "n_files": len(word_counts),
            "words": _percentiles([float(w) for w in word_counts]),
            "chars": _percentiles([float(c) for c in char_counts]),
        }

    real_codigos = set(synth_df["ORIG_CODIGO"].astype(str).str.strip().unique())
    return {
        "synthetic": _stats_for(SYNTH_TXT_DIR, codigos=None),
        "real_subset": _stats_for(REAL_TXT_DIR, codigos=real_codigos),
    }


# ---------------------------------------------------------------------------
# 8. Validator outcome
# ---------------------------------------------------------------------------
def validator_section() -> dict:
    discard_ratios: list[float] = []
    attempts: list[int] = []
    models: Counter = Counter()
    style_idx: Counter = Counter()
    if not os.path.isdir(SYNTH_RAW_DIR):
        return {"error": f"missing {SYNTH_RAW_DIR}"}
    for fname in os.listdir(SYNTH_RAW_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(SYNTH_RAW_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if "discard_ratio" in payload:
            discard_ratios.append(float(payload["discard_ratio"]))
        if "attempts" in payload:
            attempts.append(int(payload["attempts"]))
        if "model" in payload:
            models[str(payload["model"])] += 1
        if "style_variant_idx" in payload:
            style_idx[str(payload["style_variant_idx"])] += 1
    return {
        "n_documents": len(discard_ratios),
        "discard_ratio": _percentiles(discard_ratios),
        "n_docs_with_zero_discard": sum(1 for d in discard_ratios if d == 0.0),
        "attempts_distribution": dict(Counter(attempts)),
        "model_used": dict(models),
        "style_variant_distribution": dict(style_idx),
    }


# ---------------------------------------------------------------------------
# 9. Splits leakage check
# ---------------------------------------------------------------------------
def splits_section() -> dict:
    if not (os.path.isfile(TRAIN_SYNTH) and os.path.isfile(TEST_SYNTH)):
        return {"error": "synthetic split files not found"}
    train_df = pd.read_csv(TRAIN_SYNTH, dtype={"CODIGO": str, "ORIG_CODIGO": str})
    test_df = pd.read_csv(TEST_SYNTH, dtype={"CODIGO": str, "ORIG_CODIGO": str})

    train_orig = set(train_df["ORIG_CODIGO"].astype(str).str.strip())
    test_orig = set(test_df["ORIG_CODIGO"].astype(str).str.strip())
    overlap = train_orig & test_orig

    return {
        "train_codigos_unique": int(train_df["CODIGO"].nunique()),
        "test_codigos_unique": int(test_df["CODIGO"].nunique()),
        "train_orig_codigos_unique": len(train_orig),
        "test_orig_codigos_unique": len(test_orig),
        "leakage_codigos": sorted(overlap),
        "leakage_count": len(overlap),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not os.path.isfile(ANNOTATIONS_REAL):
        raise SystemExit(f"Missing real annotations: {ANNOTATIONS_REAL}")
    if not os.path.isfile(ANNOTATIONS_SYNTH):
        raise SystemExit(f"Missing synth annotations: {ANNOTATIONS_SYNTH}")

    real_df = pd.read_csv(ANNOTATIONS_REAL, dtype={"CODIGO": str})
    synth_df = pd.read_csv(
        ANNOTATIONS_SYNTH, dtype={"CODIGO": str, "ORIG_CODIGO": str}
    )

    print("[1/9] Coverage ...")
    coverage = coverage_section(real_df, synth_df)
    print("[2/9] Tuple coverage per drug ...")
    tuples = tuple_coverage_section(real_df, synth_df)
    print("[3/9] Hallucinations ...")
    hallu = hallucinations_section(real_df, synth_df)
    print("[4/9] BIO distribution (real subset vs synth) ...")
    bio = bio_distribution_section(real_df, synth_df)
    print("[5/9] Lexical diversity ...")
    diversity = diversity_section(synth_df)
    print("[6/9] Frequency normalisation ...")
    freqs = frequency_normalisation_section(synth_df)
    print("[7/9] Document length ...")
    lengths = length_section(synth_df)
    print("[8/9] Validator outcomes ...")
    validator = validator_section()
    print("[9/9] Splits leakage ...")
    splits = splits_section()
    if splits.get("leakage_count", 0):
        print(
            f"  WARNING: leakage of ORIG_CODIGO between train_synth/test_synth: "
            f"{splits['leakage_codigos']} (inherited from real splits)"
        )

    report = {
        "coverage": coverage,
        "tuple_coverage": tuples,
        "hallucinations": hallu,
        "bio_distribution": bio,
        "lexical_diversity": diversity,
        "frequency_normalisation": freqs,
        "document_length": lengths,
        "validator": validator,
        "splits": splits,
    }

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== RESUMEN ===")
    print(f"Drogas reales: {coverage['n_real_codigos_total']}")
    print(
        f"Drogas con generación sintética: {coverage['n_real_codigos_attempted']} "
        f"({coverage['real_codigos_attempted_pct']}%)"
    )
    print(f"Documentos sintéticos generados: {coverage['n_synth_documents']}")
    print(
        f"Cobertura media de tuplas (R,F,S): "
        f"{tuples.get('mean_recovery_ratio', 0):.3f} "
        f"(perfect={tuples.get('drugs_with_perfect_recovery', 0)}, "
        f"zero={tuples.get('drugs_with_zero_recovery', 0)})"
    )
    for t in ENTITY_TYPES:
        h = hallu[t]
        print(
            f"Hallucinaciones {t}: {h['hallucinated']}/{h['total_mentions']} "
            f"({h['hallucination_rate_pct']}%)"
        )
    active = diversity.get("ACTIVE", {})
    print(
        f"Diversidad ACTIVE: media {active.get('mean_surfaces_per_canonical', 0)} "
        f"superficies/canonical, {active.get('pct_canonicals_with_synonyms', 0)}% "
        f"con sinónimos"
    )
    print(
        f"Frecuencias conforme a EMA: {freqs.get('conformant_pct', 0)}% "
        f"de {freqs.get('total', 0)} menciones"
    )
    syn_words = lengths["synthetic"]["words"]
    real_words = lengths["real_subset"]["words"]
    print(
        f"Longitud sintética: media {syn_words['mean']:.0f} palabras "
        f"(p25={syn_words['p25']:.0f}, p75={syn_words['p75']:.0f})"
    )
    print(
        f"Longitud real (mismas drogas): media {real_words['mean']:.0f} palabras "
        f"(p25={real_words['p25']:.0f}, p75={real_words['p75']:.0f})"
    )
    print(
        f"Validador: discard_ratio media={validator['discard_ratio']['mean']:.3f}, "
        f"docs con discard=0: {validator['n_docs_with_zero_discard']}/{validator['n_documents']}"
    )
    print(
        f"Modelo(s) utilizado(s): {validator['model_used']}; "
        f"variantes de estilo: {validator['style_variant_distribution']}"
    )
    print(f"Splits sin leakage: train_orig∩test_orig = {splits['leakage_count']}")
    print(f"\nInforme completo: {REPORT_PATH}")


if __name__ == "__main__":
    main()
