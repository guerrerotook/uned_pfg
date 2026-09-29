"""CLI orchestrator for the synthetic NER dataset (Variante B).

Walks ``data/splits/full_annotations.csv`` (or any custom annotation table),
calls an LLM via :mod:`src.llm.azure_foundry` to generate synthetic
section-4.8 documents per drug, validates that every declared entity is
re-locatable in the synthetic text and writes the corpus to
``data/medicamentos_synth/`` so the existing
:class:`src.conll_ner_builder.ConllBuilder` and
:class:`src.supervised_training_conll.SupervisedTrainerConll` can train on
it without modification.

Examples
--------
Smoke test (2 drugs, 1 variant, 2 snippets each)::

    python run_synth_dataset.py --limit 2 --variants 1 --snippets-per-doc 2

Full corpus with default settings (GPT-5, 2 variants per drug, 3 snippets)::

    python run_synth_dataset.py

Resume an interrupted run (skips drug+variant pairs already on disk)::

    python run_synth_dataset.py --resume

Dry run (only prints the prompt of the first drug, no LLM call)::

    python run_synth_dataset.py --dry-run --limit 1
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import random
import sys
from typing import Optional

import pandas as pd

# Make sure the repo root is on PYTHONPATH when invoked as a script.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.conll_ner_builder import ConllBuilder  # noqa: E402
from src.synth_ner_generator import (  # noqa: E402
    SynthNerGenerator,
    consolidate_full_annotations,
    derive_synth_splits,
)
from src.synth_prompt_templates import (  # noqa: E402
    build_long_doc_messages,
    build_snippet_messages,
    select_snippet_tuples,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_synth_dataset")


MODEL_REGISTRY: dict[str, tuple[str, str, dict]] = {
    "gpt-5.1": (
        "src.llm.azure_foundry",
        "Gpt5AzureOpenAI",
        {"model_name": "gpt-5.1"},
    ),
}


def _load_llm_client(name: str):
    if name not in MODEL_REGISTRY:
        raise SystemExit(
            f"Unknown LLM '{name}'. Available: {sorted(MODEL_REGISTRY.keys())}"
        )
    module_name, class_name, kwargs = MODEL_REGISTRY[name]
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(**kwargs)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate the synthetic NER dataset (Variante B / cima-ner-synth-es) "
            "from data/splits/full_annotations.csv using an LLM."
        )
    )
    p.add_argument(
        "--annotations",
        default="data/splits/full_annotations.csv",
        help="Annotation table to read (default: data/splits/full_annotations.csv)",
    )
    p.add_argument(
        "--train-csv",
        default="data/splits/train.csv",
        help="Original train split (used to derive train_synth.csv)",
    )
    p.add_argument(
        "--test-csv",
        default="data/splits/test.csv",
        help="Original test split (used to derive test_synth.csv)",
    )
    p.add_argument(
        "--out-dir",
        default="artifacts/synthetic/data",
        help="Output directory for synthetic text and raw responses.",
    )
    p.add_argument(
        "--splits-out-dir",
        default="artifacts/synthetic/splits",
        help="Where to write generated synthetic split CSV files.",
    )
    p.add_argument(
        "--reports-dir",
        default="artifacts/synthetic/reports",
        help="Where to write synth_dataset_stats.json.",
    )
    p.add_argument(
        "--llm",
        default="gpt-5.1",
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Azure deployment to use (default: gpt-5.1)",
    )
    p.add_argument(
        "--variants", type=int, default=2, help="Variants per drug (default: 2)"
    )
    p.add_argument(
        "--snippets-per-doc",
        type=int,
        default=3,
        help="Synthetic short snippets to generate per (drug, variant) (default: 3)",
    )
    p.add_argument(
        "--max-unmatched-ratio",
        type=float,
        default=0.30,
        help="Max ratio of declared entities that may be unmatched before retry (default: 0.30)",
    )
    p.add_argument(
        "--max-doc-retries",
        type=int,
        default=2,
        help="Long-doc regeneration attempts after the first one (default: 2)",
    )
    p.add_argument(
        "--llm-max-attempts",
        type=int,
        default=5,
        help="Per-LLM-call retry attempts on transient errors (default: 5)",
    )
    p.add_argument(
        "--target-min-words",
        type=int,
        default=400,
        help="Soft minimum word count requested to the LLM for the long doc",
    )
    p.add_argument(
        "--target-max-words",
        type=int,
        default=900,
        help="Soft maximum word count requested to the LLM for the long doc",
    )
    p.add_argument("--seed", type=int, default=42, help="Seed for snippet selection")
    p.add_argument("--limit", type=int, default=None, help="Process at most N drugs")
    p.add_argument(
        "--codigos",
        default=None,
        help="Comma-separated list of CODIGOs to process (overrides --limit ordering)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip (drug, variant) pairs already present on disk (default behaviour also skips them)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if (drug, variant) files already exist",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the prompts and print the first one, but never call the LLM",
    )
    p.add_argument(
        "--dump-prompts",
        action="store_true",
        help=(
            "Build EVERY prompt (long doc + snippets for each variant of "
            "each selected codigo) and dump them to --prompts-out-dir as "
            "both .json (full message list) and .txt (human-readable). No "
            "LLM call is performed. Useful to review/audit prompts in bulk "
            "before spending tokens."
        ),
    )
    p.add_argument(
        "--prompts-out-dir",
        default="artifacts/synthetic/prompts",
        help="Where --dump-prompts writes the prompt files (default: data/synth_prompts)",
    )
    p.add_argument(
        "--no-consolidate",
        action="store_true",
        help="Skip the post-generation consolidation of the CSVs and stats",
    )
    p.add_argument(
        "--stats-only",
        action="store_true",
        help=(
            "Skip generation entirely; just (re)consolidate the CSVs and stats "
            "from whatever is already on disk under --out-dir/entities."
        ),
    )
    return p


def _load_annotations(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise SystemExit(f"Annotations file not found: {path}")
    df = pd.read_csv(path, dtype={"CODIGO": str})
    df["CODIGO"] = df["CODIGO"].astype(str).str.strip()
    expected = {"CODIGO", "MEDICAMENTO", "PACTIVO", "REACADV", "FRECUENCIA", "SISTEMA"}
    missing = expected - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing expected columns: {sorted(missing)}")
    return df


def _select_codigos(
    df: pd.DataFrame, limit: Optional[int], codigos_arg: Optional[str]
) -> list[str]:
    all_codigos = list(dict.fromkeys(df["CODIGO"].tolist()))
    if codigos_arg:
        wanted = [c.strip() for c in codigos_arg.split(",") if c.strip()]
        unknown = [c for c in wanted if c not in set(all_codigos)]
        if unknown:
            logger.warning(
                "Codigos not present in annotations and will be ignored: %s", unknown
            )
        result = [c for c in wanted if c in set(all_codigos)]
    else:
        result = all_codigos
    if limit is not None:
        result = result[:limit]
    return result


def _do_dry_run(df: pd.DataFrame, codigos: list[str], variants: int) -> None:
    if not codigos:
        logger.warning("No codigos to dry-run.")
        return
    codigo = codigos[0]
    rows = df[df["CODIGO"] == codigo]
    medicamento = (
        rows["MEDICAMENTO"].dropna().astype(str).iloc[0]
        if not rows["MEDICAMENTO"].dropna().empty
        else codigo
    )
    pactivos = sorted(
        {str(p).strip() for p in rows["PACTIVO"].dropna() if str(p).strip()}
    )
    messages = build_long_doc_messages(
        medicamento=medicamento,
        pactivos=pactivos,
        rows=rows,
        variante_idx=0,
    )
    print("=" * 80)
    print(f"DRY RUN — first long-doc prompt for CODIGO {codigo} (variants={variants})")
    print("=" * 80)
    for m in messages:
        print(f"\n--- [{m['role'].upper()}] ---")
        print(m["content"])


def _messages_to_text(messages: list[dict]) -> str:
    """Render a chat-message list as a human-readable, copy-pasteable file."""
    parts: list[str] = []
    for m in messages:
        role = str(m.get("role", "")).upper()
        parts.append(f"--- [{role}] ---")
        parts.append(str(m.get("content", "")))
        parts.append("")
    return "\n".join(parts).strip() + "\n"


def _dump_all_prompts(
    *,
    annotations_df: pd.DataFrame,
    codigos: list[str],
    out_dir: str,
    n_variants: int,
    n_snippets_per_doc: int,
    target_min_words: int,
    target_max_words: int,
    seed: int,
) -> None:
    """Build every prompt that the generator would send and dump it to disk.

    Mirrors the (codigo, variante, snippet) selection logic of
    :class:`SynthNerGenerator` so the dumped prompts are exactly the ones
    that would be issued by ``run_synth_dataset.py`` with the same flags
    (modulo ``--llm``, which only affects the client, not the prompt body).

    For each (codigo, variant) we write:
      - ``{codigo}_v{N}_long.json``    — full message list for the long doc
      - ``{codigo}_v{N}_long.txt``     — same, human-readable
      - ``{codigo}_v{N}_snippet_{S}_{REACT}.json`` and ``.txt`` per snippet
      - ``{codigo}_v{N}_index.json``   — small manifest with the snippet seeds
    plus a top-level ``manifest.json`` listing every emitted prompt.
    """
    os.makedirs(out_dir, exist_ok=True)
    manifest: list[dict] = []
    long_count = 0
    snippet_count = 0

    for codigo in codigos:
        rows = annotations_df[annotations_df["CODIGO"] == codigo]
        if rows.empty:
            logger.warning(
                "[dump-prompts] %s — no rows in annotations, skipping", codigo
            )
            continue

        medicamento_series = (
            rows["MEDICAMENTO"].dropna().astype(str)
            if "MEDICAMENTO" in rows.columns
            else pd.Series(dtype=str)
        )
        medicamento = (
            medicamento_series.iloc[0].strip()
            if not medicamento_series.empty
            else codigo
        )
        pactivos = sorted(
            {
                str(p).strip()
                for p in rows.get("PACTIVO", pd.Series(dtype=str)).dropna().tolist()
                if str(p).strip()
            }
        )

        for variante in range(1, n_variants + 1):
            style_idx = variante - 1
            rng = random.Random(f"{seed}|{codigo}|{variante}")
            base = f"{codigo}_v{variante}"

            # ---- Long doc prompt ----
            long_messages = build_long_doc_messages(
                medicamento=medicamento,
                pactivos=pactivos,
                rows=rows,
                variante_idx=style_idx,
                target_min_words=target_min_words,
                target_max_words=target_max_words,
            )
            long_json_path = os.path.join(out_dir, f"{base}_long.json")
            long_txt_path = os.path.join(out_dir, f"{base}_long.txt")
            with open(long_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "kind": "long_doc",
                        "codigo": codigo,
                        "variante": variante,
                        "style_variant_idx": style_idx,
                        "medicamento": medicamento,
                        "pactivos": pactivos,
                        "messages": long_messages,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            with open(long_txt_path, "w", encoding="utf-8") as f:
                f.write(_messages_to_text(long_messages))
            long_count += 1
            manifest.append(
                {
                    "kind": "long_doc",
                    "codigo": codigo,
                    "variante": variante,
                    "json": os.path.relpath(long_json_path, out_dir),
                    "txt": os.path.relpath(long_txt_path, out_dir),
                    "n_chars_user": len(long_messages[-1]["content"]),
                }
            )

            # ---- Snippet prompts (same RNG progression as the generator) ----
            seeds = select_snippet_tuples(
                rows=rows,
                pactivo=(pactivos[0] if pactivos else ""),
                n=n_snippets_per_doc,
                rng=rng,
            )
            snippet_index: list[dict] = []
            for s_idx, seed_tuple in enumerate(seeds, start=1):
                snippet_messages = build_snippet_messages(
                    medicamento=seed_tuple["medicamento"],
                    pactivo=seed_tuple["pactivo"],
                    reaccion=seed_tuple["reaccion"],
                    frecuencia=seed_tuple["frecuencia"],
                    sistema=seed_tuple["sistema"],
                    variante_idx=style_idx,
                )
                slug = (
                    "".join(
                        ch if ch.isalnum() else "_"
                        for ch in seed_tuple["reaccion"].lower()
                    )[:40].strip("_")
                    or "react"
                )
                fname_base = f"{base}_snippet_{s_idx:02d}_{slug}"
                snippet_json_path = os.path.join(out_dir, f"{fname_base}.json")
                snippet_txt_path = os.path.join(out_dir, f"{fname_base}.txt")
                with open(snippet_json_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "kind": "snippet",
                            "codigo": codigo,
                            "variante": variante,
                            "snippet_idx": s_idx,
                            "style_variant_idx": style_idx,
                            "seed": seed_tuple,
                            "messages": snippet_messages,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                with open(snippet_txt_path, "w", encoding="utf-8") as f:
                    f.write(_messages_to_text(snippet_messages))
                snippet_count += 1
                snippet_index.append(
                    {
                        "snippet_idx": s_idx,
                        "seed": seed_tuple,
                        "json": os.path.relpath(snippet_json_path, out_dir),
                        "txt": os.path.relpath(snippet_txt_path, out_dir),
                    }
                )
                manifest.append(
                    {
                        "kind": "snippet",
                        "codigo": codigo,
                        "variante": variante,
                        "snippet_idx": s_idx,
                        "json": os.path.relpath(snippet_json_path, out_dir),
                        "txt": os.path.relpath(snippet_txt_path, out_dir),
                        "n_chars_user": len(snippet_messages[-1]["content"]),
                    }
                )

            # Per-(codigo, variante) index, useful when reviewing one drug.
            with open(
                os.path.join(out_dir, f"{base}_index.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(
                    {
                        "codigo": codigo,
                        "variante": variante,
                        "long_doc": {
                            "json": f"{base}_long.json",
                            "txt": f"{base}_long.txt",
                        },
                        "snippets": snippet_index,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_drugs": len(codigos),
                "n_variants_per_drug": n_variants,
                "n_snippets_per_variant": n_snippets_per_doc,
                "n_long_doc_prompts": long_count,
                "n_snippet_prompts": snippet_count,
                "n_total_prompts": long_count + snippet_count,
                "items": manifest,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(
        "Dumped %d prompts (%d long + %d snippets) for %d drug(s) to %s",
        long_count + snippet_count,
        long_count,
        snippet_count,
        len(codigos),
        out_dir,
    )
    print(f"\nManifest: {manifest_path}")


def _consolidate_and_report(
    out_dir: str,
    splits_out_dir: str,
    reports_dir: str,
    annotations_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    os.makedirs(splits_out_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    logger.info("Consolidating entities → full_annotations_synth.csv ...")
    fa_synth = consolidate_full_annotations(
        out_dir=out_dir, annotations_df=annotations_df
    )
    fa_path = os.path.join(splits_out_dir, "full_annotations_synth.csv")
    fa_synth.to_csv(fa_path, index=False)
    logger.info("  Wrote %d rows to %s", len(fa_synth), fa_path)

    logger.info("Deriving train_synth.csv / test_synth.csv from real splits ...")
    train_synth, test_synth = derive_synth_splits(fa_synth, train_df, test_df)
    train_path = os.path.join(splits_out_dir, "train_synth.csv")
    test_path = os.path.join(splits_out_dir, "test_synth.csv")
    train_synth.to_csv(train_path, index=False)
    test_synth.to_csv(test_path, index=False)
    logger.info(
        "  train_synth=%d rows (%d codigos)  test_synth=%d rows (%d codigos)",
        len(train_synth),
        train_synth["CODIGO"].nunique(),
        len(test_synth),
        test_synth["CODIGO"].nunique(),
    )

    logger.info("Running ConllBuilder over the synthetic corpus to compute stats ...")
    txt_dir = os.path.join(out_dir, "txt")
    builder = ConllBuilder(
        txt_dir=txt_dir,
        entity_types=["REACT", "ACTIVE", "FREQ", "SYS"],
    )
    examples = builder.build_dataset(fa_synth)
    stats = builder.summarise(examples)

    # Per-split stats are useful for the eventual table.
    train_examples = builder.build_dataset(train_synth) if not train_synth.empty else []
    test_examples = builder.build_dataset(test_synth) if not test_synth.empty else []
    stats["train"] = builder.summarise(train_examples)
    stats["test"] = builder.summarise(test_examples)

    out_stats = os.path.join(reports_dir, "synth_dataset_stats.json")
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info("Wrote stats to %s", out_stats)

    print("\nSynthetic corpus summary:")
    print(
        json.dumps(
            {k: v for k, v in stats.items() if k not in ("train", "test")},
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.resume and args.force:
        raise SystemExit("--resume and --force are mutually exclusive")

    annotations_df = _load_annotations(args.annotations)

    if args.stats_only:
        train_df = pd.read_csv(args.train_csv, dtype={"CODIGO": str})
        test_df = pd.read_csv(args.test_csv, dtype={"CODIGO": str})
        _consolidate_and_report(
            out_dir=args.out_dir,
            splits_out_dir=args.splits_out_dir,
            reports_dir=args.reports_dir,
            annotations_df=annotations_df,
            train_df=train_df,
            test_df=test_df,
        )
        return

    codigos = _select_codigos(annotations_df, args.limit, args.codigos)
    logger.info(
        "About to process %d drug(s) (limit=%s, codigos=%s)",
        len(codigos),
        args.limit,
        args.codigos,
    )

    if args.dry_run:
        _do_dry_run(annotations_df, codigos, args.variants)
        return

    if args.dump_prompts:
        _dump_all_prompts(
            annotations_df=annotations_df,
            codigos=codigos,
            out_dir=args.prompts_out_dir,
            n_variants=args.variants,
            n_snippets_per_doc=args.snippets_per_doc,
            target_min_words=args.target_min_words,
            target_max_words=args.target_max_words,
            seed=args.seed,
        )
        return

    llm_client = _load_llm_client(args.llm)
    logger.info("Loaded LLM client '%s' (%s)", args.llm, type(llm_client).__name__)

    generator = SynthNerGenerator(
        llm_client=llm_client,
        out_dir=args.out_dir,
        n_variants=args.variants,
        n_snippets_per_doc=args.snippets_per_doc,
        target_min_words=args.target_min_words,
        target_max_words=args.target_max_words,
        max_unmatched_ratio=args.max_unmatched_ratio,
        max_doc_retries=args.max_doc_retries,
        llm_max_attempts=args.llm_max_attempts,
        seed=args.seed,
        model_name=getattr(llm_client, "model_name", args.llm),
    )

    success = 0
    failed = 0
    skipped = 0
    for i, codigo in enumerate(codigos, 1):
        rows = annotations_df[annotations_df["CODIGO"] == codigo]
        if rows.empty:
            logger.warning(
                "[%d/%d] %s — no rows in annotations table, skipping",
                i,
                len(codigos),
                codigo,
            )
            skipped += 1
            continue
        try:
            docs = generator.generate_for_codigo(codigo, rows, force=args.force)
        except Exception:
            logger.exception(
                "[%d/%d] %s — generation crashed; continuing", i, len(codigos), codigo
            )
            failed += 1
            continue
        if docs:
            success += len(docs)
            for d in docs:
                logger.info(
                    "[%d/%d] %s_v%d ✓ entities=%d discard_ratio=%.2f attempts=%d",
                    i,
                    len(codigos),
                    codigo,
                    d.variante,
                    len(d.entities),
                    d.discard_ratio,
                    d.attempts,
                )
        else:
            failed += 1
            logger.warning(
                "[%d/%d] %s — produced zero variants", i, len(codigos), codigo
            )

    logger.info(
        "Generation finished: success_docs=%d failed_drugs=%d skipped_drugs=%d",
        success,
        failed,
        skipped,
    )

    if args.no_consolidate:
        return

    train_df = pd.read_csv(args.train_csv, dtype={"CODIGO": str})
    test_df = pd.read_csv(args.test_csv, dtype={"CODIGO": str})
    _consolidate_and_report(
        out_dir=args.out_dir,
        splits_out_dir=args.splits_out_dir,
        reports_dir=args.reports_dir,
        annotations_df=annotations_df,
        train_df=train_df,
        test_df=test_df,
    )


if __name__ == "__main__":
    main()
