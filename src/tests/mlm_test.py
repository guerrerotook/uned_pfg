"""Intrinsic evaluation of MLM checkpoints (no NER / classifier head).

Counterpart of :mod:`src.tests.conll_test` for the base masked-language-model
checkpoints trained with :class:`src.embeddings.EmbeddingModel` (e.g.
``CIMA-BERTIN-4.8`` and ``CIMA-RoBERTa-4.8``).

What it does
------------
1. Reads the held-out CODIGOs from ``data/splits/test.csv`` — exactly the
   files that ``EmbeddingModel.load_data_set`` excluded from training, so
   there is no leakage.
2. For each CODIGO loads ``data/medicamentos/txt/{CODIGO}.txt``, splits it
   into whitespace words, tokenises it with the model's tokeniser, and
   chunks the subtoken stream into windows of ``max_length`` subtokens
   without splitting any word across windows.
3. Builds a deterministic **whole-word masked corpus** with a fixed seed:
   whole words are sampled until ~``mlm_probability`` of the subtokens are
   masked, then the standard 80/10/10 corruption is applied to every
   subtoken of the masked word (``DataCollatorForLanguageModeling`` style).
   The resulting corpus is independent of the model so it can be reused
   across checkpoints that share a tokenizer (e.g. fine-tuned + upstream).
4. Runs a no-grad forward pass and reports, at the masked positions only:
   cross-entropy loss, **pseudo-perplexity** (``exp(mean_loss)``),
   top-1 / top-5 accuracy, and the same metrics broken down by
   *medical-vocabulary* vs *generic* words.
5. Exports a JSON report with aggregate and category-specific metrics.

The class never imports any NER / classifier code: it is fully isolated.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.tests.test_shared import normalize


# A tiny inline list of Spanish stopwords. The medical-vocabulary builder
# also enforces a length>=4 filter so this list only needs to catch the
# short closed-class words that would otherwise leak in.
_SPANISH_STOPWORDS: frozenset[str] = frozenset(
    word.casefold()
    for word in (
        # articles / determiners
        "el", "la", "los", "las", "un", "una", "unos", "unas", "lo",
        # prepositions
        "de", "a", "en", "con", "por", "para", "sin", "sobre", "hasta",
        "desde", "entre", "hacia", "contra", "ante", "bajo", "durante",
        "mediante", "según", "tras",
        # conjunctions
        "y", "o", "e", "u", "pero", "sino", "ni", "que", "porque", "pues",
        "si", "aunque", "mientras", "cuando", "como",
        # pronouns / determiners
        "yo", "tu", "tú", "él", "ella", "nosotros", "vosotros", "ellos",
        "ellas", "este", "esta", "esto", "ese", "esa", "eso", "aquel",
        "aquella", "aquello", "su", "sus", "mi", "mis", "tu", "tus",
        # common verb forms
        "es", "son", "fue", "fueron", "ser", "está", "están", "estar",
        "ha", "han", "haber", "hay", "tiene", "tienen", "tener", "sea",
        "sean", "será", "fueron", "fui", "fuiste",
        # common adverbs
        "no", "sí", "también", "más", "menos", "muy", "mucho", "mucha",
        "muchos", "muchas", "poco", "todo", "todos", "toda", "todas",
        "ya", "aún", "solo", "sólo", "cada", "este", "ese", "otro",
        "otra", "otros", "otras", "mismo", "misma",
    )
)

# Pre-compiled regex used to split raw text into whitespace-separated
# words (offsets are derived from match positions for whole-word masking).
_WORD_RE = re.compile(r"\S+")


class MlmEvaluator:
    """Run intrinsic MLM evaluation on a fill-mask checkpoint.

    The class is deliberately stateless w.r.t. the masked corpus: build it
    once with :meth:`build_masked_corpus` and pass it to as many evaluator
    instances as you like. The driver in ``run_mlm_test.py`` exploits this
    to keep the masking identical across the fine-tuned and the upstream
    checkpoints of the same tokenizer family.

    Parameters
    ----------
    model_id_or_path:
        Local checkpoint directory or a Hugging
        Face hub ID (``bertin-project/bertin-base-gaussian-exp-512seqlen``).
    model_name:
        Slug used in the JSON report. Defaults to the basename of
        ``model_id_or_path``.
    batch_size:
        Per-batch size for the no-grad forward pass.
    add_prefix_space:
        Forced when loading the tokenizer so ``is_split_into_words=True``
        works for both BERTIN-style and roberta-base-bne-style BPEs (the
        latter ships with ``add_prefix_space=False``).
    """

    def __init__(
        self,
        model_id_or_path: str,
        model_name: Optional[str] = None,
        batch_size: int = 32,
        add_prefix_space: bool = True,
    ) -> None:
        self.model_id_or_path = model_id_or_path
        self.model_name = model_name or os.path.basename(
            os.path.normpath(model_id_or_path)
        )
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path, add_prefix_space=add_prefix_space, use_fast=True
        )
        if self.tokenizer.mask_token_id is None:
            raise ValueError(
                f"Tokenizer for {model_id_or_path!r} has no mask token; "
                "this evaluator only supports masked-language models."
            )
        self.device = self._select_device()
        self.model = AutoModelForMaskedLM.from_pretrained(model_id_or_path).to(
            self.device
        )
        self.model.eval()

        print(
            f"MlmEvaluator ready | model={self.model_name} "
            f"| vocab_size={self.tokenizer.vocab_size} | device={self.device}"
        )

    # ------------------------------------------------------------------
    # Setup helpers (mirror ConllNerTest._select_device)
    # ------------------------------------------------------------------

    @staticmethod
    def _select_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if (
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")

    # ------------------------------------------------------------------
    # Test corpus discovery
    # ------------------------------------------------------------------

    @staticmethod
    def load_test_codigos(
        test_csv_path: str = "data/splits/test.csv",
    ) -> List[str]:
        """Return the unique CODIGOs in the held-out test split."""
        df = pd.read_csv(test_csv_path, dtype={"CODIGO": str})
        codigos = (
            df["CODIGO"].astype(str).str.strip().dropna().unique().tolist()
        )
        # Stable order so the masked corpus is reproducible across runs.
        return sorted(codigos)

    # ------------------------------------------------------------------
    # Medical-vocabulary categoriser
    # ------------------------------------------------------------------

    @staticmethod
    def build_medical_vocab(
        full_annotations_csv_path: str = "data/splits/full_annotations.csv",
        min_word_length: int = 4,
    ) -> set[str]:
        """Build the set of casefold word forms considered *medical*.

        Source: the columns ``PACTIVO``, ``REACADV`` and ``SISTEMA`` of
        ``full_annotations.csv`` (the union of every known mention per
        CODIGO produced by :mod:`src.training_data`). Each phrase is
        normalised with :func:`src.tests.test_shared.normalize`, split on
        non-alphabetic characters, lowercased, and filtered by length so
        that closed-class Spanish words never leak in.
        """
        if not os.path.isfile(full_annotations_csv_path):
            print(
                f"Medical-vocabulary CSV not found at "
                f"{full_annotations_csv_path!r}; per-category metrics "
                "will fall back to 'generic' for every masked token."
            )
            return set()

        df = pd.read_csv(full_annotations_csv_path, dtype={"CODIGO": str})
        columns = [c for c in ("PACTIVO", "REACADV", "SISTEMA") if c in df.columns]
        if not columns:
            print(
                "full_annotations.csv has none of the expected columns "
                "(PACTIVO, REACADV, SISTEMA); medical vocab will be empty."
            )
            return set()

        vocab: set[str] = set()
        for column in columns:
            for cell in df[column].dropna().astype(str).unique():
                normalised = normalize(cell)
                # Split on any non-letter so commas, slashes, asterisks
                # and digits never leak into the vocabulary.
                for token in re.split(r"[^a-záéíóúñü]+", normalised):
                    if len(token) < min_word_length:
                        continue
                    if token in _SPANISH_STOPWORDS:
                        continue
                    vocab.add(token)
        print(
            f"Medical vocabulary: {len(vocab):,} unique word forms from "
            f"{', '.join(columns)} of {os.path.basename(full_annotations_csv_path)}."
        )
        return vocab

    # ------------------------------------------------------------------
    # Masked-corpus construction
    # ------------------------------------------------------------------

    @staticmethod
    def _word_to_category(word: str, medical_vocab: set[str]) -> str:
        """Tag ``word`` as ``medical`` or ``generic``."""
        if not medical_vocab:
            return "generic"
        normalised = normalize(word)
        # Strip non-letter characters at the edges (commas, asterisks…).
        stripped = re.sub(r"^[^a-záéíóúñü]+|[^a-záéíóúñü]+$", "", normalised)
        if stripped and stripped in medical_vocab:
            return "medical"
        return "generic"

    @classmethod
    def build_masked_corpus(
        cls,
        tokenizer,
        codigos: Sequence[str],
        txt_dir: str = "data/medicamentos/txt",
        max_length: int = 128,
        mlm_probability: float = 0.18,
        seed: int = 42,
        medical_vocab: Optional[set[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Build a deterministic whole-word masked corpus for ``codigos``.

        Returns ``(items, summary)`` where ``items`` is a list of
        per-window dicts ready for batched inference and ``summary`` is a
        small statistics block for the report header.
        """
        medical_vocab = medical_vocab or set()
        rng = np.random.default_rng(seed)

        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        mask_id = tokenizer.mask_token_id
        if cls_id is None or sep_id is None:
            raise ValueError(
                "Tokenizer must define cls/sep (or bos/eos) tokens; "
                f"got cls={cls_id} sep={sep_id}."
            )
        # The 10% "random token" branch of 80/10/10 corruption: sample
        # uniformly from the regular vocabulary, avoiding special ids.
        special_ids = set(tokenizer.all_special_ids)
        regular_ids = [
            i for i in range(tokenizer.vocab_size) if i not in special_ids
        ]
        # Cap window length to the model's hard limit if smaller.
        if hasattr(tokenizer, "model_max_length"):
            mml = tokenizer.model_max_length
            if isinstance(mml, int) and 0 < mml < max_length:
                max_length = mml
        body_capacity = max_length - 2  # reserve room for CLS / SEP

        items: List[Dict[str, Any]] = []
        missing_files: List[str] = []
        skipped_no_text: List[str] = []
        total_tokens = 0
        total_masked = 0
        category_counts: Dict[str, int] = {"medical": 0, "generic": 0}

        for codigo in codigos:
            txt_path = os.path.join(txt_dir, f"{codigo}.txt")
            if not os.path.isfile(txt_path):
                missing_files.append(codigo)
                continue
            with open(txt_path, "r", encoding="utf-8") as fh:
                text = fh.read()
            if not text.strip():
                skipped_no_text.append(codigo)
                continue

            # 1) Split the raw text into whitespace words and tokenise
            #    them as a pre-tokenised input so word_ids() is reliable
            #    for both BERTIN-BPE and roberta-base-bne tokenisers.
            words = [m.group(0) for m in _WORD_RE.finditer(text)]
            if not words:
                skipped_no_text.append(codigo)
                continue
            enc = tokenizer(
                words,
                is_split_into_words=True,
                add_special_tokens=False,
                return_attention_mask=False,
            )
            input_ids: List[int] = list(enc["input_ids"])
            word_ids: List[Optional[int]] = list(enc.word_ids())
            if not input_ids:
                skipped_no_text.append(codigo)
                continue

            # 2) Walk the subtoken stream and build word-aware windows of
            #    at most ``body_capacity`` subtokens, never splitting a
            #    word across windows.
            windows: List[Tuple[List[int], List[Optional[int]]]] = []
            cur_ids: List[int] = []
            cur_word_ids: List[Optional[int]] = []
            i = 0
            n = len(input_ids)
            while i < n:
                # Identify the next whole word's subtoken span [i, j).
                w = word_ids[i]
                j = i + 1
                while j < n and word_ids[j] == w:
                    j += 1
                word_len = j - i
                if word_len > body_capacity:
                    # Pathological word longer than the window: hard split.
                    while i < j:
                        take = min(body_capacity - len(cur_ids), j - i)
                        if take <= 0:
                            windows.append((cur_ids, cur_word_ids))
                            cur_ids, cur_word_ids = [], []
                            continue
                        cur_ids.extend(input_ids[i : i + take])
                        cur_word_ids.extend(word_ids[i : i + take])
                        i += take
                        if len(cur_ids) >= body_capacity:
                            windows.append((cur_ids, cur_word_ids))
                            cur_ids, cur_word_ids = [], []
                    continue
                if len(cur_ids) + word_len > body_capacity:
                    windows.append((cur_ids, cur_word_ids))
                    cur_ids, cur_word_ids = [], []
                cur_ids.extend(input_ids[i:j])
                cur_word_ids.extend(word_ids[i:j])
                i = j
            if cur_ids:
                windows.append((cur_ids, cur_word_ids))

            # 3) For each window, sample whole words to mask until
            #    ~mlm_probability of body subtokens are masked, then apply
            #    80/10/10 corruption.
            for window_idx, (body_ids, body_word_ids) in enumerate(windows):
                # Collect the word groups present in this window.
                groups: Dict[int, List[int]] = {}
                for pos, wid in enumerate(body_word_ids):
                    if wid is None:
                        continue
                    groups.setdefault(wid, []).append(pos)
                if not groups:
                    continue

                target_n_masked = max(1, int(round(len(body_ids) * mlm_probability)))
                ordered_word_ids = list(groups.keys())
                rng.shuffle(ordered_word_ids)

                masked_word_ids: List[int] = []
                masked_count = 0
                for wid in ordered_word_ids:
                    if masked_count >= target_n_masked:
                        break
                    masked_word_ids.append(wid)
                    masked_count += len(groups[wid])

                # Build the corrupted input + label tensors.
                input_window = list(body_ids)
                labels_window = [-100] * len(body_ids)
                masked_positions: List[int] = []
                masked_categories: List[str] = []
                for wid in masked_word_ids:
                    word_text = words[wid]
                    category = cls._word_to_category(word_text, medical_vocab)
                    # 80/10/10 sampled once per WORD (not per subtoken) so
                    # the whole word stays consistent — matches the spirit
                    # of whole-word masking in the original BERT paper.
                    rand = rng.random()
                    for pos in groups[wid]:
                        labels_window[pos] = body_ids[pos]
                        masked_positions.append(pos)
                        masked_categories.append(category)
                        if rand < 0.8:
                            input_window[pos] = mask_id
                        elif rand < 0.9:
                            input_window[pos] = int(rng.choice(regular_ids))
                        # else: keep original token id (10%)
                    category_counts[category] += len(groups[wid])

                # Prepend CLS, append SEP — and shift masked positions +1.
                final_input = [cls_id] + input_window + [sep_id]
                final_labels = [-100] + labels_window + [-100]
                shifted_positions = [p + 1 for p in masked_positions]

                items.append(
                    {
                        "codigo": codigo,
                        "window_idx": window_idx,
                        "input_ids": final_input,
                        "labels": final_labels,
                        "masked_positions": shifted_positions,
                        "masked_categories": masked_categories,
                    }
                )
                total_tokens += len(final_input)
                total_masked += len(shifted_positions)

        summary = {
            "codigos_evaluated": len(codigos)
            - len(missing_files)
            - len(skipped_no_text),
            "total_codigos_requested": len(codigos),
            "missing_txt_files": missing_files,
            "skipped_empty_files": skipped_no_text,
            "total_windows": len(items),
            "total_tokens": total_tokens,
            "total_masked_tokens": total_masked,
            "category_counts": category_counts,
            "mask_settings": {
                "max_length": max_length,
                "mlm_probability": mlm_probability,
                "seed": seed,
                "whole_word_masking": True,
            },
        }
        return items, summary

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        masked_corpus: Sequence[Dict[str, Any]],
        corpus_summary: Dict[str, Any],
        tokenizer_family: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run inference on ``masked_corpus`` and return an export-ready dict."""
        if not masked_corpus:
            raise ValueError(
                "masked_corpus is empty; build it with build_masked_corpus first."
            )

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0  # safe fallback; attention mask zeros out padding

        # Per-category accumulators.
        cats: Dict[str, Dict[str, float]] = {
            "medical": {"loss_sum": 0.0, "n": 0, "top1": 0, "top5": 0},
            "generic": {"loss_sum": 0.0, "n": 0, "top1": 0, "top5": 0},
        }
        # Per-document loss accumulator (mean loss per CODIGO).
        per_doc_loss: Dict[str, List[float]] = {}

        n_batches = math.ceil(len(masked_corpus) / self.batch_size)
        print(
            f"Running MLM evaluation on {len(masked_corpus):,} windows "
            f"(batch_size={self.batch_size}, n_batches={n_batches})…"
        )

        for batch_idx in range(n_batches):
            batch = masked_corpus[
                batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
            ]
            # Pad to the batch's longest window.
            max_len = max(len(item["input_ids"]) for item in batch)
            input_ids = torch.full(
                (len(batch), max_len), pad_id, dtype=torch.long
            )
            attention_mask = torch.zeros(
                (len(batch), max_len), dtype=torch.long
            )
            for b, item in enumerate(batch):
                seq = item["input_ids"]
                input_ids[b, : len(seq)] = torch.tensor(seq, dtype=torch.long)
                attention_mask[b, : len(seq)] = 1
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            with torch.no_grad():
                logits = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits  # (B, L, V)

            for b, item in enumerate(batch):
                positions = item["masked_positions"]
                if not positions:
                    continue
                pos_tensor = torch.as_tensor(
                    positions, dtype=torch.long, device=self.device
                )
                labels_list = [item["labels"][p] for p in positions]
                labels_tensor = torch.as_tensor(
                    labels_list, dtype=torch.long, device=self.device
                )
                pos_logits = logits[b, pos_tensor].float()  # (M, V)
                ce_per = F.cross_entropy(
                    pos_logits, labels_tensor, reduction="none"
                )
                top5_idx = pos_logits.topk(5, dim=-1).indices  # (M, 5)
                top1_hits = top5_idx[:, 0] == labels_tensor
                top5_hits = (top5_idx == labels_tensor.unsqueeze(-1)).any(
                    dim=-1
                )

                ce_cpu = ce_per.detach().cpu().tolist()
                top1_cpu = top1_hits.detach().cpu().tolist()
                top5_cpu = top5_hits.detach().cpu().tolist()

                doc_losses = per_doc_loss.setdefault(item["codigo"], [])
                for ce_val, t1, t5, cat in zip(
                    ce_cpu, top1_cpu, top5_cpu, item["masked_categories"]
                ):
                    bucket = cats.get(cat) or cats["generic"]
                    bucket["loss_sum"] += float(ce_val)
                    bucket["n"] += 1
                    if t1:
                        bucket["top1"] += 1
                    if t5:
                        bucket["top5"] += 1
                    doc_losses.append(float(ce_val))

        # ------------------------------------------------------------------
        # Aggregate
        # ------------------------------------------------------------------
        overall_n = cats["medical"]["n"] + cats["generic"]["n"]
        overall_loss_sum = (
            cats["medical"]["loss_sum"] + cats["generic"]["loss_sum"]
        )
        overall_top1 = cats["medical"]["top1"] + cats["generic"]["top1"]
        overall_top5 = cats["medical"]["top5"] + cats["generic"]["top5"]

        overall = self._summarise_bucket(
            overall_loss_sum, overall_n, overall_top1, overall_top5
        )
        per_category = {
            cat: self._summarise_bucket(
                stats["loss_sum"], stats["n"], stats["top1"], stats["top5"]
            )
            for cat, stats in cats.items()
        }

        # Per-document mean loss → distribution stats across CODIGOs.
        doc_means = [
            float(np.mean(losses)) for losses in per_doc_loss.values() if losses
        ]
        per_document = self._summarise_per_document(doc_means)

        report: Dict[str, Any] = {
            "model_name": self.model_name,
            "model_id_or_path": self.model_id_or_path,
            "tokenizer_family": tokenizer_family,
            "device": str(self.device),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": corpus_summary,
            "mlm": overall,
            "per_category": per_category,
            "per_document": per_document,
            "notes": (
                "MLM cross-entropy and pseudo-perplexity depend on the "
                "tokenizer; numbers are only directly comparable WITHIN a "
                "tokenizer family (e.g. CIMA-BERTIN-4.8 vs its upstream "
                "bertin-base-gaussian-exp-512seqlen). Comparing BERTIN to "
                "RoBERTa-bne directly is not meaningful — use top-1 / top-5 "
                "accuracy or the relative drop vs the upstream baseline."
            ),
        }
        return self._sanitize_for_json(report)

    @staticmethod
    def _summarise_bucket(
        loss_sum: float, n: int, top1: int, top5: int
    ) -> Dict[str, Any]:
        if n == 0:
            return {
                "n_masked": 0,
                "cross_entropy": None,
                "pseudo_perplexity": None,
                "top1_accuracy": None,
                "top5_accuracy": None,
            }
        ce = loss_sum / n
        return {
            "n_masked": int(n),
            "cross_entropy": round(ce, 6),
            "pseudo_perplexity": round(math.exp(ce), 6),
            "top1_accuracy": round(top1 / n, 6),
            "top5_accuracy": round(top5 / n, 6),
        }

    @staticmethod
    def _summarise_per_document(losses: List[float]) -> Dict[str, Any]:
        if not losses:
            return {
                "n_documents": 0,
                "mean_loss": None,
                "median_loss": None,
                "p25_loss": None,
                "p75_loss": None,
                "min_loss": None,
                "max_loss": None,
            }
        arr = np.asarray(losses, dtype=np.float64)
        return {
            "n_documents": int(arr.size),
            "mean_loss": round(float(arr.mean()), 6),
            "median_loss": round(float(statistics.median(arr.tolist())), 6),
            "p25_loss": round(float(np.percentile(arr, 25)), 6),
            "p75_loss": round(float(np.percentile(arr, 75)), 6),
            "min_loss": round(float(arr.min()), 6),
            "max_loss": round(float(arr.max()), 6),
        }

    # ------------------------------------------------------------------
    # JSON helpers (mirrors ConllNerTest)
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_for_json(obj: Any) -> Any:
        """Recursively normalise NaN/Inf and numpy scalars for JSON output."""
        if isinstance(obj, np.generic):
            obj = obj.item()
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: MlmEvaluator._sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [MlmEvaluator._sanitize_for_json(v) for v in obj]
        return obj

    def export_json(
        self,
        export_data: Dict[str, Any],
        output_path: str = "data/reports/mlm",
    ) -> str:
        """Write evaluation results to ``<output_path>/<model_name>_mlm_results.json``."""
        os.makedirs(output_path, exist_ok=True)
        model_name = export_data.get("model_name", self.model_name)
        file_path = os.path.join(output_path, f"{model_name}_mlm_results.json")
        safe_data = self._sanitize_for_json(export_data)
        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(safe_data, fh, ensure_ascii=False, indent=2)
        print(f"  Exported MLM results to {file_path}")
        return file_path

    @staticmethod
    def export_models_index(
        model_files: Iterable[str], output_path: str = "data/reports/mlm"
    ) -> str:
        """Write a manifest listing every result file in ``output_path``."""
        os.makedirs(output_path, exist_ok=True)
        index = {"models": [os.path.basename(f) for f in model_files]}
        file_path = os.path.join(output_path, "models_index.json")
        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(index, fh, ensure_ascii=False, indent=2)
        print(f"  Exported models index to {file_path}")
        return file_path

    # ------------------------------------------------------------------
    # Console reporting
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(report: Dict[str, Any]) -> None:
        """Print a formatted MLM evaluation report to the console."""
        sep = "=" * 80
        model_name = report.get("model_name", "<unknown>")
        family = report.get("tokenizer_family") or "—"
        summary = report.get("summary", {})
        overall = report.get("mlm", {}) or {}
        per_cat = report.get("per_category", {}) or {}
        per_doc = report.get("per_document", {}) or {}

        print(f"\n{sep}")
        print(f"  MLM Evaluation — {model_name}  (family: {family})")
        print(sep)
        print(f"  CODIGOs evaluated     : {summary.get('codigos_evaluated', 0)}")
        print(f"  Windows               : {summary.get('total_windows', 0)}")
        print(f"  Total tokens          : {summary.get('total_tokens', 0)}")
        print(f"  Masked tokens         : {summary.get('total_masked_tokens', 0)}")
        missing = summary.get("missing_txt_files") or []
        if missing:
            print(
                f"  Missing .txt for  : {len(missing)} CODIGO(s) — skipped."
            )
        print(sep)

        def _fmt(v: Any, fmt: str = "{:.4f}") -> str:
            if v is None:
                return "n/a"
            try:
                return fmt.format(float(v))
            except (TypeError, ValueError):
                return str(v)

        print("  Overall (whole-word masking, fixed seed):")
        print(
            f"    cross_entropy    = {_fmt(overall.get('cross_entropy'))}"
            f"    pseudo_perplexity = {_fmt(overall.get('pseudo_perplexity'))}"
        )
        print(
            f"    top1_accuracy    = {_fmt(overall.get('top1_accuracy'))}"
            f"    top5_accuracy     = {_fmt(overall.get('top5_accuracy'))}"
            f"    n_masked          = {overall.get('n_masked', 0)}"
        )
        print(sep)
        print("  Per-category (medical vs generic vocabulary):")
        for cat in ("medical", "generic"):
            stats = per_cat.get(cat) or {}
            print(
                f"    {cat:<8s} ce={_fmt(stats.get('cross_entropy'))}  "
                f"pppl={_fmt(stats.get('pseudo_perplexity'))}  "
                f"top1={_fmt(stats.get('top1_accuracy'))}  "
                f"top5={_fmt(stats.get('top5_accuracy'))}  "
                f"n={stats.get('n_masked', 0)}"
            )
        print(sep)
        print("  Per-document loss distribution (mean loss per CODIGO):")
        print(
            f"    n_docs={per_doc.get('n_documents', 0)}  "
            f"mean={_fmt(per_doc.get('mean_loss'))}  "
            f"median={_fmt(per_doc.get('median_loss'))}  "
            f"p25={_fmt(per_doc.get('p25_loss'))}  "
            f"p75={_fmt(per_doc.get('p75_loss'))}  "
            f"min={_fmt(per_doc.get('min_loss'))}  "
            f"max={_fmt(per_doc.get('max_loss'))}"
        )
        print(f"{sep}\n")
