# CIMA 4.8 adverse-reaction experiments

Code and reproducibility artifacts for extracting adverse drug reactions from
Spanish CIMA section 4.8 documents. The retained release covers three
experimental families:

- masked-language-model domain adaptation for BERTIN and biomedical RoBERTa;
- real and synthetic CoNLL BIO training and evaluation for `REACT`, `ACTIVE`,
  `FREQ`, and `SYS` entities;
- Azure-hosted GPT-5.4 Pro and DeepSeek V4 Pro extraction with zero-shot, few-shot, and chain-of-thought prompts.

Large model checkpoints and materialized source corpora are intentionally not
stored in this repository. Public datasets and adapted MLM models are hosted on
Hugging Face.

## Repository layout

| Path | Purpose |
|---|---|
| `notebooks/step_1.ipynb` | MLM domain-adaptation training |
| `notebooks/step_2_conll.ipynb` | Real or synthetic CoNLL NER training |
| `run_mlm_test.py` | Intrinsic whole-word-masking evaluation |
| `run_conll_test.py` | Real, synthetic, and cross-corpus NER evaluation |
| `run_prompt_strategies.py` | Azure LLM extraction |
| `run_llm_ner_test.py` | NER-style evaluation of saved LLM outputs |
| `run_synth_dataset.py` | Azure-based synthetic-corpus generation |
| `tools/materialize_datasets.py` | Restore ignored working corpora from public releases |
| `data/splits/` | Versioned real and synthetic train/test splits |
| `data/reports/` | Versioned encoder evaluation reports |
| `data/llm_outputs/` | Reference Azure LLM predictions, without metrics |
| `huggingface/` | Dataset cards, licenses, and export snapshots |

## Installation

Python 3.14.7 is the tested interpreter.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

## Datasets

The release references these public datasets:

- [CIMA-4.8-ADR](https://huggingface.co/datasets/guerrerotook/CIMA-4.8-ADR)
- [CIMA-4.8-ADR-NER](https://huggingface.co/datasets/guerrerotook/CIMA-4.8-ADR-NER)
- [CIMA-4.8-ADR-NER-EXTENDED](https://huggingface.co/datasets/guerrerotook/CIMA-4.8-ADR-NER-EXTENDED)

Materialize only the real texts referenced by the committed splits and the
complete synthetic snapshot:

```bash
python tools/materialize_datasets.py --real-scope splits
```

For MLM training, omit `--real-scope splits` to materialize the complete real
corpus:

```bash
python tools/materialize_datasets.py
```

Generated working copies under `data/medicamentos/` and
`data/medicamentos_synth/` are ignored by Git.

## Training

Run the notebooks in order:

1. `notebooks/step_1.ipynb` trains the two adapted MLM encoders under
   `artifacts/models/`.
2. `notebooks/step_2_conll.ipynb` trains three encoder families across the four single-entity configurations and the combined configuration. Set `USE_SYNTHETIC_DATA` in its configuration cell to select the corpus.

The notebooks contain no saved execution output. All generated checkpoints are
written below the ignored `artifacts/` directory.

## Encoder evaluation

Evaluate the published adapted MLM models and their upstream baselines:

```bash
python run_mlm_test.py
```

Use `--limit-codes 1` for a small smoke run. Reports are written to
`artifacts/reports/mlm/` unless `--reports-dir` is provided.

Evaluate locally trained CoNLL checkpoints:

```bash
python run_conll_test.py --models normal --test-set real
python run_conll_test.py --models synth --test-set both
```

The default checkpoint root is `artifacts/models/`. Predictions and new reports
are also written under `artifacts/` by default. The JSON files under
`data/reports/` are immutable reference results.

## Azure LLM extraction

Authentication uses `DefaultAzureCredential`; no API key is read from source
code. Authenticate with a supported Azure identity, for example `az login`, and
set the endpoint variables required by the selected deployment:

```bash
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"
export AZURE_DEEPSEEK_ENDPOINT="https://<resource>.services.ai.azure.com/openai/v1/"
# Optional override; defaults are defined by the retained Azure clients.
export AZURE_OPENAI_API_VERSION="2025-04-01-preview"
```

Inspect all prompts without constructing an Azure client or writing outputs:

```bash
python run_prompt_strategies.py \
  --model all --strategy all --split test --limit 1 --dry-run
```

Run extraction for the full union of train and test drugs:

```bash
python run_prompt_strategies.py \
  --model all --strategy all --split full
```

New responses are written under `artifacts/llm_outputs/`. The committed
reference outputs contain only the final two Azure models and three strategies:

| Model | zero-shot | few-shot | chain-of-thought |
|---|---:|---:|---:|
| GPT-5.4 Pro | 49 | 48 | 48 |
| DeepSeek V4 Pro | 49 | 49 | 49 |

Evaluate those saved responses without modifying the reference tree:

```bash
python run_llm_ner_test.py --split both
```

Generated LLM reports are intentionally excluded from this release and are
written to `artifacts/reports/llm/`. Semantic similarity defaults to
`PlanTL-GOB-ES/roberta-base-biomedical-es`; pass `--embeddings-model` to use a
compatible local adapted checkpoint instead.

## Synthetic corpus

The exported synthetic corpus is a pre-generated research artifact. Its exact
historical generation client is outside this Azure-only code release, so the
published documents are not exactly regenerable with `run_synth_dataset.py`.
The retained runner supports future Azure GPT generation and preserves the same
validation and export contract.

Build prompts without an Azure request:

```bash
python run_synth_dataset.py --dry-run --limit 1
```

## Validation

```bash
python -m compileall -q src tools *.py
python -m unittest discover -s src/tests -p 'test_*.py' -v
python -m pip check
```

The tests cover Azure runner selection, dry-run isolation, output discovery,
leading-zero medication codes, and CoNLL frequency normalization.

## Reproducibility notes

- Train and test are separated by medication code; leading zeros are preserved.
- Model weights, generated predictions, and newly generated reports belong in
  `artifacts/` and are not versioned.
- The six split CSV files and reference encoder reports are versioned.
- No precomputed LLM evaluation metrics are included.

## Licensing

No license is granted for the Python source code in this repository. Public
visibility alone does not grant permission to copy, modify, or redistribute it.

The datasets under `huggingface/` are separate works and retain their individual
CC BY 4.0 license files and attribution requirements.
