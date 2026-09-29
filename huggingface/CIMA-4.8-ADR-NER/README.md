---
language:
- es
license: cc-by-4.0
pretty_name: CIMA Sección 4.8 NER (Reacciones adversas, principio activo, frecuencia, sistema)
size_categories:
- 1K<n<10K
task_categories:
- token-classification
task_ids:
- named-entity-recognition
tags:
- biomedical
- spanish
- pharmacovigilance
- adverse-reactions
- cima
- aemps
- ner
- conll
- bio
configs:
- config_name: react
  data_files:
  - split: train
    path: data/react/train.parquet
  - split: test
    path: data/react/test.parquet
- config_name: active
  data_files:
  - split: train
    path: data/active/train.parquet
  - split: test
    path: data/active/test.parquet
- config_name: freq
  data_files:
  - split: train
    path: data/freq/train.parquet
  - split: test
    path: data/freq/test.parquet
- config_name: sys
  data_files:
  - split: train
    path: data/sys/train.parquet
  - split: test
    path: data/sys/test.parquet
- config_name: react-active-freq-sys
  default: true
  data_files:
  - split: train
    path: data/react-active-freq-sys/train.parquet
  - split: test
    path: data/react-active-freq-sys/test.parquet
---

# CIMA Sección 4.8 NER

Dataset de reconocimiento de entidades nombradas (NER) en español sobre la
**sección 4.8 ("Reacciones adversas")** de las fichas técnicas publicadas por
la [Agencia Española de Medicamentos y Productos Sanitarios (AEMPS)](https://cima.aemps.es/cima/publico/home.html)
en el Centro de Información Online de Medicamentos (CIMA).

Cada documento es la sección 4.8 completa de un medicamento, tokenizada con
spaCy (`es_core_news_sm`) y etiquetada en formato **CoNLL BIO (IOB2)** con
hasta cuatro tipos de entidad:

| Tag      | Significado                                                     | Origen (columna CSV) |
|----------|-----------------------------------------------------------------|----------------------|
| `REACT`  | Reacción adversa (p. ej. *cefalea*, *náuseas*, *trombocitopenia*) | `REACADV`            |
| `ACTIVE` | Principio activo (p. ej. *SOTALOL HIDROCLORURO*)                | `PACTIVO`            |
| `FREQ`   | Frecuencia de la reacción (p. ej. *MUY FRECUENTE*, *POCO FRECUENTE*) | `FRECUENCIA`         |
| `SYS`    | Sistema u órgano afectado (p. ej. *TRASTORNOS DEL SISTEMA NERVIOSO*) | `SISTEMA`            |

El dataset se publica como parte del Proyecto Fin de Grado (PFG) de la UNED
sobre extracción automática de información de pacto de seguridad de
medicamentos.

## Autoría

Dataset elaborado en el marco del **Trabajo Fin de Grado (PFG)** del Grado
en Ingeniería Informática de la **Universidad Nacional de Educación a
Distancia (UNED)**, curso 2025/2026.

- **Autor**: Luis Miguel Guerrero Guirado ([@guerrerotook](https://huggingface.co/guerrerotook)).
- **Director del PFG**: Salvador Ros Muñoz, UNED.

## Configuraciones

Se publican cinco configuraciones (`config_name`), una por cada combinación
de tags entrenada en el trabajo original. Todas comparten exactamente los
mismos documentos, sentencias y tokens; sólo cambia el **vocabulario de
etiquetas** y, en consecuencia, qué spans están marcados como `B-`/`I-` y
cuáles caen a `O`.

| `config_name`            | Tags incluidos                       | Labels (orden de IDs)                                                                                  |
|--------------------------|--------------------------------------|--------------------------------------------------------------------------------------------------------|
| `react`                  | REACT                                | `O`, `B-REACT`, `I-REACT`                                                                              |
| `active`                 | ACTIVE                               | `O`, `B-ACTIVE`, `I-ACTIVE`                                                                            |
| `freq`                   | FREQ                                 | `O`, `B-FREQ`, `I-FREQ`                                                                                |
| `sys`                    | SYS                                  | `O`, `B-SYS`, `I-SYS`                                                                                  |
| `react-active-freq-sys`  | REACT + ACTIVE + FREQ + SYS (default) | `O`, `B-REACT`, `I-REACT`, `B-ACTIVE`, `I-ACTIVE`, `B-FREQ`, `I-FREQ`, `B-SYS`, `I-SYS`                |

La configuración `react-active-freq-sys` es la **multi-tag** y la marcada
como `default` — es la más útil porque cualquier consumidor puede derivar
las versiones single-tag por filtrado local sin perder información.

## Estadísticas

Mismos documentos y tokens para todas las configuraciones; sólo cambia la
proporción de etiquetas no-`O`:

| Split | Fármacos | Sentencias | Tokens |
|-------|---------:|-----------:|-------:|
| train |       18 |      1 134 | 18 440 |
| test  |        8 |        552 |  9 090 |

Proporción de tokens etiquetados (no-`O`) por configuración:

| Config                     | non-O train | non-O test |
|----------------------------|------------:|-----------:|
| `react`                    |     13,44 % |    13,76 % |
| `active`                   |      0,35 % |     0,51 % |
| `freq`                     |      1,27 % |     2,20 % |
| `sys`                      |      1,12 % |     0,83 % |
| `react-active-freq-sys`    |     16,18 % |    17,28 % |

Distribución detallada por etiqueta para `react-active-freq-sys`:

| Label      | Train | Test |
|------------|------:|-----:|
| `O`        | 15 456 | 7 519 |
| `B-REACT`  |  1 349 |   698 |
| `I-REACT`  |  1 130 |   553 |
| `B-ACTIVE` |     61 |    43 |
| `I-ACTIVE` |      3 |     2 |
| `B-FREQ`   |    155 |   134 |
| `I-FREQ`   |     80 |    66 |
| `B-SYS`    |     59 |    22 |
| `I-SYS`    |    147 |    53 |

Cada `stats/{config}.json` del repositorio contiene el detalle completo.

## Esquema del dataset

Cada ejemplo es una **sentencia** (no un documento completo) con cuatro
campos:

| Campo      | Tipo                                | Descripción                                                       |
|------------|-------------------------------------|-------------------------------------------------------------------|
| `codigo`   | `string`                            | Código nacional CIMA del medicamento al que pertenece la frase    |
| `sent_idx` | `int32`                             | Índice de la sentencia dentro del documento (`0`-indexado)        |
| `tokens`   | `Sequence(string)`                  | Tokens spaCy de la sentencia                                      |
| `ner_tags` | `Sequence(ClassLabel(names=[...]))` | IDs de etiqueta BIO alineados 1:1 con `tokens`                    |

Para recuperar las etiquetas en texto:

```python
from datasets import load_dataset
ds = load_dataset("guerrerotook/cima-section48-ner", "react-active-freq-sys")
label_names = ds["train"].features["ner_tags"].feature.names
print(label_names)
# ['O', 'B-REACT', 'I-REACT', 'B-ACTIVE', 'I-ACTIVE', 'B-FREQ', 'I-FREQ', 'B-SYS', 'I-SYS']

example = ds["train"][5]
for tok, tag_id in zip(example["tokens"], example["ner_tags"]):
    print(f"{tok}\t{label_names[tag_id]}")
```

## Cómo se construyó

Pipeline determinista en cuatro pasos:

1. **Lectura del CSV de menciones**: `train.csv` / `test.csv` con columnas
   `CODIGO, MEDICAMENTO, PACTIVO, REACADV, FRECUENCIA, SISTEMA` (una fila
   por reacción adversa de cada fármaco).
2. **Tokenización word-level con spaCy** (`es_core_news_sm`) del texto bruto
   de la sección 4.8 de cada fármaco, con segmentación por
   `sentencizer` + corte adicional por saltos de línea.
3. **Etiquetado BIO** mediante matching word-boundary insensible a
   acentos/case sobre el texto normalizado (`unicodedata` + lower). Si dos
   menciones solapan, gana la más larga (longest-first ordering). Se aplica
   *promoción IOB2* (`I-X` huérfano → `B-X`).
4. **Serialización** a Parquet (`Sequence(ClassLabel)`) y, en paralelo, a
   `token<TAB>label` BIO plano (formato CoNLL clásico, blank line entre
   sentencias y `# {codigo}` como cabecera de documento).

Todo el código del builder está en [`src/conll_ner_builder.py`](https://github.com/guerrerotook/uned_pfg/blob/main/src/conll_ner_builder.py)
del repositorio original. El export se reproduce con
`python tools/export_hf_dataset.py`.

## Formatos disponibles

El mismo dataset se distribuye dos veces para máxima compatibilidad:

```
data/<config>/train.parquet   ← canónico, recomendado para HF Datasets
data/<config>/test.parquet
conll/<config>/train.conll    ← BIO plano, recomendado para seqeval/flair/spaCy
conll/<config>/test.conll
stats/<config>.json           ← estadísticas por split (label counts, etc.)
```

Los `.conll` siguen exactamente el contrato `token<TAB>label` con líneas en
blanco entre sentencias y `# {codigo}` como cabecera de documento. Son
compatibles directamente con el `parse_conll` de la mayoría de baselines NER
en español.

## Uso

### Opción 1 — Fine-tuning con `transformers` (recomendado)

```python
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from seqeval.metrics import classification_report, f1_score

ds = load_dataset("guerrerotook/cima-section48-ner", "react-active-freq-sys")
label_names = ds["train"].features["ner_tags"].feature.names
num_labels = len(label_names)

model_name = "PlanTL-GOB-ES/roberta-base-biomedical-es"
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

def tokenize_and_align(batch):
    enc = tokenizer(
        batch["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=256,
    )
    labels = []
    for i, tag_seq in enumerate(batch["ner_tags"]):
        word_ids = enc.word_ids(batch_index=i)
        prev = None
        seq = []
        for w in word_ids:
            if w is None:
                seq.append(-100)
            elif w != prev:
                seq.append(tag_seq[w])
            else:
                seq.append(-100)
            prev = w
        labels.append(seq)
    enc["labels"] = labels
    return enc

tok_ds = ds.map(tokenize_and_align, batched=True, remove_columns=ds["train"].column_names)

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label={i: n for i, n in enumerate(label_names)},
    label2id={n: i for i, n in enumerate(label_names)},
)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=2)
    true_labels = [
        [label_names[l] for l in lab if l != -100]
        for lab in p.label_ids
    ]
    pred_labels = [
        [label_names[pr] for pr, l in zip(pre, lab) if l != -100]
        for pre, lab in zip(preds, p.label_ids)
    ]
    return {"f1": f1_score(true_labels, pred_labels)}

args = TrainingArguments(
    output_dir="./cima-ner-ft",
    eval_strategy="epoch",
    per_device_train_batch_size=16,
    num_train_epochs=10,
    learning_rate=2e-5,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok_ds["train"],
    eval_dataset=tok_ds["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
)
trainer.train()
```

### Opción 2 — Descarga directa del CoNLL plano

Útil si trabajas con seqeval, flair, spaCy o un pipeline propio que ya sabe
parsear el formato `token<TAB>label`:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="guerrerotook/cima-section48-ner",
    filename="conll/react-active-freq-sys/train.conll",
    repo_type="dataset",
)

def parse_conll(path):
    sentences, current_tokens, current_tags = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#") or not line:
                if current_tokens:
                    sentences.append((current_tokens, current_tags))
                    current_tokens, current_tags = [], []
                continue
            tok, tag = line.split("\t")
            current_tokens.append(tok)
            current_tags.append(tag)
    if current_tokens:
        sentences.append((current_tokens, current_tags))
    return sentences

sentences = parse_conll(path)
print(f"{len(sentences):,} sentencias")
```

### Opción 3 — Inferencia con un modelo ya entrenado

Si publicas tu modelo NER en HF (por ejemplo
`guerrerotook/cima-roberta-ner-react-active-freq-sys`):

```python
from transformers import pipeline

ner = pipeline(
    "token-classification",
    model="guerrerotook/cima-roberta-ner-react-active-freq-sys",
    aggregation_strategy="simple",
)
text = (
    "El medicamento puede producir cefalea, mareo y, con menos frecuencia, "
    "trombocitopenia. La hipertensión es POCO FRECUENTE."
)
for ent in ner(text):
    print(f"{ent['entity_group']:<8s} {ent['score']:.3f}  {ent['word']}")
```

### Filtrar a una entidad concreta sobre la config multi-tag

```python
from datasets import load_dataset, ClassLabel, Sequence

ds = load_dataset("guerrerotook/cima-section48-ner", "react-active-freq-sys")
src = ds["train"].features["ner_tags"].feature.names  # ['O', 'B-REACT', ...]

KEEP = {"O", "B-REACT", "I-REACT"}
remap = {i: (src.index(t) if t in KEEP else 0) for i, t in enumerate(src)}

def to_react_only(example):
    example["ner_tags"] = [remap[t] if src[t] in KEEP else 0 for t in example["ner_tags"]]
    return example

ds_react = ds.map(to_react_only)
```

## Limitaciones y sesgos

- **Tamaño reducido**: 50 fármacos en total (35 train / 15 test). Apto para
  fine-tuning de modelos pre-entrenados, no para entrenar desde cero.
- **Anotación silver-standard**: las menciones provienen de las tablas
  estructuradas de la sección 4.8 (no de una anotación span-level manual);
  el etiquetado BIO se obtiene por matching exacto insensible a
  acentos/case con word boundaries.
- **Dominio cerrado**: vocabulario biomédico español, registro de ficha
  técnica. El rendimiento fuera de dominio (literatura clínica, foros de
  pacientes, etc.) no está caracterizado.
- **Multi-tag desbalanceado**: REACT representa >80 % de las menciones
  positivas. Las clases ACTIVE/FREQ/SYS son minoritarias y conviene
  evaluarlas con métricas por entidad (no sólo micro-F1).

## Citación

```bibtex
@thesis{guerreroros2026cima,
  author  = {Guerrero Guirado, Luis Miguel y Ros Munoz, Salvador},
  title   = {Detección de reacciones adversas de medicamentos en textos clínicos: Un estudio comparativo de los modelos basados en BERT y modelos LLMs},
  school  = {Universidad Nacional de Educación a Distancia (UNED)},
  year    = {2026},
  type    = {Trabajo Fin de Grado},
}
```

## Licencia y atribución

Dataset publicado bajo **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.es)**.

El texto bruto de la sección 4.8 procede de las fichas técnicas oficiales
publicadas por la [Agencia Española de Medicamentos y Productos Sanitarios
(AEMPS)](https://www.aemps.gob.es/) en su Centro de Información Online de
Medicamentos (CIMA). Al reutilizar este dataset cita tanto el dataset
original como la fuente AEMPS-CIMA.
