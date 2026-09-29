---
language:
- es
license: cc-by-4.0
pretty_name: CIMA Sección 4.8 NER Extended (corpus sintético de reacciones adversas)
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
- synthetic
- llm-generated
- synthetic-data
- data-augmentation
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
- config_name: documents
  data_files:
  - split: train
    path: data/documents/train.parquet
  - split: test
    path: data/documents/test.parquet
---

# CIMA Sección 4.8 NER Extended — corpus sintético

> ⚠️ **Aviso: todos los textos de este dataset son ficticios.** Ninguno de los
> 98 documentos que contiene es una ficha técnica real. Fueron **redactados por
> un modelo de lenguaje** imitando el estilo de la sección 4.8 ("Reacciones
> adversas") de las fichas técnicas de la AEMPS. **No deben usarse como fuente
> de información clínica, farmacológica ni regulatoria sobre ningún
> medicamento.** Lo único que procede del mundo real es el conjunto de
> anotaciones canónicas —las tuplas (reacción adversa, frecuencia, sistema
> orgánico, principio activo) extraídas de fichas técnicas reales de CIMA— que
> se usaron como semilla de la generación.

Este dataset es la **extensión sintética** del corpus NER anotado
[`guerrerotook/CIMA-4.8-ADR-NER`](https://huggingface.co/datasets/guerrerotook/CIMA-4.8-ADR-NER).
Publica el corpus conocido como *Variante B* en la memoria del PFG: por cada
uno de los 49 fármacos del corpus real se generaron **2 documentos sintéticos
independientes**, lo que duplica el volumen de texto anotado (86 599 tokens
frente a 52 202 del corpus real) con un *gold* completo por construcción.

Comparte **exactamente** el mismo esquema, el mismo constructor BIO y los
mismos nombres de configuración que el dataset real, de modo que un consumidor
puede intercambiar el `repo_id` sin tocar una línea de código.

## Autoría

Dataset elaborado en el marco del **Trabajo Fin de Grado (PFG)** del Grado
en Ingeniería Informática de la **Universidad Nacional de Educación a
Distancia (UNED)**, curso 2025/2026.

- **Autor**: Luis Miguel Guerrero Guirado ([@guerrerotook](https://huggingface.co/guerrerotook)).
- **Director del PFG**: Salvador Ros Muñoz, UNED.

## Entidades

| Tag      | Significado                                                          | Origen (columna CSV) |
|----------|----------------------------------------------------------------------|----------------------|
| `REACT`  | Reacción adversa (p. ej. *cefalea*, *náuseas*, *trombocitopenia*)     | `REACADV`            |
| `ACTIVE` | Principio activo (p. ej. *SULFADIAZINA*, *CASPOFUNGINA*)              | `PACTIVO`            |
| `FREQ`   | Frecuencia de la reacción (p. ej. *muy frecuentes*, *poco frecuentes*) | `FRECUENCIA`         |
| `SYS`    | Sistema u órgano afectado (p. ej. *TRASTORNOS DEL SISTEMA NERVIOSO*)  | `SISTEMA`            |

## Configuraciones

Seis configuraciones. Las cinco primeras son **a nivel de sentencia** en
formato BIO (IOB2) y comparten documentos, sentencias y tokens; sólo cambia el
vocabulario de etiquetas. La sexta es **a nivel de documento**.

| `config_name`           | Nivel      | Labels (orden de IDs)                                                                     |
|-------------------------|------------|--------------------------------------------------------------------------------------------|
| `react`                 | sentencia  | `O`, `B-REACT`, `I-REACT`                                                                  |
| `active`                | sentencia  | `O`, `B-ACTIVE`, `I-ACTIVE`                                                                |
| `freq`                  | sentencia  | `O`, `B-FREQ`, `I-FREQ`                                                                    |
| `sys`                   | sentencia  | `O`, `B-SYS`, `I-SYS`                                                                      |
| `react-active-freq-sys` | sentencia  | `O`, `B-REACT`, `I-REACT`, `B-ACTIVE`, `I-ACTIVE`, `B-FREQ`, `I-FREQ`, `B-SYS`, `I-SYS`    |
| `documents`             | documento  | — (texto completo + spans con offsets de carácter)                                          |

`react-active-freq-sys` es la configuración **multi-tag** y la marcada como
`default`: cualquier consumidor puede derivar las versiones single-tag por
filtrado local sin perder información.

## Estadísticas

Mismos documentos y tokens para las cinco configuraciones BIO:

| Split | Documentos | Fármacos de origen | Sentencias | Tokens |
|-------|-----------:|-------------------:|-----------:|-------:|
| train |         70 |                 35 |      4 023 | 60 386 |
| test  |         28 |                 14 |      1 577 | 26 213 |
| total |     **98** |             **49** |  **5 600** | **86 599** |

La partición es **por fármaco de origen**: las dos variantes de un mismo
fármaco caen siempre del mismo lado, y hereda la partición del corpus real
(`random_state=42`), por lo que no hay fuga de información ni entre variantes
ni respecto al dataset real.

Proporción de tokens etiquetados (no-`O`) por configuración:

| Config                  | non-O train | non-O test |
|-------------------------|------------:|-----------:|
| `react`                 |     15,67 % |    20,26 % |
| `active`                |      1,12 % |     1,15 % |
| `freq`                  |      5,25 % |     4,61 % |
| `sys`                   |      7,20 % |     6,74 % |
| `react-active-freq-sys` |     29,18 % |    32,76 % |

Distribución detallada por etiqueta para `react-active-freq-sys`:

| Label      |  Train |   Test |
|------------|-------:|-------:|
| `O`        | 42 768 | 17 625 |
| `B-REACT`  |  4 075 |  2 177 |
| `I-REACT`  |  5 385 |  3 135 |
| `B-ACTIVE` |    453 |    180 |
| `I-ACTIVE` |    222 |    121 |
| `B-FREQ`   |  1 641 |    596 |
| `I-FREQ`   |  1 527 |    612 |
| `B-SYS`    |    999 |    413 |
| `I-SYS`    |  3 316 |  1 354 |

Configuración `documents` (7 450 menciones con offsets de carácter):

| Split | Docs | Entidades | `REACT` | `ACTIVE` | `FREQ` | `SYS` | Palabras/doc |
|-------|-----:|----------:|--------:|---------:|-------:|------:|-------------:|
| train |   70 |     4 913 |   3 611 |      180 |    254 |   868 |          731 |
| test  |   28 |     2 537 |   2 022 |       73 |     95 |   347 |          785 |

Cada `stats/{config}.json` del repositorio contiene el detalle completo, y
`stats/summary.json` el resumen global.

> **Densidad de anotación frente al corpus real.** El corpus sintético etiqueta
> ~30 % de sus tokens frente al ~12 % del real. No es que el texto sea "más
> denso": es que el *gold* sintético es **completo por construcción** (el
> generador valida que cada anotación declarada aparece literalmente en el
> texto), mientras que el *gold* real es *silver-standard* y no inventaría
> todas las menciones presentes. Consúltese la sección de limitaciones.

## Esquema del dataset

### Configuraciones BIO (a nivel de sentencia)

Idéntico a `CIMA-4.8-ADR-NER`:

| Campo      | Tipo                                | Descripción                                                          |
|------------|-------------------------------------|----------------------------------------------------------------------|
| `codigo`   | `string`                            | Identificador del documento sintético: `{código CIMA}_v{variante}`   |
| `sent_idx` | `int32`                             | Índice de la sentencia dentro del documento (`0`-indexado)           |
| `tokens`   | `Sequence(string)`                  | Tokens spaCy de la sentencia                                         |
| `ner_tags` | `Sequence(ClassLabel(names=[...]))` | IDs de etiqueta BIO alineados 1:1 con `tokens`                       |

### Configuración `documents` (a nivel de documento)

| Campo           | Tipo             | Descripción                                                        |
|-----------------|------------------|--------------------------------------------------------------------|
| `codigo`        | `string`         | `{código CIMA}_v{variante}`, p. ej. `"06960_v1"`                   |
| `orig_codigo`   | `string`         | Código nacional CIMA del fármaco real que sirvió de semilla         |
| `variante`      | `int32`          | Número de variante generada (1 o 2)                                 |
| `medicamento`   | `string`         | Nombre comercial del fármaco de origen                              |
| `text`          | `string`         | Documento sintético completo                                        |
| `discard_ratio` | `float32`        | Fracción de entidades declaradas por el LLM que no se pudieron localizar en el texto (y por tanto se descartaron) |
| `entities`      | `list[struct]`   | Menciones con `type`, `canonical`, `surface`, `start`, `end`        |

`start` y `end` son **offsets de carácter** sobre `text` (Python slicing:
`text[start:end]`). El generador localizó cada mención con un matching
insensible a mayúsculas y acentos, así que se cumple:

```
strip_accents(text[start:end]).lower() == strip_accents(surface).lower()
```

pero `text[start:end]` **puede diferir de `surface` en mayúsculas o tildes**
(p. ej. `surface = "TRASTORNOS DE PIEL Y TEJIDOS"` frente al literal del texto
`"Trastornos de piel y tejidos"`). Para extraer el span literal usa siempre el
*slicing*, no el campo `surface`. `canonical` es el término normalizado tal y
como aparece en la ficha técnica real de origen.

## Cómo se construyó

### 1. Generación del texto sintético

Pipeline LLM (*Variante B*) implementado en
[`src/synth_ner_generator.py`](https://github.com/guerrerotook/uned_pfg/blob/main/src/synth_ner_generator.py)
y orquestado por
[`run_synth_dataset.py`](https://github.com/guerrerotook/uned_pfg/blob/main/run_synth_dataset.py):

1. **Entrada estructurada**: por cada `CODIGO` real se agrupan sus anotaciones
   humanas en tuplas canónicas `(REACADV, FRECUENCIA, SISTEMA, PACTIVO)`.
2. **Construcción del prompt**
   ([`src/synth_prompt_templates.py`](https://github.com/guerrerotook/uned_pfg/blob/main/src/synth_prompt_templates.py)):
   `build_long_doc_messages` arma una ficha completa "estilo EMA" con bloque
   tabulado por sistema/frecuencia; `build_snippet_messages` añade *N* casos
   descriptivos en prosa para que las entidades aparezcan también fuera de la
   tabla. Tres variantes de estilo controlan el registro; este corpus usó las
   dos más conservadoras (49 documentos cada una).
3. **Salida JSON estricta**: el modelo devuelve
   `{"text": str, "entities": [{"surface", "type", "canonical"}, …]}`.
4. **Validación**: cada `surface` se localiza dentro de `text` con el mismo
   matcher que el constructor BIO (`ConllBuilder._find_all_spans`, insensible a
   acentos y con *word boundaries*). Las menciones no localizables se descartan;
   si el `discard_ratio` supera 0,30 se rechaza el documento y se reintenta.
   Los 98 documentos publicados fueron aceptados **al primer intento con
   `discard_ratio = 0`**.
5. **Consolidación**: las entidades validadas se vuelcan a
   `full_annotations_synth.csv` con el mismo esquema que el CSV real, y los
   splits se derivan heredando la partición por fármaco del corpus real.

Modelo generador: cliente registrado como `claude-opus-4.7` en el
`MODEL_REGISTRY` del repositorio, servido vía proxy local compatible con la API
de OpenAI. El identificador exacto queda registrado en cada respuesta cruda.

### 2. Exportación a este dataset

1. **Tokenización word-level con spaCy** (`es_core_news_sm`), con segmentación
   por `sentencizer` más corte adicional por saltos de línea.
2. **Etiquetado BIO** por matching *word-boundary* insensible a acentos y
   mayúsculas. Si dos menciones solapan gana la más larga; se aplica promoción
   IOB2 (`I-X` huérfano → `B-X`).
3. **Serialización** a Parquet (`Sequence(ClassLabel)`), a `token<TAB>label` BIO
   plano y a JSON de estadísticas.

El constructor es el mismo
[`src/conll_ner_builder.py`](https://github.com/guerrerotook/uned_pfg/blob/main/src/conll_ner_builder.py)
que produce el dataset real. El export se reproduce con
`python tools/export_hf_dataset_extended.py`.

## Formatos disponibles

```
data/<config>/train.parquet     ← canónico, recomendado para HF Datasets
data/<config>/test.parquet
data/documents/{train,test}.parquet  ← texto completo + spans con offsets
conll/<config>/train.conll      ← BIO plano, recomendado para seqeval/flair/spaCy
conll/<config>/test.conll
txt/{codigo}_v{n}.txt           ← los 98 documentos sintéticos en crudo
annotations/*.csv               ← tabla de menciones y splits (trazabilidad)
stats/<config>.json             ← estadísticas por split
stats/summary.json              ← resumen global
```

Los `.conll` siguen el contrato `token<TAB>label`, con líneas en blanco entre
sentencias y `# {codigo}` como cabecera de documento.

Los CSV de `annotations/` reproducen
`data/splits/{full_annotations_synth,train_synth,test_synth}.csv` del
repositorio original, con una única diferencia: la columna `CODIGO` está
normalizada para que coincida siempre con el nombre de fichero en `txt/`
(cuatro documentos habían perdido un cero inicial al generarse; la
correspondencia queda registrada en `stats/summary.json` bajo
`codigo_normalisation`).

## Uso

### Opción 1 — Fine-tuning con `transformers`

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
from seqeval.metrics import f1_score

ds = load_dataset("guerrerotook/CIMA-4.8-ADR-NER-EXTENDED", "react-active-freq-sys")
label_names = ds["train"].features["ner_tags"].feature.names
# ['O', 'B-REACT', 'I-REACT', 'B-ACTIVE', 'I-ACTIVE', 'B-FREQ', 'I-FREQ', 'B-SYS', 'I-SYS']

model_name = "PlanTL-GOB-ES/roberta-base-biomedical-es"
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

def tokenize_and_align(batch):
    enc = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True, max_length=256)
    labels = []
    for i, tag_seq in enumerate(batch["ner_tags"]):
        word_ids = enc.word_ids(batch_index=i)
        prev, seq = None, []
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
    num_labels=len(label_names),
    id2label={i: n for i, n in enumerate(label_names)},
    label2id={n: i for i, n in enumerate(label_names)},
)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=2)
    true_labels = [[label_names[l] for l in lab if l != -100] for lab in p.label_ids]
    pred_labels = [
        [label_names[pr] for pr, l in zip(pre, lab) if l != -100]
        for pre, lab in zip(preds, p.label_ids)
    ]
    return {"f1": f1_score(true_labels, pred_labels)}

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./cima-ner-synth-ft",
        eval_strategy="epoch",
        per_device_train_batch_size=16,
        num_train_epochs=10,
        learning_rate=2e-5,
    ),
    train_dataset=tok_ds["train"],
    eval_dataset=tok_ds["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
)
trainer.train()
```

> **Importante**: si tu objetivo es medir el rendimiento sobre texto real,
> entrena aquí pero **evalúa contra el split de test de
> `CIMA-4.8-ADR-NER`**. Evaluar sobre el test sintético sobrestima la F1 en
> unos +0,22 puntos (ver limitaciones).

### Opción 2 — Aumento de datos sobre el corpus real

```python
from datasets import concatenate_datasets, load_dataset

real = load_dataset("guerrerotook/CIMA-4.8-ADR-NER", "react-active-freq-sys")
synth = load_dataset("guerrerotook/CIMA-4.8-ADR-NER-EXTENDED", "react-active-freq-sys")

# Los labels y su orden coinciden entre ambos datasets.
assert (
    real["train"].features["ner_tags"].feature.names
    == synth["train"].features["ner_tags"].feature.names
)

train_aug = concatenate_datasets([real["train"], synth["train"]]).shuffle(seed=42)
print(len(real["train"]), "+", len(synth["train"]), "=", len(train_aug))

# Evaluación honesta: sólo texto real.
eval_ds = real["test"]
```

### Opción 3 — Spans con offsets de carácter

Útil para re-tokenizar con otro tokenizador o para tareas de *span extraction*:

```python
from datasets import load_dataset

docs = load_dataset("guerrerotook/CIMA-4.8-ADR-NER-EXTENDED", "documents", split="train")
doc = docs[0]
print(doc["codigo"], doc["medicamento"], len(doc["entities"]), "entidades")

for ent in doc["entities"][:5]:
    literal = doc["text"][ent["start"]:ent["end"]]
    print(f"{ent['type']:<7s} {literal!r}  (canónico: {ent['canonical']!r})")
```

### Opción 4 — Descarga directa del CoNLL plano

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="guerrerotook/CIMA-4.8-ADR-NER-EXTENDED",
    filename="conll/react-active-freq-sys/train.conll",
    repo_type="dataset",
)

def parse_conll(path):
    sentences, tokens, tags = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#") or not line:
                if tokens:
                    sentences.append((tokens, tags))
                    tokens, tags = [], []
                continue
            tok, tag = line.split("\t")
            tokens.append(tok)
            tags.append(tag)
    if tokens:
        sentences.append((tokens, tags))
    return sentences

print(f"{len(parse_conll(path)):,} sentencias")
```

## Limitaciones y sesgos

Este corpus es una **herramienta de investigación sobre generación sintética**,
no un sustituto del corpus real. Sus limitaciones están medidas y son
importantes:

- **Los textos son ficticios.** No describen el perfil de seguridad real de
  ningún medicamento. Un modelo generativo entrenado sobre este corpus
  producirá contenido farmacológicamente inválido.
- **Métricas infladas.** Entrenar *y* evaluar sobre este corpus da una F1
  *strict* muy superior a la del corpus real: micro F1 de 0,97 frente a 0,74 en
  la configuración multi-tag (+0,22 absolutos). El motivo es doble: el *gold*
  sintético es completo por construcción y el lenguaje generado es mucho más
  regular. **Las cifras obtenidas aquí no son extrapolables** al rendimiento
  sobre fichas técnicas reales.
- **Alucinación de entidades.** El LLM introduce menciones que no estaban en el
  fármaco de origen: `FREQ` **20,4 %** (sobre todo *frecuencia no conocida*),
  `ACTIVE` 5,1 %, `REACT` 0,07 %, `SYS` 0 %. Las anotaciones siguen siendo
  correctas *respecto al texto generado*, pero la tupla
  (reacción, frecuencia, sistema) no siempre respeta la del fármaco real.
- **Baja fidelidad por tupla.** La cobertura media de las tuplas
  `(REACT, FREQ, SYS)` del fármaco original es del **8,4 %** (mediana 5 %): el
  generador selecciona un subconjunto, no reproduce la ficha completa.
- **Diversidad léxica desigual.** `ACTIVE` alcanza 3,31 superficies por término
  canónico (94,6 % de los casos con sinónimos) —el objetivo de diseño de esta
  variante—, pero `REACT` se queda en 1,07 (5,6 % con sinónimos): para las
  reacciones, la variación es esencialmente flexión y *casing*.
- **Nomenclatura de frecuencias**: sólo el **89,1 %** de las 5 978 menciones de
  `FREQ` se ajusta a las categorías estándar de la EMA.
- **Distribución de longitudes más estrecha** que la real: 746 palabras de media
  (p25 684, p75 789) frente a 816 (p25 406, p75 851) en el subconjunto real. El
  generador produce documentos mucho más homogéneos.
- **Multi-tag desbalanceado**: `REACT` concentra ~76 % de las menciones.
  Conviene evaluar por entidad y no sólo con micro-F1.
- **Sesgo del modelo generador**: todo el corpus procede de un único LLM y de
  dos variantes de estilo, por lo que hereda sus manierismos de redacción.

El análisis completo que respalda estas cifras está en
[`docs/dataset_sintetico_calidad.md`](https://github.com/guerrerotook/uned_pfg/blob/main/docs/dataset_sintetico_calidad.md)
y la comparativa experimental real/sintético en
[`docs/comparativa_ner_real_vs_sintetico.md`](https://github.com/guerrerotook/uned_pfg/blob/main/docs/comparativa_ner_real_vs_sintetico.md).

## Datasets relacionados

| Dataset | Contenido |
|---|---|
| [`guerrerotook/CIMA-4.8-ADR`](https://huggingface.co/datasets/guerrerotook/CIMA-4.8-ADR) | Corpus completo de la sección 4.8 (≈27 000 fichas reales) para pre-entrenamiento MLM |
| [`guerrerotook/CIMA-4.8-ADR-NER`](https://huggingface.co/datasets/guerrerotook/CIMA-4.8-ADR-NER) | Subconjunto NER anotado sobre **texto real** (49 fármacos, CoNLL BIO, 4 entidades) |
| [`guerrerotook/CIMA-4.8-ADR-NER-EXTENDED`](https://huggingface.co/datasets/guerrerotook/CIMA-4.8-ADR-NER-EXTENDED) | **Este dataset.** Extensión **sintética** del anterior (98 documentos generados por LLM) |

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

El texto es sintético y no procede de ninguna fuente oficial. Las anotaciones
canónicas que sirvieron de semilla derivan de las fichas técnicas publicadas
por la [Agencia Española de Medicamentos y Productos Sanitarios
(AEMPS)](https://www.aemps.gob.es/) en su Centro de Información Online de
Medicamentos (CIMA). Al reutilizar este dataset cita tanto el dataset original
como la fuente AEMPS-CIMA, e indica explícitamente la naturaleza sintética del
texto.
