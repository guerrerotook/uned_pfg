---
language:
- es
license: cc-by-4.0
pretty_name: CIMA Sección 4.8 — Reacciones Adversas (corpus completo para pre-entrenamiento)
size_categories:
- 10K<n<100K
task_categories:
- fill-mask
- text-generation
- feature-extraction
tags:
- biomedical
- spanish
- pharmacovigilance
- adverse-reactions
- cima
- aemps
- smpc
- domain-adaptation
- masked-language-modeling
- mlm
- continued-pretraining
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
---

# CIMA Sección 4.8 — Reacciones Adversas

Corpus de **texto biomédico regulatorio en español** compuesto por la
**sección 4.8 ("Reacciones adversas")** de la totalidad de las fichas
técnicas publicadas por la
[Agencia Española de Medicamentos y Productos Sanitarios (AEMPS)](https://www.aemps.gob.es/)
en su Centro de Información Online de Medicamentos
([CIMA](https://cima.aemps.es/cima/publico/home.html)).

Este recurso fue construido como base para el **pre-entrenamiento
adaptado al dominio** (*continued pre-training* / *domain-adaptive
pre-training*, DAPT) de modelos *encoder* en español sobre el lenguaje
de farmacovigilancia regulatoria, en el marco de un Proyecto Fin de
Grado (PFG) de la UNED. Es complementario al dataset NER anotado
[`guerrerotook/cima-section48-ner`](https://huggingface.co/datasets/guerrerotook/cima-section48-ner),
que contiene el subconjunto manualmente etiquetado en formato CoNLL BIO
sobre 50 fármacos.

> Hasta donde alcanza el conocimiento de los autores en el momento de su
> publicación, este es el **único recurso público en español** de texto
> regulatorio farmacológico de estas dimensiones disponible para el
> entrenamiento de modelos de lenguaje.

## Autoría

Dataset elaborado en el marco del **Trabajo Fin de Grado (PFG)** del Grado
en Ingeniería Informática de la **Universidad Nacional de Educación a
Distancia (UNED)**, curso 2025/2026.

- **Autor**: Luis Miguel Guerrero Guirado ([@guerrerotook](https://huggingface.co/guerrerotook)).
- **Director del PFG**: Salvador Ros Muñoz, UNED.

## Descripción general

La **ficha técnica** de un medicamento autorizado en España es el
equivalente nacional del *Summary of Product Characteristics* (SmPC)
europeo. Está organizada en secciones numeradas y la **sección 4.8**
recoge las reacciones adversas asociadas al principio activo,
estructuradas típicamente por sistema orgánico afectado y por categoría
de frecuencia (MUY FRECUENTE, FRECUENTE, POCO FRECUENTE, RARA, MUY RARA,
FRECUENCIA NO CONOCIDA).

Este corpus contiene exclusivamente esa sección, extraída del catálogo
completo de la AEMPS (**27 299 fichas técnicas** descargadas,
correspondientes a todos los medicamentos registrados en el sistema,
incluyendo aquellos que ya no se comercializan activamente pero
permanecen en el registro). El texto está en español, libre de marcado
HTML y normalizado a Markdown plano, lo que preserva la estructura
semántica de tablas y enumeraciones mientras elimina el ruido del
marcado de presentación.

## Cómo se construyó

Pipeline determinista en cinco pasos (descrito en detalle en la
sección 3.1 de la memoria del PFG):

1. **Lectura del catálogo oficial de la AEMPS**, que enumera la
   totalidad de los medicamentos registrados en España junto con su
   número de registro nacional (`CODIGO`), identificador único utilizado
   por la API de CIMA.
2. **Descarga de las fichas técnicas vía API REST de CIMA**: un script
   en Python recorre el catálogo completo y recupera cada ficha en
   formato JSON. Volumen total: **≈ 5,74 GB** de JSON crudo en 27 299
   ficheros.
3. **Extracción de la sección 4.8** del HTML embebido en cada JSON
   mediante una aplicación auxiliar en C#.
4. **Limpieza y normalización a Markdown** con la biblioteca
   [`ReverseMarkdown`](https://github.com/mysticmind/reversemarkdown-net),
   que preserva la estructura relacional de tablas y listas a la vez
   que elimina atributos de estilo, referencias HTML y boilerplate
   regulatorio (instrucciones estandarizadas de notificación de
   sospechas de reacciones adversas, etc.). El resultado es un fichero
   de texto plano por medicamento, nombrado con el número de registro.
5. **Consolidación a Parquet**: los ficheros de texto plano se cargan,
   concatenan y almacenan en un único fichero columnar
   (`data/train-00000-of-00001.parquet`), interoperable directamente
   con `datasets.load_dataset(...)` y con herramientas del ecosistema
   Spark/Pandas. El código exacto está en `src/upload_dataset.py` del
   repositorio del PFG.

La conversión HTML → Markdown es una decisión de diseño deliberada: una
fracción significativa de las reacciones adversas aparece en formato
tabular (sistema orgánico × frecuencia × término), y Markdown conserva
esta estructura relacional de forma legible para los modelos de
lenguaje sin arrastrar el marcado de presentación. Esto resultó
relevante tanto para el pre-entrenamiento MLM como para el etiquetado
BIO posterior que se materializa en el dataset NER asociado.

## Esquema del dataset

Una fila por medicamento, un único split `train`:

| Campo      | Tipo     | Descripción                                                                  |
|------------|----------|------------------------------------------------------------------------------|
| `file_id`  | `string` | Número de registro nacional del medicamento en CIMA (p. ej. `"85989"`)       |
| `filename` | `string` | Nombre original del fichero fuente (`{file_id}.txt`)                         |
| `text`     | `string` | Texto de la sección 4.8 normalizado a Markdown plano (tablas, listas, prosa) |

> **Importante**: `file_id` es una **cadena de caracteres**, no un
> entero. Algunos códigos AEMPS contienen ceros a la izquierda (p. ej.
> `"01196001"` para *CANCIDAS 50 MG POLVO PARA CONCENTRADO PARA SOLUCION
> PARA PERFUSION*) que se perderían si se interpretaran numéricamente.
> Al usar pandas directamente, fuerza el tipo con
> `dtype={"file_id": str}`.

El único split (`train`) contiene **todo el corpus**. La división
train/test queda en manos del consumidor: el dataset NER asociado
[`guerrerotook/cima-section48-ner`](https://huggingface.co/datasets/guerrerotook/cima-section48-ner)
publica su propia partición a nivel de fármaco (`random_state=42`) para
evitar fuga de información entre conjuntos.

## Estadísticas

- **Documentos**: ≈ 27 000 (uno por medicamento registrado en CIMA).
- **Volumen original**: ≈ 5,74 GB de JSON crudo antes de la limpieza
  HTML → Markdown.
- **Estructura**: 1 fichero Parquet en `data/train-00000-of-00001.parquet`.

Para cuantificar el corpus que descargas en tu entorno:

```python
from datasets import load_dataset

ds = load_dataset("guerrerotook/CIMA-4.8-Reacciones-Adversas", split="train")
total_chars = sum(len(t) for t in ds["text"])
total_words = sum(len(t.split()) for t in ds["text"])
print(f"Documentos: {len(ds):,}")
print(f"Caracteres totales: {total_chars:,}")
print(f"Palabras (whitespace-split): {total_words:,}")
print(f"Promedio caracteres/documento: {total_chars / len(ds):.0f}")
print(f"Promedio palabras/documento: {total_words / len(ds):.0f}")
```

## Uso

### Opción 1 — Pre-entrenamiento adaptado al dominio (MLM / DAPT)

Caso de uso principal: continuar el pre-entrenamiento de un *encoder*
preexistente (BERT, RoBERTa, BERTIN, etc.) sobre el lenguaje regulatorio
farmacéutico español para mejorar su rendimiento en tareas downstream
(NER, clasificación, recuperación) sobre fichas técnicas o textos
biomédicos relacionados.

```python
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

ds = load_dataset("guerrerotook/CIMA-4.8-Reacciones-Adversas", split="train")
ds = ds.train_test_split(test_size=0.02, seed=42)

model_name = "PlanTL-GOB-ES/roberta-base-biomedical-es"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=512,
        return_special_tokens_mask=True,
    )

tok_ds = ds.map(
    tokenize,
    batched=True,
    remove_columns=["file_id", "filename", "text"],
    num_proc=4,
)

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm_probability=0.18,  # whole-word masking; valor empleado en el PFG
)

args = TrainingArguments(
    output_dir="./cima-roberta-mlm",
    per_device_train_batch_size=16,
    learning_rate=5e-5,
    num_train_epochs=3,
    eval_strategy="steps",
    eval_steps=2_000,
    save_strategy="steps",
    save_steps=2_000,
    bf16=True,           # o fp16 según hardware
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok_ds["train"],
    eval_dataset=tok_ds["test"],
    data_collator=collator,
    tokenizer=tokenizer,
)
trainer.train()
```

Esta receta reproduce el enfoque utilizado en el PFG para entrenar los
encoders adaptados al dominio (`CIMA-BERTIN-4.8` y `CIMA-RoBERTa-4.8`),
que mejoraron sustancialmente la pseudo-perplejidad y el *top-1/top-5
accuracy* sobre un conjunto *held-out* de fichas técnicas frente a sus
respectivos *upstream*:

| *Checkpoint* | PPL ↓ | Top-1 ↑ | Top-5 ↑ |
|---|---:|---:|---:|
| `bertin-project/bertin-base-gaussian-exp-512seqlen` (upstream) | 59,71 | 44,45 % | 56,99 % |
| **CIMA-BERTIN-4.8** (adaptado a este corpus) | **11,09** | **60,82 %** | **74,12 %** |
| `PlanTL-GOB-ES/roberta-base-biomedical-es` (upstream) | 14,28 | 58,46 % | 73,40 % |
| **CIMA-RoBERTa-4.8** (adaptado a este corpus) | **7,46** | **69,09 %** | **80,43 %** |

### Opción 2 — Recuperación o búsqueda sobre fichas técnicas

Construir un índice (BM25, FAISS, etc.) para responder preguntas sobre
reacciones adversas, frecuencias o sistemas orgánicos a partir del
catálogo completo:

```python
from datasets import load_dataset

ds = load_dataset("guerrerotook/CIMA-4.8-Reacciones-Adversas", split="train")

# Búsqueda léxica simple sobre una reacción adversa concreta
hits = ds.filter(lambda ex: "trombocitopenia" in ex["text"].lower())
print(f"{len(hits)} fichas técnicas mencionan 'trombocitopenia'")
```

Para búsqueda semántica, embebe los documentos con un encoder ya
adaptado al dominio:

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("PlanTL-GOB-ES/roberta-base-biomedical-es")
embs = model.encode(ds["text"], show_progress_bar=True, batch_size=32)
index = faiss.IndexFlatIP(embs.shape[1])
index.add(np.asarray(embs, dtype="float32"))
```

### Opción 3 — Contexto para generación con LLMs

Usar el texto completo de la sección 4.8 de un fármaco como contexto en
un *prompt* a un LLM (generación de informes, resúmenes, extracción de
entidades zero-shot, etc.):

```python
from datasets import load_dataset

ds = load_dataset("guerrerotook/CIMA-4.8-Reacciones-Adversas", split="train")
doc = ds.filter(lambda ex: ex["file_id"] == "85989")[0]

prompt = (
    "A partir del siguiente texto de la sección 4.8 de la ficha técnica, "
    "lista las reacciones adversas más frecuentes en formato JSON con los "
    "campos {reaccion, frecuencia, sistema}:\n\n"
    f"{doc['text']}"
)
# … invocar tu LLM favorito con `prompt` …
```

### Opción 4 — Re-generar los ficheros `.txt` localmente

Si tu pipeline trabaja con un fichero por medicamento (p. ej., el
constructor CoNLL BIO de [`guerrerotook/cima-section48-ner`](https://huggingface.co/datasets/guerrerotook/cima-section48-ner)
o cualquier script que espere `data/medicamentos/txt/{CODIGO}.txt`):

```python
import os
from datasets import load_dataset

ds = load_dataset("guerrerotook/CIMA-4.8-Reacciones-Adversas", split="train")
os.makedirs("data/medicamentos/txt", exist_ok=True)
for ex in ds:
    with open(f"data/medicamentos/txt/{ex['file_id']}.txt", "w", encoding="utf-8") as f:
        f.write(ex["text"])
```

## Relación con otros recursos del PFG

| Recurso                                                                                                                                  | Propósito                                                                       |
|------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [`guerrerotook/CIMA-4.8-Reacciones-Adversas`](https://huggingface.co/datasets/guerrerotook/CIMA-4.8-Reacciones-Adversas)                 | **Este dataset.** Corpus completo de la sección 4.8 para pre-entrenamiento MLM  |
| [`guerrerotook/cima-section48-ner`](https://huggingface.co/datasets/guerrerotook/cima-section48-ner)                                           | Subconjunto NER anotado (CoNLL BIO, 50 fármacos, 4 entidades)                   |
| [`guerrerotook/CIMA-BERTIN-4.8`](https://huggingface.co/guerrerotook/CIMA-BERTIN-4.8)                                                          | Encoder BERTIN tras DAPT sobre este corpus                                       |
| [`guerrerotook/CIMA-RoBERTa-4.8`](https://huggingface.co/guerrerotook/CIMA-RoBERTa-4.8)                                                        | Encoder RoBERTa biomédico tras DAPT sobre este corpus                            |
| [`guerrerotook/CIMA-BERTIN-4.8-NER`](https://huggingface.co/guerrerotook/CIMA-BERTIN-4.8-NER)                                            | BERTIN adaptado + fine-tuning NER sobre `cima-section48-ner`                     |
| [`guerrerotook/CIMA-RoBERTa-4.8-NER`](https://huggingface.co/guerrerotook/CIMA-RoBERTa-4.8-NER)                                          | RoBERTa adaptado + fine-tuning NER sobre `cima-section48-ner`                    |

## Limitaciones y sesgos

- **Cobertura temporal**: los documentos reflejan el estado del
  catálogo AEMPS en la fecha de descarga, e incluyen medicamentos ya no
  comercializados pero aún registrados. No hay garantía de actualización
  con nuevas autorizaciones posteriores a esa fecha.
- **Sesgo regulatorio europeo**: el formato y la terminología siguen el
  estándar EMA-AEMPS (SmPC §4.8). Modelos entrenados sobre este corpus
  pueden generalizar peor a textos clínicos no regulatorios (notas de
  paciente, literatura científica, foros, etc.).
- **Heterogeneidad de calidad del texto fuente**: la calidad del HTML
  original varía entre laboratorios y épocas; la limpieza es robusta
  pero no perfecta. Algunos documentos pueden contener artefactos
  residuales de tablas complejas o de fichas antiguas escaneadas.
- **No es una fuente clínica primaria**: el dataset es una
  transformación textual de las fichas oficiales; cualquier uso clínico
  debe consultar la ficha técnica vigente en CIMA.
- **Idioma único**: español peninsular regulatorio. No incluye variantes
  de español latinoamericano ni traducciones a otros idiomas oficiales.
- **Sin anotación**: este corpus se distribuye exclusivamente como
  texto plano para tareas no supervisadas. Para reconocimiento de
  entidades nombradas (REACT, ACTIVE, FREQ, SYS) usa el dataset
  complementario [`guerrerotook/cima-section48-ner`](https://huggingface.co/datasets/guerrerotook/cima-section48-ner).

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

El texto bruto de la sección 4.8 procede de las fichas técnicas
oficiales publicadas por la
[Agencia Española de Medicamentos y Productos Sanitarios (AEMPS)](https://www.aemps.gob.es/)
en su Centro de Información Online de Medicamentos
([CIMA](https://cima.aemps.es/cima/publico/home.html)). Al reutilizar
este dataset cita tanto el dataset original como la fuente AEMPS-CIMA.
