import sys
import os
from typing import List
import logging
import pandas as pd
from tqdm import tqdm

from src.llm.google import Gemini3ProLlm

# Add the project root to the Python path
# This is necessary for the notebook to find the 'src' directory
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.llm.azure_foundry import (
    Llama4MaverickCloudLlm,
    Gpt5AzureOpenAI,
    Gpt5ProAzureOpenAI,
    Grok4FastReasoningAzureOpenAI,
    DeepSeekAzureOpenAI
)
from src.llm.ollama_local import OllamaLocalLlm
from src.llm.run_experiment import get_all_prompts, save_llm_output, exits_llm_output
from src.llm.utils_llm import extract_json
from src.llm.prompt_strategies import (
    PromptStrategy,
    ExampleSelector,
    build_messages,
)
from src.llm.file_manupulation import get_saved_drug, get_saved_drug_section_4_8
from src.llm.utils import prepare_p_activo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración de estrategias de prompting
# ---------------------------------------------------------------------------
# Habilitar una (o varias) estrategias de prompting:
enable_zero_shot: bool = True
enable_few_shot: bool = True
enable_chain_of_thought: bool = True

# Mostrar los prompts generados en la consola (para depuración)
SHOW_PROMPTS: bool = False

# Parámetros de Few-shot / Chain-of-Thought
N_FEW_SHOT_EXAMPLES: int = 2
MAX_EXCERPT_CHARS: int = 1500
SEED: int = 42

# ---------------------------------------------------------------------------
# Configuración de modelos
# ---------------------------------------------------------------------------
enable_llama: bool = False
enable_gpt5: bool = False
enable_gpt5_1: bool = False
enable_gpt5_2: bool = False
enable_gpt5_3_chat: bool = False
enable_gpt5_4: bool = True
enable_grok4: bool = False
enable_deep_seek: bool = False
enable_deep_seek_3_2: bool = False
enable_deep_seek_4: bool = True
enable_gemini3pro: bool = False

llama_model = Llama4MaverickCloudLlm()
gpt5_model = Gpt5AzureOpenAI(model_name="gpt-5")
gpt5_1_model = Gpt5AzureOpenAI(model_name="gpt-5.1")
gpt5_2_model = Gpt5AzureOpenAI(model_name="gpt-5.2")
gpt5_3_chat_model = Gpt5AzureOpenAI(model_name="gpt-5.3-chat")
# Los modelos *-pro sólo aceptan el Responses API (/responses); usar el
# cliente especializado para evitar `400 - The requested operation is unsupported.`
gpt5_4_model = Gpt5ProAzureOpenAI(model_name="gpt-5.4-pro")
gpt5_5_proxy = OllamaLocalLlm(model_name="gpt-5.5", base_url="http://127.0.0.1:8081/v1")

grok4_model = Grok4FastReasoningAzureOpenAI()
deep_seek_model = DeepSeekAzureOpenAI(model_name="deep-seek-r1-0528", deployment_name="deep-seek-r1-0528")
deep_seek_model_3_2 = DeepSeekAzureOpenAI(model_name="DeepSeek-V3.2", deployment_name="DeepSeek-V3.2")
deep_seek_model_4 = DeepSeekAzureOpenAI(model_name="DeepSeek-V4-Pro", deployment_name="DeepSeek-V4-Pro")
gemini3_pro_model = Gemini3ProLlm()

# Registro de modelos habilitados: (habilitado, modelo, nombre, sufijo para path)
MODELS = [
    (enable_llama,          llama_model,          "LLAMA",          "Llama-4-Maverick-17B-128E-Instruct-FP8"),
    (enable_gpt5,           gpt5_model,           "GPT-5",          "gpt-5-azure"),
    (enable_gpt5_1,         gpt5_1_model,         "GPT-5.1",        "gpt-5-1-azure"),
    (enable_gpt5_2,         gpt5_2_model,         "GPT-5.2",        "gpt-5-2-azure"),
    (enable_gpt5_3_chat,    gpt5_3_chat_model,    "GPT-5.3-CHAT",   "gpt-5-3-chat-azure"),
    (enable_gpt5_4,         gpt5_4_model,         "GPT-5.4-pro",    "gpt-5-4-pro-azure"),
    (enable_grok4,          grok4_model,           "GROK-4",         "grok-4-fast-reasoning"),
    (enable_deep_seek,      deep_seek_model,       "DEEPSEEK-R1",    "deep-seek-r1-0528"),
    (enable_deep_seek_3_2,  deep_seek_model_3_2,   "DEEPSEEK-V3.2",  "deep-seek-v3-2-speciale"),
    (enable_deep_seek_4,    deep_seek_model_4,     "DEEPSEEK-V4-PRO", "deep-seek-v4-pro"),
    (enable_gemini3pro,     gemini3_pro_model,     "GEMINI-3-PRO",   "gemini-3-pro-preview"),
]

# Registro de estrategias habilitadas
STRATEGIES = [
    (enable_zero_shot,       PromptStrategy.ZERO_SHOT,        "zero-shot"),
    (enable_few_shot,        PromptStrategy.FEW_SHOT,         "few-shot"),
    (enable_chain_of_thought, PromptStrategy.CHAIN_OF_THOUGHT, "chain-of-thought"),
]


def _build_output_path(model_suffix: str, strategy_label: str) -> str:
    """Genera el path de salida: data/resultado/{modelo}/{estrategia}"""
    return f"data/resultado/{model_suffix}/{strategy_label}"


def process_response(response: str, path_resultado: str, prompt_key: str) -> None:
    extracted_data, succeed = extract_json(response, f"{path_resultado}/failed_{prompt_key}.txt")
    if extracted_data:
        save_llm_output(extracted_data, prompt_key, path_resultado)
        print(
            f"Saved output for {prompt_key}. Raw response snippet: {response[:250]}..."
        )


def show_prompt(messages: List[dict], codigo: str, p_activo: str, strategy_label: str, model_name: str) -> None:
    """Muestra el prompt en consola si SHOW_PROMPTS está habilitado."""
    if not SHOW_PROMPTS:
        return
    print(f"\n{'='*80}")
    print(f"CODIGO: {codigo} | P. ACTIVO: {p_activo} | ESTRATEGIA: {strategy_label} | MODELO: {model_name}")
    print(f"{'='*80}")
    for msg in messages:
        print(f"\n--- [{msg['role'].upper()}] ---")
        content = msg["content"]
        print(content)
    print(f"\nTotal mensajes: {len(messages)}")
    print(f"Total caracteres: {sum(len(m['content']) for m in messages)}")


# ---------------------------------------------------------------------------
# Carga de datos y generación de información de medicamentos
# ---------------------------------------------------------------------------
df_test = pd.read_csv("data/splits/test.csv", dtype={"CODIGO": str})
df_train = pd.read_csv("data/splits/train.csv", dtype={"CODIGO": str})
df = pd.concat([df_train, df_test]).reset_index(drop=True)

# Obtener códigos únicos y datos de cada medicamento
unique_codes: List[str] = (
    df["CODIGO"].astype(str).str.strip().unique().tolist()
)

# Pre-cargar datos de cada medicamento (código -> info, texto, principios activos)
drug_data: List[tuple[str, str, str, str]] = []  # (codigo, medicamento, p_activo, texto)
for codigo in unique_codes:
    text_info = get_saved_drug(codigo, logger)
    text_completo = get_saved_drug_section_4_8(codigo, logger)
    if text_info is not None and text_completo is not None:
        text_completo = text_completo.replace("|", "")
        p_activos = prepare_p_activo(text_info["pactivos"])
        for p_activo in p_activos:
            drug_data.append((codigo, text_info["nombre"], p_activo, text_completo))
    else:
        logger.warning(f"Datos no encontrados para código {codigo}, saltando.")

# Inicializar ExampleSelector si se necesitan estrategias few-shot o CoT
example_selector = None
if enable_few_shot or enable_chain_of_thought:
    logger.info("Inicializando ExampleSelector para few-shot/chain-of-thought...")
    example_selector = ExampleSelector()
    logger.info(f"ExampleSelector listo: {len(example_selector.candidates)} candidatos")

# ---------------------------------------------------------------------------
# Generar también los prompts del modo original (legacy) para los modelos
# que usan generate_response() directamente
# ---------------------------------------------------------------------------
# Para la generación de los prompts da igual si se usa el dataset de train o de test.
# El motivo es que lo que se hace en obtener todos los prompts es buscar
# todos los códigos de medicamento únicos y generar un prompt para cada uno.
# Cómo originalmente se hizo una distribución 70/30 uniforme basado en los códigos
# de medicamento, no importa si se usan los códigos del dataset de train o de test
# para generar los prompts, ya que ambos contienen la misma variedad de códigos.
all_prompts: List[tuple[str, str]] = get_all_prompts(df)
total_legacy = len(all_prompts)

# ---------------------------------------------------------------------------
# Ejecución con estrategias de prompting
# ---------------------------------------------------------------------------
total_strategies = len(drug_data)
active_strategies = [(en, st, lb) for en, st, lb in STRATEGIES if en]
active_models = [(en, mdl, nm, sfx) for en, mdl, nm, sfx in MODELS if en]

if active_strategies:
    print(f"\n{'='*80}")
    print("MODO ESTRATEGIAS DE PROMPTING")
    print(f"Estrategias activas: {[lb for _, _, lb in active_strategies]}")
    print(f"Modelos activos: {[nm for _, _, nm, _ in active_models]}")
    print(f"Medicamentos a procesar: {total_strategies}")
    print(f"{'='*80}\n")

    for strategy_enabled, strategy, strategy_label in active_strategies:
        for model_enabled, model_runtime, model_name, model_suffix in active_models:
            path_resultado = _build_output_path(model_suffix, strategy_label)
            os.makedirs(path_resultado, exist_ok=True)

            desc = f"{model_name}/{strategy_label}"
            for idx, (codigo, medicamento, p_activo, texto) in enumerate(
                tqdm(drug_data, desc=desc, unit="drug"), start=1
            ):
                if exits_llm_output(codigo, path_resultado):
                    continue

                messages = build_messages(
                    strategy=strategy,
                    medicamento=medicamento,
                    p_activo=p_activo,
                    texto=texto,
                    example_selector=example_selector,
                    n_examples=N_FEW_SHOT_EXAMPLES,
                    max_excerpt_chars=MAX_EXCERPT_CHARS,
                    seed=SEED,
                    exclude_codigo=codigo,
                )

                show_prompt(messages, codigo, p_activo, strategy_label, model_name)

                print(f"[{model_name}] [{strategy_label}] ({idx}/{total_strategies}) Generating response for key={codigo}")
                try:
                    response = model_runtime.generate_response_messages(messages)
                    process_response(response, path_resultado, codigo)
                except Exception as e:
                    logger.error(f"[{model_name}] [{strategy_label}] Error procesando {codigo} ({p_activo}): {e}")

# ---------------------------------------------------------------------------
# Ejecución legacy (modo original sin estrategia)
# ---------------------------------------------------------------------------
print(f"\nTotal legacy prompts to process: {total_legacy}")
for idx, prompt in enumerate(
    tqdm(all_prompts, desc="Processing prompts (legacy)", unit="prompt"), start=1
):
    prompt_key, prompt_text = prompt

    for model_enabled, model_runtime, model_name, model_suffix in MODELS:
        path_resultado = f"data/resultado/{model_suffix}"
        if model_enabled and not exits_llm_output(prompt_key, path_resultado):
            print(
                f"[{model_name}] ({idx}/{total_legacy}) Generating response for key={prompt_key}"
            )
            response = model_runtime.generate_response(prompt_text)
            process_response(response, path_resultado, prompt_key)
