"""Azure clients used by the public ADR extraction pipelines."""

from __future__ import annotations

import os
from typing import Any, List

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI, OpenAI

AZURE_COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"

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


def _required_setting(explicit_value: str | None, environment_name: str) -> str:
    value = (explicit_value or os.getenv(environment_name, "")).strip()
    if not value:
        raise ValueError(
            f"Azure configuration is missing. Set {environment_name} or pass it explicitly."
        )
    return value


def _token_provider():
    credential = DefaultAzureCredential()
    return get_bearer_token_provider(credential, AZURE_COGNITIVE_SERVICES_SCOPE)


def _default_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "assistant", "content": ASSISTANT_MESSAGE},
        {"role": "user", "content": prompt},
        {"role": "user", "content": FREQUENCY_REVIEW_MESSAGE},
    ]


def _message_content(response: Any) -> str:
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Azure returned an empty response.")
    return str(content)


class Gpt5AzureOpenAI:
    """Azure OpenAI chat-completions client used for synthetic generation."""

    def __init__(
        self,
        model_name: str = "gpt-5.1",
        azure_endpoint: str | None = None,
        api_version: str | None = None,
    ):
        endpoint = _required_setting(azure_endpoint, "AZURE_OPENAI_ENDPOINT")
        self.client = AzureOpenAI(
            api_version=api_version
            or os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=endpoint,
            azure_ad_token_provider=_token_provider(),
        )
        self.model_name = model_name

    def generate_response(self, prompt: str) -> str:
        return self.generate_response_messages(_default_messages(prompt))

    def generate_response_messages(self, messages: List[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=20000,
        )
        return _message_content(response)


class Gpt5ProAzureOpenAI:
    """Azure Responses API client for GPT-5.4 Pro deployments."""

    def __init__(
        self,
        model_name: str = "gpt-5.4-pro",
        azure_endpoint: str | None = None,
        api_version: str | None = None,
    ):
        endpoint = _required_setting(azure_endpoint, "AZURE_OPENAI_ENDPOINT")
        self.client = AzureOpenAI(
            api_version=api_version
            or os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
            azure_endpoint=endpoint,
            azure_ad_token_provider=_token_provider(),
        )
        self.model_name = model_name

    @staticmethod
    def _extract_output_text(response: Any) -> str:
        text = getattr(response, "output_text", None)
        if text:
            return str(text)

        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            for part in getattr(item, "content", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    chunks.append(str(part_text))
        if not chunks:
            raise RuntimeError("Azure returned an empty response.")
        return "".join(chunks)

    def generate_response(self, prompt: str) -> str:
        return self.generate_response_messages(_default_messages(prompt))

    def generate_response_messages(self, messages: List[dict]) -> str:
        response = self.client.responses.create(
            model=self.model_name,
            input=messages,
            max_output_tokens=20000,
        )
        return self._extract_output_text(response)


class DeepSeekAzureOpenAI:
    """OpenAI-compatible Azure AI Foundry client for DeepSeek V4 Pro."""

    def __init__(
        self,
        model_name: str = "DeepSeek-V4-Pro",
        deployment_name: str | None = None,
        endpoint: str | None = None,
    ):
        self.endpoint = _required_setting(endpoint, "AZURE_DEEPSEEK_ENDPOINT")
        self.model_name = deployment_name or model_name
        self.client = OpenAI(
            base_url=self.endpoint,
            api_key=_token_provider(),
        )

    def generate_response(self, prompt: str) -> str:
        return self.generate_response_messages(_default_messages(prompt))

    def generate_response_messages(self, messages: List[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=20000,
        )
        return _message_content(response)