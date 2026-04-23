"""
AI Analysis — provider abstraction for sending backtest results to an LLM
and receiving a qualitative analysis / opinion / improvement suggestions.

Supported providers:
    - OpenRouter (default — aggregator with access to many models)
    - OpenAI
    - Anthropic
    - Local (OpenAI-compatible API — Ollama / vLLM / LM Studio / etc.)

Configuration via environment variables:
    OPENROUTER_API_KEY    — primary (default provider)
    OPENAI_API_KEY        — for OpenAI
    ANTHROPIC_API_KEY     — for Anthropic
    LOCAL_API_BASE_URL    — for local OpenAI-compatible endpoint
                            (default: http://localhost:11434/v1 for Ollama)

Output: a Markdown file with the AI's analysis.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# urllib is in the Python stdlib — no external `requests` dependency needed.
# (No try/except ImportError: stdlib imports never fail; the previous fallback
# left `urlopen` undefined and would have produced a misleading NameError
# at runtime. Keeping the imports unconditional makes failures obvious.)
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError  # noqa: F401 (surface for callers)


DEFAULT_MODELS = {
    "openrouter": "anthropic/claude-opus-4-7",
    "openai":     "gpt-4o",
    "anthropic":  "claude-opus-4-7",
    "local":      "kimi-k2-0.6b",
}


def _resolve_local_endpoint() -> str:
    """
    Resolve the Local (Ollama / vLLM / LM Studio) endpoint **at call time**
    so `LOCAL_API_BASE_URL` set or changed after module import is honored
    (important for long-lived processes and tests using `monkeypatch.setenv`).
    """
    base_url = os.environ.get("LOCAL_API_BASE_URL", "http://localhost:11434/v1").rstrip("/")
    return base_url + "/chat/completions"


class _DefaultEndpointsDict(dict):
    """
    Dict subclass where "local" resolves lazily from the environment each
    time it is read. Other keys behave as normal dict lookups.
    """
    def __getitem__(self, key):
        if key == "local":
            return _resolve_local_endpoint()
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "local":
            return _resolve_local_endpoint()
        return super().get(key, default)


DEFAULT_ENDPOINTS = _DefaultEndpointsDict({
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "openai":     "https://api.openai.com/v1/chat/completions",
    "anthropic":  "https://api.anthropic.com/v1/messages",
    # "local" is NOT cached here; _resolve_local_endpoint() is called on demand.
})

DEFAULT_ENV_VARS = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai":     "OPENAI_API_KEY",
    "anthropic":  "ANTHROPIC_API_KEY",
    "local":      None,  # usually unused or "ollama" placeholder
}

SYSTEM_PROMPT = """Sei un analista quantitativo di portafogli esperto, con background
in multi-asset allocation, risk management, e backtest reproducibility.

Ti verranno forniti i risultati di un backtest di portafoglio strutturato in 4
umbrellas defensive (gold, put-write, managed futures, options overlay) con factor
tilts (Quality + Momentum) su equity core. L'investitore vuole un parere qualificato.

Fornisci in italiano, in formato Markdown, una risposta strutturata così:

## 1. Diagnosi punti di forza
3-5 bullet points sui punti forti risultati emersi dal backtest.

## 2. Diagnosi punti di debolezza
3-5 bullet points sui punti deboli o caveat emersi.

## 3. Raccomandazioni parametriche
3-5 suggerimenti concreti con numeri specifici (es: "considerare di aumentare DBi
dal 5% al 7% NAV" o "il budget opzioni di 0.30% potrebbe essere aumentato a 0.50%
data la robustezza emersa"). Ciascuna raccomandazione deve citare il dato del
backtest che la motiva.

## 4. Caveat metodologici
2-3 limitazioni dell'analisi che l'investitore dovrebbe tenere a mente.

## 5. Verdetto finale
Una sintesi di 2-3 frasi che riassume il tuo parere complessivo.

Sii diretto, numerico, e basato sui dati forniti. Evita generalità. Se un dato
non è fornito, dillo esplicitamente.
"""


@dataclass
class AiResponse:
    content: str
    provider: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class AiAnalyzer(ABC):
    """Base class for AI provider adapters."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key

    @abstractmethod
    def analyze(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> AiResponse:
        pass


# ============================================================================
# OpenAI-compatible chat completion (also used by OpenRouter and Local)
# ============================================================================

def _openai_compatible_call(
    endpoint: str,
    model: str,
    api_key: Optional[str],
    prompt: str,
    system_prompt: str,
    extra_headers: Optional[dict] = None,
) -> dict:
    """POST to an OpenAI-compatible /chat/completions endpoint."""
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 2000,
    }
    req = Request(endpoint, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    with urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


class OpenAiAnalyzer(AiAnalyzer):
    def analyze(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> AiResponse:
        model = self.model or DEFAULT_MODELS["openai"]
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")
        data = _openai_compatible_call(
            DEFAULT_ENDPOINTS["openai"], model, api_key, prompt, system_prompt,
        )
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return AiResponse(
            content=content, provider="openai", model=model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )


class OpenRouterAnalyzer(AiAnalyzer):
    def analyze(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> AiResponse:
        model = self.model or DEFAULT_MODELS["openrouter"]
        api_key = self.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
        extra_headers = {
            "HTTP-Referer": "https://github.com/padosoft/portfolio-backtest",
            "X-Title": "Four Umbrellas Portfolio Backtest",
        }
        data = _openai_compatible_call(
            DEFAULT_ENDPOINTS["openrouter"], model, api_key, prompt, system_prompt,
            extra_headers=extra_headers,
        )
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return AiResponse(
            content=content, provider="openrouter", model=model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )


class AnthropicAnalyzer(AiAnalyzer):
    def analyze(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> AiResponse:
        model = self.model or DEFAULT_MODELS["anthropic"]
        api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        body = {
            "model": model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000,
            "temperature": 0.3,
        }
        req = Request(
            DEFAULT_ENDPOINTS["anthropic"],
            data=json.dumps(body).encode("utf-8"),
            headers=headers, method="POST",
        )
        with urlopen(req, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = data["content"][0]["text"]
        usage = data.get("usage", {})
        return AiResponse(
            content=content, provider="anthropic", model=model,
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
        )


class LocalAnalyzer(AiAnalyzer):
    """OpenAI-compatible local endpoint (Ollama, vLLM, LM Studio, etc.)."""
    def analyze(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> AiResponse:
        model = self.model or DEFAULT_MODELS["local"]
        endpoint = DEFAULT_ENDPOINTS["local"]
        # Most local endpoints (Ollama/vLLM/LM Studio) run without auth and can
        # reject unexpected Authorization headers. Only send a Bearer token if
        # the user explicitly provided one via `api_key=...`.
        api_key = self.api_key if self.api_key else None
        data = _openai_compatible_call(endpoint, model, api_key, prompt, system_prompt)
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return AiResponse(
            content=content, provider="local", model=model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )


# ============================================================================
# Factory
# ============================================================================

def get_analyzer(provider: str = "openrouter", model: Optional[str] = None) -> AiAnalyzer:
    """
    Factory: return the analyzer instance for the given provider.
    Default provider = OpenRouter (user's choice).
    """
    provider = provider.lower()
    if provider == "openai":
        return OpenAiAnalyzer(model=model)
    if provider == "openrouter":
        return OpenRouterAnalyzer(model=model)
    if provider == "anthropic":
        return AnthropicAnalyzer(model=model)
    if provider == "local":
        return LocalAnalyzer(model=model)
    raise ValueError(f"Unsupported provider: {provider}. Choose from openrouter, openai, anthropic, local.")


# ============================================================================
# Prompt building
# ============================================================================

def build_analysis_prompt(
    results_markdown: str,
    extra_context: Optional[str] = None,
) -> str:
    """
    Compose the user prompt from the REPORT.md content plus optional extras
    (sensitivity CSV summary, rolling window summary, etc.)
    """
    parts = [
        "Ecco il report del backtest:",
        "",
        results_markdown,
    ]
    if extra_context:
        parts.extend(["", "--- Contesto aggiuntivo ---", extra_context])
    parts.extend([
        "",
        "Analizza questi risultati seguendo la struttura richiesta nel system prompt.",
    ])
    return "\n".join(parts)


def save_analysis(response: AiResponse, output_path: Path) -> None:
    """Write the AI response to a Markdown file with metadata header."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# AI Analysis — Four Umbrellas Portfolio Backtest",
        "",
        f"- **Provider**: {response.provider}",
        f"- **Model**: {response.model}",
    ]
    # Use `is not None` rather than truthy: 0 tokens is a legitimate value
    # that should still be recorded in the header (prevents misleading
    # silent omission when a provider returns zero-token usage).
    if response.prompt_tokens is not None:
        header.append(f"- **Prompt tokens**: {response.prompt_tokens}")
    if response.completion_tokens is not None:
        header.append(f"- **Completion tokens**: {response.completion_tokens}")
    header.extend([
        "",
        "> *Nota: questa analisi è generata da un modello AI e deve essere valutata criticamente. "
        "Non costituisce consulenza finanziaria.*",
        "",
        "---",
        "",
    ])
    content = "\n".join(header) + response.content
    output_path.write_text(content, encoding="utf-8")
