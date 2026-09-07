"""Provider-agnostic LLM client (OpenAI / Gemini / Groq via OpenAI-compatible endpoints).

Kept separate from the RAG pipeline so everything that needs an LLM -- generation
AND the eval judge -- can import it without pulling in the retrieval stack. That
avoids import cycles and keeps the client reusable. Model names are env-configurable
(providers deprecate aggressively); on models that reject temperature=0 we retry
without it and remember.
"""

from __future__ import annotations

import os

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GEMINI_MODEL = "gemini-3-flash"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _chat_completer(client, model: str):
    """Wrap an OpenAI-style client into complete(messages) -> str."""
    supports_temperature = {"ok": True}

    def complete(messages: list[dict]) -> str:
        kwargs = {"model": model, "messages": messages}
        if supports_temperature["ok"]:
            kwargs["temperature"] = 0
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001
            if supports_temperature["ok"] and "temperature" in str(e).lower():
                supports_temperature["ok"] = False
                resp = client.chat.completions.create(model=model, messages=messages)
            else:
                raise
        return resp.choices[0].message.content or ""

    return complete


def _load_openai_llm():
    from openai import OpenAI
    return _chat_completer(OpenAI(), os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL))


def _load_gemini_llm():
    from openai import OpenAI  # Gemini via its OpenAI-compatible endpoint
    client = OpenAI(api_key=os.environ.get("GEMINI_API_KEY"), base_url=GEMINI_BASE_URL)
    return _chat_completer(client, os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))


def _load_groq_llm():
    from openai import OpenAI  # Groq via its OpenAI-compatible endpoint
    client = OpenAI(api_key=os.environ.get("GROQ_API_KEY"), base_url=GROQ_BASE_URL)
    return _chat_completer(client, os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL))


def _load_default_llm():
    loaders = {
        "openai": _load_openai_llm,
        "gemini": _load_gemini_llm,
        "groq": _load_groq_llm,
    }
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider not in loaders:
        raise ValueError(f"unknown LLM_PROVIDER: {provider!r} (use openai, gemini, or groq)")
    return loaders[provider]()