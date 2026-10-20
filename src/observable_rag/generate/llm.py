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

# Per-provider env names, so a caller can ask "which model would run, and do we
# even have a key for it?" without constructing a client.
PROVIDERS = {
    "openai": {"key": "OPENAI_API_KEY", "model_env": "OPENAI_MODEL",
               "default_model": DEFAULT_OPENAI_MODEL, "base_url": None},
    "gemini": {"key": "GEMINI_API_KEY", "model_env": "GEMINI_MODEL",
               "default_model": DEFAULT_GEMINI_MODEL, "base_url": GEMINI_BASE_URL},
    "groq": {"key": "GROQ_API_KEY", "model_env": "GROQ_MODEL",
             "default_model": DEFAULT_GROQ_MODEL, "base_url": GROQ_BASE_URL},
}
DEFAULT_PROVIDER = "gemini"


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
    provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).lower()
    if provider not in loaders:
        raise ValueError(f"unknown LLM_PROVIDER: {provider!r} (use openai, gemini, or groq)")
    return loaders[provider]()


# --- role-aware resolution (generation vs judge) ----------------------------

def has_credentials(provider: str) -> bool:
    return bool(os.getenv(PROVIDERS[provider]["key"]))


def resolve_generation() -> tuple[str, str]:
    """(provider, model) the RAG pipeline will answer with."""
    provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).lower()
    if provider not in PROVIDERS:
        raise ValueError(f"unknown LLM_PROVIDER: {provider!r} (use openai, gemini, or groq)")
    meta = PROVIDERS[provider]
    return provider, os.getenv(meta["model_env"]) or meta["default_model"]


def resolve_judge() -> tuple[str, str, bool]:
    """(provider, model, independent) for the judge.

    A judge that is the same model as the generator is grading its own output,
    which inflates faithfulness in a way no amount of prompting fixes. So unless
    JUDGE_PROVIDER/JUDGE_MODEL say otherwise, prefer any *other* provider we hold
    a key for. When there's no second provider available we fall back to the
    generator's model and report independent=False, so the number is never
    presented as an independent check when it isn't.
    """
    gen_provider, gen_model = resolve_generation()

    provider = (os.getenv("JUDGE_PROVIDER") or "").lower()
    if provider and provider not in PROVIDERS:
        raise ValueError(f"unknown JUDGE_PROVIDER: {provider!r} (use openai, gemini, or groq)")
    if not provider:
        provider = next((p for p in PROVIDERS
                         if p != gen_provider and has_credentials(p)), gen_provider)

    meta = PROVIDERS[provider]
    model = (os.getenv("JUDGE_MODEL")
             or os.getenv(meta["model_env"])
             or meta["default_model"])
    return provider, model, (provider, model) != (gen_provider, gen_model)


def load_judge_llm():
    """Build the judge's completer from resolve_judge()."""
    from openai import OpenAI

    provider, model, _ = resolve_judge()
    meta = PROVIDERS[provider]
    kwargs = {"api_key": os.environ.get(meta["key"])}
    if meta["base_url"]:
        kwargs["base_url"] = meta["base_url"]
    return _chat_completer(OpenAI(**kwargs), model)