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
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"  # llama-3.3-70b-versatile was retired
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Per-provider env names, so a caller can ask "which model would run, and do we
# even have a key for it?" without constructing a client.
#
# judge_models is an ordered preference for the eval judge. The judge only has to
# be a DIFFERENT MODEL from the generator, not a different provider, so one
# provider with two models is enough. Chasing a second provider for independence
# is what landed the judge on a free tier capped at 20 requests a day.
PROVIDERS = {
    "openai": {"key": "OPENAI_API_KEY", "model_env": "OPENAI_MODEL",
               "default_model": DEFAULT_OPENAI_MODEL, "base_url": None,
               "judge_models": ["gpt-4o-mini", "gpt-4.1-mini"]},
    "gemini": {"key": "GEMINI_API_KEY", "model_env": "GEMINI_MODEL",
               "default_model": DEFAULT_GEMINI_MODEL, "base_url": GEMINI_BASE_URL,
               "judge_models": [DEFAULT_GEMINI_MODEL]},
    "groq": {"key": "GROQ_API_KEY", "model_env": "GROQ_MODEL",
             "default_model": DEFAULT_GROQ_MODEL, "base_url": GROQ_BASE_URL,
             "judge_models": [DEFAULT_GROQ_MODEL]},
}
DEFAULT_PROVIDER = "openai"


def _chat_completer(client, model: str, recorder=None):
    """Wrap an OpenAI-style client into complete(messages) -> str.

    Token usage is recorded off each response when a recorder is given, so cost
    is measured from what the provider actually billed rather than estimated
    from a local token count.
    """
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
        if recorder is not None:
            from .usage import extract_usage
            recorder.record(extract_usage(resp, model))
        return resp.choices[0].message.content or ""

    return complete


def _load_openai_llm(recorder=None):
    from openai import OpenAI
    return _chat_completer(OpenAI(), os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
                           recorder)


def _load_gemini_llm(recorder=None):
    from openai import OpenAI  # Gemini via its OpenAI-compatible endpoint
    client = OpenAI(api_key=os.environ.get("GEMINI_API_KEY"), base_url=GEMINI_BASE_URL)
    return _chat_completer(client, os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL,
                           recorder)


def _load_groq_llm(recorder=None):
    from openai import OpenAI  # Groq via its OpenAI-compatible endpoint
    client = OpenAI(api_key=os.environ.get("GROQ_API_KEY"), base_url=GROQ_BASE_URL)
    return _chat_completer(client, os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL,
                           recorder)


def _load_default_llm(recorder=None):
    loaders = {
        "openai": _load_openai_llm,
        "gemini": _load_gemini_llm,
        "groq": _load_groq_llm,
    }
    provider = (os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).lower()
    if provider not in loaders:
        raise ValueError(f"unknown LLM_PROVIDER: {provider!r} (use openai, gemini, or groq)")
    return loaders[provider](recorder)


# --- role-aware resolution (generation vs judge) ----------------------------

def has_credentials(provider: str) -> bool:
    return bool(os.getenv(PROVIDERS[provider]["key"]))


def resolve_generation() -> tuple[str, str]:
    """(provider, model) the RAG pipeline will answer with."""
    provider = (os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).lower()
    if provider not in PROVIDERS:
        raise ValueError(f"unknown LLM_PROVIDER: {provider!r} (use openai, gemini, or groq)")
    meta = PROVIDERS[provider]
    return provider, os.getenv(meta["model_env"]) or meta["default_model"]


def resolve_judge() -> tuple[str, str, bool]:
    """(provider, model, independent) for the judge.

    A judge that is the same model as the generator is grading its own output,
    which inflates faithfulness in a way no amount of prompting fixes. What it
    takes to fix that is a different MODEL -- a different provider is one way to
    get one, not the requirement.

    Preference order:

      1. JUDGE_PROVIDER / JUDGE_MODEL, if set.
      2. A second model on the generator's own provider. Same key, same billing,
         nothing extra to configure.
      3. Another provider we hold a key for.
      4. The generator's own model, reported as independent=False.

    (2) comes before (3) deliberately. Reaching for a second provider sounds
    more independent, but in practice it reached for a free tier that ran out
    mid-run, and a judge that silently stops answering is worse than one that
    shares a vendor with the generator.
    """
    gen_provider, gen_model = resolve_generation()

    env_provider = (os.getenv("JUDGE_PROVIDER") or "").lower()
    env_model = os.getenv("JUDGE_MODEL") or None  # CI renders an unset var as ""
    if env_provider and env_provider not in PROVIDERS:
        raise ValueError(
            f"unknown JUDGE_PROVIDER: {env_provider!r} (use openai, gemini, or groq)")

    if env_provider or env_model:
        provider = env_provider or gen_provider
        meta = PROVIDERS[provider]
        model = env_model or os.getenv(meta["model_env"]) or meta["default_model"]
        return provider, model, (provider, model) != (gen_provider, gen_model)

    # (2) a different model on the provider already in use
    for candidate in PROVIDERS[gen_provider].get("judge_models", []):
        if candidate != gen_model:
            return gen_provider, candidate, True

    # (3) any other provider we have a key for
    for provider in PROVIDERS:
        if provider != gen_provider and has_credentials(provider):
            meta = PROVIDERS[provider]
            model = os.getenv(meta["model_env"]) or meta["default_model"]
            if (provider, model) != (gen_provider, gen_model):
                return provider, model, True

    # (4) nothing else available: say so rather than imply independence
    return gen_provider, gen_model, False


def load_judge_llm():
    """Build the judge's completer from resolve_judge()."""
    from openai import OpenAI

    provider, model, _ = resolve_judge()
    meta = PROVIDERS[provider]
    kwargs = {"api_key": os.environ.get(meta["key"])}
    if meta["base_url"]:
        kwargs["base_url"] = meta["base_url"]
    return _chat_completer(OpenAI(**kwargs), model)