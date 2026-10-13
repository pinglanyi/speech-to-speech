"""Resolve dedicated speech-service credentials without reading the LLM key."""
import os


def resolve_speech_api_key(explicit: str | None, env_name: str) -> str | None:
    if explicit is not None:
        return explicit
    # Preserve None for legacy callers; the launcher sets empty values for
    # unauthenticated services to explicitly suppress OPENAI_API_KEY fallback.
    return os.environ[env_name] if env_name in os.environ else None
