"""Shared logging setup and provider-client singletons."""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def setup_logging(name: str) -> logging.Logger:
    """Configure root logging once and return a module-scoped logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(name)


_clients: dict[str, Any] = {}
_clients_lock = threading.Lock()


def _require_env(var: str) -> str:
    val = os.getenv(var)
    if not val:
        sys.exit(f"{var} is not set")
    return val


def openai_client() -> Any:
    with _clients_lock:
        if "openai" not in _clients:
            import openai

            _clients["openai"] = openai.OpenAI(api_key=_require_env("OPENAI_API_KEY"))
        return _clients["openai"]


def xai_client() -> Any:
    with _clients_lock:
        if "xai" not in _clients:
            import openai

            _clients["xai"] = openai.OpenAI(
                api_key=_require_env("XAI_API_KEY"),
                base_url="https://api.x.ai/v1",
            )
        return _clients["xai"]


def openrouter_client() -> Any:
    with _clients_lock:
        if "openrouter" not in _clients:
            import openai

            _clients["openrouter"] = openai.OpenAI(
                api_key=_require_env("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )
        return _clients["openrouter"]


def anthropic_client() -> Any:
    with _clients_lock:
        if "anthropic" not in _clients:
            from anthropic import Anthropic

            _clients["anthropic"] = Anthropic(api_key=_require_env("ANTHROPIC_API_KEY"))
        return _clients["anthropic"]


def google_client() -> Any:
    with _clients_lock:
        if "google" not in _clients:
            from google import genai

            _clients["google"] = genai.Client(api_key=_require_env("GEMINI_API_KEY"))
        return _clients["google"]
