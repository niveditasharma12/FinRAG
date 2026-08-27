"""
Provider-agnostic LLM factory.

The rest of the codebase calls get_llm() instead of hardcoding a vendor class,
so switching providers (Anthropic / Groq / anything OpenAI-compatible) is just
an env-var change:

  LLM_PROVIDER=groq   GROQ_API_KEY=...   [LLM_MODEL=llama-3.3-70b-versatile]
  LLM_PROVIDER=anthropic   ANTHROPIC_API_KEY=...   [LLM_MODEL=claude-sonnet-4-5]

If LLM_PROVIDER is unset we auto-detect: Groq if GROQ_API_KEY is present,
otherwise Anthropic.
"""
import os
import time
import threading
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Ensure provider keys from .env are visible even when this module is used
# without going through src.config first (idempotent).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class RateLimitedLLM:
    """Wrapper around a LangChain chat model that enforces a minimum interval
    between LLM calls to avoid hitting per-minute token rate limits (e.g. on
    the Groq free tier at 8 000 TPM).

    Also retries automatically on 429 / RateLimitError with exponential backoff.
    """

    def __init__(self, llm: Any, min_interval: float = 6.0, max_retries: int = 10):
        self._llm = llm
        self._min_interval = min_interval
        self._max_retries = max_retries
        self._last_call_time = 0.0
        self._lock = threading.Lock()

    def _wait_for_rate_limit(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call_time = time.monotonic()

    def invoke(self, input: Any, **kwargs: Any) -> Any:
        import openai

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            self._wait_for_rate_limit()
            try:
                return self._llm.invoke(input, **kwargs)
            except (openai.RateLimitError, openai.APIStatusError) as exc:
                last_exc = exc
                if isinstance(exc, openai.RateLimitError) and attempt < self._max_retries - 1:
                    delay = min(0.5 * (2 ** attempt), 30.0)
                    print(f"[RateLimitedLLM] 429 hit, retry {attempt+1}/{self._max_retries} in {delay:.1f}s")
                    time.sleep(delay)
                else:
                    raise
        raise last_exc  # type: ignore[misc]

    # Proxy common attributes so code that accesses e.g. llm.model_name still works
    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


_llm_instance: Any | None = None
_is_groq: bool | None = None


def is_groq_provider() -> bool:
    """Return True if the active provider is Groq (free tier has 8 000 TPM)."""
    global _is_groq
    if _is_groq is None:
        provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
        if not provider:
            provider = "groq" if os.environ.get("GROQ_API_KEY") else "anthropic"
        _is_groq = provider == "groq"
    return _is_groq


def get_llm():
    """Return a LangChain chat model for the configured provider.

    Returns a **singleton** so that rate-limit tracking is shared across all
    pipeline components (router, grader, generator, verifier). When running on
    the Groq free tier (8 000 TPM), the returned model is wrapped with
    rate-limit-aware retry logic so the pipeline doesn't crash on 429 errors.
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if not provider:
        provider = "groq" if os.environ.get("GROQ_API_KEY") else "anthropic"

    if provider == "groq":
        # Groq exposes an OpenAI-compatible API, so the standard ChatOpenAI
        # client works — no extra integration package needed.
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=os.environ.get("LLM_MODEL", "openai/gpt-oss-120b"),
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            temperature=0,
            max_retries=0,  # inner retries disabled — RateLimitedLLM handles retries
        )
        # Groq free tier = 8 000 TPM. With retry-on-429 we can lower the
        # interval from 65s to 10s, relying on exponential backoff instead
        # of pre-waiting to stay within the quota.
        _llm_instance = RateLimitedLLM(llm, min_interval=10.0, max_retries=10)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        _llm_instance = ChatAnthropic(
            model=os.environ.get("LLM_MODEL", "claude-sonnet-4-5"),
            temperature=0,
            max_retries=5,
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider!r} (use 'groq' or 'anthropic')")

    return _llm_instance
