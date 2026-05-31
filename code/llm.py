"""
Unified LLM client wrapping Anthropic and HuggingFace backends.

Provides a single interface for the agent module regardless of which
LLM provider is available at runtime.
"""

from __future__ import annotations

import logging
import socket
from urllib.parse import urlparse

from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_BASE_URL,
    HF_LLM_MODEL,
    LLM_MODEL,
)
from embeddings import _get_hf_token

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client supporting Anthropic and HuggingFace inference."""

    def __init__(self, provider: str, client, model: str):
        self.provider = provider
        self.client = client
        self.model = model

    def generate(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 2000,
        temperature: float = 0,
    ) -> str:
        """Send a chat completion request and return the raw text response."""
        if self.provider == "anthropic":
            return self._generate_anthropic(system, messages, max_tokens, temperature)
        elif self.provider == "huggingface":
            return self._generate_huggingface(system, messages, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _generate_anthropic(
        self, system: str, messages: list[dict], max_tokens: int, temperature: float,
    ) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
        )
        return response.content[0].text

    def _generate_huggingface(
        self, system: str, messages: list[dict], max_tokens: int, temperature: float,
    ) -> str:
        hf_messages = [{"role": "system", "content": system}] + messages
        response = self.client.chat_completion(
            messages=hf_messages,
            max_tokens=max_tokens,
            temperature=max(temperature, 0.01),  # HF may reject exactly 0
        )
        return response.choices[0].message.content


def _check_host_reachable(url: str, timeout: float = 3.0) -> bool:
    """Check if the host:port in a URL is reachable via TCP."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except (OSError, socket.timeout):
        return False


def create_llm_client() -> LLMClient:
    """
    Factory: try Anthropic proxy first, fall back to HuggingFace Inference API.

    Returns an LLMClient wrapping whichever backend is available.
    Raises RuntimeError if neither is available.
    """
    # 1. Try Anthropic local proxy
    if ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL:
        if _check_host_reachable(ANTHROPIC_BASE_URL):
            try:
                import anthropic

                client = anthropic.Anthropic(
                    base_url=ANTHROPIC_BASE_URL,
                    api_key=ANTHROPIC_API_KEY,
                )
                logger.info(
                    "Using Anthropic backend: %s / %s", ANTHROPIC_BASE_URL, LLM_MODEL,
                )
                return LLMClient(provider="anthropic", client=client, model=LLM_MODEL)
            except Exception as e:
                logger.warning("Anthropic client creation failed: %s", e)
        else:
            logger.info("Anthropic proxy not reachable at %s", ANTHROPIC_BASE_URL)

    # 2. Fall back to HuggingFace Inference API
    hf_token = _get_hf_token()
    if hf_token:
        try:
            from huggingface_hub import InferenceClient

            client = InferenceClient(model=HF_LLM_MODEL, token=hf_token)
            logger.info("Using HuggingFace backend: %s", HF_LLM_MODEL)
            return LLMClient(provider="huggingface", client=client, model=HF_LLM_MODEL)
        except Exception as e:
            logger.warning("HuggingFace client creation failed: %s", e)

    # 3. Last resort: Anthropic without reachability check (proxy may start later)
    if ANTHROPIC_API_KEY:
        import anthropic

        client = anthropic.Anthropic(
            base_url=ANTHROPIC_BASE_URL,
            api_key=ANTHROPIC_API_KEY,
        )
        logger.warning(
            "Falling back to Anthropic without reachability check at %s",
            ANTHROPIC_BASE_URL,
        )
        return LLMClient(provider="anthropic", client=client, model=LLM_MODEL)

    raise RuntimeError(
        "No LLM backend available. Set ANTHROPIC_API_KEY with a reachable "
        "ANTHROPIC_BASE_URL, or set HG_TOKEN/HF_TOKEN for HuggingFace Inference API."
    )
