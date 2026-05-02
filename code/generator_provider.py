"""Response generator provider pattern with LLM-first, deterministic fallback."""

from __future__ import annotations

import json
import os

from code.config import USE_LLM
from code.generator import generate_grounded_response
from code.llm_client import LLMClient
from code.models import RetrievedDoc, Ticket


class BaseGeneratorProvider:
    def __init__(self) -> None:
        self.last_used_provider = "unknown"

    def generate(self, ticket: Ticket, docs: list[RetrievedDoc]) -> str:
        raise NotImplementedError


class DeterministicGeneratorProvider(BaseGeneratorProvider):
    def __init__(self) -> None:
        super().__init__()

    def generate(self, ticket: Ticket, docs: list[RetrievedDoc]) -> str:
        self.last_used_provider = "deterministic"
        return generate_grounded_response(docs)


class MultiProviderLLMGeneratorProvider(BaseGeneratorProvider):
    """Uses multi-provider LLM client to generate ticket-specific, context-grounded responses.
    
    Supports: OpenAI, Anthropic, Google, Azure, and custom OpenAI-compatible endpoints.
    Auto-detects provider from environment and falls back to deterministic if LLM fails.
    """

    def __init__(self) -> None:
        super().__init__()
        self.det = DeterministicGeneratorProvider()
        self.client = LLMClient()

    @property
    def available(self) -> bool:
        return self.client.available

    def generate(self, ticket: Ticket, docs: list[RetrievedDoc]) -> str:
        if not docs:
            self.last_used_provider = "deterministic"
            return ""
        if not self.available:
            self.last_used_provider = "deterministic"
            return self.det.generate(ticket, docs)

        doc_blob = "\n\n".join(
            f"[{os.path.basename(d.source.split('#')[0])}]\n{d.content}" for d in docs
        )
        system_prompt = (
            "You are a helpful customer support agent. "
            "Answer the user's question using ONLY the provided documentation excerpts. "
            "Be specific and directly address what the user asked. "
            "If the docs don't cover the question adequately, say so briefly. "
            "Keep your response concise (2-4 sentences). "
            "Do not mention file names or 'source:' labels in your reply."
        )
        user_prompt = (
            f"Customer question: {ticket.user_text}\n\n"
            f"Relevant documentation:\n{doc_blob}\n\n"
            "Please provide a helpful, specific response to this customer."
        )

        try:
            # LLMClient.generate_json expects JSON response, but we just want text.
            # So we'll use a modified approach for text generation.
            result = self._generate_text(system_prompt, user_prompt)
            if result:
                self.last_used_provider = f"llm_{self.client.provider_name}"
                return result
        except Exception:
            pass  # Fall through to deterministic

        self.last_used_provider = "deterministic"
        return self.det.generate(ticket, docs)

    def _generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Generate plain text response (not JSON) from LLM."""
        import urllib.error
        import urllib.request

        if self.client.provider == "anthropic":
            return self._generate_text_anthropic(system_prompt, user_prompt)
        elif self.client.provider == "google":
            return self._generate_text_google(system_prompt, user_prompt)
        else:
            # For OpenAI-compatible providers (OpenAI, Azure, custom, etc.)
            return self._generate_text_openai_compatible(system_prompt, user_prompt)

    def _generate_text_openai_compatible(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text using OpenAI-compatible endpoint."""
        import urllib.error
        import urllib.request

        body = {
            "model": self.client.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 300,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"{self.client.config.auth_header_prefix} {self.client.config.api_key}",
        }

        request = urllib.request.Request(
            url=self.client.config.base_url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
                content = payload["choices"][0]["message"]["content"].strip()
                return content
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError):
            return ""

    def _generate_text_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text using Anthropic Claude API."""
        import urllib.error
        import urllib.request

        body = {
            "model": self.client.config.model,
            "max_tokens": 300,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.client.config.api_key,
            "anthropic-version": "2023-06-01",
        }

        request = urllib.request.Request(
            url=self.client.config.base_url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
                text = payload["content"][0]["text"].strip()
                return text
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError):
            return ""

    def _generate_text_google(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text using Google Gemini API (native REST API)."""
        import urllib.error
        import urllib.request

        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": f"{system_prompt}\n\n{user_prompt}"
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 300,
            }
        }

        # Google requires API key as query parameter
        url_with_key = f"{self.client.config.base_url}?key={self.client.config.api_key}"

        headers = {
            "Content-Type": "application/json",
        }

        request = urllib.request.Request(
            url=url_with_key,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
                content = payload["candidates"][0]["content"]["parts"][0]["text"].strip()
                return content
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError):
            return ""

class AnthropicGeneratorProvider(MultiProviderLLMGeneratorProvider):
    """Deprecated: use MultiProviderLLMGeneratorProvider instead."""
    pass


class LLMGeneratorProvider(MultiProviderLLMGeneratorProvider):
    """Deprecated: use MultiProviderLLMGeneratorProvider instead."""
    pass


def get_generator_provider() -> BaseGeneratorProvider:
    if USE_LLM:
        return MultiProviderLLMGeneratorProvider()
    return DeterministicGeneratorProvider()
