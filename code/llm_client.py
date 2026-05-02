"""Multi-provider LLM client with automatic provider detection.

Supports: OpenAI, Anthropic, Google Gemini, Azure OpenAI, and custom OpenAI-compatible endpoints.
Uses only os.environ-driven configuration for Doppler/env compatibility.
Auto-detects provider from available API keys in priority order.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ProviderConfig:
    """Provider-specific configuration."""
    provider: Literal["openai", "anthropic", "google", "azure", "custom"]
    api_key: str
    base_url: str
    model: str
    auth_header_prefix: str


def _detect_provider() -> ProviderConfig | None:
    """Auto-detect LLM provider from environment variables.
    
    Priority order: OpenAI > Anthropic > Google > Azure > Custom endpoint
    Falls back to legacy TRIAGE_API_KEY if present.
    """
    timeout_sec = int(os.environ.get("TRIAGE_LLM_TIMEOUT_SEC", "12"))
    
    # 1. OpenAI (OPENAI_API_KEY)
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai_key:
        return ProviderConfig(
            provider="openai",
            api_key=openai_key,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions").strip(),
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip(),
            auth_header_prefix="Bearer",
        )
    
    # 2. Anthropic (ANTHROPIC_API_KEY)
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key:
        return ProviderConfig(
            provider="anthropic",
            api_key=anthropic_key,
            base_url=os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1/messages").strip(),
            model=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip(),
            auth_header_prefix="Bearer",
        )
    
    # 3. Google Gemini (GOOGLE_API_KEY)
    google_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if google_key:
        return ProviderConfig(
            provider="google",
            api_key=google_key,
            base_url=os.environ.get("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent").strip(),
            model=os.environ.get("GOOGLE_MODEL", "gemini-2.0-flash").strip(),
            auth_header_prefix="Bearer",
        )
    
    # 4. Azure OpenAI (AZURE_OPENAI_API_KEY)
    azure_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    if azure_key:
        return ProviderConfig(
            provider="azure",
            api_key=azure_key,
            base_url=os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip(),
            model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini").strip(),
            auth_header_prefix="Bearer",
        )
    
    # 5. Custom OpenAI-compatible endpoint (TRIAGE_LLM_URL + TRIAGE_API_KEY)
    custom_url = os.environ.get("TRIAGE_LLM_URL", "").strip()
    custom_key = os.environ.get("TRIAGE_API_KEY", "").strip()
    if custom_url and custom_key:
        return ProviderConfig(
            provider="custom",
            api_key=custom_key,
            base_url=custom_url,
            model=os.environ.get("TRIAGE_LLM_MODEL", "gpt-4o-mini").strip(),
            auth_header_prefix="Bearer",
        )
    
    return None


class LLMClient:
    """Multi-provider LLM client."""
    
    def __init__(self) -> None:
        """Initialize with auto-detected provider."""
        self.config = _detect_provider()
        self.timeout_sec = int(os.environ.get("TRIAGE_LLM_TIMEOUT_SEC", "12"))

    @property
    def available(self) -> bool:
        """Check if LLM is configured and available."""
        return self.config is not None

    @property
    def provider_name(self) -> str:
        """Get the name of the active provider."""
        return self.config.provider if self.config else "none"

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Generate JSON response from LLM.
        
        Returns parsed JSON object, or raises RuntimeError on failure.
        """
        if not self.available:
            raise RuntimeError(
                "No LLM provider configured. Set one of: "
                "OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, AZURE_OPENAI_API_KEY, "
                "or TRIAGE_LLM_URL + TRIAGE_API_KEY"
            )

        if self.config.provider == "anthropic":
            return self._generate_anthropic(system_prompt, user_prompt)
        elif self.config.provider == "google":
            return self._generate_google(system_prompt, user_prompt)
        else:
            # OpenAI-compatible: OpenAI, Azure, custom endpoints
            return self._generate_openai_compatible(system_prompt, user_prompt)

    def _generate_openai_compatible(self, system_prompt: str, user_prompt: str) -> dict:
        """Generate JSON using OpenAI-compatible endpoint."""
        body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"{self.config.auth_header_prefix} {self.config.api_key}",
        }

        request = urllib.request.Request(
            url=self.config.base_url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            raise RuntimeError(f"LLM request failed ({self.config.provider}): {exc}") from exc

        try:
            content = payload["choices"][0]["message"]["content"]
            return json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LLM response parsing failed ({self.config.provider}).") from exc

    def _generate_anthropic(self, system_prompt: str, user_prompt: str) -> dict:
        """Generate JSON using Anthropic (Claude) API."""
        body = {
            "model": self.config.model,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
        }

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
        }

        request = urllib.request.Request(
            url=self.config.base_url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            raise RuntimeError(f"LLM request failed (anthropic): {exc}") from exc

        try:
            # Anthropic returns content as a list
            content = payload["content"][0]["text"]
            return json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError("LLM response parsing failed (anthropic).") from exc

    def _generate_google(self, system_prompt: str, user_prompt: str) -> dict:
        """Generate JSON using Google Gemini API (native REST API)."""
        # Google Gemini native API format
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": f"{system_prompt}\n\n{user_prompt}\n\nRespond ONLY with valid JSON."
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 4096,
            }
        }

        # Add API key as query parameter for Google
        url_with_key = f"{self.config.base_url}?key={self.config.api_key}"

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
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            raise RuntimeError(f"LLM request failed (google): {exc}") from exc

        try:
            # Google returns content in a different structure
            content = payload["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError("LLM response parsing failed (google).") from exc
