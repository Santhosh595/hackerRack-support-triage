#!/usr/bin/env python3
"""Quick test script to verify LLM provider detection and Doppler integration."""

import os
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent))

from code.llm_client import LLMClient
from code.config import USE_LLM


def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def main():
    print_header("LLM Configuration Check")

    # Check if any provider is configured
    client = LLMClient()
    
    print("🔍 Environment Status:")
    print(f"  USE_LLM:           {USE_LLM}")
    print(f"  Provider Detected: {client.provider_name}")
    print(f"  LLM Available:     {client.available}")
    
    if not client.available:
        print("\n⚠️  No LLM provider configured. Running in deterministic mode.")
        print("   To enable LLM, set one of these environment variables:")
        print("     - OPENAI_API_KEY (OpenAI)")
        print("     - ANTHROPIC_API_KEY (Anthropic)")
        print("     - GOOGLE_API_KEY (Google)")
        print("     - AZURE_OPENAI_API_KEY (Azure)")
        print("     - TRIAGE_LLM_URL + TRIAGE_API_KEY (Custom)")
        return 1

    print(f"\n✅ LLM Provider Active: {client.provider_name.upper()}")
    
    if client.config:
        print(f"   Model:     {client.config.model}")
        print(f"   Base URL:  {client.config.base_url[:50]}...")
        print(f"   Timeout:   {client.timeout_sec}s")

    # Try a quick test if LLM is available
    print_header("Testing LLM Connectivity")
    
    try:
        print("Sending test request to LLM...")
        result = client.generate_json(
            system_prompt="You are a helpful assistant. Respond with JSON.",
            user_prompt='Return this JSON: {"status": "ok", "message": "Connection successful"}'
        )
        
        print(f"✅ LLM Response Received:")
        print(f"   {result}")
        print(f"\n✅ LLM is working correctly!")
        return 0
        
    except RuntimeError as e:
        print(f"❌ LLM Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Check API key is valid")
        print("  2. Check internet connection")
        print("  3. Check endpoint URL is correct")
        print("  4. Check rate limits haven't been exceeded")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
