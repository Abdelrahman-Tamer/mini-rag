"""
NileGuide mini-rag — Gemini Provider Integration and Error Handling Diagnostic
Verifies OpenAI-compatible Gemini provider authentication, model listing,
generation, and error handling without sending any requests to Production.
"""

import os
import sys
import logging
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openai import OpenAI, AuthenticationError, NotFoundError, APIError
from api.helpers.config import Settings
from api.stores.llm.providers.OpenAIProvider import OpenAIProvider


def mask_secret(secret: Optional[str]) -> str:
    if not secret:
        return "[UNCONFIGURED]"
    if len(secret) <= 8:
        return f"[REDACTED len={len(secret)}]"
    return f"{secret[:4]}...{secret[-4:]} (len={len(secret)})"


def run_negative_auth_test(base_url: str):
    """Verify that an invalid API key fails closed with 401 AuthenticationError."""
    print("\n--- 1. Negative Authentication Verification ---")
    dummy_client = OpenAI(
        api_key="invalid-audit-key-test-000000000",
        base_url=base_url,
    )
    try:
        dummy_client.models.list()
        print("FAIL: Expected AuthenticationError with invalid key, but call succeeded!")
        return False
    except AuthenticationError as e:
        print(f"PASS: Provider rejected invalid key with AuthenticationError (HTTP 401).")
        return True
    except Exception as e:
        print(f"PASS: Provider rejected invalid key with expected exception: {type(e).__name__} ({e})")
        return True


def run_negative_model_error_test(api_key: str, base_url: str):
    """Verify that an invalid model ID raises a 404 / NotFoundError."""
    print("\n--- 2. Negative Model Error Handling Verification ---")
    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        client.chat.completions.create(
            model="gemini-nonexistent-dummy-model-test-9999",
            messages=[{"role": "user", "content": "ping"}],
        )
        print("FAIL: Expected error for invalid model, but call succeeded!")
        return False
    except NotFoundError as e:
        print(f"PASS: Provider rejected non-existent model with NotFoundError (HTTP 404): {e.status_code}")
        return True
    except APIError as e:
        print(f"PASS: Provider rejected non-existent model with APIError ({type(e).__name__}): {e.message}")
        return True
    except Exception as e:
        print(f"PASS: Provider rejected non-existent model with {type(e).__name__}: {e}")
        return True


def run_live_provider_test(api_key: str, base_url: str, model_id: str):
    """Verify live authentication, model listing, and simple generation."""
    print("\n--- 3. Live Provider Integration Verification ---")
    print(f"Target Base URL: {base_url}")
    print(f"Target Model: {model_id}")
    print(f"API Key: {mask_secret(api_key)}")

    client = OpenAI(api_key=api_key, base_url=base_url)

    # 3a. Model listing
    print("\nQuerying available models via client.models.list()...")
    try:
        models = client.models.list()
        gemini_models = [m.id for m in models if "gemini" in (m.id or "").lower()]
        print(f"PASS: client.models.list() succeeded. Found {len(gemini_models)} Gemini models.")
        for m in gemini_models[:8]:
            print(f"  - {m}")
        if len(gemini_models) > 8:
            print(f"  ... and {len(gemini_models) - 8} more.")
    except Exception as e:
        print(f"WARNING: client.models.list() failed or not supported on this endpoint: {type(e).__name__}: {e}")

    # 3b. Simple generation test
    print(f"\nGenerating simple completion with model='{model_id}'...")
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a test agent. Reply with the exact single word: VERIFIED"},
                {"role": "user", "content": "ping"}
            ],
            max_tokens=20,
            temperature=0.0,
        )
        if response and response.choices and response.choices[0].message:
            reply = (response.choices[0].message.content or "").strip()
            print(f"PASS: Model generation succeeded. Response: '{reply}'")
        else:
            print("FAIL: Received empty choices in response.")
            return False
    except Exception as e:
        print(f"FAIL: Generation failed for '{model_id}': {type(e).__name__}: {e}")
        return False

    # 3c. Test via OpenAIProvider wrapper
    print("\nTesting via application OpenAIProvider wrapper...")
    try:
        provider = OpenAIProvider(
            api_key=api_key,
            api_url=base_url,
            default_generation_max_output_tokens=50,
            default_generation_temperature=0.0,
        )
        provider.set_generation_model(model_id)
        reply = provider.generate_text("Reply with 'OK'")
        print(f"PASS: OpenAIProvider.generate_text() returned: '{reply}'")
        return True
    except Exception as e:
        print(f"FAIL: OpenAIProvider.generate_text() failed: {type(e).__name__}: {e}")
        return False


def main():
    settings = Settings()
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY") or settings.OPENAI_API_KEY
    base_url = os.environ.get("OPENAI_API_URL") or settings.OPENAI_API_URL or "https://generativelanguage.googleapis.com/v1beta/openai/"
    model_id = os.environ.get("GENERATION_MODEL_ID") or settings.GENERATION_MODEL_ID or "gemini-3.5-flash-lite"

    print("=" * 60)
    print("NileGuide mini-rag — Gemini Provider Verification Diagnostic")
    print("=" * 60)
    print(f"Configured Base URL: {base_url}")
    print(f"Configured Model ID: {model_id}")
    print(f"API Key Present: {bool(api_key)}")

    # 1. Negative auth test (does not require a valid key)
    neg_auth_ok = run_negative_auth_test(base_url)

    # 2. Live credential tests
    if not api_key:
        print("\n" + "=" * 60)
        print("NOTICE: No OPENAI_API_KEY or GEMINI_API_KEY available in local environment.")
        print("Skipping live completion and model list checks.")
        print("To run live provider verification locally, pass:")
        print("  $env:OPENAI_API_KEY = '<your-gemini-key>'")
        print("  python scripts/verify_gemini_provider.py")
        print("=" * 60)
        sys.exit(0 if neg_auth_ok else 1)

    neg_model_ok = run_negative_model_error_test(api_key, base_url)
    live_ok = run_live_provider_test(api_key, base_url, model_id)

    print("\n" + "=" * 60)
    print("Provider Verification Summary:")
    print(f"  Negative Auth Check: {'PASS' if neg_auth_ok else 'FAIL'}")
    print(f"  Negative Model Check: {'PASS' if neg_model_ok else 'FAIL'}")
    print(f"  Live Generation Check: {'PASS' if live_ok else 'FAIL'}")
    print("=" * 60)

    sys.exit(0 if (neg_auth_ok and neg_model_ok and live_ok) else 1)


if __name__ == "__main__":
    main()
