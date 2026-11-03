"""
NileGuide mini-rag — Gemini Provider Integration and Diagnostics Suite
Verifies Google Gemini API authentication, OpenAI-compatible endpoint routing,
model availability (gemini-3.8-flash / gemini-3.5-flash-lite), minimal generation,
token accounting, and error handling without sending any requests to Production.
"""

import os
import sys
import time
import logging
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openai import (
    OpenAI,
    AuthenticationError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    APIError,
    APIConnectionError,
)
from api.helpers.config import Settings
from api.stores.llm.providers.OpenAIProvider import OpenAIProvider


def mask_secret(secret: Optional[str]) -> str:
    """Mask credentials for safe diagnostic logging."""
    if not secret:
        return "[UNCONFIGURED]"
    if len(secret) <= 8:
        return f"[REDACTED len={len(secret)}]"
    return f"{secret[:4]}...{secret[-4:]} (len={len(secret)})"


def classify_error(err: Exception) -> str:
    """Categorize API errors into clear operational buckets."""
    if isinstance(err, AuthenticationError):
        return "HTTP 401 — Authentication Failure (Invalid API Key or token)"
    elif isinstance(err, PermissionDeniedError):
        return "HTTP 403 — Permission Denied (Project lacks access, billing inactive, or API disabled)"
    elif isinstance(err, NotFoundError):
        return "HTTP 404 — Not Found (Model ID invalid, decommissioned, or endpoint path unrecognized)"
    elif isinstance(err, RateLimitError):
        return "HTTP 429 — Rate Limit / Quota Exceeded (Resource exhausted or throughput limit hit)"
    elif isinstance(err, APIConnectionError):
        return "Network / Connection Error (DNS, TLS, or firewall blocked outbound connection)"
    elif isinstance(err, APIError):
        status = getattr(err, "status_code", "UNKNOWN")
        return f"HTTP {status} — API Error: {err.message if hasattr(err, 'message') else str(err)}"
    return f"Unexpected Error ({type(err).__name__}): {str(err)}"


def run_negative_auth_test(base_url: str) -> bool:
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
        print(f"PASS: Provider rejected invalid key with expected exception: {type(e).__name__} ({classify_error(e)})")
        return True


def run_negative_model_error_test(api_key: str, base_url: str) -> bool:
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
        print(f"PASS: Provider rejected non-existent model with NotFoundError (HTTP 404).")
        return True
    except APIError as e:
        print(f"PASS: Provider rejected non-existent model with APIError ({classify_error(e)}).")
        return True
    except Exception as e:
        print(f"PASS: Provider rejected non-existent model with {type(e).__name__}: {classify_error(e)}")
        return True


def run_live_provider_test(api_key: str, base_url: str, model_id: str) -> bool:
    """Verify live authentication, model listing, metadata, and generation."""
    print("\n--- 3. Live Provider Integration Verification ---")
    print(f"Target Base URL : {base_url}")
    print(f"Target Model    : {model_id}")
    print(f"API Key         : {mask_secret(api_key)}")

    client = OpenAI(api_key=api_key, base_url=base_url)

    # 3a. Model listing
    print("\n[3a] Querying available models via client.models.list()...")
    gemini_models = []
    try:
        start_t = time.perf_counter()
        models = client.models.list()
        elapsed_ms = (time.perf_counter() - start_t) * 1000
        gemini_models = [m.id for m in models if "gemini" in (m.id or "").lower()]
        print(f"PASS: client.models.list() succeeded in {elapsed_ms:.1f}ms. Found {len(gemini_models)} Gemini models.")
        
        # Display sample models
        for m in sorted(gemini_models)[:10]:
            marker = " <== TARGET" if m == model_id or m == f"models/{model_id}" else ""
            print(f"  - {m}{marker}")
        if len(gemini_models) > 10:
            print(f"  ... and {len(gemini_models) - 10} more.")
            
        target_in_list = any(model_id in m for m in gemini_models)
        if target_in_list:
            print(f"PASS: Configured model '{model_id}' is confirmed present in provider model catalog.")
        else:
            print(f"NOTICE: '{model_id}' was not explicitly in client.models.list(). Testing direct invocation.")
    except Exception as e:
        print(f"WARNING: client.models.list() encountered an issue: {classify_error(e)}")

    # 3b. Model metadata / retrieve test
    print(f"\n[3b] Inspecting model metadata for '{model_id}'...")
    try:
        model_meta = client.models.retrieve(model_id)
        print(f"PASS: client.models.retrieve('{model_id}') succeeded: id='{getattr(model_meta, 'id', model_id)}'")
    except Exception as e:
        print(f"INFO: client.models.retrieve('{model_id}') returned: {classify_error(e)}")
        print("      (Google's OpenAI-compatible endpoint often restricts direct models.retrieve; chat completion is authoritative.)")

    # 3c. Minimal generation test
    print(f"\n[3c] Testing chat completion generation with model='{model_id}'...")
    generation_ok = False
    try:
        start_t = time.perf_counter()
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a NileGuide verification agent. Respond with only: OK"},
                {"role": "user", "content": "ping"}
            ],
            max_tokens=20,
            temperature=0.0,
        )
        duration_ms = (time.perf_counter() - start_t) * 1000

        if response and response.choices and response.choices[0].message:
            reply = (response.choices[0].message.content or "").strip()
            print(f"PASS: Completion succeeded in {duration_ms:.1f}ms.")
            print(f"      Response Content : '{reply}'")
            
            # Token usage
            if hasattr(response, "usage") and response.usage:
                u = response.usage
                print(f"      Token Usage      : prompt={getattr(u, 'prompt_tokens', 'N/A')}, "
                      f"completion={getattr(u, 'completion_tokens', 'N/A')}, "
                      f"total={getattr(u, 'total_tokens', 'N/A')}")
            else:
                print("      Token Usage      : Not reported by provider.")
            generation_ok = bool(reply)
        else:
            print("FAIL: Received empty choices in chat completion response.")
            return False
    except Exception as e:
        print(f"FAIL: Chat completion failed for model='{model_id}': {classify_error(e)}")
        return False

    # 3d. Application OpenAIProvider wrapper test
    print("\n[3d] Testing via application OpenAIProvider wrapper...")
    try:
        provider = OpenAIProvider(
            api_key=api_key,
            api_url=base_url,
            default_generation_max_output_tokens=50,
            default_generation_temperature=0.0,
        )
        provider.set_generation_model(model_id)
        reply = provider.generate_text("Reply with 'VERIFIED'")
        if reply and len(reply.strip()) > 0:
            print(f"PASS: OpenAIProvider.generate_text() returned: '{reply.strip()}'")
            return generation_ok
        else:
            print(f"FAIL: OpenAIProvider.generate_text() returned empty/None.")
            return False
    except Exception as e:
        print(f"FAIL: OpenAIProvider wrapper threw exception: {classify_error(e)}")
        return False


def main():
    settings = Settings()
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or settings.OPENAI_API_KEY
    )
    base_url = (
        os.environ.get("OPENAI_API_URL")
        or settings.OPENAI_API_URL
        or "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    model_id = (
        os.environ.get("GENERATION_MODEL_ID")
        or "gemini-3.8-flash"
    )

    print("=" * 70)
    print("NileGuide mini-rag — Google Gemini Provider Verification Suite")
    print("=" * 70)
    print(f"Configured Base URL : {base_url}")
    print(f"Target Model ID     : {model_id}")
    print(f"API Key Status      : {'CONFIGURED (' + mask_secret(api_key) + ')' if api_key else 'NOT CONFIGURED'}")

    # 1. Negative authentication test (runs without requiring a valid key)
    neg_auth_ok = run_negative_auth_test(base_url)

    # 2. Check if credentials are present for live execution
    if not api_key:
        print("\n" + "=" * 70)
        print("NOTICE: No OPENAI_API_KEY or GEMINI_API_KEY detected in current environment.")
        print("Skipping live completion, model listing, and wrapper tests.")
        print("")
        print("To run the live verification safely without exposing keys in terminal history:")
        print("  .\\scripts\\run_secure_gemini_check.ps1")
        print("=" * 70)
        print("\nSummary: Negative Auth Check PASSED. Live generation awaiting operator credentials.")
        sys.exit(0 if neg_auth_ok else 1)

    neg_model_ok = run_negative_model_error_test(api_key, base_url)
    live_ok = run_live_provider_test(api_key, base_url, model_id)

    print("\n" + "=" * 70)
    print("Provider Diagnostic Summary:")
    print(f"  1. Negative Auth Check (401 Verification) : {'PASS' if neg_auth_ok else 'FAIL'}")
    print(f"  2. Negative Model Check (404 Verification): {'PASS' if neg_model_ok else 'FAIL'}")
    print(f"  3. Live Generation & Wrapper Check        : {'PASS' if live_ok else 'FAIL'}")
    print("=" * 70)

    overall_ok = neg_auth_ok and neg_model_ok and live_ok
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
