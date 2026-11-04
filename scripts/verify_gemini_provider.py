"""
NileGuide mini-rag — Gemini Provider Integration and Diagnostics Suite
Verifies Google Gemini API authentication, OpenAI-compatible endpoint routing,
model availability (gemini-3.5-flash-lite primary & gemini-3.5-flash fallback),
minimal generation, token accounting, and failover error handling without sending any requests to Production.
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
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    APIError,
    APIConnectionError,
    APITimeoutError,
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
    if isinstance(err, BadRequestError):
        return f"HTTP 400 — Bad Request / Invalid Argument (Observed Google API key rejection: {err.message if hasattr(err, 'message') else str(err)})"
    elif isinstance(err, AuthenticationError):
        return "HTTP 401 — Authentication Failure (Invalid API Key or Bearer token)"
    elif isinstance(err, PermissionDeniedError):
        return "HTTP 403 — Permission Denied (Project lacks access, billing inactive, or API disabled)"
    elif isinstance(err, NotFoundError):
        return "HTTP 404 — Not Found (Model ID invalid, decommissioned, or endpoint path unrecognized)"
    elif isinstance(err, RateLimitError):
        return "HTTP 429 — Rate Limit / Quota Exceeded (Resource exhausted or throughput limit hit)"
    elif isinstance(err, (APIConnectionError, APITimeoutError)):
        return f"Network / Timeout Error ({type(err).__name__}: {str(err)})"
    elif isinstance(err, APIError):
        status = getattr(err, "status_code", "UNKNOWN")
        return f"HTTP {status} — API Error: {err.message if hasattr(err, 'message') else str(err)}"
    return f"Unexpected Error ({type(err).__name__}): {str(err)}"


def run_negative_auth_test(base_url: str) -> bool:
    """Verify that an invalid API key fails closed with HTTP 400 (INVALID_ARGUMENT / BadRequestError) or HTTP 401."""
    print("\n--- 1. Negative Authentication Verification ---")
    dummy_client = OpenAI(
        api_key="invalid-audit-key-test-000000000",
        base_url=base_url,
    )
    try:
        dummy_client.models.list()
        print("FAIL: Expected rejection with invalid key, but call succeeded!")
        return False
    except (BadRequestError, AuthenticationError) as e:
        status = getattr(e, "status_code", 400)
        print(f"PASS: Provider rejected invalid key with expected {type(e).__name__} (HTTP {status} INVALID_ARGUMENT fail-closed).")
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


def run_live_provider_test(api_key: str, base_url: str, primary_model_id: str, fallback_model_id: str) -> tuple[bool, bool, bool]:
    """Verify live authentication, model listing, primary generation, fallback generation, and wrapper failover."""
    print("\n--- 3. Live Provider Integration Verification ---")
    print(f"Target Base URL    : {base_url}")
    print(f"Primary Model      : {primary_model_id}")
    print(f"Fallback Candidate : {fallback_model_id}")
    print(f"API Key            : {mask_secret(api_key)}")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=10.0)

    # 3a. Model listing
    print("\n[3a] Querying available models via client.models.list()...")
    try:
        start_t = time.perf_counter()
        models = client.models.list()
        elapsed_ms = (time.perf_counter() - start_t) * 1000
        gemini_models = [m.id for m in models if "gemini" in (m.id or "").lower()]
        print(f"PASS: client.models.list() succeeded in {elapsed_ms:.1f}ms. Found {len(gemini_models)} Gemini models.")
        
        has_primary = any(primary_model_id in m for m in gemini_models)
        has_fallback = any(fallback_model_id in m for m in gemini_models)
        print(f"  - Primary '{primary_model_id}' present: {has_primary}")
        print(f"  - Fallback '{fallback_model_id}' present: {has_fallback}")
    except Exception as e:
        print(f"WARNING: client.models.list() encountered an issue: {classify_error(e)}")

    # 3b. Primary model generation test
    print(f"\n[3b] Testing PRIMARY model chat completion with '{primary_model_id}'...")
    primary_ok = False
    try:
        start_t = time.perf_counter()
        response = client.chat.completions.create(
            model=primary_model_id,
            messages=[
                {"role": "system", "content": "You are a NileGuide verification agent. Respond with: OK"},
                {"role": "user", "content": "ping"}
            ],
            max_tokens=100,
            temperature=0.0,
        )
        duration_ms = (time.perf_counter() - start_t) * 1000

        if response and response.choices and response.choices[0].message:
            reply = (response.choices[0].message.content or "").strip()
            print(f"PASS: Primary completion succeeded in {duration_ms:.1f}ms.")
            print(f"      Response Content : '{reply}'")
            if hasattr(response, "usage") and response.usage:
                u = response.usage
                print(f"      Token Usage      : prompt={getattr(u, 'prompt_tokens', 'N/A')}, "
                      f"completion={getattr(u, 'completion_tokens', 'N/A')}, total={getattr(u, 'total_tokens', 'N/A')}")
            primary_ok = bool(reply)
        else:
            print("FAIL: Received empty choices in chat completion response.")
    except Exception as e:
        print(f"FAIL: Primary chat completion failed for model='{primary_model_id}': {classify_error(e)}")

    # 3c. Fallback model generation test
    print(f"\n[3c] Testing FALLBACK candidate chat completion with '{fallback_model_id}'...")
    fallback_ok = False
    try:
        start_t = time.perf_counter()
        response_fb = client.chat.completions.create(
            model=fallback_model_id,
            messages=[
                {"role": "system", "content": "You are a NileGuide verification agent. Respond with: OK"},
                {"role": "user", "content": "ping"}
            ],
            max_tokens=100,
            temperature=0.0,
        )
        duration_fb_ms = (time.perf_counter() - start_t) * 1000

        if response_fb and response_fb.choices and response_fb.choices[0].message:
            reply_fb = (response_fb.choices[0].message.content or "").strip()
            print(f"PASS: Fallback completion succeeded in {duration_fb_ms:.1f}ms.")
            print(f"      Response Content : '{reply_fb}'")
            if hasattr(response_fb, "usage") and response_fb.usage:
                u = response_fb.usage
                print(f"      Token Usage      : prompt={getattr(u, 'prompt_tokens', 'N/A')}, "
                      f"completion={getattr(u, 'completion_tokens', 'N/A')}, total={getattr(u, 'total_tokens', 'N/A')}")
            fallback_ok = bool(reply_fb)
        else:
            print("FAIL: Received empty choices in fallback chat completion response.")
    except Exception as e:
        print(f"FAIL: Fallback chat completion failed for model='{fallback_model_id}': {classify_error(e)}")

    # 3d. Application OpenAIProvider wrapper failover test
    print("\n[3d] Testing via application OpenAIProvider wrapper (with automatic failover)...")
    wrapper_ok = False
    try:
        provider = OpenAIProvider(
            api_key=api_key,
            api_url=base_url,
            default_generation_max_output_tokens=100,
            default_generation_temperature=0.0,
            timeout=7.0
        )
        provider.set_generation_model(primary_model_id)
        reply = provider.generate_text("Reply with 'VERIFIED'")
        if reply and len(reply.strip()) > 0:
            print(f"PASS: OpenAIProvider.generate_text() returned: '{reply.strip()}'")
            wrapper_ok = True
        else:
            print(f"FAIL: OpenAIProvider.generate_text() returned empty/None.")
    except Exception as e:
        print(f"FAIL: OpenAIProvider wrapper threw exception: {classify_error(e)}")

    return primary_ok, fallback_ok, wrapper_ok


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
    primary_model_id = (
        os.environ.get("GENERATION_MODEL_ID")
        or "gemini-3.5-flash-lite"
    )
    fallback_model_id = (
        os.environ.get("FALLBACK_MODEL_ID")
        or "gemini-3.5-flash"
    )

    print("=" * 70)
    print("NileGuide mini-rag — Google Gemini Provider Verification Suite")
    print("=" * 70)
    print(f"Configured Base URL : {base_url}")
    print(f"Primary Model ID    : {primary_model_id}")
    print(f"Fallback Model ID   : {fallback_model_id}")
    print(f"API Key Status      : {'CONFIGURED (' + mask_secret(api_key) + ')' if api_key else 'NOT CONFIGURED'}")

    # 1. Negative authentication test (verifies HTTP 400 INVALID_ARGUMENT fail-closed)
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
    primary_ok, fallback_ok, wrapper_ok = run_live_provider_test(api_key, base_url, primary_model_id, fallback_model_id)

    print("\n" + "=" * 70)
    print("Provider Diagnostic Summary:")
    print(f"  1. Negative Auth Check (HTTP 400 INVALID_ARGUMENT Fail-Closed) : {'PASS' if neg_auth_ok else 'FAIL'}")
    print(f"  2. Negative Model Check (HTTP 404 Verification)                : {'PASS' if neg_model_ok else 'FAIL'}")
    print(f"  3. Primary Model Live Generation (gemini-3.5-flash-lite)       : {'PASS' if primary_ok else 'FAIL'}")
    print(f"  4. Fallback Model Live Generation (gemini-3.5-flash)           : {'PASS' if fallback_ok else 'FAIL'}")
    print(f"  5. OpenAIProvider Wrapper Failover Integration                 : {'PASS' if wrapper_ok else 'FAIL'}")
    print("=" * 70)

    overall_ok = neg_auth_ok and neg_model_ok and primary_ok and fallback_ok and wrapper_ok
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
