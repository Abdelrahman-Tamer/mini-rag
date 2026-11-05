import pytest
from unittest.mock import MagicMock, patch
from stores.llm.providers.OpenAIProvider import OpenAIProvider


def create_mock_completion(content_text="Mock response"):
    choice = MagicMock()
    choice.message.content = content_text
    response = MagicMock()
    response.choices = [choice]
    return response


def test_primary_succeeds_fallback_not_called():
    provider = OpenAIProvider(
        api_key="mock-key",
        api_url="https://mock.api",
        fallback_model_id="gemini-3.5-flash",
    )
    provider.set_generation_model("gemini-3.5-flash-lite")
    
    mock_response = create_mock_completion("Valid generated answer")
    provider.client = MagicMock()
    provider.client.chat.completions.create.return_value = mock_response

    result = provider.generate_text(prompt="What is Giza?")

    assert result == "Valid generated answer"
    assert provider.client.chat.completions.create.call_count == 1
    call_kwargs = provider.client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gemini-3.5-flash-lite"


def test_primary_fails_fallback_differs_invoked():
    provider = OpenAIProvider(
        api_key="mock-key",
        api_url="https://mock.api",
        fallback_model_id="gemini-3.5-flash",
    )
    provider.set_generation_model("gemini-3.5-flash-lite")

    mock_fallback_response = create_mock_completion("Fallback generated answer")
    provider.client = MagicMock()
    provider.client.chat.completions.create.side_effect = [
        Exception("503 Service Unavailable"),
        mock_fallback_response,
    ]

    result = provider.generate_text(prompt="What is Giza?")

    assert result == "Fallback generated answer"
    assert provider.client.chat.completions.create.call_count == 2
    
    first_call_model = provider.client.chat.completions.create.call_args_list[0].kwargs["model"]
    second_call_model = provider.client.chat.completions.create.call_args_list[1].kwargs["model"]
    assert first_call_model == "gemini-3.5-flash-lite"
    assert second_call_model == "gemini-3.5-flash"


def test_primary_promoted_duplicate_fallback_skipped():
    """
    When GENERATION_MODEL_ID is promoted to gemini-3.5-flash and FALLBACK_MODEL_ID is gemini-3.5-flash,
    never call gemini-3.5-flash twice. If primary fails, skip duplicate fallback immediately.
    """
    provider = OpenAIProvider(
        api_key="mock-key",
        api_url="https://mock.api",
        fallback_model_id="gemini-3.5-flash",
    )
    provider.set_generation_model("gemini-3.5-flash")

    provider.client = MagicMock()
    provider.client.chat.completions.create.side_effect = Exception("503 Service Unavailable")

    result = provider.generate_text(prompt="What is Giza?")

    assert result is None
    # Crucial: Exactly 1 call was made, duplicate fallback skipped!
    assert provider.client.chat.completions.create.call_count == 1
    called_model = provider.client.chat.completions.create.call_args.kwargs["model"]
    assert called_model == "gemini-3.5-flash"


def test_both_failure_paths_return_none_and_log():
    provider = OpenAIProvider(
        api_key="mock-key",
        api_url="https://mock.api",
        fallback_model_id="gemini-1.5-pro",
    )
    provider.set_generation_model("gemini-3.5-flash")

    provider.client = MagicMock()
    provider.client.chat.completions.create.side_effect = [
        Exception("503 Service Unavailable primary"),
        Exception("503 Service Unavailable fallback"),
    ]

    with patch.object(provider.logger, "error") as mock_logger_error:
        result = provider.generate_text(prompt="What is Giza?")
        assert result is None
        assert provider.client.chat.completions.create.call_count == 2
        assert mock_logger_error.call_count >= 1


def test_set_fallback_model_dynamically():
    provider = OpenAIProvider(
        api_key="mock-key",
        api_url="https://mock.api",
    )
    provider.set_generation_model("gemini-3.5-flash")
    provider.set_fallback_model("gemini-3.5-flash")

    provider.client = MagicMock()
    provider.client.chat.completions.create.side_effect = Exception("503 Service Unavailable")

    result = provider.generate_text(prompt="What is Giza?")
    assert result is None
    assert provider.client.chat.completions.create.call_count == 1
