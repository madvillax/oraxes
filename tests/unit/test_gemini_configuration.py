import pytest

from oraxes.llm.gemini import GeminiConfigurationError, GeminiProvider, SynthesisOutput


def test_provider_rejects_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(GeminiConfigurationError, match="missing"):
        GeminiProvider()


def test_provider_rejects_example_api_key() -> None:
    with pytest.raises(GeminiConfigurationError, match="placeholder"):
        GeminiProvider(api_key="replace-with-your-key")


def test_provider_explains_unsupported_function_response_model() -> None:
    provider = GeminiProvider(api_key="test-key", model="gemini-2.5-flash")

    error = provider._translate_request_error(
        RuntimeError("Multimodal function responses are not supported for this model.")
    )

    assert error is not None
    assert "gemini-2.5-flash" in str(error)
    assert "gemini-3.5-flash-lite" in str(error)


def test_synthesis_schema_excludes_application_owned_citations() -> None:
    schema = SynthesisOutput.model_json_schema()

    assert "citations" not in schema["properties"]
    assert "format" not in str(schema)
    assert "pattern" not in str(schema)


def test_provider_parses_gemini_retry_delay() -> None:
    error = RuntimeError("Please retry in 2.038401149s. code: too_many_requests")

    assert GeminiProvider._is_rate_limit_error(error)
    assert GeminiProvider._retry_delay_seconds(error, attempt=0) == pytest.approx(2.538401149)
