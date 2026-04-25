"""
Unit tests for src/ai_analyzer.py — provider factory, prompt building,
response saving. HTTP calls are mocked (no network required).
"""

from unittest.mock import patch, MagicMock

import pytest

from src.ai_analyzer import (
    get_analyzer,
    build_analysis_prompt,
    save_analysis,
    AiResponse,
    OpenAiAnalyzer,
    OpenRouterAnalyzer,
    AnthropicAnalyzer,
    LocalAnalyzer,
    DEFAULT_MODELS,
    DEFAULT_ENDPOINTS,
    SYSTEM_PROMPT,
)


class TestFactory:
    def test_default_is_openrouter(self):
        """Calling get_analyzer() with no args must return OpenRouter
        (the documented default). Avoids the prior misleading call
        `get_analyzer('openrouter')` that tested the explicit string
        rather than the default."""
        analyzer = get_analyzer()
        assert isinstance(analyzer, OpenRouterAnalyzer)

    def test_openai_factory(self):
        analyzer = get_analyzer("openai")
        assert isinstance(analyzer, OpenAiAnalyzer)

    def test_anthropic_factory(self):
        analyzer = get_analyzer("anthropic")
        assert isinstance(analyzer, AnthropicAnalyzer)

    def test_local_factory(self):
        analyzer = get_analyzer("local")
        assert isinstance(analyzer, LocalAnalyzer)

    def test_unsupported_raises(self):
        with pytest.raises(ValueError):
            get_analyzer("unknown")

    def test_case_insensitive(self):
        analyzer = get_analyzer("OPENROUTER")
        assert isinstance(analyzer, OpenRouterAnalyzer)


class TestDefaults:
    def test_default_models_complete(self):
        expected = {"openrouter", "openai", "anthropic", "local"}
        assert set(DEFAULT_MODELS.keys()) == expected

    def test_endpoints_valid_urls(self):
        for provider, url in DEFAULT_ENDPOINTS.items():
            assert url.startswith("http")

    def test_openrouter_is_default_in_docs(self):
        # Verify OpenRouter default model is documented / sensible
        assert "anthropic" in DEFAULT_MODELS["openrouter"] or "gpt" in DEFAULT_MODELS["openrouter"].lower() \
               or "openai" in DEFAULT_MODELS["openrouter"].lower()


class TestPromptBuilding:
    def test_includes_report(self):
        prompt = build_analysis_prompt("## Test report content")
        assert "## Test report content" in prompt

    def test_extra_context_appended(self):
        prompt = build_analysis_prompt(
            "## Report",
            extra_context="### Additional sensitivity\nfoo=1",
        )
        assert "Additional sensitivity" in prompt

    def test_prompt_mentions_structure(self):
        prompt = build_analysis_prompt("## Report")
        # Prompt should tell the model to respond per the system prompt's structure
        assert "struttura" in prompt.lower() or "struttura" in SYSTEM_PROMPT.lower()


class TestSaveAnalysis:
    def test_save_creates_file(self, tmp_path):
        response = AiResponse(
            content="## Analysis\n\nSome markdown text.",
            provider="openrouter",
            model="claude-opus-4-7",
            prompt_tokens=1000,
            completion_tokens=500,
        )
        out = tmp_path / "AI_ANALYSIS.md"
        save_analysis(response, out)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "# AI Analysis" in content
        assert "claude-opus-4-7" in content
        assert "Some markdown text" in content
        assert "Prompt tokens" in content or "1,000" in content

    def test_save_handles_no_token_counts(self, tmp_path):
        response = AiResponse(
            content="Some analysis",
            provider="local",
            model="kimi-k2",
        )
        out = tmp_path / "AI.md"
        save_analysis(response, out)
        content = out.read_text(encoding="utf-8")
        assert "local" in content

    def test_save_records_zero_tokens(self, tmp_path):
        """Regression (Copilot PR #10): prompt_tokens=0 must appear in the
        header. Previously a truthy-check silently omitted it."""
        response = AiResponse(
            content="analysis", provider="local", model="kimi-k2",
            prompt_tokens=0, completion_tokens=0,
        )
        out = tmp_path / "AI.md"
        save_analysis(response, out)
        content = out.read_text(encoding="utf-8")
        assert "Prompt tokens" in content
        assert "Completion tokens" in content


class TestLocalEndpointResolution:
    """Regression (Copilot PR #10): LOCAL_API_BASE_URL must be resolved
    lazily on each lookup, not frozen at module-import time."""

    def test_local_endpoint_follows_env_changes(self, monkeypatch):
        from src.ai_analyzer import DEFAULT_ENDPOINTS
        monkeypatch.setenv("LOCAL_API_BASE_URL", "http://example.com:9999/v1")
        endpoint = DEFAULT_ENDPOINTS["local"]
        assert endpoint == "http://example.com:9999/v1/chat/completions"
        # Change env var — next lookup must reflect the new value
        monkeypatch.setenv("LOCAL_API_BASE_URL", "http://other.local:8080/v1")
        endpoint2 = DEFAULT_ENDPOINTS["local"]
        assert endpoint2 == "http://other.local:8080/v1/chat/completions"

    def test_local_endpoint_default_when_unset(self, monkeypatch):
        from src.ai_analyzer import DEFAULT_ENDPOINTS
        monkeypatch.delenv("LOCAL_API_BASE_URL", raising=False)
        endpoint = DEFAULT_ENDPOINTS["local"]
        assert endpoint == "http://localhost:11434/v1/chat/completions"

    def test_local_endpoint_trims_trailing_slash(self, monkeypatch):
        from src.ai_analyzer import DEFAULT_ENDPOINTS
        monkeypatch.setenv("LOCAL_API_BASE_URL", "http://example.com:9999/v1/")
        assert DEFAULT_ENDPOINTS["local"] == "http://example.com:9999/v1/chat/completions"


class TestLocalAnalyzerNoAuth:
    """Regression (Copilot PR #10): LocalAnalyzer must not force a bogus
    'ollama' Bearer token, because many local servers reject unexpected
    Authorization headers."""

    @patch("src.ai_analyzer.urlopen")
    def test_no_bearer_header_when_api_key_is_none(self, mock_urlopen, monkeypatch):
        from src.ai_analyzer import LocalAnalyzer
        monkeypatch.setenv("LOCAL_API_BASE_URL", "http://localhost:11434/v1")
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"choices":[{"message":{"content":"ok"}}],"usage":{}}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        analyzer = LocalAnalyzer()  # api_key=None by default
        analyzer.analyze("test")

        # Inspect the Request object passed to urlopen
        call_args = mock_urlopen.call_args
        req = call_args[0][0]  # first positional arg
        # Authorization header must NOT be present when api_key is None
        assert not req.has_header("Authorization"), \
            "LocalAnalyzer should not send Authorization header by default"

    @patch("src.ai_analyzer.urlopen")
    def test_bearer_header_when_user_provides_key(self, mock_urlopen):
        from src.ai_analyzer import LocalAnalyzer
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"choices":[{"message":{"content":"ok"}}],"usage":{}}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        analyzer = LocalAnalyzer(api_key="my-token")
        analyzer.analyze("test")
        req = mock_urlopen.call_args[0][0]
        assert req.has_header("Authorization")


class TestMockedCalls:
    """Test that analyzers correctly invoke the HTTP machinery when keys are set."""

    @patch("src.ai_analyzer.urlopen")
    def test_openrouter_call_structure(self, mock_urlopen, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-12345")
        # Mock response
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"choices": [{"message": {"content": "mock analysis"}}], "usage": {"prompt_tokens": 100, "completion_tokens": 50}}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        analyzer = OpenRouterAnalyzer(model="test-model")
        result = analyzer.analyze("test prompt")
        assert result.content == "mock analysis"
        assert result.provider == "openrouter"
        assert result.model == "test-model"
        assert result.prompt_tokens == 100

    @patch("src.ai_analyzer.urlopen")
    def test_anthropic_call_structure(self, mock_urlopen, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-xyz")
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"content": [{"text": "anthropic analysis"}], "usage": {"input_tokens": 200, "output_tokens": 100}}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        analyzer = AnthropicAnalyzer(model="claude-opus-4-7")
        result = analyzer.analyze("test prompt")
        assert result.content == "anthropic analysis"
        assert result.prompt_tokens == 200
        assert result.completion_tokens == 100

    def test_openrouter_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        analyzer = OpenRouterAnalyzer()
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            analyzer.analyze("prompt")

    def test_openai_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        analyzer = OpenAiAnalyzer()
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            analyzer.analyze("prompt")
