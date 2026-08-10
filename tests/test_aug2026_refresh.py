"""Tests for the August 2026 refresh: non-retryable error fast-fail,
Codex ChatGPT-account preflight, and the Antigravity CLI provider."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import models
import providers
from models import (
    CODEX_CHATGPT_HINT,
    call_single_model,
    is_non_retryable_error,
    is_reasoning_model,
    resolve_antigravity_model,
)
from providers import (
    CODEX_CHATGPT_MODELS,
    codex_auth_mode,
    warn_codex_chatgpt_model_support,
)


class TestNonRetryableErrors:
    def test_chatgpt_account_model_rejection(self):
        msg = (
            '{"type":"error","status":400,"error":{"type":"invalid_request_error",'
            '"message":"The \'gpt-5.3-codex\' model is not supported when using '
            'Codex with a ChatGPT account."}}'
        )
        assert is_non_retryable_error(msg)

    def test_invalid_request_error(self):
        assert is_non_retryable_error("litellm.BadRequestError: invalid_request_error")

    def test_model_not_found(self):
        assert is_non_retryable_error("model_not_found: no such model")
        assert is_non_retryable_error(
            "The model `gpt-9` does not exist or you do not have access to it"
        )

    def test_auth_errors(self):
        assert is_non_retryable_error("litellm.AuthenticationError: bad key")
        assert is_non_retryable_error("Incorrect API key provided")

    def test_antigravity_deterministic_errors(self):
        assert is_non_retryable_error(
            "Antigravity CLI is not authenticated. Run `agy` interactively once"
        )
        assert is_non_retryable_error(
            "Antigravity CLI returned status ERROR: invalid model selection"
        )

    def test_bedrock_rewritten_errors(self):
        assert is_non_retryable_error(
            "Model not enabled in your Bedrock account: claude-opus-5"
        )
        assert is_non_retryable_error("Invalid Bedrock model ID: bogus.model")

    def test_transient_errors_still_retry(self):
        assert not is_non_retryable_error("rate limit exceeded")
        assert not is_non_retryable_error("connection reset by peer")
        assert not is_non_retryable_error("timed out after 600s")
        assert not is_non_retryable_error("500 Internal Server Error")

    def test_call_single_model_fails_fast_on_non_retryable(self):
        calls = []

        def failing_call(*args, **kwargs):
            calls.append(1)
            raise RuntimeError(
                "The 'gpt-5.3-codex' model is not supported when using Codex "
                "with a ChatGPT account."
            )

        with (
            patch("models.call_codex_model", side_effect=failing_call),
            patch("models.time.sleep") as mock_sleep,
        ):
            result = call_single_model(
                "codex/gpt-5.3-codex", "sys", "user", "investor", "progress"
            )

        assert len(calls) == 1  # single attempt, no retries
        mock_sleep.assert_not_called()
        assert result.error is not None
        assert CODEX_CHATGPT_HINT in result.error

    def test_call_single_model_retries_transient_errors(self):
        calls = []

        def failing_call(*args, **kwargs):
            calls.append(1)
            raise RuntimeError("rate limit exceeded")

        with (
            patch("models.call_codex_model", side_effect=failing_call),
            patch("models.time.sleep"),
        ):
            result = call_single_model(
                "codex/gpt-5.5", "sys", "user", "investor", "progress"
            )

        assert len(calls) == models.MAX_RETRIES
        assert result.error is not None
        assert "rate limit" in result.error

    def test_call_single_model_success_passthrough(self):
        with patch(
            "models.call_codex_model", return_value=("review text", 10, 5)
        ):
            result = call_single_model(
                "codex/gpt-5.5", "sys", "user", "investor", "progress"
            )
        assert result.error is None
        assert result.response == "review text"
        assert result.input_tokens == 10
        assert result.persona == "investor"
        assert result.dimension == "progress"


class TestCodexChatGPTPreflight:
    def test_chatgpt_lineup_membership(self):
        assert "gpt-5.6-sol" in CODEX_CHATGPT_MODELS
        assert "gpt-5.5" in CODEX_CHATGPT_MODELS
        assert "gpt-5.3-codex" not in CODEX_CHATGPT_MODELS
        assert "gpt-5.5-pro" not in CODEX_CHATGPT_MODELS

    def test_auth_mode_chatgpt(self, tmp_path, monkeypatch):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text(
            json.dumps({"auth_mode": "chatgpt", "tokens": {"x": 1}})
        )
        monkeypatch.setattr(providers.Path, "home", staticmethod(lambda: tmp_path))
        assert codex_auth_mode() == "chatgpt"

    def test_auth_mode_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(providers.Path, "home", staticmethod(lambda: tmp_path))
        assert codex_auth_mode() is None

    def test_warns_on_unsupported_model(self, capsys):
        with patch("providers.codex_auth_mode", return_value="chatgpt"):
            warn_codex_chatgpt_model_support(["codex/gpt-5.3-codex", "gpt-5.5"])
        err = capsys.readouterr().err
        assert "gpt-5.3-codex" in err
        assert "ChatGPT account" in err

    def test_silent_on_supported_models(self, capsys):
        with patch("providers.codex_auth_mode", return_value="chatgpt"):
            warn_codex_chatgpt_model_support(["codex/gpt-5.6-sol", "codex/gpt-5.5"])
        assert capsys.readouterr().err == ""

    def test_silent_on_apikey_auth(self, capsys):
        with patch("providers.codex_auth_mode", return_value="apikey"):
            warn_codex_chatgpt_model_support(["codex/gpt-5.3-codex"])
        assert capsys.readouterr().err == ""

    def test_silent_without_codex_models(self, capsys):
        with patch("providers.codex_auth_mode", return_value="chatgpt"):
            warn_codex_chatgpt_model_support(["gpt-5.5", "claude-opus-5"])
        assert capsys.readouterr().err == ""


class TestAntigravityProvider:
    def test_resolve_slug_passthrough(self):
        assert (
            resolve_antigravity_model("antigravity/gemini-3.1-pro-high")
            == "gemini-3.1-pro-high"
        )
        assert (
            resolve_antigravity_model("antigravity/claude-sonnet-4-6")
            == "claude-sonnet-4-6"
        )

    def test_resolve_bare_antigravity_uses_default(self):
        assert resolve_antigravity_model("antigravity") is None

    def test_call_parses_agy_json(self):
        payload = {
            "conversation_id": "abc",
            "status": "SUCCESS",
            "response": "OK\n",
            "usage": {"input_tokens": 17579, "output_tokens": 26},
        }
        fake = type(
            "P", (), {"returncode": 0, "stdout": json.dumps(payload), "stderr": ""}
        )()
        with (
            patch("models.ANTIGRAVITY_AVAILABLE", True),
            patch("models.ANTIGRAVITY_PATH", "/usr/bin/agy"),
            patch("models.subprocess.run", return_value=fake) as mock_run,
        ):
            text, inp, out = models.call_antigravity_model(
                "sys", "user", "antigravity/gemini-3.1-pro-high"
            )
        assert text == "OK"
        assert (inp, out) == (17579, 26)
        cmd = mock_run.call_args[0][0]
        assert "--model" in cmd and "gemini-3.1-pro-high" in cmd
        assert "--output-format" in cmd and "json" in cmd
        # Prompt must travel via stdin, never argv — large project contexts
        # exceed the Linux 128 KiB per-argument limit.
        stdin_input = mock_run.call_args.kwargs["input"]
        assert "SYSTEM INSTRUCTIONS:" in stdin_input and "user" in stdin_input
        assert not any("USER REQUEST" in arg for arg in cmd)

    def test_call_raises_on_error_status(self):
        payload = {
            "status": "ERROR",
            "response": "",
            "error": "invalid model selection",
        }
        fake = type(
            "P", (), {"returncode": 0, "stdout": json.dumps(payload), "stderr": ""}
        )()
        with (
            patch("models.ANTIGRAVITY_AVAILABLE", True),
            patch("models.ANTIGRAVITY_PATH", "/usr/bin/agy"),
            patch("models.subprocess.run", return_value=fake),
        ):
            with pytest.raises(RuntimeError, match="invalid model selection"):
                models.call_antigravity_model("sys", "user", "antigravity/bogus")

    def test_call_detects_unauthenticated(self):
        fake = type(
            "P",
            (),
            {
                "returncode": 1,
                "stdout": "Waiting for authentication (timeout 30s)...",
                "stderr": "",
            },
        )()
        with (
            patch("models.ANTIGRAVITY_AVAILABLE", True),
            patch("models.ANTIGRAVITY_PATH", "/usr/bin/agy"),
            patch("models.subprocess.run", return_value=fake),
        ):
            with pytest.raises(RuntimeError, match="not authenticated"):
                models.call_antigravity_model("sys", "user", "antigravity")

    def test_call_rejects_json_without_response_field(self):
        payload = {"conversation_id": "abc", "status": "SUCCESS"}
        fake = type(
            "P", (), {"returncode": 0, "stdout": json.dumps(payload), "stderr": ""}
        )()
        with (
            patch("models.ANTIGRAVITY_AVAILABLE", True),
            patch("models.ANTIGRAVITY_PATH", "/usr/bin/agy"),
            patch("models.subprocess.run", return_value=fake),
        ):
            with pytest.raises(RuntimeError, match="No response text"):
                models.call_antigravity_model("sys", "user", "antigravity")

    def test_call_falls_back_to_raw_text_output(self):
        fake = type(
            "P", (), {"returncode": 0, "stdout": "plain text answer", "stderr": ""}
        )()
        with (
            patch("models.ANTIGRAVITY_AVAILABLE", True),
            patch("models.ANTIGRAVITY_PATH", "/usr/bin/agy"),
            patch("models.subprocess.run", return_value=fake),
        ):
            text, _, _ = models.call_antigravity_model("sys", "user", "antigravity")
        assert text == "plain text answer"

    def test_bedrock_fable_5_mapping(self):
        assert (
            providers.resolve_bedrock_model("claude-fable-5")
            == "anthropic.claude-fable-5"
        )

    def test_validate_credentials_antigravity(self):
        with patch("providers.ANTIGRAVITY_AVAILABLE", True):
            valid, invalid = providers.validate_model_credentials(
                ["antigravity/gemini-3.1-pro-high"]
            )
        assert valid == ["antigravity/gemini-3.1-pro-high"]
        with patch("providers.ANTIGRAVITY_AVAILABLE", False):
            valid, invalid = providers.validate_model_credentials(
                ["antigravity/gemini-3.1-pro-high"]
            )
        assert invalid == ["antigravity/gemini-3.1-pro-high"]

    def test_antigravity_cost_is_free(self):
        assert providers.get_model_cost("antigravity/gemini-3.1-pro-high") == {
            "input": 0.0,
            "output": 0.0,
        }
        assert providers.get_model_cost("antigravity") == {
            "input": 0.0,
            "output": 0.0,
        }


class TestCodexStructuredErrors:
    def _fake_proc(self, stdout, returncode=0, stderr=""):
        return type(
            "P", (), {"returncode": returncode, "stdout": stdout, "stderr": stderr}
        )()

    def test_error_event_surfaces_structured_message(self):
        stdout = json.dumps(
            {
                "type": "error",
                "message": "The 'gpt-5.3-codex' model is not supported when using "
                "Codex with a ChatGPT account.",
            }
        )
        with (
            patch("models.CODEX_AVAILABLE", True),
            patch("models.CODEX_PATH", "/usr/bin/codex"),
            patch("models.subprocess.run", return_value=self._fake_proc(stdout)),
        ):
            with pytest.raises(RuntimeError, match="not supported when using Codex"):
                models.call_codex_model("sys", "user", "codex/gpt-5.3-codex")

    def test_turn_failed_nested_error_message(self):
        stdout = json.dumps(
            {"type": "turn.failed", "error": {"message": "model_not_found: gpt-9"}}
        )
        with (
            patch("models.CODEX_AVAILABLE", True),
            patch("models.CODEX_PATH", "/usr/bin/codex"),
            patch("models.subprocess.run", return_value=self._fake_proc(stdout)),
        ):
            with pytest.raises(RuntimeError, match="model_not_found"):
                models.call_codex_model("sys", "user", "codex/gpt-9")

    def test_structured_error_preferred_over_stderr_noise(self):
        stdout = json.dumps({"type": "error", "message": "invalid_request_error: bad model"})
        proc = self._fake_proc(stdout, returncode=1, stderr="DeprecationWarning: noise")
        with (
            patch("models.CODEX_AVAILABLE", True),
            patch("models.CODEX_PATH", "/usr/bin/codex"),
            patch("models.subprocess.run", return_value=proc),
        ):
            with pytest.raises(RuntimeError, match="invalid_request_error"):
                models.call_codex_model("sys", "user", "codex/bogus")

    def test_success_path_unaffected(self):
        lines = [
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "review text"},
                }
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 11, "output_tokens": 7},
                }
            ),
        ]
        with (
            patch("models.CODEX_AVAILABLE", True),
            patch("models.CODEX_PATH", "/usr/bin/codex"),
            patch("models.subprocess.run", return_value=self._fake_proc("\n".join(lines))),
        ):
            text, inp, out = models.call_codex_model("sys", "user", "codex/gpt-5.5")
        assert (text, inp, out) == ("review text", 11, 7)


class TestBedrockErrorRewrite:
    def test_access_denied_keeps_original_and_appends_hint(self):
        def raise_access_denied(**kwargs):
            raise RuntimeError(
                "AccessDeniedException: You don't have access to the model with the specified model ID."
            )

        with (
            patch("models.completion", side_effect=raise_access_denied),
            patch("models.time.sleep") as mock_sleep,
        ):
            result = models.call_single_model(
                "claude-opus-5", "sys", "user", "investor", "progress",
                bedrock_mode=True,
            )
        mock_sleep.assert_not_called()  # classifier fast-fails on the hint phrase
        assert "AccessDeniedException" in result.error  # original preserved
        assert "model not enabled in your Bedrock account" in result.error

    def test_validation_exception_keeps_original_and_appends_hint(self):
        def raise_validation(**kwargs):
            raise RuntimeError("ValidationException: malformed input request")

        with (
            patch("models.completion", side_effect=raise_validation),
            patch("models.time.sleep") as mock_sleep,
        ):
            result = models.call_single_model(
                "bogus.model", "sys", "user", "investor", "progress",
                bedrock_mode=True,
            )
        mock_sleep.assert_not_called()
        assert "ValidationException" in result.error
        assert "invalid Bedrock model ID" in result.error


class TestInvalidModelMessaging:
    def test_antigravity_message_mentions_agy_not_credentials(self):
        lines = providers.describe_invalid_models(["antigravity/gemini-3.1-pro-high"])
        assert len(lines) == 1
        assert "agy" in lines[0]
        assert "credentials" not in lines[0].lower()
        assert "api key" not in lines[0].lower()

    def test_codex_message_mentions_install(self):
        lines = providers.describe_invalid_models(["codex/gpt-5.5"])
        assert "npm install -g @openai/codex" in lines[0]

    def test_api_key_message_for_litellm_models(self):
        lines = providers.describe_invalid_models(["zai/glm-5.2"])
        assert "missing API key" in lines[0]


class TestReasoningModelDetection:
    def test_kimi_k3_is_reasoning(self):
        assert is_reasoning_model("moonshot/kimi-k3")
        assert is_reasoning_model("moonshot/kimi-k2.5")

    def test_kimi_k2_is_not_reasoning(self):
        assert not is_reasoning_model("moonshot/kimi-k2")
