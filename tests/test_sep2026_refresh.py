"""Tests for the September 2026 model refresh: new defaults, gpt-6-astra and
Fable 5.1 handling, and live Anthropic / gpt-6 model discovery."""

from __future__ import annotations

import argparse
import io
import json
from unittest.mock import patch

import crucible
import models
import providers


def test_xai_default_is_grok_47(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "x")
    defaults = {name: model for name, _, model in providers.get_available_providers()}
    assert defaults["xAI"] == "xai/grok-4.7"


def test_fable_51_omits_temperature():
    assert models.claude_version("claude-fable-5-1") == (5, 1)
    assert models.is_reasoning_model("claude-fable-5-1")
    assert not models.uses_max_completion_tokens("claude-fable-5-1")


def test_gpt6_astra_is_reasoning_model():
    assert models.is_reasoning_model("gpt-6-astra")
    assert models.uses_max_completion_tokens("gpt-6-astra")
    assert models.is_reasoning_model("codex/gpt-6-astra")
    assert "gpt-6-astra" in providers.CODEX_CHATGPT_MODELS


def _only(monkeypatch, var):
    for v in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY",
              "ZAI_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY", "DEEPSEEK_API_KEY",
              "MOONSHOT_API_KEY", "MINIMAX_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv(var, "k")
    monkeypatch.setattr(providers, "ANTIGRAVITY_AVAILABLE", False)


def test_discover_models_anthropic_is_live(monkeypatch, capsys):
    _only(monkeypatch, "ANTHROPIC_API_KEY")
    body = json.dumps({"data": [{"id": "claude-fable-5-1"}, {"id": "claude-opus-5"}]})

    def fake_urlopen(req, timeout=10):
        assert req.full_url.startswith("https://api.anthropic.com/v1/models")
        assert req.get_header("X-api-key") == "k"
        return io.BytesIO(body.encode())

    with patch("urllib.request.urlopen", fake_urlopen):
        assert crucible.cmd_discover_models(argparse.Namespace()) == 0
    out = capsys.readouterr().out
    assert "Anthropic:" in out
    assert "- claude-fable-5-1" in out and "- claude-opus-5" in out


def test_discover_models_openai_includes_gpt6(monkeypatch, capsys):
    _only(monkeypatch, "OPENAI_API_KEY")
    body = json.dumps({"data": [
        {"id": "gpt-6-astra"}, {"id": "gpt-5.6-sol"}, {"id": "gpt-6-astra-realtime"}, {"id": "dall-e-3"},
    ]})
    with patch("urllib.request.urlopen", lambda req, timeout=10: io.BytesIO(body.encode())):
        assert crucible.cmd_discover_models(argparse.Namespace()) == 0
    out = capsys.readouterr().out
    assert "- gpt-6-astra\n" in out and "- gpt-5.6-sol" in out
    assert "realtime" not in out and "dall-e" not in out


def test_codex_call_uses_sandbox_flag_not_full_auto(monkeypatch):
    # codex-cli 0.154 removed --full-auto; every codex/ call failed with
    # "unexpected argument '--full-auto' found" until this was fixed.
    monkeypatch.setattr(models, "CODEX_AVAILABLE", True)
    monkeypatch.setattr(models, "CODEX_PATH", "/bin/codex")
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        class R:
            returncode = 0
            stdout = json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "OK"}}) + "\n"
            stderr = ""
        return R()

    with patch("subprocess.run", fake_run):
        models.call_codex_model("sys", "user", "codex/gpt-6-astra")
    assert "--full-auto" not in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("--sandbox") + 1] == "workspace-write"
    assert "gpt-6-astra" in seen["cmd"]
