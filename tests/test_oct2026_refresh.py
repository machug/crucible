"""Tests for the October 2026 model refresh: Gemini 4 Argon discovery, the
GPT-6 Sol/Luna lineup, Claude 5.5 temperature handling, new defaults, Codex
max/ultra effort, and the gpt-5.5 retirement from ChatGPT-account Codex."""

from __future__ import annotations

import argparse
import io
import json
import sys
from unittest.mock import patch

import pytest

import crucible
import models
import providers


def _only(monkeypatch, var):
    for v in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY",
              "ZAI_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY", "DEEPSEEK_API_KEY",
              "MOONSHOT_API_KEY", "MINIMAX_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv(var, "k")
    monkeypatch.setattr(providers, "ANTIGRAVITY_AVAILABLE", False)


def test_discover_models_gemini_includes_gemini4(monkeypatch, capsys):
    _only(monkeypatch, "GEMINI_API_KEY")
    body = json.dumps({"models": [
        {"name": "models/gemini-4-argon"},
        {"name": "models/gemini-3.8-flash"},
        {"name": "models/gemma-4-31b-it"},
    ]})
    with patch("urllib.request.urlopen", lambda req, timeout=10: io.BytesIO(body.encode())):
        assert crucible.cmd_discover_models(argparse.Namespace()) == 0
    out = capsys.readouterr().out
    assert "- gemini/gemini-4-argon" in out
    assert "gemma" not in out


def test_gemini4_argon_keeps_temperature():
    # verified 2026-10-14: gemini-4-argon accepts temperature and max_tokens
    assert not models.is_reasoning_model("gemini/gemini-4-argon")


def test_gpt6_sol_luna_are_reasoning_models():
    for m in ("gpt-6.1-sol", "gpt-6-sol", "gpt-6-luna", "codex/gpt-6.1-sol"):
        assert models.is_reasoning_model(m)
        assert models.uses_max_completion_tokens(m)


def test_claude_55_omits_temperature():
    for m in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"):
        assert models.claude_version(m) == (5, 5)
        assert models.is_reasoning_model(m)
        assert not models.uses_max_completion_tokens(m)


def test_codex_chatgpt_lineup():
    for m in ("gpt-6.1-sol", "gpt-6-sol", "gpt-6-luna", "gpt-6-astra"):
        assert m in providers.CODEX_CHATGPT_MODELS
    assert "gpt-5.5" not in providers.CODEX_CHATGPT_MODELS


def test_new_defaults(monkeypatch):
    for v in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(v, "k")
    monkeypatch.setattr(providers, "CODEX_AVAILABLE", True)
    monkeypatch.setattr(providers, "ANTIGRAVITY_AVAILABLE", True)
    defaults = {name: model for name, _, model in providers.get_available_providers()}
    assert defaults["OpenAI"] == "gpt-6.1-sol"
    assert defaults["Anthropic"] == "claude-opus-5-5"
    assert defaults["Google"] == "gemini/gemini-4-argon"
    assert defaults["Codex CLI"] == "codex/gpt-6.1-sol"
    assert defaults["Antigravity CLI"] == "antigravity/gemini-3.8-flash-high"


def test_codex_reasoning_levels(monkeypatch):
    assert providers.CODEX_REASONING_LEVELS[-2:] == ("max", "ultra")
    assert "minimal" not in providers.CODEX_REASONING_LEVELS
    assert providers.DEFAULT_CODEX_REASONING == "xhigh"
    monkeypatch.setattr(sys, "argv", ["crucible.py", "review", "--models", "x", "--codex-reasoning", "minimal"])
    with pytest.raises(SystemExit) as exc:
        crucible.main()
    assert exc.value.code == 2
