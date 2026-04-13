"""Provider configuration, cost tracking, and profile management for crucible.

Reuses the same LLM provider infrastructure as spec-debate.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

PROFILES_DIR = Path.home() / ".config" / "crucible" / "profiles"
GLOBAL_CONFIG_PATH = Path.home() / ".claude" / "crucible" / "config.json"

# Use LiteLLM's community-maintained model cost registry at runtime.
try:
    from litellm import model_cost as _litellm_model_cost
except ImportError:
    _litellm_model_cost = {}

# CLI tools aren't in LiteLLM's registry (subscription/account-based)
_CLI_COSTS = {
    "codex/": {"input": 0.0, "output": 0.0},
    "gemini-cli/": {"input": 0.0, "output": 0.0},
}

DEFAULT_COST = {"input": 5.00, "output": 15.00}


def get_model_cost(model: str) -> dict[str, float]:
    """Get cost per 1M tokens for a model, using LiteLLM's registry."""
    for prefix, cost in _CLI_COSTS.items():
        if model.startswith(prefix):
            return cost

    litellm_key = model.split("/", 1)[1] if "/" in model and model.split("/")[0] in (
        "gemini", "xai", "mistral", "groq", "deepseek", "openrouter"
    ) else model
    for key in (model, litellm_key):
        if key in _litellm_model_cost:
            entry = _litellm_model_cost[key]
            return {
                "input": entry.get("input_cost_per_token", 0) * 1_000_000,
                "output": entry.get("output_cost_per_token", 0) * 1_000_000,
            }

    return DEFAULT_COST


# Check CLI tool availability
CODEX_PATH = shutil.which("codex")
GEMINI_CLI_PATH = shutil.which("gemini")
CODEX_AVAILABLE = CODEX_PATH is not None
GEMINI_CLI_AVAILABLE = GEMINI_CLI_PATH is not None

DEFAULT_CODEX_REASONING = "xhigh"

# Bedrock model mapping
BEDROCK_MODEL_MAP = {
    "claude-sonnet-4.6": "anthropic.claude-sonnet-4-6-20250627-v1:0",
    "claude-opus-4.6": "anthropic.claude-opus-4-6-20250627-v1:0",
    "claude-sonnet-4": "anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-opus-4": "anthropic.claude-opus-4-20250514-v1:0",
    "claude-haiku-4.5": "anthropic.claude-haiku-4-5-20251001-v1:0",
    "llama-3-8b": "meta.llama3-8b-instruct-v1:0",
    "llama-3-70b": "meta.llama3-70b-instruct-v1:0",
    "llama-3.1-70b": "meta.llama3-1-70b-instruct-v1:0",
    "llama-3.1-405b": "meta.llama3-1-405b-instruct-v1:0",
    "mistral-large": "mistral.mistral-large-2402-v1:0",
    "mixtral-8x7b": "mistral.mixtral-8x7b-instruct-v0:1",
}


def load_global_config() -> dict:
    """Load global config from ~/.claude/crucible/config.json."""
    if not GLOBAL_CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(GLOBAL_CONFIG_PATH.read_text())
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in global config: {e}", file=sys.stderr)
        return {}


def save_global_config(config: dict):
    """Save global config."""
    GLOBAL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    GLOBAL_CONFIG_PATH.write_text(json.dumps(config, indent=2))


def is_bedrock_enabled() -> bool:
    config = load_global_config()
    return config.get("bedrock", {}).get("enabled", False)


def get_bedrock_config() -> dict:
    config = load_global_config()
    return config.get("bedrock", {})


def resolve_bedrock_model(friendly_name: str, config: Optional[dict] = None) -> Optional[str]:
    if "." in friendly_name and not friendly_name.startswith("bedrock/"):
        return friendly_name
    if friendly_name in BEDROCK_MODEL_MAP:
        return BEDROCK_MODEL_MAP[friendly_name]
    if config is None:
        config = get_bedrock_config()
    custom_aliases = config.get("custom_aliases", {})
    if friendly_name in custom_aliases:
        return custom_aliases[friendly_name]
    return None


def validate_bedrock_models(models: list[str], config: Optional[dict] = None) -> tuple[list[str], list[str]]:
    if config is None:
        config = get_bedrock_config()
    available = config.get("available_models", [])
    valid, invalid = [], []
    for model in models:
        if model in available:
            resolved = resolve_bedrock_model(model, config)
            if resolved:
                valid.append(resolved)
            else:
                invalid.append(model)
        else:
            resolved = resolve_bedrock_model(model, config)
            if resolved:
                for avail in available:
                    if resolve_bedrock_model(avail, config) == resolved:
                        valid.append(resolved)
                        break
                else:
                    invalid.append(model)
            else:
                invalid.append(model)
    return valid, invalid


def get_available_providers() -> list[tuple[str, Optional[str], str]]:
    """Get list of providers with configured API keys."""
    providers = [
        ("OpenAI", "OPENAI_API_KEY", "gpt-5.4"),
        ("Anthropic", "ANTHROPIC_API_KEY", "claude-opus-4-6"),
        ("Google", "GEMINI_API_KEY", "gemini/gemini-3.1-pro-preview"),
        ("xAI", "XAI_API_KEY", "xai/grok-4.20-0309-reasoning"),
        ("Mistral", "MISTRAL_API_KEY", "mistral/mistral-large"),
        ("Groq", "GROQ_API_KEY", "groq/llama-3.3-70b-versatile"),
        ("OpenRouter", "OPENROUTER_API_KEY", "openrouter/openai/gpt-5.2-pro"),
        ("Deepseek", "DEEPSEEK_API_KEY", "deepseek/deepseek-chat"),
        ("ZAI (GLM)", "ZAI_API_KEY", "zai/glm-5.1"),
        ("Moonshot (Kimi)", "MOONSHOT_API_KEY", "moonshot/kimi-k2.5"),
    ]

    available: list[tuple[str, Optional[str], str]] = []
    for name, key, model in providers:
        if os.environ.get(key):
            available.append((name, key, model))

    if CODEX_AVAILABLE:
        available.append(("Codex CLI", None, "codex/gpt-5.3-codex"))
    if GEMINI_CLI_AVAILABLE:
        available.append(("Gemini CLI", None, "gemini-cli/gemini-3.1-pro-preview"))

    return available


def get_default_model() -> Optional[str]:
    config = get_bedrock_config()
    if config.get("enabled"):
        available_models = config.get("available_models", [])
        if available_models:
            return available_models[0]
    available = get_available_providers()
    if available:
        return available[0][2]
    return None


def validate_model_credentials(models: list[str]) -> tuple[list[str], list[str]]:
    """Validate that API keys are available for requested models."""
    bedrock_config = get_bedrock_config()
    if bedrock_config.get("enabled"):
        return validate_bedrock_models(models, bedrock_config)

    valid, invalid = [], []
    provider_map = {
        "gpt-": "OPENAI_API_KEY",
        "o1": "OPENAI_API_KEY",
        "o3": "OPENAI_API_KEY",
        "o4": "OPENAI_API_KEY",
        "claude-": "ANTHROPIC_API_KEY",
        "gemini/": "GEMINI_API_KEY",
        "xai/": "XAI_API_KEY",
        "foundry/": "AZURE_AI_API_KEY",
        "mistral/": "MISTRAL_API_KEY",
        "groq/": "GROQ_API_KEY",
        "deepseek/": "DEEPSEEK_API_KEY",
        "zai/": "ZAI_API_KEY",
        "moonshot/": "MOONSHOT_API_KEY",
        "codex/": None,
        "gemini-cli/": None,
    }

    for model in models:
        if model.startswith("codex/"):
            (valid if CODEX_AVAILABLE else invalid).append(model)
            continue
        if model.startswith("gemini-cli/"):
            (valid if GEMINI_CLI_AVAILABLE else invalid).append(model)
            continue
        required_key = None
        for prefix, key in provider_map.items():
            if model.startswith(prefix):
                required_key = key
                break
        if required_key is None:
            valid.append(model)
            continue
        if os.environ.get(required_key):
            valid.append(model)
        else:
            invalid.append(model)

    return valid, invalid


def list_providers():
    """List all supported providers and their API key status."""
    providers = [
        ("OpenAI", "OPENAI_API_KEY", "gpt-5.4, gpt-5.4-pro, o3-pro, o4-mini"),
        ("Anthropic", "ANTHROPIC_API_KEY", "claude-opus-4-6, claude-sonnet-4-6"),
        ("Google", "GEMINI_API_KEY", "gemini/gemini-3.1-pro-preview, gemini/gemini-2.5-pro"),
        ("xAI", "XAI_API_KEY", "xai/grok-4.20-0309-reasoning, xai/grok-4-0709"),
        ("Azure AI", "AZURE_AI_API_KEY", "foundry/<deployment-name>"),
        ("Mistral", "MISTRAL_API_KEY", "mistral/mistral-large, mistral/codestral"),
        ("Groq", "GROQ_API_KEY", "groq/llama-3.3-70b-versatile"),
        ("OpenRouter", "OPENROUTER_API_KEY", "openrouter/openai/gpt-5.2-pro"),
        ("Deepseek", "DEEPSEEK_API_KEY", "deepseek/deepseek-chat"),
        ("ZAI (GLM)", "ZAI_API_KEY", "zai/glm-5.1, zai/glm-5-turbo"),
        ("Moonshot", "MOONSHOT_API_KEY", "moonshot/kimi-k2.5"),
    ]

    print("Supported providers:\n")
    for name, key, models in providers:
        status = "[set]" if os.environ.get(key) else "[not set]"
        print(f"  {name:12} {key:24} {status}")
        print(f"             Example models: {models}")
        print()

    codex_status = "[installed]" if CODEX_AVAILABLE else "[not installed]"
    print(f"  {'Codex CLI':12} {'(ChatGPT subscription)':24} {codex_status}")
    print("             Example models: codex/gpt-5.3-codex")
    print()

    gemini_cli_status = "[installed]" if GEMINI_CLI_AVAILABLE else "[not installed]"
    print(f"  {'Gemini CLI':12} {'(Google account)':24} {gemini_cli_status}")
    print("             Example models: gemini-cli/gemini-3.1-pro-preview")
    print()


def load_profile(profile_name: str) -> dict:
    """Load a saved review profile by name."""
    profile_path = PROFILES_DIR / f"{profile_name}.json"
    if not profile_path.exists():
        print(f"Error: Profile '{profile_name}' not found", file=sys.stderr)
        sys.exit(2)
    try:
        return json.loads(profile_path.read_text())
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in profile '{profile_name}': {e}", file=sys.stderr)
        sys.exit(2)


def save_profile(profile_name: str, config: dict):
    """Save a review profile to disk."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = PROFILES_DIR / f"{profile_name}.json"
    profile_path.write_text(json.dumps(config, indent=2))
    print(f"Profile saved to {profile_path}")


def list_profiles():
    """List all saved profiles."""
    print("Saved Profiles:\n")
    if not PROFILES_DIR.exists():
        print("  No profiles found.")
        return
    profiles = list(PROFILES_DIR.glob("*.json"))
    if not profiles:
        print("  No profiles found.")
        return
    for p in sorted(profiles):
        try:
            config = json.loads(p.read_text())
            print(f"  {p.stem}")
            print(f"    models: {config.get('models', 'not set')}")
            print(f"    dimensions: {config.get('dimensions', 'all')}")
            print()
        except Exception:
            print(f"  {p.stem} [error reading]")
