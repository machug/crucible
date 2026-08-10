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
    "antigravity": {"input": 0.0, "output": 0.0},
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
ANTIGRAVITY_PATH = shutil.which("agy")
CODEX_AVAILABLE = CODEX_PATH is not None
GEMINI_CLI_AVAILABLE = GEMINI_CLI_PATH is not None
ANTIGRAVITY_AVAILABLE = ANTIGRAVITY_PATH is not None

DEFAULT_CODEX_REASONING = "xhigh"

# Models Codex CLI serves when authenticated with a ChatGPT account (not an
# API key). Rotates with OpenAI's ChatGPT lineup — see
# https://developers.openai.com/codex/models. Last verified 2026-08-10.
CODEX_CHATGPT_MODELS = {
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.4",  # retires 2026-08-31
    "gpt-5.4-mini",  # retires 2026-08-31
    "gpt-5.3-codex-spark",  # ChatGPT Pro only
}


def codex_auth_mode() -> Optional[str]:
    """Return Codex CLI auth mode ("chatgpt" or "apikey") or None if unknown."""
    auth_path = Path.home() / ".codex" / "auth.json"
    try:
        data = json.loads(auth_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    mode = data.get("auth_mode")
    if mode:
        return mode
    if data.get("OPENAI_API_KEY"):
        return "apikey"
    if data.get("tokens"):
        return "chatgpt"
    return None


def warn_codex_chatgpt_model_support(models: list[str]) -> None:
    """Warn upfront when a codex/ model won't work with ChatGPT-account auth.

    ChatGPT-account Codex serves only the current ChatGPT lineup; other models
    (gpt-5.3-codex, gpt-5.5-pro, ...) hard-fail with a 400. The supported set
    rotates, so this warns rather than blocks.
    """
    codex_models = [
        m.split("/", 1)[1] for m in models if m.startswith("codex/") and "/" in m
    ]
    if not codex_models or codex_auth_mode() != "chatgpt":
        return
    unsupported = [m for m in codex_models if m not in CODEX_CHATGPT_MODELS]
    if unsupported:
        print(
            f"Warning: Codex CLI is authenticated with a ChatGPT account, which "
            f"likely rejects: {', '.join(unsupported)}. ChatGPT-account models "
            f"(as of 2026-08): gpt-5.6-sol, gpt-5.6-terra, gpt-5.6-luna, gpt-5.5. "
            f"Other models need Codex API-key auth or the OPENAI_API_KEY route.\n",
            file=sys.stderr,
        )


# Bedrock model mapping
BEDROCK_MODEL_MAP = {
    # Note: claude-fable-5 requires opting into data sharing via Bedrock's
    # Data Retention API (provider_data_share) before invocation succeeds.
    "claude-fable-5": "anthropic.claude-fable-5",
    "claude-opus-5": "anthropic.claude-opus-5",
    "claude-sonnet-5": "anthropic.claude-sonnet-5",
    "claude-opus-4.7": "anthropic.claude-opus-4-7-20260416-v1:0",
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
        ("OpenAI", "OPENAI_API_KEY", "gpt-5.6-sol"),
        ("Anthropic", "ANTHROPIC_API_KEY", "claude-opus-5"),
        ("Google", "GEMINI_API_KEY", "gemini/gemini-3.1-pro-preview"),
        ("xAI", "XAI_API_KEY", "xai/grok-4.5"),
        ("Mistral", "MISTRAL_API_KEY", "mistral/mistral-large"),
        ("Groq", "GROQ_API_KEY", "groq/llama-3.3-70b-versatile"),
        ("OpenRouter", "OPENROUTER_API_KEY", "openrouter/openai/gpt-5.5-pro"),
        ("Deepseek", "DEEPSEEK_API_KEY", "deepseek/deepseek-v4-pro"),
        ("ZAI (GLM)", "ZAI_API_KEY", "zai/glm-5.2"),
        ("Moonshot (Kimi)", "MOONSHOT_API_KEY", "moonshot/kimi-k3"),
        ("MiniMax", "MINIMAX_API_KEY", "minimax/MiniMax-M3"),
    ]

    available: list[tuple[str, Optional[str], str]] = []
    for name, key, model in providers:
        if os.environ.get(key):
            available.append((name, key, model))

    if CODEX_AVAILABLE:
        available.append(("Codex CLI", None, "codex/gpt-5.6-sol"))
    # Antigravity CLI is Gemini CLI's successor (consumer gemini-cli retired
    # 2026-06-18); gemini-cli/ still works if requested explicitly but is no
    # longer auto-selected.
    if ANTIGRAVITY_AVAILABLE:
        available.append(("Antigravity CLI", None, "antigravity/gemini-3.1-pro-high"))

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
        "minimax/": "MINIMAX_API_KEY",
        "codex/": None,
        "gemini-cli/": None,
        "antigravity/": None,  # Uses Google account via agy CLI
    }

    for model in models:
        if model.startswith("codex/"):
            (valid if CODEX_AVAILABLE else invalid).append(model)
            continue
        if model == "antigravity" or model.startswith("antigravity/"):
            (valid if ANTIGRAVITY_AVAILABLE else invalid).append(model)
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


def describe_invalid_models(invalid: list[str]) -> list[str]:
    """Human-actionable failure lines for models that failed validation.

    CLI providers (codex, antigravity, gemini-cli) have no API key — telling
    the user "missing credentials" sends them to the wrong remedy.
    """
    lines = []
    for m in invalid:
        if m.startswith("codex/"):
            lines.append(
                f"{m}: Codex CLI not installed. Install: npm install -g @openai/codex && codex login"
            )
        elif m == "antigravity" or m.startswith("antigravity/"):
            lines.append(
                f"{m}: Antigravity CLI (agy) not installed. Install: "
                f"curl -fsSL https://antigravity.google/cli/install.sh | bash "
                f"— then run `agy` once to sign in."
            )
        elif m.startswith("gemini-cli/"):
            lines.append(
                f"{m}: Gemini CLI not installed (consumer service retired 2026-06-18). "
                f"Use antigravity/<model> or gemini/<model> instead."
            )
        else:
            lines.append(f"{m}: missing API key (run 'crucible.py providers' for setup)")
    return lines


def list_providers():
    """List all supported providers and their API key status."""
    providers = [
        ("OpenAI", "OPENAI_API_KEY", "gpt-5.6-sol, gpt-5.6-terra, gpt-5.6-luna, gpt-5.5, gpt-5.5-pro"),
        ("Anthropic", "ANTHROPIC_API_KEY", "claude-fable-5, claude-opus-5, claude-sonnet-5, claude-haiku-4-5"),
        ("Google", "GEMINI_API_KEY", "gemini/gemini-3.1-pro-preview, gemini/gemini-3.6-flash"),
        ("xAI", "XAI_API_KEY", "xai/grok-4.5, xai/grok-4.3, xai/grok-4.20-0309-reasoning"),
        ("Azure AI", "AZURE_AI_API_KEY", "foundry/<deployment-name>"),
        ("Mistral", "MISTRAL_API_KEY", "mistral/mistral-large, mistral/codestral"),
        ("Groq", "GROQ_API_KEY", "groq/llama-3.3-70b-versatile"),
        ("OpenRouter", "OPENROUTER_API_KEY", "openrouter/openai/gpt-5.5-pro, openrouter/anthropic/claude-opus-5"),
        ("Deepseek", "DEEPSEEK_API_KEY", "deepseek/deepseek-v4-pro, deepseek/deepseek-v4-flash"),
        ("ZAI (GLM)", "ZAI_API_KEY", "zai/glm-5.2, zai/glm-5.1, zai/glm-5-turbo"),
        ("Moonshot", "MOONSHOT_API_KEY", "moonshot/kimi-k3, moonshot/kimi-k2.7-code"),
        ("MiniMax", "MINIMAX_API_KEY", "minimax/MiniMax-M3, minimax/MiniMax-M2.7"),
    ]

    print("Supported providers:\n")
    for name, key, models in providers:
        status = "[set]" if os.environ.get(key) else "[not set]"
        print(f"  {name:12} {key:24} {status}")
        print(f"             Example models: {models}")
        print()

    codex_status = "[installed]" if CODEX_AVAILABLE else "[not installed]"
    auth_mode = codex_auth_mode()
    print(f"  {'Codex CLI':12} {'(ChatGPT subscription)':24} {codex_status}")
    if auth_mode:
        print(f"             Auth mode: {auth_mode}")
    print("             Example models: codex/gpt-5.6-sol, codex/gpt-5.6-terra, codex/gpt-5.5")
    print("             Note: ChatGPT-account auth serves only the ChatGPT lineup (gpt-5.6-sol/terra/luna,")
    print("                   gpt-5.5). gpt-5.3-codex and gpt-5.5-pro need API-key auth or OPENAI_API_KEY.")
    print()

    agy_status = "[installed]" if ANTIGRAVITY_AVAILABLE else "[not installed]"
    print(f"  {'Antigravity':12} {'(Google account)':24} {agy_status}")
    print("             Example models: antigravity/gemini-3.6-flash-high, antigravity/gemini-3.1-pro-high,")
    print("             antigravity/claude-sonnet-4-6, antigravity/gpt-oss-120b-medium (`agy models` lists all)")
    print("             Install: curl -fsSL https://antigravity.google/cli/install.sh | bash")
    print("             Auth: run `agy` once interactively (Google sign-in), then headless works")
    print()

    gemini_cli_status = "[installed]" if GEMINI_CLI_AVAILABLE else "[not installed]"
    print(f"  {'Gemini CLI':12} {'(RETIRED 2026-06-18)':24} {gemini_cli_status}")
    print("             Consumer service ended; enterprise licenses only. Use antigravity/ or gemini/ instead.")
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
