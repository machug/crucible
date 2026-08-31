"""Model calling, cost tracking, and response handling for crucible."""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

os.environ["LITELLM_LOG"] = "ERROR"

try:
    import litellm
    from litellm import completion

    litellm.suppress_debug_info = True
except ImportError:
    print(
        "Error: litellm package not installed. Run: pip install litellm",
        file=sys.stderr,
    )
    sys.exit(1)

from providers import (
    ANTIGRAVITY_AVAILABLE,
    ANTIGRAVITY_PATH,
    CODEX_AVAILABLE,
    CODEX_PATH,
    DEFAULT_CODEX_REASONING,
    GEMINI_CLI_AVAILABLE,
    GEMINI_CLI_PATH,
    get_model_cost,
)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0

# Error substrings that retrying cannot fix: bad model id, wrong auth mode,
# rejected/revoked credentials. These are deterministic 4xx-class failures —
# retrying just burns time and spams warnings.
NON_RETRYABLE_PATTERNS = (
    "not supported when using codex with a chatgpt account",
    "invalid_request_error",
    "model_not_found",
    "does not exist or you do not have access",
    "authenticationerror",
    "invalid api key",
    "incorrect api key",
    "notfounderror",
    # Antigravity CLI deterministic failures
    "is not authenticated",
    "invalid model selection",
    # Bedrock messages rewritten in the retry loop below
    "model not enabled in your bedrock account",
    "invalid bedrock model id",
)

CODEX_CHATGPT_HINT = (
    "Codex is authenticated with a ChatGPT account, which only serves: "
    "gpt-5.6-sol, gpt-5.6-terra, gpt-5.6-luna, gpt-5.5 "
    "(gpt-5.4/-mini retire 2026-08-31; gpt-5.3-codex-spark needs ChatGPT Pro). "
    "For other models authenticate Codex with an API key or use the "
    "OPENAI_API_KEY litellm route (e.g. --models gpt-5.5-pro)."
)


def is_non_retryable_error(error_msg: str) -> bool:
    """Whether an error is deterministic (4xx-class) and not worth retrying."""
    lower = error_msg.lower()
    return any(p in lower for p in NON_RETRYABLE_PATTERNS)


# Anthropic models from this version up reject any temperature but 1
# (verified 2026-08-31: claude-opus-4-7/-4-8, claude-opus-5, claude-sonnet-5 and
# claude-fable-5 all raise UnsupportedParamsError on temperature=0.7; sonnet-4-6,
# opus-4-6 and haiku-4-5 still accept it).
CLAUDE_FIXED_TEMPERATURE_FROM = (4, 7)

# Matches "claude-opus-5", "claude-opus-4-8", "claude-sonnet-4-6-20250627-v1:0",
# "anthropic.claude-opus-4-7-...", "antigravity/claude-sonnet-4-6". Deliberately
# does NOT match the legacy "claude-3-5-sonnet" ordering, which is pre-4.7.
_CLAUDE_VERSION_RE = re.compile(r"claude-(?:opus|sonnet|haiku|fable)-(\d+)(?:[-.](\d+))?")


def claude_version(model: str) -> Optional[tuple[int, int]]:
    """Return (major, minor) for a Claude model id, or None if not one."""
    m = _CLAUDE_VERSION_RE.search(model.lower())
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2) or 0))


def is_reasoning_model(model: str) -> bool:
    """Check if a model is a reasoning model (o-series, gpt-5, Claude 4.7+)."""
    model_lower = model.lower()
    if model_lower.startswith(("o1", "o3", "o4")) or "/o1" in model_lower or "/o3" in model_lower or "/o4" in model_lower:
        return True
    if "gpt-5" in model_lower:
        return True
    if "xai/" in model_lower and model_lower.endswith("-reasoning") and not model_lower.endswith("-non-reasoning"):
        return True
    # Moonshot Kimi reasoning models (kimi-k2.5 and later reject temperature,
    # only allow 1). Anchor on the version segment after "kimi-k" so arbitrary
    # "k3" substrings elsewhere in a model id don't match.
    if "moonshot/" in model_lower:
        m = re.search(r"kimi-k(\d+(?:\.\d+)?)", model_lower)
        if m and float(m.group(1)) >= 2.5:
            return True
    # Anthropic Claude 4.7 and newer only accept temperature=1
    version = claude_version(model_lower)
    if version and version >= CLAUDE_FIXED_TEMPERATURE_FROM:
        return True
    return False


def uses_max_completion_tokens(model: str) -> bool:
    """Check if a model uses max_completion_tokens instead of max_tokens."""
    if not is_reasoning_model(model):
        return False
    if model.lower().startswith(("xai/", "moonshot/")):
        return False
    # Anthropic takes max_tokens, not max_completion_tokens
    if claude_version(model):
        return False
    return True


@dataclass
class ReviewResponse:
    """Response from a model review."""

    model: str
    persona: str
    dimension: str
    response: str
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


@dataclass
class CostTracker:
    """Track token usage and costs across model calls."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    by_model: dict = field(default_factory=dict)

    def add(self, model: str, input_tokens: int, output_tokens: int) -> float:
        costs = get_model_cost(model)
        cost = (input_tokens / 1_000_000 * costs["input"]) + (
            output_tokens / 1_000_000 * costs["output"]
        )
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        if model not in self.by_model:
            self.by_model[model] = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
        self.by_model[model]["input_tokens"] += input_tokens
        self.by_model[model]["output_tokens"] += output_tokens
        self.by_model[model]["cost"] += cost
        return cost

    def summary(self) -> str:
        lines = ["", "=== Cost Summary ==="]
        lines.append(
            f"Total tokens: {self.total_input_tokens:,} in / {self.total_output_tokens:,} out"
        )
        lines.append(f"Total cost: ${self.total_cost:.4f}")
        if len(self.by_model) > 1:
            lines.append("")
            lines.append("By model:")
            for model, data in self.by_model.items():
                lines.append(
                    f"  {model}: ${data['cost']:.4f} ({data['input_tokens']:,} in / {data['output_tokens']:,} out)"
                )
        return "\n".join(lines)


cost_tracker = CostTracker()


def call_foundry_model(
    system_prompt: str, user_message: str, model: str, timeout: int = 600,
) -> tuple[str, int, int]:
    """Call Azure AI Foundry using the azure-ai-inference SDK."""
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import SystemMessage, UserMessage
    from azure.core.credentials import AzureKeyCredential

    api_key = os.environ.get("AZURE_AI_API_KEY")
    api_base = os.environ.get("AZURE_AI_API_BASE", "")
    if not api_key:
        raise ValueError("AZURE_AI_API_KEY environment variable not set")

    endpoint = api_base.rstrip("/")
    if not endpoint.endswith("/models"):
        parts = endpoint.split(".services.ai.azure.com")
        if len(parts) == 2:
            endpoint = parts[0] + ".services.ai.azure.com/models"
        else:
            endpoint = endpoint + "/models"

    deployment_name = model.split("/", 1)[1] if "/" in model else model
    client = ChatCompletionsClient(
        endpoint=endpoint, credential=AzureKeyCredential(api_key),
    )
    response = client.complete(
        messages=[SystemMessage(content=system_prompt), UserMessage(content=user_message)],
        model=deployment_name,
    )
    content = response.choices[0].message.content or ""
    input_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.completion_tokens if response.usage else 0
    return content, input_tokens, output_tokens


def call_codex_model(
    system_prompt: str, user_message: str, model: str,
    reasoning_effort: str = DEFAULT_CODEX_REASONING, timeout: int = 600,
) -> tuple[str, int, int]:
    """Call Codex CLI in headless mode."""
    if not CODEX_AVAILABLE:
        raise RuntimeError("Codex CLI not found. Install with: npm install -g @openai/codex")

    actual_model = model.split("/", 1)[1] if "/" in model else model
    full_prompt = f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\nUSER REQUEST:\n{user_message}"

    try:
        cmd = [
            CODEX_PATH, "exec", "--json", "--full-auto", "--skip-git-repo-check",
            "--model", actual_model, "-c", f'model_reasoning_effort="{reasoning_effort}"',
            full_prompt,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL,
        )

        # Parse JSONL output to extract agent messages and structured errors.
        # Codex CLI emits API errors as `{"type":"error",...}` events on stdout
        # while stderr carries deprecation warnings and unrelated noise —
        # prefer the structured error over raw stderr.
        response_text, input_tokens, output_tokens = "", 0, 0
        structured_error: Optional[str] = None
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_type = event.get("type")
            if event_type == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    response_text = item.get("text", "")
            elif event_type == "turn.completed":
                usage = event.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
            elif event_type in ("error", "turn.failed"):
                msg = event.get("message") or event.get("error", {}).get("message")
                if msg:
                    structured_error = msg

        if result.returncode != 0 or structured_error:
            error_msg = (
                structured_error
                or result.stderr.strip()
                or f"Codex exited with code {result.returncode}"
            )
            raise RuntimeError(f"Codex CLI failed: {error_msg}")

        if not response_text:
            raise RuntimeError("No agent message found in Codex output")
        return response_text, input_tokens, output_tokens
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Codex CLI timed out after {timeout}s")


def call_gemini_cli_model(
    system_prompt: str, user_message: str, model: str, timeout: int = 600,
) -> tuple[str, int, int]:
    """Call Gemini CLI for model inference (retired for consumer accounts)."""
    if not GEMINI_CLI_AVAILABLE:
        raise RuntimeError(
            "Gemini CLI not found. Note: Gemini CLI was retired for consumer "
            "accounts on 2026-06-18 — use antigravity/<model> (agy CLI) or "
            "gemini/<model> (GEMINI_API_KEY) instead."
        )

    print(
        "Warning: Gemini CLI consumer service was retired 2026-06-18 in favor of "
        "Antigravity CLI. If this call fails, switch to antigravity/<model> "
        "(agy CLI) or gemini/<model> (GEMINI_API_KEY).",
        file=sys.stderr,
    )

    actual_model = model.split("/", 1)[1] if "/" in model else model
    full_prompt = f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\nUSER REQUEST:\n{user_message}"

    try:
        cmd = [GEMINI_CLI_PATH, "-m", actual_model, "-y"]
        result = subprocess.run(cmd, input=full_prompt, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(f"Gemini CLI failed: {result.stderr.strip() or f'exit code {result.returncode}'}")

        response_text = result.stdout.strip()
        lines = response_text.split("\n")
        filtered = [l for l in lines if not l.startswith(("Loaded cached", "Server ", "Loading extension"))]
        response_text = "\n".join(filtered).strip()

        if not response_text:
            raise RuntimeError("No response from Gemini CLI")
        input_tokens = len(full_prompt) // 4
        output_tokens = len(response_text) // 4
        return response_text, input_tokens, output_tokens
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Gemini CLI timed out after {timeout}s")


def resolve_antigravity_model(model: str) -> Optional[str]:
    """Extract the agy model slug from an antigravity/<slug> model string.

    `agy --model` accepts slugs exactly as listed by `agy models`
    (e.g. gemini-3.1-pro-high, claude-sonnet-4-6, gpt-oss-120b-medium).
    Returns None for a bare "antigravity" (use agy's default model).
    """
    slug = model.split("/", 1)[1] if "/" in model else ""
    return slug or None


def call_antigravity_model(
    system_prompt: str, user_message: str, model: str, timeout: int = 600,
) -> tuple[str, int, int]:
    """Call Antigravity CLI (agy) in headless print mode using Google account auth.

    Sign in once interactively (`agy`) before headless use — print mode reuses
    cached credentials and cannot complete the OAuth flow itself.
    Token counts come from agy JSON metadata when present, else estimated.
    """
    if not ANTIGRAVITY_AVAILABLE:
        raise RuntimeError(
            "Antigravity CLI not found. Install with: "
            "curl -fsSL https://antigravity.google/cli/install.sh | bash "
            "— then run `agy` once to sign in."
        )

    agy_model = resolve_antigravity_model(model)
    full_prompt = f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\nUSER REQUEST:\n{user_message}"

    # Prompt goes via stdin, not argv — project contexts can exceed the OS
    # per-argument size limit (128 KiB on Linux). Piped stdin puts agy in
    # print mode.
    cmd = [
        ANTIGRAVITY_PATH,
        "--output-format", "json",
        "--print-timeout", f"{timeout}s",
    ]
    if agy_model:
        cmd.extend(["--model", agy_model])

    try:
        result = subprocess.run(
            cmd, input=full_prompt, capture_output=True, text=True,
            timeout=timeout + 30,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Antigravity CLI timed out after {timeout}s")
    except FileNotFoundError:
        raise RuntimeError("Antigravity CLI not found in PATH")

    stdout = result.stdout.strip()

    # agy prints an interactive OAuth prompt when credentials are missing —
    # detect its fixed prompt strings, not URL fragments (which could appear
    # in legitimate model output).
    if (
        "Waiting for authentication" in result.stdout
        or "paste the authorization code" in result.stdout
    ):
        raise RuntimeError(
            "Antigravity CLI is not authenticated. Run `agy` interactively once "
            "to complete Google sign-in, then retry."
        )

    if result.returncode != 0:
        error_msg = (
            result.stderr.strip()
            or stdout
            or f"Antigravity CLI exited with code {result.returncode}"
        )
        raise RuntimeError(f"Antigravity CLI failed: {error_msg}")

    response_text, input_tokens, output_tokens = "", 0, 0

    # JSON output is a single object; schema may evolve, so probe common keys.
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        status = payload.get("status", "")
        if status and status != "SUCCESS":
            raise RuntimeError(
                f"Antigravity CLI returned status {status}: "
                f"{payload.get('error') or payload.get('response') or stdout[:200]}"
            )
        for key in ("response", "result", "text", "output", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                response_text = value.strip()
                break
        usage = payload.get("usage") or payload.get("metadata") or {}
        if isinstance(usage, dict):
            input_tokens = int(usage.get("input_tokens", 0) or 0)
            output_tokens = int(usage.get("output_tokens", 0) or 0)

    if not response_text and payload is None:
        # Fall back to raw stdout only when it wasn't JSON at all
        # (e.g. --output-format ignored by an older agy)
        response_text = stdout

    if not response_text:
        raise RuntimeError(
            "No response text in Antigravity CLI output: " + stdout[:200]
        )

    if not input_tokens:
        input_tokens = len(full_prompt) // 4
    if not output_tokens:
        output_tokens = len(response_text) // 4

    return response_text, input_tokens, output_tokens


def call_single_model(
    model: str,
    system_prompt: str,
    user_message: str,
    persona: str,
    dimension: str,
    codex_reasoning: str = DEFAULT_CODEX_REASONING,
    timeout: int = 600,
    bedrock_mode: bool = False,
    bedrock_region: Optional[str] = None,
) -> ReviewResponse:
    """Send review request to a single model with retry on failure."""
    actual_model = model
    if bedrock_mode:
        if bedrock_region:
            os.environ["AWS_REGION"] = bedrock_region
        if not model.startswith("bedrock/"):
            actual_model = f"bedrock/{model}"

    last_error = None
    display_model = model

    # Route to specialized handlers
    for attempt in range(MAX_RETRIES):
        try:
            if model.startswith("codex/"):
                content, input_tokens, output_tokens = call_codex_model(
                    system_prompt, user_message, model, codex_reasoning, timeout,
                )
            elif model == "antigravity" or model.startswith("antigravity/"):
                content, input_tokens, output_tokens = call_antigravity_model(
                    system_prompt, user_message, model, timeout,
                )
            elif model.startswith("gemini-cli/"):
                content, input_tokens, output_tokens = call_gemini_cli_model(
                    system_prompt, user_message, model, timeout,
                )
            elif model.startswith("foundry/"):
                content, input_tokens, output_tokens = call_foundry_model(
                    system_prompt, user_message, model, timeout,
                )
            else:
                # Standard litellm path
                completion_kwargs = {
                    "model": actual_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "timeout": timeout,
                }
                if uses_max_completion_tokens(actual_model):
                    completion_kwargs["max_completion_tokens"] = 16000
                else:
                    completion_kwargs["max_tokens"] = 16000
                if not is_reasoning_model(actual_model):
                    completion_kwargs["temperature"] = 0.7

                response = completion(**completion_kwargs)
                content = response.choices[0].message.content or ""
                input_tokens = response.usage.prompt_tokens if response.usage else 0
                output_tokens = response.usage.completion_tokens if response.usage else 0

            cost = cost_tracker.add(display_model, input_tokens, output_tokens)
            return ReviewResponse(
                model=display_model, persona=persona, dimension=dimension,
                response=content, input_tokens=input_tokens,
                output_tokens=output_tokens, cost=cost,
            )

        except Exception as e:
            last_error = str(e)
            # Bedrock: keep the original service diagnostic (AccessDenied can be
            # IAM/SCP, Marketplace, or data-share; Validation can be bad request
            # params) and append a hint. The hint phrasing doubles as the
            # non-retryable classifier match so these still fail fast.
            if bedrock_mode:
                if "AccessDeniedException" in last_error:
                    last_error = (
                        f"{last_error}\n  Hint: model not enabled in your Bedrock "
                        f"account ({display_model})? Also check IAM/SCP, Marketplace "
                        f"subscription, and data-share opt-in (claude-fable-5)."
                    )
                elif "ValidationException" in last_error:
                    last_error = (
                        f"{last_error}\n  Hint: invalid Bedrock model ID "
                        f"({display_model})? Or invalid request parameters."
                    )
            if "not supported when using codex with a chatgpt account" in last_error.lower():
                last_error = f"{last_error}\n  Hint: {CODEX_CHATGPT_HINT}"
            if is_non_retryable_error(last_error):
                print(f"Error: {display_model} failed (non-retryable): {last_error}", file=sys.stderr)
                break
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"Warning: {display_model} failed (attempt {attempt + 1}/{MAX_RETRIES}): {last_error}. Retrying in {delay:.1f}s...", file=sys.stderr)
                time.sleep(delay)
            else:
                print(f"Error: {display_model} failed after {MAX_RETRIES} attempts: {last_error}", file=sys.stderr)

    return ReviewResponse(
        model=display_model, persona=persona, dimension=dimension,
        response="", error=last_error,
    )


def call_models_parallel(
    assignments: list[dict],
    codex_reasoning: str = DEFAULT_CODEX_REASONING,
    timeout: int = 600,
    bedrock_mode: bool = False,
    bedrock_region: Optional[str] = None,
) -> list[ReviewResponse]:
    """Call multiple models in parallel with their assigned personas and dimensions.

    Args:
        assignments: List of dicts with keys: model, system_prompt, user_message, persona, dimension
    """
    if not assignments:
        return []

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(assignments)) as executor:
        future_to_assignment = {
            executor.submit(
                call_single_model,
                a["model"], a["system_prompt"], a["user_message"],
                a["persona"], a["dimension"],
                codex_reasoning, timeout, bedrock_mode, bedrock_region,
            ): a
            for a in assignments
        }
        for future in concurrent.futures.as_completed(future_to_assignment):
            results.append(future.result())

    return results
