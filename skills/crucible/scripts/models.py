"""Model calling, cost tracking, and response handling for crucible."""

from __future__ import annotations

import concurrent.futures
import json
import os
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
    CODEX_AVAILABLE,
    CODEX_PATH,
    DEFAULT_CODEX_REASONING,
    GEMINI_CLI_AVAILABLE,
    GEMINI_CLI_PATH,
    get_model_cost,
)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0


def is_reasoning_model(model: str) -> bool:
    """Check if a model is a reasoning model (o-series, gpt-5)."""
    model_lower = model.lower()
    if model_lower.startswith(("o1", "o3", "o4")) or "/o1" in model_lower or "/o3" in model_lower or "/o4" in model_lower:
        return True
    if "gpt-5" in model_lower:
        return True
    if "xai/" in model_lower and model_lower.endswith("-reasoning") and not model_lower.endswith("-non-reasoning"):
        return True
    if "moonshot/" in model_lower and "k2.5" in model_lower:
        return True
    return False


def uses_max_completion_tokens(model: str) -> bool:
    """Check if a model uses max_completion_tokens instead of max_tokens."""
    if not is_reasoning_model(model):
        return False
    if model.lower().startswith(("xai/", "moonshot/")):
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(f"Codex CLI failed: {result.stderr.strip() or f'exit code {result.returncode}'}")

        response_text, input_tokens, output_tokens = "", 0, 0
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                if event.get("type") == "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message":
                        response_text = item.get("text", "")
                if event.get("type") == "turn.completed":
                    usage = event.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
            except json.JSONDecodeError:
                continue

        if not response_text:
            raise RuntimeError("No agent message found in Codex output")
        return response_text, input_tokens, output_tokens
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Codex CLI timed out after {timeout}s")


def call_gemini_cli_model(
    system_prompt: str, user_message: str, model: str, timeout: int = 600,
) -> tuple[str, int, int]:
    """Call Gemini CLI for model inference."""
    if not GEMINI_CLI_AVAILABLE:
        raise RuntimeError("Gemini CLI not found. Install with: npm install -g @google/gemini-cli")

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
