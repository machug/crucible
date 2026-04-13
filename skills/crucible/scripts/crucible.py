#!/usr/bin/env python3
"""
Crucible: multi-LLM adversarial project review.

Gathers project context and sends it to multiple LLMs, each reviewing through
a different adversarial persona and dimension. Produces a structured report
with severity ratings and actionable recommendations.

Usage:
    python3 crucible.py review --models gpt-5.4,xai/grok-4-0709 --context context.json
    python3 crucible.py review --models gpt-5.4 --context context.json --dimensions progress,code-quality
    python3 crucible.py challenge --models gpt-5.4,xai/grok-4-0709 --reviews reviews.json
    python3 crucible.py providers
    python3 crucible.py discover-models
    python3 crucible.py dimensions
    python3 crucible.py personas

Exit codes:
    0 - Success
    1 - API error
    2 - Missing API key or config error
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
os.environ["LITELLM_LOG"] = "ERROR"

try:
    from litellm import completion
except ImportError:
    print("Error: litellm not installed. Run: pip install litellm", file=sys.stderr)
    sys.exit(1)

from models import call_models_parallel, cost_tracker, ReviewResponse
from prompts import (
    REVIEW_DIMENSIONS,
    PERSONAS,
    SYSTEM_PROMPT,
    REVIEW_PROMPT_TEMPLATE,
    CHALLENGE_PROMPT_TEMPLATE,
)
from providers import (
    get_available_providers,
    list_providers,
    validate_model_credentials,
)


def cmd_review(args: argparse.Namespace) -> int:
    """Run adversarial review with multiple models."""
    models = [m.strip() for m in args.models.split(",")]

    # Validate credentials
    valid, invalid = validate_model_credentials(models)
    if invalid:
        print(f"Error: Missing credentials for: {', '.join(invalid)}", file=sys.stderr)
        if not valid:
            sys.exit(2)
        print(f"Continuing with: {', '.join(valid)}", file=sys.stderr)
        models = valid

    # Load context from stdin or file
    if args.context and args.context != "-":
        context = Path(args.context).read_text()
    else:
        context = sys.stdin.read()

    if not context.strip():
        print("Error: No project context provided. Pipe context via stdin or --context file.", file=sys.stderr)
        sys.exit(1)

    # Determine which dimensions to review
    if args.dimensions:
        dimensions = [d.strip() for d in args.dimensions.split(",")]
        for d in dimensions:
            if d not in REVIEW_DIMENSIONS:
                print(f"Error: Unknown dimension '{d}'. Use 'crucible.py dimensions' to list.", file=sys.stderr)
                sys.exit(2)
    else:
        dimensions = list(REVIEW_DIMENSIONS.keys())

    # Assign personas to models round-robin
    persona_names = list(PERSONAS.keys())
    assignments = []

    for i, dim in enumerate(dimensions):
        model = models[i % len(models)]
        persona_key = persona_names[i % len(persona_names)]

        system = SYSTEM_PROMPT
        if args.persona:
            # Override with custom persona for all
            persona_text = args.persona
        else:
            persona_text = PERSONAS[persona_key]

        user_msg = REVIEW_PROMPT_TEMPLATE.format(
            round=1,
            persona=persona_text,
            dimension=REVIEW_DIMENSIONS[dim],
            focus_section="",
            context=context,
        )

        assignments.append({
            "model": model,
            "system_prompt": system,
            "user_message": user_msg,
            "persona": persona_key,
            "dimension": dim,
        })

    print(f"Starting review: {len(assignments)} dimensions across {len(set(models))} models...", file=sys.stderr)
    results = call_models_parallel(
        assignments,
        codex_reasoning=args.codex_reasoning if hasattr(args, "codex_reasoning") else "xhigh",
        timeout=args.timeout,
    )

    # Output results as JSON for Claude to parse
    output = {
        "timestamp": datetime.now().isoformat(),
        "models": models,
        "dimensions": dimensions,
        "reviews": [],
    }

    for r in results:
        review = {
            "model": r.model,
            "persona": r.persona,
            "dimension": r.dimension,
            "response": r.response,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "cost": r.cost,
        }
        if r.error:
            review["error"] = r.error
        output["reviews"].append(review)

    print(json.dumps(output, indent=2))
    print(cost_tracker.summary(), file=sys.stderr)
    return 0


def cmd_challenge(args: argparse.Namespace) -> int:
    """Run cross-examination round where models challenge each other."""
    models = [m.strip() for m in args.models.split(",")]

    valid, invalid = validate_model_credentials(models)
    if invalid:
        print(f"Error: Missing credentials for: {', '.join(invalid)}", file=sys.stderr)
        if not valid:
            sys.exit(2)
        models = valid

    # Load previous reviews
    if args.reviews and args.reviews != "-":
        reviews_data = json.loads(Path(args.reviews).read_text())
    else:
        reviews_data = json.loads(sys.stdin.read())

    reviews = reviews_data.get("reviews", [])
    if not reviews:
        print("Error: No reviews to challenge.", file=sys.stderr)
        sys.exit(1)

    persona_names = list(PERSONAS.keys())
    assignments = []

    for i, review in enumerate(reviews):
        if review.get("error"):
            continue

        model = models[i % len(models)]
        persona_key = review.get("persona", persona_names[i % len(persona_names)])
        persona_text = PERSONAS.get(persona_key, PERSONAS["staff-engineer"])

        # Gather other reviews for cross-examination
        other_reviews = "\n\n---\n\n".join(
            f"**{r['dimension']}** (by {r['persona']}):\n{r['response']}"
            for r in reviews
            if r.get("dimension") != review["dimension"] and not r.get("error")
        )

        user_msg = CHALLENGE_PROMPT_TEMPLATE.format(
            round=2,
            persona=persona_text,
            own_review=review["response"],
            other_reviews=other_reviews,
        )

        assignments.append({
            "model": model,
            "system_prompt": SYSTEM_PROMPT,
            "user_message": user_msg,
            "persona": persona_key,
            "dimension": review["dimension"],
        })

    print(f"Starting cross-examination: {len(assignments)} challenges...", file=sys.stderr)
    results = call_models_parallel(assignments, timeout=args.timeout)

    output = {
        "timestamp": datetime.now().isoformat(),
        "round": "challenge",
        "challenges": [],
    }

    for r in results:
        challenge = {
            "model": r.model,
            "persona": r.persona,
            "dimension": r.dimension,
            "response": r.response,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "cost": r.cost,
        }
        if r.error:
            challenge["error"] = r.error
        output["challenges"].append(challenge)

    print(json.dumps(output, indent=2))
    print(cost_tracker.summary(), file=sys.stderr)
    return 0


def cmd_providers(args: argparse.Namespace) -> int:
    list_providers()
    return 0


def cmd_discover_models(args: argparse.Namespace) -> int:
    """Query provider APIs for available models."""
    from providers import get_available_providers
    import urllib.request

    results = {}

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            req = urllib.request.Request(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            models = sorted(
                m["id"] for m in data["data"]
                if any(k in m["id"] for k in ("gpt-4.1", "gpt-5", "o1", "o3", "o4"))
                and not any(k in m["id"] for k in ("audio", "realtime", "tts", "transcribe", "search", "image", "embed"))
            )
            results["OpenAI"] = models
        except Exception as e:
            results["OpenAI"] = [f"[error: {e}]"]

    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            req = urllib.request.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            )
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            models = sorted(
                m["name"].replace("models/", "") for m in data["models"]
                if any(k in m["name"] for k in ("gemini-2.5", "gemini-3", "gemini-pro", "gemini-flash"))
                and not any(k in m["name"] for k in ("tts", "image", "audio", "live"))
            )
            results["Google Gemini"] = [f"gemini/{m}" for m in models]
        except Exception as e:
            results["Google Gemini"] = [f"[error: {e}]"]

    api_key = os.environ.get("XAI_API_KEY")
    if api_key:
        try:
            req = urllib.request.Request(
                "https://api.x.ai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            models = sorted(
                m["id"] for m in data["data"]
                if "grok" in m["id"] and not any(k in m["id"] for k in ("image", "video"))
            )
            results["xAI (Grok)"] = [f"xai/{m}" for m in models]
        except Exception as e:
            results["xAI (Grok)"] = [f"[error: {e}]"]

    api_key = os.environ.get("ZAI_API_KEY")
    if api_key:
        try:
            req = urllib.request.Request(
                "https://open.bigmodel.cn/api/paas/v4/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            models = sorted(m["id"] for m in data["data"] if "glm" in m["id"])
            results["ZAI (GLM)"] = [f"zai/{m}" for m in models]
        except Exception as e:
            results["ZAI (GLM)"] = [f"[error: {e}]"]

    api_key = os.environ.get("MOONSHOT_API_KEY")
    if api_key:
        results["Moonshot (Kimi)"] = ["moonshot/kimi-k2.5", "moonshot/kimi-k2-thinking"]

    if os.environ.get("ANTHROPIC_API_KEY"):
        results["Anthropic"] = ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"]

    if not results:
        print("No API keys configured. Run 'crucible.py providers' for setup info.", file=sys.stderr)
        return 2

    print("Available models:\n")
    for provider, models in sorted(results.items()):
        print(f"  {provider}:")
        for m in models:
            print(f"    - {m}")
        print()
    return 0


def cmd_dimensions(args: argparse.Namespace) -> int:
    """List available review dimensions."""
    print("Available review dimensions:\n")
    for name, desc in REVIEW_DIMENSIONS.items():
        first_line = desc.strip().split("\n")[0].replace("**DIMENSION:", "").replace("**", "").strip()
        print(f"  {name:15} {first_line}")
    print()
    return 0


def cmd_personas(args: argparse.Namespace) -> int:
    """List available adversarial personas."""
    print("Available adversarial personas:\n")
    for name, desc in PERSONAS.items():
        print(f"  {name}")
        print(f"    {desc[:100]}...")
        print()
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Crucible: multi-LLM adversarial project review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # review
    p_review = sub.add_parser("review", help="Run adversarial project review")
    p_review.add_argument("--models", required=True, help="Comma-separated model list")
    p_review.add_argument("--context", help="Path to context JSON file (or stdin)")
    p_review.add_argument("--dimensions", help="Comma-separated dimensions to review (default: all)")
    p_review.add_argument("--persona", help="Override persona for all reviewers")
    p_review.add_argument("--codex-reasoning", default="xhigh", help="Codex reasoning effort")
    p_review.add_argument("--timeout", type=int, default=600, help="Per-model timeout in seconds")

    # challenge
    p_challenge = sub.add_parser("challenge", help="Cross-examination round")
    p_challenge.add_argument("--models", required=True, help="Comma-separated model list")
    p_challenge.add_argument("--reviews", help="Path to reviews JSON from review round (or stdin)")
    p_challenge.add_argument("--timeout", type=int, default=600, help="Per-model timeout in seconds")

    # utility commands
    sub.add_parser("providers", help="List supported providers and API key status")
    sub.add_parser("discover-models", help="Query provider APIs for available models")
    sub.add_parser("dimensions", help="List review dimensions")
    sub.add_parser("personas", help="List adversarial personas")

    args = parser.parse_args()

    if args.command == "review":
        sys.exit(cmd_review(args))
    elif args.command == "challenge":
        sys.exit(cmd_challenge(args))
    elif args.command == "providers":
        sys.exit(cmd_providers(args))
    elif args.command == "discover-models":
        sys.exit(cmd_discover_models(args))
    elif args.command == "dimensions":
        sys.exit(cmd_dimensions(args))
    elif args.command == "personas":
        sys.exit(cmd_personas(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
