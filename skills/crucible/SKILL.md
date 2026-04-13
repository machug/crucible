---
name: crucible
description: Multi-LLM adversarial project review. Evaluates progress, tech stack, code quality, CI/CD, pipeline, and plan drift through parallel critique from multiple AI models. Use when user says "crucible", "project review", "adversarial review", "roast my project", "health check", "plan drift", "tech audit", "code audit".
allowed-tools: Bash, Read, Write, Edit, Agent, AskUserQuestion, WebFetch, WebSearch, Grep, Glob
---

# crucible

**If the user asks to see the banner, NFO, or splash screen, display the following:**

```
          ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
          ░                                                       ░
          ░     ▄████▄  ██▀███  █    ██ ▄████▄  ██▓ ▄▄▄▄  ██▓   ░
          ░    ▒██▀ ▀█ ▓██ ▒ ██▒██  ▓██▒██▀ ▀█ ▓██▒▓█████▄▓██▒   ░
          ░    ▒▓█    ▄▓██ ░▄█ ▒██  ▒██▒▓█    ▄▒██▒▒██▒ ▄█▒██░   ░
          ░    ▒▓▓▄ ▄██▒██▀▀█▄ ░██  ░██▒▓▓▄ ▄██░██░▒██░█▀ ▒██░   ░
          ░    ▒ ▓███▀ ░██▓ ▒██░ ████▓▒▒ ▓███▀ ░██░░▓█  ▀█░██████░
          ░    ░ ░▒ ▒  ░ ▒▓ ░▒▓░ ▒░▒░▒░░ ░▒ ▒  ░▓  ░▒▓███▀░ ▒░▓  ░
          ░      ░  ▒    ░▒ ░ ▒░ ░ ▒ ▒░  ░  ▒   ▒ ░▒░▒   ░░ ░ ▒  ░
          ░                                                       ░
          ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
          ╔═══════════════════════════════════════════════════════╗
          ║               RELEASE INFORMATION                     ║
          ╠═══════════════════════════════════════════════════════╣
          ║                                                       ║
          ║  Skill.......: crucible                               ║
          ║  Author......: machug          (hughtec.com)          ║
          ║  Version.....: 0.1.0                                  ║
          ║  Released....: 2026                                   ║
          ║  License.....: MIT                                    ║
          ║  Requires....: Python 3.10+, litellm                  ║
          ║                                                       ║
          ╠═══════════════════════════════════════════════════════╣
          ║                PIPELINE OVERVIEW                       ║
          ╠═══════════════════════════════════════════════════════╣
          ║                                                       ║
          ║  gather ──> review ──> challenge ──> synthesize        ║
          ║                                                       ║
          ║  Adversarial project review via multi-LLM panel.      ║
          ║  Each model plays a different adversarial persona.     ║
          ║  N models enter. 1 honest report leaves.              ║
          ║                                                       ║
          ╚═══════════════════════════════════════════════════════╝
```

Multi-LLM adversarial project review. Gathers project context and sends it to multiple AI models, each reviewing through a different adversarial persona and dimension. Produces a structured report with severity ratings and actionable recommendations.

**Important: Claude is an active participant in this review, not just an orchestrator.** You (Claude) will provide your own independent assessment alongside the external models, challenge findings you disagree with, and synthesize the final report.

## Requirements

- Python 3.10+ with `litellm` package installed
- API key for at least one provider (set via environment variable)

**IMPORTANT: Do NOT install the `llm` package.** This skill uses `litellm` for API providers.

## Supported Providers

| Provider   | API Key Env Var        | Example Models |
|------------|------------------------|----------------|
| OpenAI     | `OPENAI_API_KEY`       | `gpt-5.4`, `gpt-5.4-pro`, `o3-pro`, `o4-mini` |
| Anthropic  | `ANTHROPIC_API_KEY`    | `claude-opus-4-6`, `claude-sonnet-4-6` |
| Google     | `GEMINI_API_KEY`       | `gemini/gemini-3.1-pro-preview`, `gemini/gemini-2.5-pro` |
| xAI        | `XAI_API_KEY`          | `xai/grok-4.20-0309-reasoning`, `xai/grok-4-0709` |
| Azure AI   | `AZURE_AI_API_KEY`     | `foundry/<deployment-name>` |
| Mistral    | `MISTRAL_API_KEY`      | `mistral/mistral-large` |
| Groq       | `GROQ_API_KEY`         | `groq/llama-3.3-70b-versatile` |
| OpenRouter | `OPENROUTER_API_KEY`   | `openrouter/openai/gpt-5.2-pro` |
| Deepseek   | `DEEPSEEK_API_KEY`     | `deepseek/deepseek-chat` |
| ZAI (GLM)  | `ZAI_API_KEY`          | `zai/glm-5.1`, `zai/glm-5-turbo` |
| Moonshot   | `MOONSHOT_API_KEY`     | `moonshot/kimi-k2.5` |
| Codex CLI  | (ChatGPT subscription) | `codex/gpt-5.3-codex` |
| Gemini CLI | (Google account)       | `gemini-cli/gemini-3.1-pro-preview` |

**Discover latest models:** Run `cd ${CLAUDE_PLUGIN_ROOT}/skills/crucible/scripts && python3 crucible.py discover-models`

Run `cd ${CLAUDE_PLUGIN_ROOT}/skills/crucible/scripts && python3 crucible.py providers` to see which keys are set.

## Review Dimensions

Each dimension is a focused lens for project evaluation:

| Dimension | Focus |
|-----------|-------|
| `progress` | Plan drift, milestone tracking, velocity, scope creep |
| `tech-stack` | Framework choices, architecture, dependencies, complexity |
| `code-quality` | Type safety, test coverage, dead code, error handling |
| `pipeline` | CI/CD health, deploy frequency, build reproducibility |
| `security` | Dependency vulns, secret management, OWASP risks |
| `devex` | Onboarding friction, README accuracy, project structure |

## Adversarial Personas

Each model is assigned a different adversarial persona for maximum critique diversity:

| Persona | Perspective |
|---------|-------------|
| `investor` | Skeptical VC doing due diligence — progress vs burn, PMF evidence |
| `staff-engineer` | Seen it all — architecture sustainability, tech debt, scaling risks |
| `devops-lead` | Will be paged at 3am — deploy reliability, observability, rollback |
| `qa-skeptic` | Zero tolerance — test coverage, edge cases, regression prevention |
| `new-hire` | Day one developer — setup friction, documentation accuracy, tribal knowledge |

## Process

### Step 1: Gather Project Context

Automatically collect project context. This is the critical input — the richer the context, the better the review. Gather ALL of the following that exist:

**Git & History:**
```bash
git log --oneline -30
git log --format="%h %s" --since="2 weeks ago"
git diff --stat HEAD~10..HEAD
git branch -a
```

**Project Structure:**
```bash
ls -la
find . -name "*.json" -maxdepth 2 -not -path "*/node_modules/*" | head -20
```

**Dependencies (read the relevant files):**
- `package.json` / `requirements.txt` / `Cargo.toml` / `go.mod`
- Lock files (check existence, freshness)

**CI/CD Configuration (read if they exist):**
- `.github/workflows/` / `Jenkinsfile` / `.gitlab-ci.yml` / `vercel.json`

**Issue Tracker:**
- If beads is available: `bd list --status=open` and `bd stats`
- If there's a TODO.md or similar: read it

**Code Metrics (sample):**
- Count files by type: `find . -name "*.ts" -not -path "*/node_modules/*" | wc -l`
- Check for test files: `find . -name "*.test.*" -o -name "*.spec.*" | head -10`
- Check for type config: `tsconfig.json`, `mypy.ini`, `.eslintrc`

**Build & Deploy:**
- Try `npm run` / `yarn` / `make` to see available scripts
- Check for Dockerfile, docker-compose.yml
- Check for deploy scripts or configs

Compile ALL gathered context into a single JSON structure:

```json
{
  "project_name": "...",
  "gathered_at": "ISO timestamp",
  "git": {
    "recent_commits": "...",
    "branches": "...",
    "recent_diff_stats": "..."
  },
  "structure": {
    "top_level_files": "...",
    "file_counts_by_type": {}
  },
  "dependencies": {
    "package_json": "... or null",
    "lock_file_exists": true
  },
  "ci_cd": {
    "github_actions": "... or null",
    "deploy_config": "... or null"
  },
  "issues": {
    "open_count": 0,
    "in_progress": [],
    "recent_closed": []
  },
  "code_quality": {
    "test_files_count": 0,
    "type_config_exists": true,
    "lint_config_exists": true
  }
}
```

Write this context to a temporary file (e.g., `/tmp/crucible-context-{project}.json`).

### Step 2: Select Models

First, discover available models:

```bash
cd ${CLAUDE_PLUGIN_ROOT}/skills/crucible/scripts && python3 crucible.py discover-models
```

If that fails, fall back to:
```bash
cd ${CLAUDE_PLUGIN_ROOT}/skills/crucible/scripts && python3 crucible.py providers
```

Present available models to the user using AskUserQuestion with multiSelect. Build options from discover-models output. Recommend at least 3 models for good adversarial coverage.

Also ask which dimensions to review (default: all 6), or let the user focus on specific areas.

### Step 3: Run Review Round

Send the context to all selected models in parallel, each assigned a different persona and dimension:

```bash
cd ${CLAUDE_PLUGIN_ROOT}/skills/crucible/scripts && python3 crucible.py review --models MODEL1,MODEL2,MODEL3 --context /tmp/crucible-context-{project}.json
```

The script outputs JSON with all reviews. Parse it and present each review to the user.

**Claude's Own Review:** After receiving model reviews, provide YOUR OWN independent assessment. You have advantages the models don't: you can read the actual code, run commands, and verify claims. Use this to:
- Confirm or challenge model findings with evidence
- Add findings the models missed
- Flag any model hallucinations (e.g., claiming a file exists that doesn't)

### Step 4: Cross-Examination (Optional)

Ask the user: "Want to run a cross-examination round where models challenge each other's findings?"

If yes, save the reviews to a file and run:

```bash
cd ${CLAUDE_PLUGIN_ROOT}/skills/crucible/scripts && python3 crucible.py challenge --models MODEL1,MODEL2 --reviews /tmp/crucible-reviews-{project}.json
```

### Step 5: Synthesize Report

Consolidate all reviews (model + Claude's own) into a structured final report:

```markdown
# Crucible Report: {Project Name}
**Date:** {date}
**Models:** {list}
**Dimensions reviewed:** {list}

## Executive Summary
{2-3 sentence overall assessment}

## Scorecard

| Dimension | Score | Severity | Key Finding |
|-----------|-------|----------|-------------|
| Progress  | X/10  | ...      | ...         |
| Tech Stack| X/10  | ...      | ...         |
| ...       | ...   | ...      | ...         |

**Overall: X/10**

## Critical Findings
{Findings rated CRITICAL by any reviewer, with cross-examination results}

## Warnings
{Findings rated WARNING}

## Healthy Areas
{What's working well — important for morale}

## Recommended Actions
{Prioritized list of concrete next steps}

## Reviewer Disagreements
{Areas where models disagreed — these often reveal the most interesting insights}

## Cost Summary
{Token usage and cost breakdown from the review}
```

Write the report to `./crucible-report-{date}.md` in the project directory.

### Step 6: Export Actions (Optional)

Ask the user if they want to export critical findings and recommendations as issues:
- If beads is available: create beads issues with `bd create`
- Otherwise: list them as actionable items

## CLI Reference

```bash
# Run review
cd ${CLAUDE_PLUGIN_ROOT}/skills/crucible/scripts && python3 crucible.py review --models gpt-5.4,xai/grok-4-0709 --context context.json

# Specific dimensions only
cd ${CLAUDE_PLUGIN_ROOT}/skills/crucible/scripts && python3 crucible.py review --models gpt-5.4 --context ctx.json --dimensions progress,code-quality

# Cross-examination round
cd ${CLAUDE_PLUGIN_ROOT}/skills/crucible/scripts && python3 crucible.py challenge --models gpt-5.4,xai/grok-4-0709 --reviews reviews.json

# List providers
cd ${CLAUDE_PLUGIN_ROOT}/skills/crucible/scripts && python3 crucible.py providers

# Discover models
cd ${CLAUDE_PLUGIN_ROOT}/skills/crucible/scripts && python3 crucible.py discover-models

# List dimensions
cd ${CLAUDE_PLUGIN_ROOT}/skills/crucible/scripts && python3 crucible.py dimensions

# List personas
cd ${CLAUDE_PLUGIN_ROOT}/skills/crucible/scripts && python3 crucible.py personas
```

## Key Principles

1. **Evidence over opinion** — every finding must cite specific files, commits, or configs
2. **Actionable over academic** — recommendations should be concrete and prioritized
3. **Honest over kind** — the point is to find problems before they find you
4. **Diverse perspectives** — different models have different biases; that's a feature
5. **Claude verifies** — you have the ability to actually read code and run commands; use it to fact-check model claims
