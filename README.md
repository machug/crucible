# crucible

Multi-LLM adversarial project review for [Claude Code](https://claude.ai/claude-code). Evaluates progress, tech stack, code quality, CI/CD, and plan drift through parallel critique from multiple AI models.

```
gather ──> review ──> challenge ──> synthesize
```

Each model plays a different adversarial persona. N models enter. 1 honest report leaves.

## Install

Add the marketplace (once):

```
/plugin marketplace add machug/marketplace
```

Install the plugin:

```
/plugin install crucible@machug
```

## Usage

Run `/crucible` in any project directory. The skill will:

1. **Gather** project context (git history, dependencies, CI/CD config, issues, code metrics)
2. **Review** by fanning out to multiple LLMs, each playing a different adversarial persona
3. **Challenge** with an optional cross-examination round where models critique each other
4. **Synthesize** into a structured report with severity ratings and prioritised actions

## Review Dimensions

| Dimension | Focus |
|-----------|-------|
| `progress` | Plan drift, milestone tracking, velocity, scope creep |
| `tech-stack` | Framework choices, architecture, dependencies, complexity |
| `code-quality` | Type safety, test coverage, dead code, error handling |
| `pipeline` | CI/CD health, deploy frequency, build reproducibility |
| `security` | Dependency vulns, secret management, OWASP risks |
| `devex` | Onboarding friction, README accuracy, project structure |

## Adversarial Personas

| Persona | Perspective |
|---------|-------------|
| `investor` | Skeptical VC doing due diligence — progress vs burn, PMF evidence |
| `staff-engineer` | Seen it all — architecture sustainability, tech debt, scaling risks |
| `devops-lead` | Will be paged at 3am — deploy reliability, observability, rollback |
| `qa-skeptic` | Zero tolerance — test coverage, edge cases, regression prevention |
| `new-hire` | Day one developer — setup friction, documentation accuracy, tribal knowledge |

## Supported Providers

Any provider supported by [litellm](https://github.com/BerriAI/litellm), including:

| Provider | Env Var | Example Models |
|----------|---------|----------------|
| OpenAI | `OPENAI_API_KEY` | `gpt-5.4`, `o3-pro`, `o4-mini` |
| Google | `GEMINI_API_KEY` | `gemini/gemini-3.1-pro-preview` |
| xAI | `XAI_API_KEY` | `xai/grok-4.20-0309-reasoning` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-opus-4-6`, `claude-sonnet-4-6` |
| Azure AI Foundry | `AZURE_AI_API_KEY` | `foundry/<deployment>` |
| Mistral | `MISTRAL_API_KEY` | `mistral/mistral-large` |
| Groq | `GROQ_API_KEY` | `groq/llama-3.3-70b-versatile` |
| Deepseek | `DEEPSEEK_API_KEY` | `deepseek/deepseek-chat` |
| ZAI (GLM) | `ZAI_API_KEY` | `zai/glm-5.1` |
| Moonshot (Kimi) | `MOONSHOT_API_KEY` | `moonshot/kimi-k2.5` |
| Codex CLI | (ChatGPT sub) | `codex/gpt-5.3-codex` |
| Gemini CLI | (Google account) | `gemini-cli/gemini-3.1-pro-preview` |

## Requirements

- Python 3.10+
- `pip install litellm`
- At least one provider API key

## CLI

```bash
# Check providers
python3 crucible.py providers

# Discover available models
python3 crucible.py discover-models

# List dimensions and personas
python3 crucible.py dimensions
python3 crucible.py personas

# Run review
echo '{"context": "..."}' | python3 crucible.py review --models gpt-5.4,xai/grok-4-0709

# Cross-examination
python3 crucible.py challenge --models gpt-5.4,xai/grok-4-0709 --reviews reviews.json
```

## Part of the machug marketplace

- [fact-checker](https://github.com/machug/fact-checker) — Multi-LLM fact verification
- [spec-debate](https://github.com/machug/spec-debate) — Adversarial spec refinement
- **crucible** — Adversarial project review
- [gtm-forge](https://github.com/machug/gtm-forge) — GTM strategy stress-testing

## Author

[Hugh McIntyre](https://hughtec.com/) ([X](https://x.com/mmhughmm))

## License

MIT
