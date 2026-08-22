# Support Triage Agent - Hackathon Submission 

An evaluator-ready, production-style support triage system for three domains:
- HackerRank
- Claude
- Visa

The project is built for reliability-first judging: deterministic by default, safety-gated, and modular.

## What this solves

Support queues receive mixed-quality tickets: safe product questions, ambiguous requests, high-risk incidents, and policy-exfiltration attempts.
This system triages requests in a strict pipeline so unsafe or uncertain cases escalate instead of hallucinating.

## Canonical Runtime (single architecture)

Only these entrypoints are active:
- `main.py` - terminal demo runner
- `evaluate.py` - benchmark/evaluation runner
- `code/pipeline.py` - authoritative orchestration

Legacy v2 paths are deprecated and excluded from runtime behavior.

## Pipeline Order (strict, preserved)

1. Prompt injection / jailbreak detection
2. High-risk / sensitive escalation detection
3. Domain classification
4. Request type classification
5. Authority / responsibility check
6. Domain-specific retrieval
7. Retrieval confidence validation
8. Response generation OR escalation
9. Structured logging

## Safety Guarantees

- Prompt-injection and policy-exfiltration attempts are refused.
- High-risk and sensitive financial/security/account cases escalate.
- Unsupported admin override requests escalate/refuse.
- Multi-domain ambiguity escalates for manual handling.
- Low retrieval confidence escalates instead of guessing.
- Optional LLM mode cannot bypass deterministic safety stages.

## Evaluator Taxonomy (fixed labels)

### Domains
- `HackerRank`
- `Claude`
- `Visa`
- `None`

### Request Types
- `product_issue`
- `feature_request`
- `bug`
- `invalid`

## Retrieval Strategy

- Domain-separated corpora:
  - `data/hackerrank`
  - `data/claude`
  - `data/visa`
- Method: TF-IDF + cosine similarity
- Retrieval is restricted to predicted domain
- Indices/vectorizers are precomputed at startup for stable latency

## Optional LLM Augmentation

Deterministic fallback is the default for reliability and reproducible evaluation.

The agent auto-detects an LLM provider from the environment, in priority order:

1. `OPENAI_API_KEY` (OpenAI)
2. `ANTHROPIC_API_KEY` (Anthropic)
3. `GOOGLE_API_KEY` (Google Gemini)
4. `AZURE_OPENAI_API_KEY` (Azure OpenAI)
5. `TRIAGE_LLM_URL` + `TRIAGE_API_KEY` set together (custom OpenAI-compatible endpoint)

- No provider configured -> deterministic mode
- Provider configured -> grounded LLM augmentation behind the same safety pipeline
- Any LLM failure (network, auth, parse) -> falls back to the deterministic grounded response

A startup warning is printed if an LLM provider is active but `TRIAGE_LLM_URL`
is not set (informational only — built-in providers use their default endpoints).

Provider selection (set one):
- `OPENAI_API_KEY` (+ optional `OPENAI_BASE_URL`, `OPENAI_MODEL`)
- `ANTHROPIC_API_KEY` (+ optional `ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL`)
- `GOOGLE_API_KEY` (+ optional `GOOGLE_BASE_URL`, `GOOGLE_MODEL`)
- `AZURE_OPENAI_API_KEY` (+ `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`)
- `TRIAGE_LLM_URL` + `TRIAGE_API_KEY` (+ optional `TRIAGE_LLM_MODEL`) for local/self-hosted OpenAI-compatible servers

Tuning (optional):
- `TRIAGE_LLM_TIMEOUT_SEC` (default 12)
- `RETRIEVAL_CONFIDENCE_THRESHOLD` (default 0.07)
- `RETRIEVAL_TOP_K` (default 3)

See `.env.example` for the full annotated list.

Environment loading is plain `os.environ` (Doppler-compatible, no SDK lock-in).

## Output Contract

Default CSV output:
- `support_tickets/output.csv`

Required columns:
- `status`
- `product_area`
- `response`
- `justification`
- `request_type`

## Demo UX

Interactive terminal mode includes:
- colored status badges
- startup banner
- progress indicator
- boxed result cards
- formatted justification trace

Presentation changes do not alter evaluator logic or CSV schema.

## Repository Layout

- `main.py`
- `evaluate.py`
- `test_llm_config.py` - LLM provider-detection self-check
- `requirements.txt`
- `.env.example` - copy to `.env`; never commit `.env`
- `AGENTS.md` - coding-agent working agreement (Orchestrate starter)
- `chat_transcript.txt` - interactive session transcript (appended at runtime)
- `support_tickets/`
  - `support_tickets.csv` - input tickets (batch mode)
  - `sample_support_tickets.csv` - provided sample set
  - `output.csv` - written results
- `logs/` - structured JSONL run logs (`triage_structured.jsonl`, `eval_v3.jsonl`)
- `outputs/` - archived result snapshots
- `code/`
  - `__init__.py`
  - `config.py`
  - `models.py`
  - `classifiers.py`
  - `classifier_provider.py`
  - `retriever.py`
  - `generator.py`
  - `generator_provider.py`
  - `llm_client.py`
  - `escalation.py`
  - `pipeline.py`
  - `logger.py`
  - `utils.py`
  - `main.py` - entry-point shim delegating to the root runner (evaluator path)
  - `README.md` - code-level notes
- `data/`
  - `hackerrank/`
  - `claude/`
  - `visa/`

## Quick Start

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Ensure corpus docs exist in each domain folder under `data/`
3. Run terminal demo (interactive):
   - `python main.py`
4. Run batch over the ticket CSV (evaluator path — reads `support_tickets/support_tickets.csv`, writes `support_tickets/output.csv`):
   - `python main.py --batch`
5. Run evaluation:
   - `python evaluate.py -v`

## Why this scores well for judging

- Reliability-first: deterministic baseline, no network dependency required.
- Safety-first: escalation over guessing for risky/uncertain cases.
- Clear modularity: provider pattern allows optional LLM enhancement without architectural churn.
- Reproducibility: fixed taxonomy, fixed pipeline order, fixed output schema.
