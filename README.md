# Support Triage Agent - Hackathon Submission (v3)

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

- `TRIAGE_API_KEY` present -> optional LLM augmentation enabled
- `TRIAGE_API_KEY` missing -> deterministic mode
- `TRIAGE_API_KEY` set but `TRIAGE_LLM_URL` missing -> startup warning + deterministic fallback

Optional env vars:
- `TRIAGE_API_KEY`
- `TRIAGE_LLM_URL`
- `TRIAGE_LLM_MODEL`
- `TRIAGE_LLM_TIMEOUT_SEC`

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
- `requirements.txt`
- `code/`
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
- `data/`
  - `hackerrank/`
  - `claude/`
  - `visa/`

## Quick Start

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Ensure corpus docs exist in each domain folder under `data/`
3. Run terminal demo:
   - `python main.py`
4. Run evaluation:
   - `python evaluate.py -v`

## Why this scores well for hackathon judging

- Reliability-first: deterministic baseline, no network dependency required.
- Safety-first: escalation over guessing for risky/uncertain cases.
- Clear modularity: provider pattern allows optional LLM enhancement without architectural churn.
- Reproducibility: fixed taxonomy, fixed pipeline order, fixed output schema.
