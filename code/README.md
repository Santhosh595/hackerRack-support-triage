# Support Triage Agent — Code

Multi-domain AI support triage agent for HackerRank, Claude, and Visa tickets.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

## Run (batch mode — evaluator path)

```bash
python main.py --batch
# Reads:  support_tickets/support_tickets.csv
# Writes: support_tickets/output.csv
```

## Run (interactive mode)

```bash
python main.py
# or
python code/main.py
```

## Pipeline stages

1. **Prompt injection detection** — blocks policy-exfiltration attempts
2. **High-risk escalation** — fraud, identity theft, account takeover → escalated
3. **Domain classification** — keyword + company signal → HackerRank / Claude / Visa
4. **Multi-domain ambiguity check** — only escalates on genuinely tied signals
5. **Request type classification** — product_issue / feature_request / bug / invalid
6. **Authority check** — requests outside agent scope → escalated
7. **TF-IDF retrieval** — top-3 docs from 800+ file corpus
8. **Response generation** — Claude API (LLM mode) or extractive fallback
9. **Structured logging** — every decision traced to `logs/triage_structured.jsonl`

## Architecture

```
main.py                 ← CLI entry + batch runner
code/
  pipeline.py           ← Orchestrates all stages
  classifiers.py        ← Deterministic safety + routing gates
  classifier_provider.py← Deterministic + optional LLM classification
  retriever.py          ← TF-IDF domain retriever (800+ corpus files)
  generator.py          ← Extractive fallback generator
  generator_provider.py ← Anthropic Claude API generator with fallback
  escalation.py         ← Escalation/refuse helpers
  models.py             ← Shared dataclasses
  config.py             ← Central config (env-driven)
  utils.py              ← CSV I/O
  logger.py             ← Structured JSONL logger
```

## Environment variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Enables LLM generation via Claude API |
| `RETRIEVAL_CONFIDENCE_THRESHOLD` | Min cosine similarity (default: 0.18) |
| `RETRIEVAL_TOP_K` | Docs retrieved per query (default: 3) |
