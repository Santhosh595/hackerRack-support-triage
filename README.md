# hackerRack Support Triage

> **Support ticket triage system** — Deterministic, safety-gated, modular pipeline for HackerRank, Claude, and Visa support domains.

---

## What This Solves

Support queues receive mixed-quality tickets: safe product questions, ambiguous requests, high-risk incidents, and policy-exfiltration attempts. This system triages requests in a strict pipeline so unsafe or uncertain cases escalate instead of hallucinating.

## Pipeline Order (Strict, Preserved)

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

- Prompt-injection and policy-exfiltration attempts are refused
- High-risk and sensitive financial/security/account cases escalate
- Unsupported admin override requests escalate/refuse
- Multi-domain ambiguity escalates for manual handling
- Low retrieval confidence escalates instead of guessing
- Optional LLM mode cannot bypass deterministic safety stages

## Quick Start

```bash
pip install -r requirements.txt
python main.py         # Interactive terminal demo
python evaluate.py -v  # Benchmark evaluation
```

## Tech

Python, TF-IDF + cosine similarity retrieval, domain-separated corpora, deterministic safety-first pipeline with optional LLM augmentation.

## 📄 License

MIT
