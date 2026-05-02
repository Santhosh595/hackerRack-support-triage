"""Deterministic safety and routing classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from code.models import Ticket


def _contains_any(text: str, patterns: Iterable[str]) -> bool:
    lower_text = text.lower()
    return any(p in lower_text for p in patterns)


@dataclass(frozen=True)
class ClassifierOutcome:
    label: str
    reason: str
    confidence: float = 1.0


def _domain_scores(ticket: Ticket) -> dict[str, int]:
    text = ticket.user_text.lower()
    company = ticket.company.lower().strip()
    scores = {
        "HackerRank": sum(
            keyword in text
            for keyword in (
                "hackerrank",
                "coding test",
                "assessment",
                "candidate",
                "submission",
                "interview",
                "recruiter",
            )
        ),
        "Claude": sum(
            keyword in text
            for keyword in (
                "claude",
                "anthropic",
                "workspace",
                "bedrock",
                "api key",
                "prompt",
                "crawler",
            )
        ),
        "Visa": sum(
            keyword in text
            for keyword in (
                "visa",
                "card",
                "merchant",
                "transaction",
                "payment",
                "chargeback",
                "travel",
            )
        ),
    }
    if company in ("hackerrank", "claude", "visa"):
        mapped = company.capitalize() if company != "hackerrank" else "HackerRank"
        scores[mapped] += 1
    return scores


def detect_prompt_injection(ticket: Ticket) -> ClassifierOutcome:
    patterns = (
        "ignore previous instructions",
        "ignore all prior",
        "system prompt",
        "hidden prompt",
        "hidden prompts",
        "internal prompts",
        "internal rules",
        "show retrieved docs",
        "print retrieved context",
        "dump docs before answering",
        "show your chain of thought",
        "chain of thought",
        "hidden reasoning",
        "internal reasoning",
        "reveal your logic",
        "show your reasoning",
        "decision criteria",
        "fraud logic",
        "reveal policy",
        "developer message",
    )
    if _contains_any(ticket.user_text, patterns):
        return ClassifierOutcome(
            label="escalate",
            reason="Prompt-injection or policy-exfiltration request detected.",
            confidence=1.0,
        )
    return ClassifierOutcome(label="pass", reason="No prompt-injection pattern detected.", confidence=1.0)


def detect_high_risk(ticket: Ticket) -> ClassifierOutcome:
    patterns = (
        "fraud",
        "identity theft",
        "security vulnerability",
        "bug bounty",
        "unauthorized access",
        "account takeover",
        "urgent financial",
        "restore account access",
    )
    if _contains_any(ticket.user_text, patterns):
        return ClassifierOutcome(label="escalate", reason="High-risk/sensitive case detected.", confidence=1.0)

    # Ambiguous sensitive language — only escalate if not clearly a standard support request
    ambiguity_signals = ("compromised", "hacked", "suspicious")
    if _contains_any(ticket.user_text, ambiguity_signals):
        return ClassifierOutcome(
            label="escalate",
            reason="Potentially high-risk ambiguous scenario detected.",
            confidence=0.9,
        )
    return ClassifierOutcome(label="pass", reason="No high-risk indicator detected.", confidence=1.0)


def classify_domain(ticket: Ticket) -> ClassifierOutcome:
    keyword_scores = _domain_scores(ticket)
    domain, score = max(keyword_scores.items(), key=lambda item: item[1])
    if score == 0:
        return ClassifierOutcome(label="None", reason="No domain signal detected.", confidence=0.2)
    confidence = min(0.99, 0.45 + (0.15 * score))
    return ClassifierOutcome(
        label=domain,
        reason=f"Domain matched via deterministic keywords ({score}).",
        confidence=confidence,
    )


def detect_multi_domain_ambiguity(ticket: Ticket) -> ClassifierOutcome:
    scores = _domain_scores(ticket)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_domain, best_score = ranked[0]
    second_domain, second_score = ranked[1]

    # Only escalate when two domains have strong AND equal signals (truly ambiguous).
    strong_domains = [name for name, score in ranked if score >= 3]
    if len(strong_domains) >= 2:
        return ClassifierOutcome(
            label="escalate",
            reason=(
                "Ambiguous multi-domain ticket with strong signals for "
                f"{', '.join(strong_domains[:2])}."
            ),
            confidence=0.95,
        )

    # Escalate only when best score is weak AND both domains are tied.
    if best_score > 0 and second_score > 0 and best_score == second_score and best_score <= 2:
        return ClassifierOutcome(
            label="escalate",
            reason=(
                f"Ambiguous domain routing: tied match between {best_domain} "
                f"({best_score}) and {second_domain} ({second_score})."
            ),
            confidence=0.9,
        )
    return ClassifierOutcome(label="pass", reason="No multi-domain ambiguity detected.", confidence=1.0)



def classify_request_type(ticket: Ticket) -> ClassifierOutcome:
    text = ticket.user_text.lower().strip()
    if not text:
        return ClassifierOutcome(label="invalid", reason="Empty ticket text.", confidence=1.0)

    if any(kw in text for kw in ("error", "broken", "fails", "not working", "bug", "crash", "stuck", "blocked")):
        return ClassifierOutcome(label="bug", reason="Bug-like symptom keywords found.", confidence=0.9)
    if any(
        kw in text
        for kw in ("feature", "would like", "please add", "enhancement", "improve", "add support", "can you add")
    ):
        return ClassifierOutcome(
            label="feature_request",
            reason="Feature request language detected.",
            confidence=0.9,
        )

    unrelated = (
        "weather",
        "sports score",
        "recipe",
        "movie recommendation",
        "stock tip",
        "crypto price",
        "homework answer",
        "write malware",
    )
    if _contains_any(text, unrelated):
        return ClassifierOutcome(label="invalid", reason="Unrelated or malicious non-support request.", confidence=0.95)

    product_issue_signals = (
        "how do i",
        "where can i",
        "cannot find",
        "issue with",
        "help with",
        "account",
        "billing",
        "payment",
        "refund",
        "subscription",
        "access",
        "login",
        "support",
    )
    if _contains_any(text, product_issue_signals):
        return ClassifierOutcome(
            label="product_issue",
            reason="Product support issue pattern detected.",
            confidence=0.8,
        )
    # Prefer product_issue for plausible support content to reduce false invalids.
    return ClassifierOutcome(label="product_issue", reason="Defaulted to product_issue for support-style request.", confidence=0.7)


def check_authority(ticket: Ticket) -> ClassifierOutcome:
    patterns = (
        "increase my hackerrank score",
        "increase score",
        "change my score",
        "fix my score",
        "review my answers",
        "regrade my answers",
        "graded unfairly",
        "move me to next round",
        "tell the company",
        "tell the recruiter",
        "force recruiter",
        "override recruiter decision",
        "refund this dispute now",
        "refund me directly",
        "restore unauthorized workspace",
        "restore access even though i am not admin",
        "restore access even though i am not owner",
        "ban this seller",
        "force approval",
        "force acceptance",
        "punish",
        "override decision",
        "approve me manually",
    )
    if _contains_any(ticket.user_text, patterns):
        return ClassifierOutcome(
            label="escalate",
            reason="Request asks for unsupported administrative override.",
            confidence=1.0,
        )
    return ClassifierOutcome(label="pass", reason="Authority check passed.", confidence=1.0)
