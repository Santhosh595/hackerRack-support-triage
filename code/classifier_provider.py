"""Classifier provider pattern: deterministic + optional LLM augmentation."""

from __future__ import annotations

from dataclasses import dataclass

from code.classifiers import ClassifierOutcome, classify_domain, classify_request_type
from code.config import TRIAGE_API_KEY, USE_LLM, VALID_DOMAINS, VALID_REQUEST_TYPES
from code.llm_client import LLMClient
from code.models import Ticket


@dataclass(frozen=True)
class ClassificationBundle:
    domain: ClassifierOutcome
    request_type: ClassifierOutcome
    provider_name: str


class BaseClassifierProvider:
    def classify(self, ticket: Ticket) -> ClassificationBundle:
        raise NotImplementedError


class DeterministicClassifierProvider(BaseClassifierProvider):
    def classify(self, ticket: Ticket) -> ClassificationBundle:
        return ClassificationBundle(
            domain=classify_domain(ticket),
            request_type=classify_request_type(ticket),
            provider_name="deterministic",
        )


class LLMClassifierProvider(BaseClassifierProvider):
    """LLM can improve confidence for ambiguous cases but not safety gates."""

    def __init__(self) -> None:
        self.det = DeterministicClassifierProvider()
        self.client = LLMClient()  # Auto-detects provider from environment

    def classify(self, ticket: Ticket) -> ClassificationBundle:
        baseline = self.det.classify(ticket)
        if not self.client.available:
            return baseline

        system_prompt = (
            "You are a strict classifier. Return only JSON keys: "
            "domain, request_type, confidence, reason. "
            "Allowed domain values: HackerRank, Claude, Visa, None. "
            "Allowed request_type values: product_issue, feature_request, bug, invalid."
        )
        user_prompt = (
            f"company={ticket.company}\n"
            f"ticket={ticket.user_text}\n"
            "Use ticket content first, company as a weak prior."
        )
        try:
            result = self.client.generate_json(system_prompt=system_prompt, user_prompt=user_prompt)
            domain = str(result.get("domain", baseline.domain.label))
            request_type = str(result.get("request_type", baseline.request_type.label))
            conf = float(result.get("confidence", 0.6))
            reason = str(result.get("reason", "LLM-assisted interpretation."))

            if domain not in VALID_DOMAINS:
                domain = baseline.domain.label
            if request_type not in VALID_REQUEST_TYPES:
                request_type = baseline.request_type.label

            return ClassificationBundle(
                domain=ClassifierOutcome(
                    label=domain,
                    reason=f"{reason} [llm_augmented:{self.client.provider_name}]",
                    confidence=max(0.0, min(1.0, conf)),
                ),
                request_type=ClassifierOutcome(
                    label=request_type,
                    reason=f"{reason} [llm_augmented:{self.client.provider_name}]",
                    confidence=max(0.0, min(1.0, conf)),
                ),
                provider_name=f"llm_{self.client.provider_name}",
            )
        except Exception:
            # Deterministic fallback is mandatory for reliability in evaluation.
            return baseline


def get_classifier_provider() -> BaseClassifierProvider:
    if USE_LLM:
        return LLMClassifierProvider()
    return DeterministicClassifierProvider()
