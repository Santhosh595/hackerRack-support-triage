"""End-to-end triage pipeline in required stage order."""

from __future__ import annotations

from code.classifiers import (
    check_authority,
    detect_high_risk,
    detect_multi_domain_ambiguity,
    detect_prompt_injection,
)
from code.classifier_provider import BaseClassifierProvider, get_classifier_provider
from code.config import RETRIEVAL_CONFIDENCE_THRESHOLD
from code.escalation import escalate, refuse
from code.generator import build_justification
from code.generator_provider import BaseGeneratorProvider, get_generator_provider
from code.logger import StructuredLogger
from code.models import Ticket, TriageResult
from code.retriever import DomainRetriever


class TriagePipeline:
    """Implements all required stages as deterministic gates."""

    def __init__(
        self,
        retriever: DomainRetriever,
        logger: StructuredLogger,
        classifier_provider: BaseClassifierProvider | None = None,
        generator_provider: BaseGeneratorProvider | None = None,
    ) -> None:
        self.retriever = retriever
        self.logger = logger
        self.classifier_provider = classifier_provider or get_classifier_provider()
        self.generator_provider = generator_provider or get_generator_provider()

    def run(self, ticket: Ticket) -> TriageResult:
        stage_reasons: list[str] = []
        domain_label = self._domain_from_company(ticket.company)
        request_type = "invalid"

        # 1) Prompt injection detection
        injection = detect_prompt_injection(ticket)
        stage_reasons.append(f"prompt_injection={injection.label}")
        if injection.label == "escalate":
            result = refuse(domain_label, request_type, injection.reason)
            self._log(ticket, result, stage_reasons)
            return result

        # 2) High-risk escalation detection
        high_risk = detect_high_risk(ticket)
        stage_reasons.append(f"high_risk={high_risk.label}")
        if high_risk.label == "escalate":
            result = escalate(domain_label, request_type, high_risk.reason)
            self._log(ticket, result, stage_reasons)
            return result

        # 3) Domain classification
        classifications = self.classifier_provider.classify(ticket)
        domain = classifications.domain
        domain_label = domain.label
        stage_reasons.append(f"domain={domain_label}")
        stage_reasons.append(f"domain_confidence={domain.confidence:.2f}")
        if domain_label == "None":
            # Vague outage/down reports with no domain: escalate to human
            outage_signals = ("site is down", "not working", "down", "outage", "inaccessible", "pages are accessible")
            from code.classifiers import _contains_any
            if _contains_any(ticket.user_text, outage_signals):
                result = escalate("general", "bug", "Vague outage report with no identifiable product domain.")
                self._log(ticket, result, stage_reasons)
                return result
            # Other out-of-scope/unrecognisable tickets: reply as invalid
            result = TriageResult(
                status="replied",
                product_area="general",
                request_type="invalid",
                response="I'm sorry, I can only assist with support queries related to HackerRank, Claude, or Visa. Please contact the relevant support team for your request.",
                justification="No product domain detected; ticket classified as out-of-scope.",
            )
            self._log(ticket, result, stage_reasons)
            return result

        domain_ambiguity = detect_multi_domain_ambiguity(ticket)
        stage_reasons.append(f"domain_ambiguity={domain_ambiguity.label}")
        if domain_ambiguity.label == "escalate":
            result = escalate(domain_label, request_type, domain_ambiguity.reason)
            self._log(ticket, result, stage_reasons)
            return result

        # 4) Request type classification
        req_type = classifications.request_type
        request_type = req_type.label
        stage_reasons.append(f"request_type={request_type}")
        stage_reasons.append(f"request_confidence={req_type.confidence:.2f}")
        stage_reasons.append(f"classifier_provider={classifications.provider_name}")

        # 5) Authority / responsibility check
        authority = check_authority(ticket)
        stage_reasons.append(f"authority={authority.label}")
        if authority.label == "escalate":
            result = escalate(domain_label, request_type, authority.reason)
            self._log(ticket, result, stage_reasons)
            return result

        # 6) Domain-specific retrieval
        docs = self.retriever.retrieve(domain_label, ticket.user_text)
        stage_reasons.append(f"retrieval_docs={len(docs)}")

        # 7) Retrieval confidence validation
        max_conf = max((d.score for d in docs), default=0.0)
        stage_reasons.append(f"retrieval_confidence={max_conf:.3f}")
        if max_conf < RETRIEVAL_CONFIDENCE_THRESHOLD:
            result = escalate(
                domain_label,
                request_type,
                "Insufficient retrieval confidence. Escalated to avoid hallucination.",
            )
            self._log(ticket, result, stage_reasons)
            return result

        # 8) Response generation OR escalation
        response = self.generator_provider.generate(ticket=ticket, docs=docs)
        stage_reasons.append(
            f"generator_provider={getattr(self.generator_provider, 'last_used_provider', 'unknown')}"
        )
        if not response.strip():
            result = escalate(domain_label, request_type, "Documentation insufficient for grounded answer.")
            self._log(ticket, result, stage_reasons)
            return result

        justification = build_justification(docs, stage_reasons)
        result = TriageResult(
            status="replied",
            product_area=domain_label,
            request_type=request_type,
            response=response,
            justification=justification,
            metadata={"max_confidence": max_conf},
        )

        # 9) Structured logging
        self._log(ticket, result, stage_reasons)
        return result

    def _log(self, ticket: Ticket, result: TriageResult, stage_reasons: list[str]) -> None:
        self.logger.log(
            {
                "ticket_id": ticket.ticket_id,
                "company": ticket.company,
                "status": result.status,
                "product_area": result.product_area,
                "request_type": result.request_type,
                "justification": result.justification,
                "pipeline_trace": stage_reasons,
            }
        )

    @staticmethod
    def _domain_from_company(company: str) -> str:
        mapping = {
            "hackerrank": "HackerRank",
            "claude": "Claude",
            "visa": "Visa",
        }
        return mapping.get(company.lower().strip(), "None")
