"""Escalation response helpers."""

from __future__ import annotations

from code.models import TriageResult


def escalate(product_area: str, request_type: str, reason: str) -> TriageResult:
    return TriageResult(
        status="escalated",
        product_area=product_area,
        request_type=request_type,
        response="We cannot safely resolve this automatically. A human specialist has been engaged.",
        justification=reason,
        metadata={"escalation_reason": reason},
    )


def refuse(product_area: str, request_type: str, reason: str) -> TriageResult:
    return TriageResult(
        status="escalated",
        product_area=product_area,
        request_type=request_type,
        response=(
            "I cannot help with that request. Please ask a standard product-support question "
            "or contact official support for further assistance."
        ),
        justification=reason,
        metadata={"refusal_reason": reason},
    )
