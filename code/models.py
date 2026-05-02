"""Datamodels shared across pipeline components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Ticket:
    """Incoming support ticket."""

    ticket_id: str
    user_text: str
    company: str = ""


@dataclass(frozen=True)
class RetrievedDoc:
    """Single retrieved document chunk."""

    source: str
    content: str
    score: float


@dataclass
class TriageResult:
    """Final output record with required CSV schema."""

    status: str
    product_area: str
    response: str
    justification: str
    request_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
