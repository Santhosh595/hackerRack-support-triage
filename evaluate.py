#!/usr/bin/env python3
"""Evaluator for the canonical v3 pipeline in `code/`."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from code.logger import StructuredLogger
from code.models import Ticket
from code.pipeline import TriagePipeline
from code.retriever import DomainRetriever


@dataclass(frozen=True)
class Fixture:
    ticket_id: str
    company: str
    text: str
    expected_domain: str
    expected_request_type: str
    expected_status: str


FIXTURES = [
    Fixture(
        ticket_id="FX-001",
        company="HackerRank",
        text="My coding test page shows a UI bug and fails to load timer.",
        expected_domain="HackerRank",
        expected_request_type="bug",
        expected_status="replied",
    ),
    Fixture(
        ticket_id="FX-002",
        company="Claude",
        text="How do I recover workspace access after lockout?",
        expected_domain="Claude",
        expected_request_type="product_issue",
        expected_status="replied",
    ),
    Fixture(
        ticket_id="FX-003",
        company="Visa",
        text="There is fraud on my card and I need urgent help.",
        expected_domain="Visa",
        expected_request_type="invalid",
        expected_status="escalated",
    ),
    Fixture(
        ticket_id="FX-004",
        company="HackerRank",
        text="Please add a feature to export all assessment attempts as CSV.",
        expected_domain="HackerRank",
        expected_request_type="feature_request",
        expected_status="replied",
    ),
]


def run_fixture(pipeline: TriagePipeline, fixture: Fixture) -> tuple[bool, bool, bool, float]:
    ticket = Ticket(ticket_id=fixture.ticket_id, user_text=fixture.text, company=fixture.company)
    t0 = time.perf_counter()
    result = pipeline.run(ticket)
    latency_ms = (time.perf_counter() - t0) * 1000

    domain_ok = result.product_area == fixture.expected_domain
    request_ok = result.request_type == fixture.expected_request_type
    status_ok = result.status == fixture.expected_status
    return domain_ok, request_ok, status_ok, latency_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v3 triage evaluator fixtures.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-fixture checks")
    args = parser.parse_args()

    # Canonical v3 stack only: no imports from legacy modules/.
    pipeline = TriagePipeline(
        retriever=DomainRetriever(),
        logger=StructuredLogger(Path("logs/eval_v3.jsonl")),
    )

    domain_hits = 0
    request_hits = 0
    status_hits = 0
    latencies: list[float] = []

    for fixture in FIXTURES:
        domain_ok, request_ok, status_ok, latency_ms = run_fixture(pipeline, fixture)
        latencies.append(latency_ms)
        domain_hits += int(domain_ok)
        request_hits += int(request_ok)
        status_hits += int(status_ok)
        if args.verbose:
            print(
                f"{fixture.ticket_id}: domain={'OK' if domain_ok else 'FAIL'} "
                f"request_type={'OK' if request_ok else 'FAIL'} "
                f"status={'OK' if status_ok else 'FAIL'} "
                f"latency_ms={latency_ms:.1f}"
            )

    n = len(FIXTURES)
    print("\n=== V3 Evaluation Summary ===")
    print(f"fixtures: {n}")
    print(f"domain_accuracy: {domain_hits / n:.3f}")
    print(f"request_type_accuracy: {request_hits / n:.3f}")
    print(f"status_accuracy: {status_hits / n:.3f}")
    print(f"avg_latency_ms: {sum(latencies) / max(len(latencies), 1):.1f}")


if __name__ == "__main__":
    main()
