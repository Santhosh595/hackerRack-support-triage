"""I/O helpers for CSV outputs."""

from __future__ import annotations

import csv
from pathlib import Path

from code.models import TriageResult


CSV_FIELDS = ["status", "product_area", "response", "justification", "request_type"]


def append_result_csv(path: Path, result: TriageResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "status": result.status,
                "product_area": result.product_area,
                "response": result.response,
                "justification": result.justification,
                "request_type": result.request_type,
            }
        )
