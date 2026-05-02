"""Terminal entrypoint for the support triage agent.

Usage:
    python main.py              # Interactive mode
    python main.py --batch      # Batch: read support_tickets/support_tickets.csv
                                #        write support_tickets/output.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import uuid

from colorama import Fore, Style, init

from code.config import LOG_PATH, OUTPUT_CSV, SUPPORT_TICKETS_DIR, TRIAGE_API_KEY, TRIAGE_LLM_URL, USE_LLM
from code.logger import StructuredLogger
from code.models import Ticket
from code.pipeline import TriagePipeline
from code.retriever import DomainRetriever
from code.utils import append_result_csv

init(autoreset=True)


def _prompt(text: str) -> str:
    return input(f"{Fore.CYAN}{text}{Style.RESET_ALL}").strip()


def _badge(status: str) -> str:
    if status == "replied":
        return f"{Fore.GREEN}[REPLIED]{Style.RESET_ALL}"
    if status == "escalated":
        return f"{Fore.YELLOW}[ESCALATED]{Style.RESET_ALL}"
    return f"{Fore.RED}[{status.upper()}]{Style.RESET_ALL}"


def _print_banner() -> None:
    print(f"{Style.BRIGHT}{Fore.BLUE}+------------------------------------------------------+")
    print("|        AI SUPPORT TRIAGE AGENT - HACKATHON v3       |")
    print(f"+------------------------------------------------------+{Style.RESET_ALL}")


def _print_progress() -> None:
    steps = ["Safety checks", "Classification", "Retrieval", "Response"]
    print(f"{Style.BRIGHT}Pipeline Progress{Style.RESET_ALL}")
    for step in steps:
        print(f"  {Fore.MAGENTA}*{Style.RESET_ALL} {step} ...", end="")
        time.sleep(0.06)
        print(f" {Fore.GREEN}done{Style.RESET_ALL}")


def _print_result_card(result_status: str, product_area: str, request_type: str, response: str, justification: str) -> None:
    print(f"\n{Style.BRIGHT}{Fore.WHITE}+------------------ TRIAGE RESULT ------------------+{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}Status:{Style.RESET_ALL} {_badge(result_status)}")
    print(f"{Style.BRIGHT}Product Area:{Style.RESET_ALL} {product_area}")
    print(f"{Style.BRIGHT}Request Type:{Style.RESET_ALL} {request_type}")
    print(f"{Style.BRIGHT}Response:{Style.RESET_ALL} {response}")
    print(f"{Style.BRIGHT}Justification Trace:{Style.RESET_ALL}")
    for item in [seg.strip() for seg in justification.split(";") if seg.strip()]:
        print(f"  - {item}")
    print(f"{Style.BRIGHT}{Fore.WHITE}+----------------------------------------------------+{Style.RESET_ALL}\n")


def run_batch(pipeline: TriagePipeline) -> None:
    """Read support_tickets.csv, process every row, write output.csv."""
    input_csv = SUPPORT_TICKETS_DIR / "support_tickets.csv"
    output_csv = OUTPUT_CSV

    if not input_csv.exists():
        print(f"{Fore.RED}ERROR: {input_csv} not found.{Style.RESET_ALL}")
        sys.exit(1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    CSV_FIELDS = ["status", "product_area", "response", "justification", "request_type"]

    with (
        input_csv.open(encoding="utf-8", newline="") as in_f,
        output_csv.open("w", encoding="utf-8", newline="") as out_f,
    ):
        reader = csv.DictReader(in_f)
        writer = csv.DictWriter(out_f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        rows = list(reader)
        total = len(rows)
        print(f"\n{Style.BRIGHT}Processing {total} tickets...{Style.RESET_ALL}\n")

        for i, row in enumerate(rows, start=1):
            issue = (row.get("Issue") or row.get("issue") or "").strip()
            subject = (row.get("Subject") or row.get("subject") or "").strip()
            company = (row.get("Company") or row.get("company") or "").strip()

            # Combine issue + subject for richer retrieval signal
            combined_text = issue
            if subject and subject.lower() not in issue.lower():
                combined_text = f"{subject}. {issue}"

            ticket = Ticket(
                ticket_id=str(uuid.uuid4()),
                user_text=combined_text,
                company=company,
            )

            result = pipeline.run(ticket)

            writer.writerow({
                "status": result.status,
                "product_area": result.product_area,
                "response": result.response,
                "justification": result.justification,
                "request_type": result.request_type,
            })
            out_f.flush()

            status_badge = "✓" if result.status == "replied" else "↑"
            print(f"  [{i:02d}/{total}] {status_badge} {result.status:10s}  {result.product_area:25s}  {issue[:60]}")

    print(f"\n{Fore.GREEN}{Style.BRIGHT}Done! Output written to: {output_csv}{Style.RESET_ALL}\n")


def run_interactive(pipeline: TriagePipeline) -> None:
    """Interactive REPL mode."""
    _print_banner()
    print("Enter 'exit' as ticket text to stop.\n")

    if USE_LLM and not TRIAGE_LLM_URL:
        print(
            "WARNING: TRIAGE_API_KEY is set but TRIAGE_LLM_URL is missing. "
            "LLM augmentation will fallback to deterministic mode."
        )
    elif TRIAGE_API_KEY:
        print("LLM augmentation is enabled.\n")

    while True:
        company = _prompt("Company [HackerRank/Claude/Visa/blank] > ")
        ticket_text = _prompt("Ticket > ")
        if ticket_text.lower() == "exit":
            print(f"{Fore.CYAN}Session ended.{Style.RESET_ALL}")
            break

        ticket = Ticket(
            ticket_id=str(uuid.uuid4()),
            user_text=ticket_text,
            company=company,
        )
        _print_progress()
        result = pipeline.run(ticket)
        append_result_csv(OUTPUT_CSV, result)
        _print_result_card(
            result_status=result.status,
            product_area=result.product_area,
            request_type=result.request_type,
            response=result.response,
            justification=result.justification,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Support Triage Agent")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: read support_tickets.csv, write output.csv",
    )
    args = parser.parse_args()

    retriever = DomainRetriever()
    logger = StructuredLogger(LOG_PATH)
    pipeline = TriagePipeline(retriever=retriever, logger=logger)

    if args.batch:
        run_batch(pipeline)
    else:
        run_interactive(pipeline)


if __name__ == "__main__":
    main()
