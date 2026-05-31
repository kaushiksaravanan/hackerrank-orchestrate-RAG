"""
main.py -- Entry point for the support triage agent.

Reads a CSV of support tickets, processes each through the agent pipeline,
and writes structured results to output.csv.
"""

import argparse
import csv
import sys
import traceback
from pathlib import Path

import pandas as pd

from config import REPO_ROOT
from embeddings import create_embedder
from indexer import build_all_indices
from llm import create_llm_client
from retriever import Retriever
from agent import process_ticket


def parse_args():
    parser = argparse.ArgumentParser(
        description="Support triage agent -- resolve tickets from CSV"
    )
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / ".." / "support_tickets" / "support_tickets.csv"
    default_output = script_dir / ".." / "support_tickets" / "output.csv"

    parser.add_argument(
        "--input",
        type=str,
        default=str(default_input),
        help="Path to the input support tickets CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(default_output),
        help="Path to write the output CSV",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuild of all vector indices",
    )
    return parser.parse_args()


def make_default_result(issue, subject, company):
    """Return a default escalated result when processing fails."""
    return {
        "issue": issue,
        "subject": subject,
        "company": company,
        "response": "Unable to process this request. Escalating to human support.",
        "product_area": "",
        "status": "escalated",
        "request_type": "bug",
        "justification": "Automated processing failed; escalated for manual review.",
    }


def _get_column(row, name: str) -> str:
    """Case-insensitive column access from a DataFrame row."""
    # Try exact match first
    if name in row.index:
        val = row[name]
        return str(val) if val is not None and str(val) != "nan" else ""
    # Try case-insensitive match
    for col in row.index:
        if col.lower() == name.lower():
            val = row[col]
            return str(val) if val is not None and str(val) != "nan" else ""
    return ""


def main():
    # Load environment variables from repo root .env
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")

    args = parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Initialize LLM client (auto-detects Anthropic vs HuggingFace)
    print("Initializing LLM client...")
    llm_client = create_llm_client()
    print(f"  Provider: {llm_client.provider} ({llm_client.model})")

    # Initialize embedder
    print("Initializing embedder...")
    embedder = create_embedder()
    print(f"  Model: {embedder.model_name} ({embedder.dimension}d)")

    # Build or load indices
    print("Building/loading indices...")
    indices = build_all_indices(embedder, force_rebuild=args.rebuild_index)

    # Create retriever
    retriever = Retriever(indices, embedder)
    print("Retriever ready.\n")

    # Read input CSV
    df = pd.read_csv(input_path, dtype=str)
    df = df.fillna("")

    total = len(df)
    print(f"Processing {total} tickets from: {input_path}\n")

    results = []
    replied_count = 0
    escalated_count = 0

    for i, row in df.iterrows():
        issue = _get_column(row, "Issue")
        subject = _get_column(row, "Subject")
        company = _get_column(row, "Company")

        try:
            agent_result = process_ticket(
                issue, subject, company, retriever, llm_client,
            )

            result = {
                "issue": issue,
                "subject": subject,
                "company": company,
                "response": agent_result.get("response", ""),
                "product_area": agent_result.get("product_area", ""),
                "status": str(agent_result.get("status", "escalated")).lower(),
                "request_type": str(agent_result.get("request_type", "bug")).lower(),
                "justification": agent_result.get("justification", ""),
            }
        except Exception as e:
            print(f"  ERROR processing ticket {i+1}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            result = make_default_result(issue, subject, company)

        # Track counts
        if result["status"] == "escalated":
            escalated_count += 1
        else:
            replied_count += 1

        # Progress
        subj_display = subject[:40] if subject else "(no subject)"
        print(f"[{i+1}/{total}] {subj_display} -> {result['status']}")

        results.append(result)

    # Build output DataFrame with exact column order
    output_columns = [
        "issue",
        "subject",
        "company",
        "response",
        "product_area",
        "status",
        "request_type",
        "justification",
    ]
    out_df = pd.DataFrame(results, columns=output_columns)

    # Write output CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)

    print(f"\nDone. Replied: {replied_count}, Escalated: {escalated_count}")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
