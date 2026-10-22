"""Run with python -m evaluation.run --document-id UUID (server must be running)."""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from uuid import UUID

import httpx

SCORER_VERSION = "2-quote-context"

def normalize(text):
    return " ".join(text.casefold().split())


def source_found(source, matches):
    return any(
        match["filename"] == source["filename"]
        and match["page_number"] == source["page_number"]
        and normalize(source["passage"]) in normalize(match["text"])
        for match in matches
    )


def score(case, matches, answer):
    gold = case["sources"]
    citations = answer["citations"]
    # This checks source location + quote presence, NOT logical entailment.
    citation_checks = [any(
        citation["filename"] == source["filename"]
        and citation["page_number"] == source["page_number"]
        and bool(normalize(citation["quote"]))
        and (
            normalize(citation["quote"]) in normalize(source["passage"])
            or normalize(source["passage"]) in normalize(citation["quote"])
        )
        and any(
            match["filename"] == citation["filename"]
            and match["page_number"] == citation["page_number"]
            and normalize(citation["quote"]) in normalize(match["text"])
            for match in matches
        )
        for source in gold
    ) for citation in citations]
    return {
        "retrieval_recall_at_k": sum(source_found(s, matches) for s in gold) / len(gold) if gold else None,
        "answer_phrase_match": (
            not answer["abstained"]
            and all(normalize(p) in normalize(answer["answer"]) for p in case["required_phrases"])
        ) if case["answerable"] else None,
        "abstention_correct": answer["abstained"] == (not case["answerable"]),
        "citation_location_quote_passes": sum(citation_checks),
        "citation_count": len(citations),
        "manual_answer_correct": None,
        "manual_faithfulness": None,
        "manual_citation_correct": None,
    }


def summarize(rows):
    successful = [row for row in rows if row["status"] == "ok"]
    def average(field):
        values = [r["scores"][field] for r in successful if r["scores"][field] is not None]
        return {"value": mean(values) if values else None, "cases": len(values)}
    citation_count = sum(r["scores"]["citation_count"] for r in successful)
    unsupported = [r for r in successful if not r["answerable"]]
    return {
        "attempted_cases": len(rows), "successful_cases": len(successful),
        "error_cases": len(rows) - len(successful),
        "macro_retrieval_recall_at_k": average("retrieval_recall_at_k"),
        "answer_phrase_match_rate": average("answer_phrase_match"),
        "abstention_decision_accuracy": average("abstention_correct"),
        "unanswerable_abstention_rate": {
            "value": mean(r["answer"]["abstained"] for r in unsupported) if unsupported else None,
            "cases": len(unsupported),
        },
        "citation_location_quote_rate": {
            "value": sum(r["scores"]["citation_location_quote_passes"] for r in successful) / citation_count if citation_count else None,
            "citations": citation_count,
        },
        "mean_search_latency_ms": mean(r["search_latency_ms"] for r in successful) if successful else None,
        "mean_ask_latency_ms": mean(r["ask_latency_ms"] for r in successful) if successful else None,
        "token_usage": None,
        "token_usage_note": "Not exposed by these endpoints; not measured.",
        "manual_metrics_note": "Answer correctness, faithfulness, and semantic citation correctness require review; not scored automatically.",
    }


async def run(args):
    dataset_bytes = args.dataset.read_bytes()
    dataset = json.loads(dataset_bytes)
    rows = []
    async with httpx.AsyncClient(base_url=args.base_url, timeout=180) as client:
        # Verify the selected document contains the gold source passages before spending quota.
        document = await client.get(f"/documents/{args.document_id}/chunks")
        document.raise_for_status()
        stored = document.json()["chunks"]
        if not all(source_found(s, stored) for case in dataset["cases"] for s in case["sources"]):
            raise ValueError("Selected document does not contain every dataset source passage. Ingest the original synthetic-policy.pdf.")
        for case in dataset["cases"]:
            row = {"case_id": case["id"], "question": case["question"],
                   "expected_answer": case["expected_answer"], "answerable": case["answerable"],
                   "gold_sources": case["sources"], "status": "error"}
            phase = "search"
            try:
                payload = {"question": case["question"], "top_k": args.top_k}
                started = perf_counter()
                search = await client.post(f"/documents/{args.document_id}/search", json=payload)
                row["search_latency_ms"] = round((perf_counter() - started) * 1000, 2)
                search.raise_for_status()
                row["matches"] = search.json()["matches"]
                phase = "ask"
                started = perf_counter()
                answer = await client.post(f"/documents/{args.document_id}/ask", json=payload)
                row["ask_latency_ms"] = round((perf_counter() - started) * 1000, 2)
                answer.raise_for_status()
                row["answer"] = answer.json()
                row["scores"] = score(case, row["matches"], row["answer"])
                row["status"] = "ok"
            except httpx.HTTPStatusError as exc:
                row["error"] = {"phase": phase, "http_status": exc.response.status_code}
            except httpx.RequestError:
                row["error"] = {"phase": phase, "type": "network_error"}
            rows.append(row)
            print(f"{case['id']}: {row['status']}")
    report = {
        "dataset_id": dataset["dataset_id"], "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "scorer_version": SCORER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "document_id": str(args.document_id), "top_k": args.top_k,
        "summary": summarize(rows), "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved {args.output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-id", type=UUID, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--top-k", type=int, choices=range(1, 11), default=2)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("policy_cases.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/latest.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists. Choose a new --output path to preserve previous results.")
    try:
        asyncio.run(run(args))
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as exc:
        # No raw provider bodies, URLs, headers, or credentials in console output.
        raise SystemExit(f"Evaluation could not start ({type(exc).__name__}). Check server, document ID, and dataset passages.") from None


if __name__ == "__main__":
    main()
