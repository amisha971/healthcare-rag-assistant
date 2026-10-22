import json
from pathlib import Path

import pytest

from evaluation.run import score, summarize

SOURCE = {"filename": "synthetic-policy.pdf", "page_number": 3,
          "passage": "Report missing inspection entries to the shift coordinator."}
CASE = {"sources": [SOURCE], "answerable": True, "required_phrases": ["shift coordinator"]}
MATCH = {**SOURCE, "text": SOURCE["passage"]}
ANSWER = {"answer": "Notify the shift coordinator. [1]", "abstained": False,
          "citations": [{"filename": SOURCE["filename"], "page_number": 3, "quote": SOURCE["passage"]}]}


def test_gold_retrieval_and_partial_recall():
    two_sources = {**CASE, "sources": [SOURCE, {**SOURCE, "page_number": 1}]}
    assert score(two_sources, [MATCH], ANSWER)["retrieval_recall_at_k"] == 0.5
    assert score(CASE, [{**MATCH, "filename": "other.pdf"}], ANSWER)["retrieval_recall_at_k"] == 0


def test_wrong_citation_and_missing_answer_phrase():
    answer = {**ANSWER, "answer": "Notify the doctor.", "citations": [{**ANSWER["citations"][0], "page_number": 1}]}
    result = score(CASE, [MATCH], answer)
    assert result["answer_phrase_match"] is False
    assert result["citation_location_quote_passes"] == 0
    assert result["manual_citation_correct"] is None


def test_quote_can_include_heading_when_it_is_present_in_source():
    text = "Equipment Workflow\n" + SOURCE["passage"]
    answer = {**ANSWER, "citations": [{**ANSWER["citations"][0], "quote": text}]}
    assert score(CASE, [{**MATCH, "text": text}], answer)["citation_location_quote_passes"] == 1


def test_longer_quote_with_fabricated_context_is_rejected():
    answer = {**ANSWER, "citations": [{**ANSWER["citations"][0], "quote": "Invented header " + SOURCE["passage"]}]}
    assert score(CASE, [MATCH], answer)["citation_location_quote_passes"] == 0


def test_unrelated_quote_on_gold_page_is_rejected():
    answer = {**ANSWER, "citations": [{**ANSWER["citations"][0], "quote": "Equipment Workflow"}]}
    assert score(CASE, [{**MATCH, "text": "Equipment Workflow\n" + SOURCE["passage"]}], answer)["citation_location_quote_passes"] == 0


def test_unanswerable_has_no_retrieval_denominator():
    result = score({"sources": [], "answerable": False, "required_phrases": []}, [],
                   {"answer": "Cannot answer", "abstained": True, "citations": []})
    assert result["retrieval_recall_at_k"] is None
    assert result["answer_phrase_match"] is None
    assert result["abstention_correct"] is True


def test_errors_are_counted_separately():
    rows = [{"status": "ok", "answerable": True, "answer": ANSWER,
             "scores": score(CASE, [MATCH], ANSWER), "search_latency_ms": 12, "ask_latency_ms": 80},
            {"status": "error", "error": {"http_status": 503}}]
    summary = summarize(rows)
    assert summary["successful_cases"] == 1 and summary["error_cases"] == 1
    assert summary["macro_retrieval_recall_at_k"] == {"value": 1, "cases": 1}
    assert summary["mean_ask_latency_ms"] == 80
    assert summary["unanswerable_abstention_rate"]["value"] is None
    assert summary["token_usage"] is None


def test_empty_summary_has_no_invented_scores():
    summary = summarize([])
    assert summary["macro_retrieval_recall_at_k"]["value"] is None
    assert summary["citation_location_quote_rate"]["value"] is None
    assert summary["mean_ask_latency_ms"] is None


def test_dataset_gold_matches_actual_synthetic_extraction():
    import pymupdf
    from app.services.chunker import chunk_pages
    from app.services.pdf_extractor import extract_pdf_pages
    from evaluation.run import source_found
    dataset = json.loads((Path(__file__).parents[1] / "evaluation/policy_cases.json").read_text())
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Staff must record equipment inspections in the daily checklist.")
        document.new_page()
        page = document.new_page()
        page.insert_text((72, 72), "Report missing inspection entries to the shift coordinator.")
        chunks = chunk_pages(extract_pdf_pages(document.tobytes()), "synthetic-policy.pdf")
    assert len({case["id"] for case in dataset["cases"]}) == 6
    assert all(source_found(s, chunks) for case in dataset["cases"] for s in case["sources"])
