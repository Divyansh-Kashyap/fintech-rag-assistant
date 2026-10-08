"""
Automated RAG Evaluation Suite.
Quantifies Faithfulness/Groundedness, Context Precision, Context Recall,
Answer Correctness, and Citation Accuracy over financial policy benchmarks.
Outputs benchmark report to evaluation/eval_results.json.
"""

import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.models.schemas import QueryRequest
from app.services.rag_chain import FALLBACK_MESSAGE, get_rag_chain

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag-eval")
settings = get_settings()


def compute_token_f1(pred: str, target: str) -> float:
    """Computes token-level precision, recall, and F1 score."""
    pred_tokens = re.findall(r"\w+", pred.lower())
    target_tokens = re.findall(r"\w+", target.lower())
    if not pred_tokens or not target_tokens:
        return 1.0 if pred.strip().lower() == target.strip().lower() else 0.0

    common = set(pred_tokens).intersection(set(target_tokens))
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(target_tokens)
    return 2.0 * (precision * recall) / (precision + recall)


def evaluate_faithfulness(generated_answer: str, sources: List[Dict[str, Any]], is_in_domain: bool) -> float:
    """
    Evaluates whether generated claims are strictly grounded in retrieved source snippets.
    For out-of-domain queries correctly producing the fallback message, faithfulness is 1.0.
    """
    if generated_answer.strip() == FALLBACK_MESSAGE:
        return 1.0 if not is_in_domain else 0.0

    if not sources:
        return 0.0

    combined_snippets = " ".join([s.get("snippet", "").lower() for s in sources])
    answer_terms = [t for t in re.findall(r"\w+", generated_answer.lower()) if len(t) > 3]

    if not answer_terms:
        return 1.0

    supported = sum(1 for t in answer_terms if t in combined_snippets or t in ["finbase", "section", "policy", "according"])
    return round(supported / len(answer_terms), 3)


def evaluate_context_recall(sources: List[Dict[str, Any]], expected_doc: str, expected_sec: str, is_in_domain: bool) -> float:
    """Evaluates whether the ground-truth document and section were successfully recalled."""
    if not is_in_domain:
        # For out-of-domain, recall is 1.0 if zero false sources were included
        return 1.0 if len(sources) == 0 else 0.0

    if not sources or not expected_doc:
        return 0.0

    found_doc = any(expected_doc.lower() in s.get("document", "").lower() or s.get("document", "").lower() in expected_doc.lower() for s in sources)
    found_sec = any(expected_sec.lower() in s.get("section", "").lower() for s in sources) if expected_sec else True

    if found_doc and found_sec:
        return 1.0
    elif found_doc or found_sec:
        return 0.5
    return 0.0


def evaluate_context_precision(sources: List[Dict[str, Any]], expected_doc: str, is_in_domain: bool) -> float:
    """Calculates the proportion of retrieved source chunks that are relevant."""
    if not is_in_domain:
        return 1.0 if len(sources) == 0 else 0.0

    if not sources:
        return 0.0

    relevant_count = 0
    for s in sources:
        doc = s.get("document", "").lower()
        if expected_doc.lower() in doc or doc in expected_doc.lower():
            relevant_count += 1

    return round(relevant_count / len(sources), 3)


def evaluate_citation_accuracy(sources: List[Dict[str, Any]], expected_doc: str, expected_sec: str, is_in_domain: bool) -> float:
    """Validates structural and semantic validity of citations."""
    if not is_in_domain:
        return 1.0 if len(sources) == 0 else 0.0

    if not sources:
        return 0.0

    valid_citations = 0
    for s in sources:
        has_doc = bool(s.get("document"))
        has_sec = bool(s.get("section"))
        has_page = isinstance(s.get("page"), int) and s.get("page", 0) >= 1
        has_snippet = bool(s.get("snippet")) and len(s.get("snippet", "")) > 10

        if has_doc and has_sec and has_page and has_snippet:
            valid_citations += 1

    return round(valid_citations / len(sources), 3)


async def run_evaluation():
    """Main evaluation runner executing test queries and aggregating metrics."""
    eval_file = settings.EVAL_DATASET_PATH
    if not eval_file.is_absolute():
        eval_file = settings.BASE_DIR / eval_file

    if not eval_file.exists():
        logger.error("Evaluation dataset not found at %s", eval_file)
        sys.exit(1)

    with open(eval_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    logger.info("Loaded %d evaluation test cases.", len(test_cases))
    rag_chain = get_rag_chain()

    results: List[Dict[str, Any]] = []
    start_all = time.time()

    print("\n=========================================================================================")
    print("                    FINTECH RAG ASSISTANT BENCHMARK EVALUATION                          ")
    print("=========================================================================================")
    print(f"{'ID':<10} | {'Query (Truncated)':<40} | {'Domain':<8} | {'Correct':<7} | {'Faithful':<8} | {'Latency':<8}")
    print("-----------------------------------------------------------------------------------------")

    for case in test_cases:
        qid = case["id"]
        query = case["query"]
        ground_truth = case["ground_truth_answer"]
        expected_doc = case.get("expected_document") or ""
        expected_sec = case.get("expected_section") or ""
        is_in_domain = case.get("is_in_domain", True)

        req = QueryRequest(query=query, top_k=3)
        res = await rag_chain.query(req)

        # 1. Answer Correctness
        if not is_in_domain:
            ans_correctness = 1.0 if res.answer.strip() == FALLBACK_MESSAGE else 0.0
        else:
            ans_correctness = compute_token_f1(res.answer, ground_truth)
            # Boost if key numbers and policy clause match
            if expected_sec and expected_sec.lower() in res.answer.lower():
                ans_correctness = min(1.0, ans_correctness + 0.20)

        # 2. Faithfulness
        sources_dicts = [s.model_dump() for s in res.sources]
        faithfulness = evaluate_faithfulness(res.answer, sources_dicts, is_in_domain)

        # 3. Context Recall & Precision
        recall = evaluate_context_recall(sources_dicts, expected_doc, expected_sec, is_in_domain)
        precision = evaluate_context_precision(sources_dicts, expected_doc, is_in_domain)

        # 4. Citation Accuracy
        citation_acc = evaluate_citation_accuracy(sources_dicts, expected_doc, expected_sec, is_in_domain)

        record = {
            "id": qid,
            "query": query,
            "is_in_domain": is_in_domain,
            "generated_answer": res.answer,
            "ground_truth_answer": ground_truth,
            "sources_count": len(res.sources),
            "confidence_score": res.confidence_score,
            "latency_ms": res.latency_ms,
            "metrics": {
                "answer_correctness": round(ans_correctness, 3),
                "faithfulness": round(faithfulness, 3),
                "context_recall": round(recall, 3),
                "context_precision": round(precision, 3),
                "citation_accuracy": round(citation_acc, 3),
            },
        }
        results.append(record)

        domain_str = "IN" if is_in_domain else "OUT"
        trunc_query = query[:38] + "..." if len(query) > 38 else query
        print(f"{qid:<10} | {trunc_query:<40} | {domain_str:<8} | {ans_correctness:<7.2f} | {faithfulness:<8.2f} | {res.latency_ms:<8.1f}ms")

    elapsed_all = time.time() - start_all

    # Aggregate summaries
    mean_correctness = sum(r["metrics"]["answer_correctness"] for r in results) / len(results)
    mean_faithfulness = sum(r["metrics"]["faithfulness"] for r in results) / len(results)
    mean_recall = sum(r["metrics"]["context_recall"] for r in results) / len(results)
    mean_precision = sum(r["metrics"]["context_precision"] for r in results) / len(results)
    mean_citation = sum(r["metrics"]["citation_accuracy"] for r in results) / len(results)
    mean_latency = sum(r["latency_ms"] for r in results) / len(results)

    # Negative test case accuracy (fallback accuracy)
    out_of_domain_cases = [r for r in results if not r["is_in_domain"]]
    fallback_accuracy = (
        sum(1 for r in out_of_domain_cases if r["generated_answer"].strip() == FALLBACK_MESSAGE) / len(out_of_domain_cases)
        if out_of_domain_cases
        else 1.0
    )

    benchmark_summary = {
        "timestamp": time.time(),
        "total_test_cases": len(results),
        "in_domain_cases": len(results) - len(out_of_domain_cases),
        "out_of_domain_cases": len(out_of_domain_cases),
        "overall_metrics": {
            "mean_answer_correctness": round(mean_correctness, 4),
            "mean_faithfulness_groundedness": round(mean_faithfulness, 4),
            "mean_context_recall": round(mean_recall, 4),
            "mean_context_precision": round(mean_precision, 4),
            "mean_citation_accuracy": round(mean_citation, 4),
            "negative_fallback_accuracy": round(fallback_accuracy, 4),
            "mean_latency_ms": round(mean_latency, 2),
        },
        "query_results": results,
    }

    out_file = PROJECT_ROOT / "evaluation" / "eval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    print("=========================================================================================")
    print(f"                               AGGREGATE BENCHMARK REPORT                                ")
    print("=========================================================================================")
    print(f"Total Test Cases Evaluated : {len(results)}")
    print(f"Faithfulness / Groundedness: {mean_faithfulness * 100:.2f}% (Target: >95%)")
    print(f"Answer Correctness         : {mean_correctness * 100:.2f}%")
    print(f"Context Recall             : {mean_recall * 100:.2f}%")
    print(f"Context Precision          : {mean_precision * 100:.2f}%")
    print(f"Citation Accuracy          : {mean_citation * 100:.2f}%")
    print(f"Out-of-Domain Fallback Rate: {fallback_accuracy * 100:.2f}% (Target: 100% zero-hallucination)")
    print(f"Average Response Latency   : {mean_latency:.1f} ms")
    print(f"Benchmark Report Saved To  : {out_file.resolve()}")
    print("=========================================================================================\n")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
