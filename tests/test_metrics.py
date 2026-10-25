from cv_rag.evaluation.metrics import (candidate_precision, candidate_recall, hit, keyword_recall,
                                       reciprocal_rank, safe_mean)
from cv_rag.evaluation.runner import score_example, summarise, to_markdown


def test_keyword_recall():
    assert keyword_recall("Studied at NIT Trichy", ["trichy", "UCL"]) == 0.5
    assert keyword_recall("anything", []) is None


def test_candidate_recall_and_precision():
    assert candidate_recall(["A", "A", "B"], ["A", "C"]) == 0.5
    assert candidate_precision(["A", "A", "B", "C"], ["A"]) == 0.5


def test_reciprocal_rank_and_hit():
    ctxs = ["Bob knows Java", "Alice knows Kafka", "Alice knows Spark"]
    cands = ["Bob", "Alice", "Alice"]
    assert reciprocal_rank(ctxs, cands, ["Alice"], ["spark"]) == 1 / 3
    assert hit(ctxs, cands, ["Alice"], ["spark"]) == 1.0
    assert reciprocal_rank(ctxs, cands, ["Alice"], ["rust"]) == 0.0
    assert reciprocal_rank(ctxs, cands, [], ["x"]) is None


def test_safe_mean_ignores_none():
    assert safe_mean([1.0, None, 0.0]) == 0.5
    assert safe_mean([None]) is None


def test_score_and_summarise_roundtrip():
    example = {"id": "q1", "category": "single_fact", "expected_candidates": ["Alice"],
               "expected_keywords": ["Kafka"]}
    result = {"answer": "Alice used Kafka [1].", "latency_s": 1.2,
              "retrieved_contexts": ["Alice knows Kafka"], "retrieved_metadata": [{"candidate": "Alice"}]}
    row = {"id": "q1", "category": "single_fact", **score_example(example, result), "correctness": 5}
    assert row["retrieval_hit"] == 1.0 and row["answer_keyword_recall"] == 1.0
    summary = summarise([row])
    assert summary["overall"]["correctness"] == 5
    md = to_markdown(summary, [row], {"timestamp": "t"})
    assert "| q1 | single_fact |" in md


def test_off_topic_skips_retrieval_metrics():
    row = score_example({"category": "off_topic", "expected_candidates": [], "expected_keywords": []},
                        {"answer": "I only answer CV questions."})
    assert row["retrieval_hit"] is None and row["candidate_recall"] is None
