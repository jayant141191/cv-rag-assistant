"""Graph routing tests with a scripted fake LLM — no API key needed."""

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_openai")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from cv_rag.config import Settings  # noqa: E402
from cv_rag.graph import (CVRagGraph, GroundednessCheck, QueryAnalysis,  # noqa: E402
                          RelevantPassages, RewrittenQuery)


class FakeLLM:
    """Returns queued responses per structured-output schema."""

    def __init__(self, responses: dict, answer: str = "Alice worked at Acme [1]."):
        self.responses = {k: list(v) for k, v in responses.items()}
        self.answer = answer

    def with_structured_output(self, schema):
        llm = self

        class _Runner:
            def invoke(self, _messages):
                return llm.responses[schema].pop(0)

        return _Runner()

    def invoke(self, _messages):
        return AIMessage(self.answer)


DOC = Document(page_content="Candidate: Alice | Section: Experience\nAcme 2020-2024",
               metadata={"candidate": "Alice", "section": "Experience", "source": "alice.md"})


def make_graph(responses, docs_per_call):
    calls = iter(docs_per_call)
    return CVRagGraph(retriever=lambda q, c: next(calls), llm=FakeLLM(responses), settings=Settings())


def test_happy_path():
    g = make_graph({
        QueryAnalysis: [QueryAnalysis(is_cv_question=True, standalone_query="Alice employer",
                                      candidates=["Alice"])],
        RelevantPassages: [RelevantPassages(relevant_indices=[0])],
        GroundednessCheck: [GroundednessCheck(grounded=True)],
    }, [[DOC]])
    out = g.ask("Where did Alice work?", known_candidates=["Alice"])
    assert "Acme" in out["answer"] and out["grounded"] is True
    assert out["sources"][0]["candidate"] == "Alice"


def test_off_topic_short_circuits():
    g = make_graph({QueryAnalysis: [QueryAnalysis(is_cv_question=False, standalone_query="capital")]}, [])
    out = g.ask("What's the capital of France?", known_candidates=["Alice"])
    assert "CV" in out["answer"] and out["sources"] == []


def test_rewrite_then_no_context():
    g = make_graph({
        QueryAnalysis: [QueryAnalysis(is_cv_question=True, standalone_query="salary")],
        RelevantPassages: [RelevantPassages(relevant_indices=[]), RelevantPassages(relevant_indices=[])],
        RewrittenQuery: [RewrittenQuery(query="expected compensation")],
    }, [[DOC], [DOC]])
    out = g.ask("What salary does Alice expect?", known_candidates=["Alice"])
    assert any(step.startswith("rewrite") for step in out["trace"])
    assert "couldn't find" in out["answer"]


def test_ungrounded_answer_is_regenerated():
    g = make_graph({
        QueryAnalysis: [QueryAnalysis(is_cv_question=True, standalone_query="Alice")],
        RelevantPassages: [RelevantPassages(relevant_indices=[0])],
        GroundednessCheck: [GroundednessCheck(grounded=False, reason="x"), GroundednessCheck(grounded=True)],
    }, [[DOC]])
    out = g.ask("Tell me about Alice", known_candidates=["Alice"])
    assert sum(step == "generate" for step in out["trace"]) == 2


def test_unknown_candidate_names_are_dropped():
    g = make_graph({
        QueryAnalysis: [QueryAnalysis(is_cv_question=True, standalone_query="Zed", candidates=["Zed"])],
        RelevantPassages: [RelevantPassages(relevant_indices=[0])],
        GroundednessCheck: [GroundednessCheck(grounded=True)],
    }, [[DOC]])
    out = g.ask("Tell me about Zed", known_candidates=["Alice"])
    assert "candidates=[]" in out["trace"][0]
