"""LangGraph RAG pipeline for CV question answering.

Flow (corrective / self-checking RAG):

    analyze ──► off_topic ──► END
       │
       ▼
    retrieve ──► grade ──► generate ──► check_grounded ──► END
       ▲           │           ▲              │
       │           ▼           └── (retry) ───┘
       └──────  rewrite          (not grounded)
                   │
                   ▼ (out of rewrites)
               no_context ──► END
"""

from __future__ import annotations

import time
from typing import Callable, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from . import prompts
from .config import Settings, get_settings

Retriever = Callable[[str, list[str] | None], list[Document]]


# --------------------------------------------------------------------- schemas
class QueryAnalysis(BaseModel):
    is_cv_question: bool = Field(description="True if the message is about the CVs/candidates.")
    standalone_query: str = Field(description="Self-contained search query.")
    candidates: list[str] = Field(default_factory=list, description="Exact names of targeted candidates.")


class RelevantPassages(BaseModel):
    relevant_indices: list[int] = Field(default_factory=list)


class RewrittenQuery(BaseModel):
    query: str


class GroundednessCheck(BaseModel):
    grounded: bool
    reason: str = ""


# ----------------------------------------------------------------------- state
class RAGState(TypedDict, total=False):
    question: str
    chat_history: list[tuple[str, str]]          # [("user"|"assistant", text), ...]
    known_candidates: list[str]
    candidate_filter: list[str]                   # set explicitly by the UI (optional)

    query: str
    target_candidates: list[str]
    is_cv_question: bool
    documents: list[Document]
    retrieved_documents: list[Document]           # before grading, kept for evaluation
    rewrites: int
    generations: int
    strict: bool
    answer: str
    grounded: bool
    trace: list[str]


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] (source: {d.metadata.get('source', '?')})\n{d.page_content}"
                       for i, d in enumerate(docs, start=1))


def _history_messages(history: list[tuple[str, str]] | None, max_turns: int = 6) -> list[BaseMessage]:
    msgs: list[BaseMessage] = []
    for role, text in (history or [])[-max_turns * 2:]:
        msgs.append(HumanMessage(text) if role == "user" else AIMessage(text))
    return msgs


def _trace(state: RAGState, step: str) -> list[str]:
    return [*state.get("trace", []), step]


# ----------------------------------------------------------------------- graph
class CVRagGraph:
    def __init__(
        self,
        retriever: Retriever,
        llm: BaseChatModel | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        if llm is None:
            self.settings.require_api_key()
            llm = ChatOpenAI(model=self.settings.chat_model, temperature=0,
                             api_key=self.settings.openai_api_key)
        self.llm = llm
        self.retriever = retriever
        self.app = self._build()

    # ------------------------------------------------------------- nodes
    def analyze(self, state: RAGState) -> RAGState:
        known = state.get("known_candidates", [])
        planner = self.llm.with_structured_output(QueryAnalysis)
        result: QueryAnalysis = planner.invoke([
            SystemMessage(prompts.ANALYZE_SYSTEM.format(candidates=", ".join(known) or "(none)")),
            *_history_messages(state.get("chat_history")),
            HumanMessage(state["question"]),
        ])
        # The UI filter wins; otherwise keep only names that really exist in the index.
        targets = state.get("candidate_filter") or [c for c in result.candidates if c in known]
        return {
            "query": result.standalone_query or state["question"],
            "is_cv_question": result.is_cv_question,
            "target_candidates": targets,
            "rewrites": 0,
            "generations": 0,
            "strict": False,
            "trace": _trace(state, f"analyze → query='{result.standalone_query}', candidates={targets}"),
        }

    def retrieve(self, state: RAGState) -> RAGState:
        targets = state.get("target_candidates") or None
        docs = self.retriever(state["query"], targets)
        if not docs and targets and not state.get("candidate_filter"):
            docs = self.retriever(state["query"], None)  # name matching may have been too strict
        return {"documents": docs, "retrieved_documents": docs,
                "trace": _trace(state, f"retrieve → {len(docs)} chunks")}

    def grade(self, state: RAGState) -> RAGState:
        docs = state.get("documents", [])
        if not docs:
            return {"documents": [], "trace": _trace(state, "grade → nothing to grade")}
        grader = self.llm.with_structured_output(RelevantPassages)
        listing = "\n\n".join(f"<passage index={i}>\n{d.page_content}\n</passage>" for i, d in enumerate(docs))
        result: RelevantPassages = grader.invoke([
            SystemMessage(prompts.GRADE_SYSTEM),
            HumanMessage(f"Question: {state['query']}\n\n{listing}"),
        ])
        keep = [docs[i] for i in sorted(set(result.relevant_indices)) if 0 <= i < len(docs)]
        return {"documents": keep, "trace": _trace(state, f"grade → kept {len(keep)}/{len(docs)}")}

    def rewrite(self, state: RAGState) -> RAGState:
        rewriter = self.llm.with_structured_output(RewrittenQuery)
        result: RewrittenQuery = rewriter.invoke([
            SystemMessage(prompts.REWRITE_SYSTEM),
            HumanMessage(state["query"]),
        ])
        return {"query": result.query, "rewrites": state.get("rewrites", 0) + 1,
                "trace": _trace(state, f"rewrite → '{result.query}'")}

    def generate(self, state: RAGState) -> RAGState:
        system = prompts.GENERATE_SYSTEM.format(
            strict_note=prompts.STRICT_NOTE if state.get("strict") else "")
        response = self.llm.invoke([
            SystemMessage(system),
            *_history_messages(state.get("chat_history")),
            HumanMessage(prompts.GENERATE_USER.format(
                context=format_context(state["documents"]), question=state["question"])),
        ])
        return {"answer": response.content, "generations": state.get("generations", 0) + 1,
                "trace": _trace(state, "generate")}

    def check_grounded(self, state: RAGState) -> RAGState:
        checker = self.llm.with_structured_output(GroundednessCheck)
        result: GroundednessCheck = checker.invoke([
            SystemMessage(prompts.GROUNDEDNESS_SYSTEM),
            HumanMessage(f"CV excerpts:\n{format_context(state['documents'])}\n\nAnswer:\n{state['answer']}"),
        ])
        return {"grounded": result.grounded, "strict": not result.grounded,
                "trace": _trace(state, f"check_grounded → {result.grounded} {result.reason}".strip())}

    def off_topic(self, state: RAGState) -> RAGState:
        return {"answer": prompts.OFF_TOPIC_REPLY, "documents": [], "grounded": True,
                "trace": _trace(state, "off_topic")}

    def no_context(self, state: RAGState) -> RAGState:
        return {"answer": prompts.NO_CONTEXT_REPLY, "documents": [], "grounded": True,
                "trace": _trace(state, "no_context")}

    # ----------------------------------------------------------- routers
    @staticmethod
    def _after_analyze(state: RAGState) -> Literal["retrieve", "off_topic"]:
        return "retrieve" if state.get("is_cv_question", True) else "off_topic"

    def _after_grade(self, state: RAGState) -> Literal["generate", "rewrite", "no_context"]:
        if state.get("documents"):
            return "generate"
        if state.get("rewrites", 0) < self.settings.max_query_rewrites:
            return "rewrite"
        return "no_context"

    def _after_check(self, state: RAGState) -> Literal["generate", "__end__"]:
        if state.get("grounded") or state.get("generations", 0) > self.settings.max_generation_retries:
            return END
        return "generate"

    # ------------------------------------------------------------- build
    def _build(self):
        g = StateGraph(RAGState)
        for name in ("analyze", "retrieve", "grade", "rewrite", "generate",
                     "check_grounded", "off_topic", "no_context"):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "analyze")
        g.add_conditional_edges("analyze", self._after_analyze)
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges("grade", self._after_grade)
        g.add_edge("rewrite", "retrieve")
        g.add_edge("generate", "check_grounded")
        g.add_conditional_edges("check_grounded", self._after_check)
        g.add_edge("off_topic", END)
        g.add_edge("no_context", END)
        return g.compile()

    # --------------------------------------------------------------- run
    def ask(
        self,
        question: str,
        chat_history: list[tuple[str, str]] | None = None,
        known_candidates: list[str] | None = None,
        candidate_filter: list[str] | None = None,
    ) -> dict:
        start = time.perf_counter()
        final: RAGState = self.app.invoke({
            "question": question,
            "chat_history": chat_history or [],
            "known_candidates": known_candidates or [],
            "candidate_filter": candidate_filter or [],
            "trace": [],
        })
        docs = final.get("documents", [])
        return {
            "answer": final.get("answer", ""),
            "sources": [{"index": i, "candidate": d.metadata.get("candidate"),
                         "section": d.metadata.get("section"), "source": d.metadata.get("source"),
                         "text": d.page_content} for i, d in enumerate(docs, start=1)],
            "contexts": [d.page_content for d in docs],
            "retrieved_contexts": [d.page_content for d in final.get("retrieved_documents", [])],
            "retrieved_metadata": [d.metadata for d in final.get("retrieved_documents", [])],
            "query": final.get("query", question),
            "grounded": final.get("grounded"),
            "trace": final.get("trace", []),
            "latency_s": round(time.perf_counter() - start, 3),
        }


def build_default_graph(settings: Settings | None = None) -> tuple[CVRagGraph, "CVVectorStore"]:  # noqa: F821
    """Wire the graph to the persistent Chroma store."""
    from .vectorstore import CVVectorStore

    settings = settings or get_settings()
    store = CVVectorStore(settings)
    graph = CVRagGraph(retriever=lambda q, cands: store.search(q, cands), settings=settings)
    return graph, store
