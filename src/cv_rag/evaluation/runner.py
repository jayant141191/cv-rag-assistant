"""Run the evaluation dataset through the graph and score it."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .metrics import (candidate_precision, candidate_recall, hit, keyword_recall,
                      reciprocal_rank, safe_mean)

METRIC_KEYS = [
    "retrieval_hit", "retrieval_mrr", "candidate_recall", "candidate_precision",
    "context_keyword_recall", "answer_keyword_recall",
    "correctness", "faithfulness", "relevance", "latency_s",
]


def load_dataset(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def score_example(example: dict, result: dict) -> dict:
    """Deterministic metrics for one example (judge scores are added separately)."""
    exp_cands = example.get("expected_candidates", [])
    keywords = example.get("expected_keywords", [])
    ctxs = result.get("retrieved_contexts", [])
    cands = [m.get("candidate") for m in result.get("retrieved_metadata", [])]
    retrieval_applies = example.get("category") != "off_topic"

    return {
        "retrieval_hit": hit(ctxs, cands, exp_cands, keywords) if retrieval_applies else None,
        "retrieval_mrr": reciprocal_rank(ctxs, cands, exp_cands, keywords) if retrieval_applies else None,
        "candidate_recall": candidate_recall(cands, exp_cands) if retrieval_applies else None,
        "candidate_precision": candidate_precision(cands, exp_cands) if retrieval_applies else None,
        "context_keyword_recall": keyword_recall(" ".join(ctxs), keywords),
        "answer_keyword_recall": keyword_recall(result.get("answer", ""), keywords),
        "latency_s": result.get("latency_s"),
    }


def summarise(rows: list[dict]) -> dict:
    overall = {k: safe_mean([r.get(k) for r in rows]) for k in METRIC_KEYS}
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r.get("category", "uncategorised")].append(r)
    per_category = {cat: {"n": len(rs), **{k: safe_mean([r.get(k) for r in rs]) for k in METRIC_KEYS}}
                    for cat, rs in sorted(by_cat.items())}
    return {"n": len(rows), "overall": overall, "per_category": per_category}


def _fmt(v) -> str:
    return "–" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def to_markdown(summary: dict, rows: list[dict], meta: dict) -> str:
    lines = [
        "# CV RAG — Evaluation report", "",
        f"- Run: {meta.get('timestamp')}",
        f"- Chat model: `{meta.get('chat_model')}` · Judge: `{meta.get('judge_model')}` · "
        f"Embeddings: `{meta.get('embedding_model')}`",
        f"- Examples: {summary['n']}", "",
        "## Overall", "", "| Metric | Score |", "|---|---|",
    ]
    lines += [f"| {k} | {_fmt(v)} |" for k, v in summary["overall"].items()]
    cols = ["n", "retrieval_hit", "retrieval_mrr", "candidate_recall", "answer_keyword_recall",
            "correctness", "faithfulness", "relevance"]
    lines += ["", "## By category", "", "| Category | " + " | ".join(cols) + " |",
              "|---" * (len(cols) + 1) + "|"]
    for cat, m in summary["per_category"].items():
        lines.append(f"| {cat} | " + " | ".join(_fmt(m.get(c)) for c in cols) + " |")
    lines += ["", "## Per question", "",
              "| ID | Category | Hit | MRR | Correct | Faithful | Judge note |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        note = (r.get("judge_explanation") or "").replace("|", "/").replace("\n", " ")[:140]
        lines.append(f"| {r['id']} | {r.get('category')} | {_fmt(r.get('retrieval_hit'))} | "
                     f"{_fmt(r.get('retrieval_mrr'))} | {_fmt(r.get('correctness'))} | "
                     f"{_fmt(r.get('faithfulness'))} | {note} |")
    return "\n".join(lines) + "\n"


def run_evaluation(dataset_path: str | Path, cv_dir: str | Path, out_dir: str | Path,
                   use_judge: bool = True, reingest: bool = True) -> dict:
    """End-to-end evaluation. Uses its own Chroma collection so your real index is untouched."""
    from langchain_openai import ChatOpenAI

    from ..config import get_settings
    from ..graph import CVRagGraph
    from ..loaders import SUPPORTED_EXTENSIONS
    from ..vectorstore import CVVectorStore
    from .judge import judge_answer

    settings = replace(get_settings(), collection_name=get_settings().collection_name + "_eval")
    settings.require_api_key()
    store = CVVectorStore(settings)
    if reingest:
        store.reset()
        for f in sorted(Path(cv_dir).iterdir()):
            if f.suffix.lower() in SUPPORTED_EXTENSIONS:
                store.add_path(f)
    candidates = store.list_candidates()
    graph = CVRagGraph(retriever=lambda q, c: store.search(q, c), settings=settings)
    judge_llm = ChatOpenAI(model=settings.judge_model, temperature=0,
                           api_key=settings.openai_api_key) if use_judge else None

    rows: list[dict] = []
    for ex in load_dataset(dataset_path):
        result = graph.ask(ex["question"], known_candidates=candidates)
        row = {"id": ex["id"], "category": ex.get("category"), "question": ex["question"],
               "answer": result["answer"], "query": result["query"], "trace": result["trace"],
               **score_example(ex, result)}
        if judge_llm is not None:
            scores = judge_answer(judge_llm, ex["question"], ex["reference_answer"],
                                  result["contexts"], result["answer"])
            row.update(correctness=scores.correctness, faithfulness=scores.faithfulness,
                       relevance=scores.relevance, judge_explanation=scores.explanation)
        rows.append(row)
        print(f"[{ex['id']}] hit={_fmt(row['retrieval_hit'])} correct={_fmt(row.get('correctness'))} "
              f"{ex['question'][:60]}")

    summary = summarise(rows)
    meta = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "chat_model": settings.chat_model, "judge_model": settings.judge_model if use_judge else None,
            "embedding_model": settings.embedding_model, "dataset": str(dataset_path)}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / f"eval_{stamp}.json").write_text(
        json.dumps({"meta": meta, "summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    report = to_markdown(summary, rows, meta)
    (out / f"eval_{stamp}.md").write_text(report, encoding="utf-8")
    (out / "latest.md").write_text(report, encoding="utf-8")
    return {"meta": meta, "summary": summary, "rows": rows, "report_path": str(out / f"eval_{stamp}.md")}
