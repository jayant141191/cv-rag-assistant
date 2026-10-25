"""LLM-as-judge scoring of generated answers."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

JUDGE_SYSTEM = """You are a strict evaluator of a CV question-answering system.
Score the ANSWER on three 1-5 scales and explain briefly.

correctness (vs. the REFERENCE answer):
  5 = all key facts match the reference, nothing wrong
  3 = partially correct or missing important facts
  1 = wrong, or answers a different question
  For questions the CVs cannot answer, a clear "not mentioned / not available" answer
  scores 5 and any invented answer scores 1. For off-topic questions, a polite redirect
  to CV topics scores 5.

faithfulness (vs. the CONTEXT the system retrieved):
  5 = every claim is supported by the context (or the answer correctly abstains)
  1 = contains fabricated claims not present in the context

relevance (vs. the QUESTION):
  5 = directly and completely addresses the question, no padding
  1 = off-target"""

JUDGE_USER = """QUESTION:
{question}

REFERENCE ANSWER:
{reference}

RETRIEVED CONTEXT:
{context}

ANSWER TO EVALUATE:
{answer}"""


class JudgeScores(BaseModel):
    correctness: int = Field(ge=1, le=5)
    faithfulness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    explanation: str


def judge_answer(llm: BaseChatModel, question: str, reference: str,
                 contexts: list[str], answer: str) -> JudgeScores:
    context = "\n\n---\n\n".join(contexts) if contexts else "(no context retrieved)"
    return llm.with_structured_output(JudgeScores).invoke([
        SystemMessage(JUDGE_SYSTEM),
        HumanMessage(JUDGE_USER.format(question=question, reference=reference,
                                       context=context, answer=answer)),
    ])
