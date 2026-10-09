"""Retrieval-augmented answers: retrieve passages, ask the model to answer only from them, and
check that every citation points at a passage that was actually retrieved."""

from __future__ import annotations

import json
from typing import cast

from pydantic import BaseModel, Field

from agentqa.guards.redaction import redact
from agentqa.llm.prompts import load_prompt
from agentqa.llm.router import LLMClient
from agentqa.llm.types import LLMError
from agentqa.obs import tracing
from agentqa.veroniqa.knowledge import Passage

NOT_FOUND = (
    "I could not find this in the project's knowledge. Upload a document or add a link that "
    "covers it, and ask again."
)


class AnswerOut(BaseModel):
    answer: str
    cited: list[int] = Field(default_factory=list, description="numbers of the passages used")
    found: bool = Field(description="false when the passages do not answer the question")


class Answer(BaseModel):
    text: str
    citations: list[Passage] = Field(default_factory=list)
    grounded: bool = False
    note: str = ""


def answer(question: str, passages: list[Passage], client: LLMClient) -> Answer:
    if not passages:
        return Answer(text=NOT_FOUND)
    prompt = load_prompt("veroniqa_answer")
    payload = {
        "question": question,
        "passages": [
            {"n": p.n, "source": p.source, "section": p.section, "text": p.text[:1800]}
            for p in passages
        ],
    }
    try:
        with tracing.span("veroniqa.answer", "agent", **{"agentqa.agent": "veroniqa"}):
            res = client.complete(
                prompt.render(payload=json.dumps(redact(payload), indent=1)),
                response_schema=AnswerOut,
                max_tokens=700,
                metadata=prompt.metadata("veroniqa"),
            )
    except LLMError as exc:
        return Answer(
            text=f"The model is unavailable ({exc}). Closest passages are listed below.",
            citations=passages[:3],
        )
    out = cast(AnswerOut, res.parsed)
    known = {p.n: p for p in passages}
    cited = [known[n] for n in dict.fromkeys(out.cited) if n in known]
    invented = [n for n in out.cited if n not in known]
    if not out.found:
        return Answer(text=out.answer or NOT_FOUND, citations=[], grounded=False)
    note = ""
    if invented:
        note = f"Dropped citations to passages that were not retrieved: {invented}."
    if not cited:
        note = (note + " The answer cites no retrieved passage; treat it as unverified.").strip()
    return Answer(text=out.answer, citations=cited, grounded=bool(cited), note=note)
