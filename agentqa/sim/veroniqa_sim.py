"""Simulated responders for Veroniqa's prompts: keyword routing and extractive answers.

Deterministic on purpose: a chat interface that randomly misroutes would only make the demo and
the tests flaky. Real models replace both with `AGENTQA_PROFILE=free|mixed|premium`.
"""

from __future__ import annotations

import math
import re
from typing import Any

from agentqa.llm.simulated import SimRequest, responder

URL = re.compile(r"https?://\S+")
RUN_ID = re.compile(r"\brun-\d{8}-\d{6}-[0-9a-f]{6}\b")
BUGS = re.compile(r"\bB(0?[1-9]|1[0-2])\b", re.IGNORECASE)
STOP = set(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "of",
        "on",
        "or",
        "our",
        "should",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    ]
)


def _stem(word: str) -> str:
    for suffix, repl in (
        ("ied", "y"),
        ("ies", "y"),
        ("ing", ""),
        ("ed", ""),
        ("es", ""),
        ("s", ""),
    ):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + repl
    return word


def _words(text: str) -> set[str]:
    return {
        _stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 2
    }


@responder("veroniqa_route")
def route(req: SimRequest) -> dict[str, Any]:
    msg = str(req.payload().get("message", ""))
    low = msg.lower()
    urls = URL.findall(msg)
    if urls and (
        msg.strip() == urls[0]
        or re.search(r"\b(add|link|read|learn|ingest|include|import|save)\b", low)
    ):
        return {"action": "add_link", "argument": urls[0], "bugs": ""}
    if re.search(
        r"\b(run|execute|start|launch)\b.*\b(tests?|suite|scan)\b|\btest (the|my|this) api\b", low
    ):
        if re.search(r"\b(clean|no bugs|without bugs)\b", low):
            bugs = "clean"
        else:
            ids = sorted({f"B{int(m.group(1)):02d}" for m in BUGS.finditer(msg)})
            bugs = ",".join(ids) if ids else "all"
        return {"action": "run_tests", "argument": "", "bugs": bugs}
    if re.search(r"\b(list|show|previous|past|earlier)\b.*\bruns\b|\brun history\b", low):
        return {"action": "list_runs", "argument": "", "bugs": ""}
    if re.search(r"\b(report|summary|findings)\b", low) and re.search(
        r"\b(show|open|latest|last|see|read)\b", low
    ):
        run = RUN_ID.findall(msg)
        return {"action": "show_report", "argument": run[0] if run else "", "bugs": ""}
    if re.search(r"\b(sources|documents|docs|knowledge base)\b", low) and re.search(
        r"\b(list|which|what|show)\b", low
    ):
        return {"action": "list_sources", "argument": "", "bugs": ""}
    if re.search(r"^(help|\?)$|what can you do|how do i use you", low):
        return {"action": "help", "argument": "", "bugs": ""}
    return {"action": "ask", "argument": msg, "bugs": ""}


def _sentences(text: str) -> list[str]:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)  # code blocks are not prose
    text = re.sub(r"^#+ .*$", " ", text, flags=re.M)  # headings
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.M)  # list bullets
    text = " ".join(text.split())
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]


@responder("veroniqa_answer")
def answer(req: SimRequest) -> dict[str, Any]:
    """Extractive: the sentences that best match the question's rarer terms, with citations."""
    p = req.payload()
    q = _words(str(p.get("question", "")))
    sentences = [
        (passage["n"], i, sentence, _words(sentence))
        for passage in p.get("passages", [])
        for i, sentence in enumerate(_sentences(passage["text"]))
    ]
    df = {t: sum(t in words for *_, words in sentences) for t in q}
    idf = {t: math.log((1 + len(sentences)) / (1 + df[t])) + 1 for t in q}
    ranked = sorted(
        ((sum(idf[t] for t in q & words), -n, -i, sentence) for n, i, sentence, words in sentences),
        reverse=True,
    )
    covered = len(q & _words(ranked[0][3])) if ranked else 0
    if not ranked or ranked[0][0] == 0 or covered < max(1, len(q) // 3):
        return {
            "answer": "The project's knowledge does not answer this.",
            "cited": [],
            "found": False,
        }
    best = ranked[0][0]
    picked = [r for r in ranked if r[0] >= best * 0.7][:3]
    parts = [f"{sentence} [{-neg_n}]" for _, neg_n, _, sentence in picked]
    cited = list(dict.fromkeys(-neg_n for _, neg_n, _, _ in picked))
    return {"answer": " ".join(parts), "cited": cited, "found": True}
