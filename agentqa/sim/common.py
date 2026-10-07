"""Helpers shared by the simulated responders."""

from __future__ import annotations

import re

DOC_RE = re.compile(r'<untrusted_document id="([^"]+)">\n(.*?)\n</untrusted_document>', re.S)


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("**", "")).lower()


def documents(text: str) -> dict[str, str]:
    """{chunk_id: normalised text} for every delimited document in a prompt."""
    return {cid: normalise(body) for cid, body in DOC_RE.findall(text)}
