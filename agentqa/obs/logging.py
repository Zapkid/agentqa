"""structlog JSON logging with trace/run correlation and central redaction."""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from contextvars import ContextVar
from typing import Any

import structlog

from agentqa.guards.redaction import redact
from agentqa.obs.tracing import current_trace_id

run_id_var: ContextVar[str | None] = ContextVar("agentqa_run_id", default=None)
_configured = False


def _add_correlation(
    _: Any, __: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    trace_id = current_trace_id()
    if trace_id:
        event_dict.setdefault("trace_id", trace_id)
    run_id = run_id_var.get()
    if run_id:
        event_dict.setdefault("run_id", run_id)
    return event_dict


def _redact_processor(
    _: Any, __: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    cleaned: MutableMapping[str, Any] = redact(dict(event_dict))
    return cleaned


def configure_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=level)
    for noisy in ("httpx", "httpcore", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _add_correlation,
            _redact_processor,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str = "agentqa") -> Any:
    configure_logging()
    return structlog.get_logger(name)
