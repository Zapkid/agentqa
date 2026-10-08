"""Tasks API: a small multi-user task tracker used as a second, differently shaped test target.

It differs from the bundled Orders API on purpose: authentication is an ``X-API-Key`` header
(not a bearer token), and the published contract is a Swagger 2.0 document (``swagger.json``).
State is in memory. Seeded defects are switched on with ``BUGS=T01,T02,...`` (empty = clean).
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
BUGS = {b.strip() for b in os.environ.get("BUGS", "").split(",") if b.strip()}

NAMESPACE = uuid.UUID("5f0c2a6e-8d1b-4c53-9a77-0c1d5e6f7a88")


def _uid(name: str) -> str:
    return str(uuid.uuid5(NAMESPACE, name))


# Sandbox fixtures, not secrets: the keys only work against this in-memory demo service.
USERS: dict[str, dict[str, str]] = {
    "key-admin": {"id": _uid("admin"), "role": "admin"},
    "key-alice": {"id": _uid("alice"), "role": "user"},
    "key-bob": {"id": _uid("bob"), "role": "user"},
}

Priority = Literal["low", "medium", "high"]
Status = Literal["open", "done"]


class TaskIn(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=1000)
    priority: Priority = "medium"
    due_date: date | None = None


class TaskPatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=1000)
    priority: Priority | None = None
    due_date: date | None = None


TASKS: dict[str, dict[str, Any]] = {}


def _seed() -> None:
    TASKS.clear()
    owners = [_uid("alice"), _uid("bob")]
    for i in range(30):
        tid = _uid(f"task-{i}")
        TASKS[tid] = {
            "id": tid,
            "title": f"Seed task {i:02d}",
            "description": None,
            "priority": ("low", "medium", "high")[i % 3],
            "status": "open",
            "owner_id": owners[i % 2],
            "due_date": None,
            "created_at": f"2026-01-01T00:{i:02d}:00Z",
        }


_seed()
api_key = APIKeyHeader(name="X-API-Key", auto_error=False)


def current_user(key: Annotated[str | None, Security(api_key)] = None) -> dict[str, str]:
    user = USERS.get(key or "")
    if user is None:
        raise HTTPException(status_code=401, detail="missing or invalid API key")
    return user


User = Annotated[dict[str, str], Depends(current_user)]
app = FastAPI(title="Tasks API", version="1.0.0")


def _visible(user: dict[str, str], task: dict[str, Any]) -> bool:
    return user["role"] == "admin" or task["owner_id"] == user["id"] or "T01" in BUGS


def _get(task_id: str, user: dict[str, str]) -> dict[str, Any]:
    try:
        uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="task_id must be a UUID") from None
    task = TASKS.get(task_id)
    if task is None or not _visible(user, task):
        raise HTTPException(status_code=404, detail="task not found")
    return task


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/swagger.json", include_in_schema=False)
def swagger() -> JSONResponse:
    return JSONResponse(json.loads((HERE / "swagger.json").read_text(encoding="utf-8")))


@app.post("/__reset", include_in_schema=False)
def reset() -> dict[str, str]:
    _seed()
    return {"status": "reset"}


@app.get("/tasks")
def list_tasks(
    user: User,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    status: str | None = None,
) -> dict[str, Any]:
    if status is not None and status not in ("open", "done") and "T02" not in BUGS:
        raise HTTPException(status_code=422, detail="status must be open or done")
    rows = [t for t in TASKS.values() if user["role"] == "admin" or t["owner_id"] == user["id"]]
    if status in ("open", "done"):  # with T02 an unknown status is silently ignored
        rows = [t for t in rows if t["status"] == status]
    rows.sort(key=lambda t: (t["created_at"], t["id"]))
    start = (page - 1) * page_size
    if "T04" in BUGS and page > 1:
        start += 1  # off-by-one: the first item of every later page is skipped
    return {
        "items": rows[start : start + page_size],
        "page": page,
        "page_size": page_size,
        "total": len(rows),
    }


@app.post("/tasks", status_code=201)
def create_task(body: TaskIn, user: User) -> dict[str, Any]:
    if body.due_date is not None and body.due_date < datetime.now(UTC).date():
        raise HTTPException(status_code=422, detail="due_date must not be in the past")
    tid = str(uuid.uuid4())
    task = {
        "id": tid,
        "title": body.title,
        "description": body.description,
        "priority": body.priority,
        "status": "open",
        "owner_id": user["id"],
        "due_date": body.due_date.isoformat() if body.due_date else None,
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    TASKS[tid] = task
    return task


@app.get("/tasks/{task_id}")
def read_task(task_id: str, user: User) -> dict[str, Any]:
    return _get(task_id, user)


@app.patch("/tasks/{task_id}")
def update_task(task_id: str, body: TaskPatch, user: User) -> dict[str, Any]:
    task = _get(task_id, user)
    if task["status"] == "done":
        raise HTTPException(status_code=409, detail="a completed task cannot be edited")
    changes = body.model_dump(exclude_unset=True)
    if "due_date" in changes and changes["due_date"] is not None:
        changes["due_date"] = changes["due_date"].isoformat()
    task.update(changes)
    return task


@app.delete("/tasks/{task_id}", status_code=204)
def delete_task(task_id: str, user: User) -> None:
    task = _get(task_id, user)
    del TASKS[task["id"]]


@app.post("/tasks/{task_id}/complete")
def complete_task(task_id: str, user: User) -> dict[str, Any]:
    task = _get(task_id, user)
    if task["status"] == "done" and "T03" not in BUGS:
        raise HTTPException(status_code=409, detail="task is already completed")
    task["status"] = "done"
    return task
