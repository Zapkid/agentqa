"""Target configuration: how to authenticate as each role and sign webhooks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from agentqa.config import REPO_ROOT


class RoleAuth(BaseModel):
    token: str
    customer_id: str | None = None


class AuthConfig(BaseModel):
    """``bearer`` sends ``Authorization: Bearer <token>``; ``api_key`` sends ``<header>: <token>``."""

    scheme: Literal["bearer", "api_key"] = "bearer"
    header: str = "X-API-Key"  # only used by the api_key scheme
    roles: dict[str, RoleAuth] = Field(default_factory=dict)


class WebhookConfig(BaseModel):
    header: str = "X-Signature"
    algorithm: str = "hmac-sha256"
    secret_env: str = "TARGET_WEBHOOK_SECRET"
    secret_default: str = ""

    @property
    def secret(self) -> str:
        return os.environ.get(self.secret_env, self.secret_default)


class TargetConfig(BaseModel):
    name: str
    spec: str
    docs: str | None = None
    sandbox: bool = False
    auth: AuthConfig = Field(default_factory=AuthConfig)
    webhook: WebhookConfig | None = None

    def headers(self, role: str) -> dict[str, str]:
        r = self.auth.roles[role]
        if self.auth.scheme == "api_key":
            return {self.auth.header: r.token}
        return {"Authorization": f"Bearer {r.token}"}

    @property
    def spec_path(self) -> Path:
        return (REPO_ROOT / self.spec) if not Path(self.spec).is_absolute() else Path(self.spec)

    @property
    def docs_path(self) -> Path | None:
        if not self.docs:
            return None
        return (REPO_ROOT / self.docs) if not Path(self.docs).is_absolute() else Path(self.docs)


def load_target(
    path: str | Path = REPO_ROOT / "target_api" / "agentqa_target.yaml",
) -> TargetConfig:
    return TargetConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
