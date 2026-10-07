"""Write the target's OpenAPI document to target_api/openapi.json (identical for every build)."""

from __future__ import annotations

import json
from pathlib import Path

from target_api.app.main import app

if __name__ == "__main__":
    out = Path(__file__).parent / "openapi.json"
    out.write_text(json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
