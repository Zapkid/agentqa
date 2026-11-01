"""Dockerfile.vercel is built so each deployment adds little to Vercel's image registry.

A registry stores a layer once, keyed by its content, so the dependencies must be a layer of their
own that does not change when the code does, and that is byte-identical across builds. There is no
Docker daemon in CI, so these check the file's structure; `docs/DEPLOY.md` says how the real
thing was measured."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT = (ROOT / "Dockerfile.vercel").read_text()


def _instructions() -> list[str]:
    """Each Dockerfile instruction on one line (continuations joined, comments dropped)."""
    joined = re.sub(r"\\\n\s*", " ", TEXT)
    return [ln.strip() for ln in joined.splitlines() if ln.strip() and not ln.startswith("#")]


def _stage(name: str) -> list[str]:
    ins, out, inside = _instructions(), [], False
    for line in ins:
        if line.startswith("FROM "):
            inside = line.endswith(f" AS {name}")
        elif inside:
            out.append(line)
    return out


def _final_stage() -> list[str]:
    ins = _instructions()
    last_from = max(i for i, ln in enumerate(ins) if ln.startswith("FROM "))
    return ins[last_from:]


def test_dependencies_are_their_own_layer_and_the_code_comes_after() -> None:
    final = _final_stage()
    deps_at = next(i for i, ln in enumerate(final) if ln.startswith("COPY --from=deps /app/.venv "))
    later = [i for i, ln in enumerate(final) if re.match(r"COPY --from=(src|assets) ", ln)]
    assert len(later) >= 3 and all(i > deps_at for i in later)  # media, then the code
    # never the whole of /app from another stage: that would put dependencies and code in one layer
    assert not any(re.match(r"COPY --from=\S+ /app/? /app\b", ln) for ln in final)
    # nothing comes straight from the build context in the last stage: a checkout's file times
    # differ every build, so such a layer would never repeat; the code and media pass through
    # stages that fix them first
    assert not any(re.match(r"COPY (?!--from)", ln) for ln in final)
    assert not any(ln.startswith("COPY . ") for ln in _instructions())


def test_media_and_code_stages_fix_file_times_and_precompile_reproducibly() -> None:
    assets, src = " ".join(_stage("assets")), " ".join(_stage("src"))
    assert "touch -h -d @0" in assets and "media/videos/veroniqa-intro.mp4" in assets
    assert "touch -h -d @0" in src
    assert "PYTHONHASHSEED=0" in src and "--invalidation-mode unchecked-hash" in src


def test_the_dependency_layer_is_reproducible_and_independent_of_the_code() -> None:
    deps = " ".join(_stage("deps"))
    assert "COPY pyproject.toml uv.lock" in deps and "--locked" in deps
    assert "--no-install-project" in deps and "--no-dev" in deps  # no project metadata inside
    assert "UV_COMPILE_BYTECODE=0" in deps  # bytecode is compiled below, by content hash
    assert "--invalidation-mode unchecked-hash" in deps  # a .pyc that records no file time
    # -f: venv creation leaves one timestamp-based .pyc behind, which compileall would skip
    assert "compileall -q -f " in deps
    # set constants are marshalled in hash order: without a fixed seed the same source compiles
    # to different bytes from one build to the next (seen with pycparser's c_parser)
    assert "PYTHONHASHSEED=0" in deps
    # one process, in order: with -j the same source compiles to different bytes depending on
    # which worker compiled which files before it (seen in pycparser's c_parser)
    assert "-j" not in deps.split("compileall", 1)[1].split("&&")[0]
    assert "touch -h -d @0" in deps  # every file time fixed, so the layer's bytes repeat
    assert not re.search(r"COPY (agentqa|target_api|config|prompts|media)\b", deps)  # no code
    assert "UV_NO_CACHE=1" in deps  # uv's download cache is as large as the environment


def test_the_image_trims_unused_packages_and_checks_it_can_start() -> None:
    deps = " ".join(_stage("deps"))
    for package in ("locust", "matplotlib", "kubernetes", "onnxruntime"):
        assert package in deps
    final = " ".join(_final_stage())
    assert "PYTHONPATH=/app" in final  # the code runs from /app (the project is not installed)
    assert 'python -c "import agentqa.veroniqa.server"' in final  # fails the build, not the deploy
    assert _final_stage()[-2].startswith("EXPOSE") and _instructions()[-1].startswith("CMD")
    users = [ln.split()[1] for ln in _instructions() if ln.startswith("USER ")]
    assert users and users[-1] == "10001"  # the smoke test runs as root, the app does not


# ------------------------------------------------------------------ builds Vercel skips


def _skips(ref: str) -> bool:
    """Run vercel.json's ignoreCommand as Vercel does: exit 0 skips the build, 1 builds it."""
    import json
    import subprocess

    command = json.loads((ROOT / "vercel.json").read_text())["ignoreCommand"]
    env = {"PATH": "/usr/bin:/bin", "VERCEL_GIT_COMMIT_REF": ref}
    done = subprocess.run(["sh", "-c", command], env=env, capture_output=True, check=False)  # noqa: S603, S607
    assert done.returncode in (0, 1), done.stderr  # anything else would also build, but is a bug
    return done.returncode == 0


def test_dependency_bump_branches_do_not_build_an_image_but_everything_else_does() -> None:
    # Each Dependabot branch changes the lockfile, so it would store a whole new dependency layer
    # that nobody looks at; GitHub's CI already tests it. Production and ordinary branches build.
    for ref in ("dependabot/uv/pydantic-2.14.1", "dependabot/github_actions/actions/checkout-7"):
        assert _skips(ref), ref
    for ref in ("main", "claude/new-session-lof4dk", "ccr-f6304afe-lh80hr", "dependabot", ""):
        assert not _skips(ref), ref
