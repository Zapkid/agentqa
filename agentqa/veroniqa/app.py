"""VeroniQA's web interface (Streamlit).

Run it with `make veroniqa` or `uv run agentqa veroniqa`: it binds to 127.0.0.1. Everything the UI
does goes through agentqa.veroniqa, so the chat, the buttons and the tests share one code path.
Markdown is rendered with Streamlit's default escaping (no raw HTML), because documents, pages and
reports are untrusted. With VERONIQA_HOSTED=1 it runs as a public demo (see hosted.py).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import streamlit as st

from agentqa.veroniqa import (
    Reply,
    VeroniQA,
    create_demo_project,
    create_project,
    hosted,
    list_projects,
)
from agentqa.veroniqa.fetch import FetchError
from agentqa.veroniqa.knowledge import UPLOAD_TYPES
from agentqa.veroniqa.projects import Project, load_project
from agentqa.veroniqa.runs import NotRunnable, project_runs

PROFILES = ["simulated", "free", "mixed", "premium"]
BUG_IDS = [f"B{i:02d}" for i in range(1, 13)]
HOSTED = hosted.hosted()

st.set_page_config(page_title="VeroniQA · AgentQA", page_icon="🧪", layout="wide")


@st.cache_resource(show_spinner=False, max_entries=64)
def agent(project_root: str, profile: str) -> VeroniQA:
    """One agent (and vector-store client) per project folder and profile."""
    return VeroniQA(Project(Path(project_root)), profile=profile)


def show_reply(reply: Reply) -> None:
    st.markdown(reply.text)
    if reply.note:
        st.caption(reply.note)
    if reply.citations:
        with st.expander(f"Sources ({len(reply.citations)})"):
            for c in reply.citations:
                where = f" · {c.url}" if c.url else ""
                st.markdown(f"**[{c.n}] {c.source}** · {c.section}{where}")
                st.text(c.text[:700])
    if reply.report_html and Path(reply.report_html).exists():
        st.download_button(
            "Download the full report (HTML)",
            Path(reply.report_html).read_bytes(),
            file_name=f"{reply.run_id or 'report'}.html",
            mime="text/html",
            key=f"dl-{reply.run_id}-{id(reply)}",
        )


def workspace() -> Path | None:
    """The projects folder for this visitor: a private workspace when hosted, else the default."""
    if not HOSTED:
        return None
    wid = st.query_params.get("w", "")
    if not hosted.WORKSPACE_ID.match(wid):
        wid = hosted.new_workspace_id()
        st.query_params["w"] = wid
    if "pruned" not in st.session_state:
        hosted.prune_workspaces(keep=wid)
        st.session_state.pruned = True
    hosted.touch(wid)
    root = hosted.workspace_root(wid)
    if not list_projects(root):
        with st.spinner("Setting up your workspace with the Orders API demo project..."):
            create_demo_project(root)
    return root


root = workspace()

# ---------------------------------------------------------------- sidebar: projects and settings

with st.sidebar:
    st.title("VeroniQA")
    st.caption("Your AgentQA assistant: projects, knowledge, tests.")
    if HOSTED:
        profile = "simulated"
        st.info(
            "Public demo. Models are simulated: answers and runs show the mechanics, not a real "
            "model. Your workspace is private to this link and temporary, so do not upload "
            "confidential documents."
        )
        st.link_button("Watch the introduction", "/", width="stretch")
    else:
        profile = st.selectbox(
            "Model profile",
            PROFILES,
            index=PROFILES.index(os.environ.get("AGENTQA_PROFILE", "simulated"))
            if os.environ.get("AGENTQA_PROFILE", "simulated") in PROFILES
            else 0,
            help="simulated needs no API keys. The others read keys from .env.",
        )
        if profile == "simulated":
            st.info("Simulated models: answers and runs show the mechanics, not a real model.")
    projects = list_projects(root)
    slugs = [p.slug for p in projects]
    names = {p.slug: p.config.name for p in projects}
    # A project created on the previous run is selected here, before the selectbox exists:
    # Streamlit forbids changing a widget's state after the widget is drawn.
    if "select_project" in st.session_state:
        st.session_state.project = st.session_state.pop("select_project")
    if "project" not in st.session_state or st.session_state.project not in slugs:
        st.session_state.project = slugs[0] if slugs else None
    if slugs:
        st.selectbox("Project", slugs, key="project", format_func=lambda s: names.get(s, s))
    with st.expander("New project", expanded=not slugs):
        with st.form("new-project", clear_on_submit=True):
            name = st.text_input("Name")
            description = st.text_area("What is this API?", height=80)
            submitted = st.form_submit_button("Create project")
        if submitted:
            try:
                created = create_project(name, description, root=root)
                st.session_state.select_project = created.slug
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if st.button("Create the Orders API demo project"):
            with st.spinner("Copying the demo docs into a new project..."):
                st.session_state.select_project = create_demo_project(root).slug
            st.rerun()

slug: str | None = st.session_state.get("project")
if not slug:
    st.header("Welcome to VeroniQA")
    st.write(
        "Create a project in the sidebar, or start with the demo project: the bundled Orders "
        "API, its requirement documents and 12 seeded bugs."
    )
    st.stop()

project: Project = load_project(slug, root)
v = agent(str(project.root), profile)
st.header(project.config.name)
if project.config.description:
    st.caption(project.config.description)

chat_tab, knowledge_tab, runs_tab, settings_tab = st.tabs(
    ["Chat", "Knowledge", "Test runs", "Settings"]
)

# ---------------------------------------------------------------- chat

with chat_tab:
    history: list[dict[str, Any]] = st.session_state.setdefault(f"chat-{slug}", [])
    conversation = st.container()  # drawn above the input box, which renders inline in a tab
    prompt = st.chat_input("Ask about the API, add a link, or say 'run the tests'")
    with conversation:
        for turn in history:
            with st.chat_message(turn["role"]):
                if turn["role"] == "user":
                    st.markdown(turn["text"])
                else:
                    show_reply(Reply.model_validate(turn["reply"]))
        if prompt:
            history.append({"role": "user", "text": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"), st.spinner("Working..."):
                reply = v.chat(prompt)
                show_reply(reply)
            history.append({"role": "assistant", "reply": reply.model_dump()})

# ---------------------------------------------------------------- knowledge

with knowledge_tab:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Add knowledge")
        files = st.file_uploader(
            "Upload documents (requirements, specs in prose, runbooks)",
            type=sorted(t.lstrip(".") for t in UPLOAD_TYPES),
            accept_multiple_files=True,
            key=f"upload-{slug}",
        )
        if files and st.button("Add the uploaded files"):
            for f in files:
                try:
                    src = v.kb.add_file(f.name, f.getvalue())
                    msg = f"{src.name}: {src.chunks} passages"
                    if src.quarantined:
                        st.warning(
                            f"{msg}, {src.quarantined} quarantined (possible prompt injection)"
                        )
                    else:
                        st.success(msg)
                except ValueError as exc:
                    st.error(f"{f.name}: {exc}")
        with st.form("add-link", clear_on_submit=True):
            url = st.text_input("Link to a page (http or https)")
            if st.form_submit_button("Add link") and url:
                try:
                    src = v.kb.add_link(url.strip(), v.fetcher)
                    st.success(f"Added {src.name} ({src.chunks} passages)")
                except (FetchError, ValueError) as exc:
                    st.error(f"Could not add the link: {exc}")
        st.subheader("Try retrieval")
        query = st.text_input("Search the knowledge base", key=f"search-{slug}")
        if query:
            for p in v.kb.retrieve(query):
                st.markdown(f"**[{p.n}] {p.source}** · {p.section}")
                st.text(p.text[:500])
    with right:
        st.subheader("Sources")
        sources = project.sources()
        if not sources:
            st.write("No sources yet.")
        for s in sources:
            cols = st.columns([3, 1])
            flag = f" · {s.quarantined} quarantined" if s.quarantined else ""
            noun = "passage" if s.chunks == 1 else "passages"
            cols[0].markdown(f"**{s.name}**  \n{s.kind} · {s.chunks} {noun}{flag}")
            if cols[1].button("Remove", key=f"rm-{s.id}"):
                v.kb.remove(s.id)
                st.rerun()

# ---------------------------------------------------------------- test runs

with runs_tab:
    if project.config.local_demo:
        st.write("Run AgentQA against the bundled Orders API with the seeded bugs you choose.")
        clean = st.toggle("Clean build (no bugs)", key=f"clean-{slug}")
        chosen = st.multiselect(
            "Seeded bugs (empty means all 12)", BUG_IDS, disabled=clean, key=f"bugs-{slug}"
        )
        bugs = "clean" if clean else ",".join(chosen)
    else:
        bugs = ""
        st.write("Runs use the spec, target and requirement documents set for this project.")
    if st.button("Run the tests", type="primary"):
        with st.spinner("Planning, generating, running and triaging..."):
            try:
                out = v.runner(project, profile, bugs)
                st.success(f"Run {out.report.run_id} finished: {out.report.outcome_counts()}")
            except NotRunnable as exc:
                st.error(str(exc))
    runs = project_runs(project)
    if runs:
        st.dataframe(
            [
                {
                    "run": r.run_id,
                    "when": r.created_at,
                    "findings": r.findings,
                    "product bugs": r.product_bugs,
                    "tokens": r.tokens,
                }
                for r in runs
            ],
            hide_index=True,
            use_container_width=True,
        )
        picked = st.selectbox("Open a run", [r.run_id for r in runs], key=f"run-{slug}")
        run = next(r for r in runs if r.run_id == picked)
        if Path(run.summary_md).exists():
            summary = Path(run.summary_md).read_text(encoding="utf-8")
            st.markdown(re.sub(r"^(#{1,3}) ", r"###\1 ", summary, flags=re.M))  # demote headings
        if Path(run.report_html).exists():
            st.download_button(
                "Download the full report (HTML)",
                Path(run.report_html).read_bytes(),
                file_name=f"{run.run_id}.html",
                mime="text/html",
            )
    else:
        st.write("No runs yet.")

# ---------------------------------------------------------------- settings

with settings_tab:
    cfg = project.config
    if HOSTED:
        st.info(
            "In the public demo, tests only run against the bundled Orders API, so this server "
            "cannot be used to send traffic to other APIs. To test your own API, run VeroniQA "
            "yourself (`make veroniqa`)."
        )
    else:
        with st.form("settings"):
            spec = st.text_input(
                "OpenAPI or Swagger spec (file path or http(s) URL)", cfg.spec or ""
            )
            base_url = st.text_input("Base URL of the API under test", cfg.base_url or "")
            target = st.text_input(
                "Target config (YAML: auth scheme, role tokens, sandbox flag)",
                cfg.target_config or "",
            )
            if st.form_submit_button("Save settings"):
                project.update(
                    spec=spec.strip() or None,
                    base_url=base_url.strip() or None,
                    target_config=target.strip() or None,
                )
                st.success("Saved.")
        spec_file = st.file_uploader("…or upload a spec file", type=["json", "yaml", "yml"])
        if spec_file is not None and st.button("Use this spec"):
            dest = project.root / f"spec{Path(spec_file.name).suffix.lower()}"
            dest.write_bytes(spec_file.getvalue()[:5_000_000])
            project.update(spec=str(dest))
            st.success(f"Spec saved to {dest.name}.")
        st.caption(
            "Without `sandbox: true` in the target config, runs only use read-only methods. "
            "Keys and tokens belong in .env, not in project files."
        )
