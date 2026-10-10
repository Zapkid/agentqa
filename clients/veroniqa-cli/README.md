# veroniqa

The official command-line client for [VeroniQA](https://veroniqa.vercel.app/), the AI assistant
for API testing. It calls VeroniQA's public API, which needs no key, and it has no dependencies
beyond the Python standard library.

## Install

```bash
# from the repository (works today)
pipx install "git+https://github.com/Zapkid/agentqa#subdirectory=clients/veroniqa-cli"
# or run it without installing
uvx --from "git+https://github.com/Zapkid/agentqa#subdirectory=clients/veroniqa-cli" veroniqa status
```

## Use

```bash
veroniqa status                       # service status and version
veroniqa results                      # the measured results shown on the home page
veroniqa ask "What is the p95 latency target for the order list?"
veroniqa openapi > veroniqa.json      # the public API's OpenAPI description
veroniqa demo-spec > orders.json      # the OpenAPI spec of the bundled demo API
veroniqa ask "..." --json             # raw JSON for scripts and agents
```

Options: `--base-url URL` (or `VERONIQA_URL`) to point at another deployment, `--timeout SECONDS`,
and `--json` for machine-readable output.

Exit codes: `0` success, `1` the API returned an error (an RFC 9457 problem document, printed with
its `code` and `hint`), `2` usage error, `3` network error.

API guide: <https://veroniqa.vercel.app/developers>
