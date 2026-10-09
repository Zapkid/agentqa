# Deploying the VeroniQA public demo

The public demo is VeroniQA on its own URL (planned: <https://veroniqa.vercel.app>), with the
videos on the same site:

| path | what |
|---|---|
| `/` | the VeroniQA app (Streamlit): chat, knowledge, test runs, and a **Watch** tab |
| `/watch` | a plain page with the narrated introduction (captions on) and the three short films |
| `/videos/<name>.mp4` | the videos themselves, served with range requests so they stream and seek |

One ASGI app serves all three: `agentqa/veroniqa/server.py` wraps the Streamlit script in
Streamlit's `st.App` and adds the extra routes. It runs as a container image on Vercel
(`Dockerfile.vercel`). A plain Python function would not do: the locked dependencies are about
830 MB installed (Streamlit and pyarrow, Chroma and its runtime), and the app needs a WebSocket
for the life of a session.

## What "hosted mode" changes

`Dockerfile.vercel` sets `VERONIQA_HOSTED=1`, which turns the local tool into a safe public demo
(`agentqa/veroniqa/hosted.py`):

- **Private, temporary workspaces.** Each visitor gets their own projects and vector stores,
  named by a random id kept in the page URL (`?w=...`). Reloading or sharing that link brings the
  same workspace back while the server instance lives; idle workspaces are deleted after 12 hours.
  A new workspace starts with the Orders API demo project and its documents.
- **Simulated models only.** The server holds no API keys, so a visitor cannot run up a bill.
- **Test runs only against the bundled Orders API**, at most two at a time. A public server that
  ran generated tests against any URL a visitor typed could be used to send traffic to other
  people's APIs, so the project settings for external APIs are locked and the run function
  refuses non-demo projects.
- Uploads (5 MB, type allowlist), link fetching (public addresses only) and the injection scan
  work exactly as they do locally.

Storage is the container's temporary disk. Workspaces do not survive a new deployment or a cold
start. This is a demo, not a place to keep work.

## One-time setup on Vercel

1. In the Vercel dashboard, **Add New → Project**, import the GitHub repository `Zapkid/agentqa`,
   and name the project **`veroniqa`**. That name gives the production URL
   `veroniqa.vercel.app` when it is free.
2. Leave the root directory as the repository root. Vercel finds `Dockerfile.vercel` there and
   builds the image; no build command, environment variables or secrets are needed.
3. Deploy. From then on every push to `main` deploys to production, and every other branch gets a
   preview URL.
4. If `veroniqa.vercel.app` is not assigned automatically, add it under **Settings → Domains**.

The image runs as an unprivileged user, listens on `$PORT` and writes only under `/tmp`.

## Run the same thing locally

```bash
make veroniqa-hosted     # hosted mode on http://127.0.0.1:8502 (the UI, /watch and /videos)
make veroniqa            # the normal single-user app on http://127.0.0.1:8501
```
