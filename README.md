# Sidekick - Personal AI Assistant

Sidekick is a FastAPI + LangGraph personal assistant with a single-page web UI, streaming chat responses, optional web search tools, and per-thread file-aware context retrieval.

## Features

- LangGraph worker/evaluator loop with retry ceiling.
- Streaming chat over Server-Sent Events (SSE).
- File uploads with chunking and per-thread retrieval (RAG-style context injection).
- Tavily web search tool integration.
- Optional Playwright browser tools locally; graceful fallback when unavailable.
- Thread persistence via SQLite checkpoints and document chunk storage.
- Dark/light theme UI with conversation reset support.

## Architecture

- `main.py`: Main FastAPI app, graph lifecycle, tools, SSE chat stream, file upload, and reset endpoints.
- `api/index.py`: Vercel serverless entrypoint that loads and exposes `app` from `main.py`.
- `public/index.html`: Chat UI page.
- `public/app.js`: Frontend chat/upload/reset logic and SSE event handling.
- `public/styles.css`: Styling and theme rules.
- `vercel.json`: Rewrites and function bundling config for deployment.

## Requirements

- Python 3.13 recommended (the repository includes `.python-version` set to `3.13`).
- Node.js + npm (for formatting/lint tooling only).
- `OPENAI_API_KEY` (required for chat responses).
- `TAVILY_API_KEY` (optional for Tavily search quality).

## Local Setup

### Python dependencies

Use either `uv` or `pip`.

```bash
# uv workflow (recommended if you use uv.lock)
uv sync

# or pip workflow
pip install -r requirements.txt
```

### Frontend tooling dependencies

```bash
npm install
```

### Run the app

```bash
uvicorn main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000).

## Environment Variables

| Variable | Required | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes | API key used by `ChatOpenAI` models. |
| `TAVILY_API_KEY` | No | API key for Tavily search tool. |

Notes:

- In serverless environments (`VERCEL`, `VERCEL_ENV`, or Lambda vars), Sidekick switches to serverless-safe behavior.
- Playwright tools are attempted only in non-serverless mode and may fall back to Tavily-only automatically if browser startup fails.

## API Endpoints

- `GET /`: Serves the chat UI.
- `GET /styles.css`: Serves frontend styles.
- `GET /app.js`: Serves frontend script.
- `GET /logo.svg`: Serves logo asset.
- `POST /api/upload`: Upload and index files for a thread.
- `POST /api/chat`: Stream assistant output over SSE (`meta`, `status`, `assistant`, `error`, `done` events).
- `POST /api/reset`: Returns a new `thread_id`; optionally accepts an old `thread_id` query param for cleanup.

### `POST /api/chat` request body

```json
{
  "message": "string",
  "success_criteria": "string (optional)",
  "thread_id": "string (optional)"
}
```

### `POST /api/upload` behavior

- Accepted extensions include text/code/docs such as `.txt`, `.md`, `.py`, `.json`, `.csv`, `.html`, `.js`, `.ts`, `.css`, `.log`, `.pdf`, and `.docx`.
- Files over 5 MB are ignored.
- Indexed chunks are stored per thread in SQLite and reused across turns.

## Deployment (Vercel)

The repository includes `vercel.json` rewrites that route `/` and `/api/*` to `api/index.py`.

```bash
vercel
vercel --prod
```

Set environment variables in Vercel:

- `OPENAI_API_KEY` (required)
- `TAVILY_API_KEY` (optional)

## Developer Commands

```bash
# Format frontend/docs
npm run format

# Check formatting
npm run format:check

# Lint frontend JavaScript
npm run lint:js
```

See `FORMATTING.md` for formatting conventions.

## Current Gaps

- There is no committed automated test suite in this repository yet.
- `.docx` parsing expects `python-docx`; if not installed, `.docx` files are reported as ignored during upload.
