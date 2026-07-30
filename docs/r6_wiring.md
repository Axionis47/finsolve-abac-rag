# R6 wiring: `finrag serve` and `finrag ui`

R6 built the serving layer but touched no existing file. This is the list of
edits that hook it up. Every one of them is in `src/finrag/cli.py`, plus two
optional one-liners elsewhere.

## What R6 added

| File | What it is |
|---|---|
| `src/finrag/llm/vertex.py` | `VertexLLM` (the real `LLMBackend`) and `build_llm(settings)` |
| `src/finrag/prompting.py` | the grounded prompt, citation parsing, the extractive quote |
| `src/finrag/generate.py` | `Answerer`, `answer(question, role, mode, k)` |
| `src/finrag/api/app.py` | `create_app(answerer=None) -> FastAPI`: `/health`, `/modes`, `/roles`, `/ask` |
| `ui/app.py`, `ui/api_client.py`, `ui/pages/{1_ask,2_compare,3_policy}.py` | the Streamlit demo |
| `tests/test_{vertex,prompting,generate,api}.py` | 57 offline tests |

The app lives at `src/finrag/api/app.py`, not `src/finrag/api.py`, because
`src/finrag/api/` already exists as a package and the two names collide.

## 1. cli.py: imports

Add next to the existing imports:

```python
from finrag.api.app import create_app
```

Nothing else needs importing at module level. `uvicorn` and `streamlit` are
imported inside the two command functions below, so `finrag ingest` and
`finrag eval` still run in an install without the `ui` group.

## 2. cli.py: drop `serve` from the pending list

```python
PENDING_COMMANDS: dict[str, str] = {
    "graph": "R5 (graph schema and the structural or LLM builds)",
}
```

## 3. cli.py: the two subparsers

Replace the `subparsers.add_parser("serve", ...)` line with:

```python
    serve_parser = subparsers.add_parser("serve", help="run the API on localhost")
    serve_parser.add_argument("--host", default="127.0.0.1", help="interface to bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="port to bind")
    serve_parser.add_argument(
        "--reload", action="store_true", help="restart on a code change (development only)"
    )

    ui_parser = subparsers.add_parser("ui", help="run the Streamlit UI (needs the API running)")
    ui_parser.add_argument("--port", type=int, default=8501, help="port for the UI")
    ui_parser.add_argument(
        "--api",
        default="http://127.0.0.1:8000",
        help="base URL of the finrag API the pages call",
    )
```

## 4. cli.py: the two command functions

Add near `command_eval`:

```python
def command_serve(host: str, port: int, reload: bool) -> int:
    """Run the FastAPI app with uvicorn.

    Passed as an import string with `factory=True` so uvicorn builds the app in
    the worker process; `--reload` needs that anyway, and it keeps this function
    from loading 318 chunks in a process that is about to be replaced.
    """
    import uvicorn

    settings = Settings.from_env()
    if not settings.chunks_path.exists():
        print(f"No chunks at {settings.chunks_path}. Run `finrag ingest` first.")
        return 1

    print(f"Serving finrag on http://{host}:{port}  (docs at /docs)")
    uvicorn.run(
        "finrag.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )
    return 0


def command_ui(port: int, api: str) -> int:
    """Run the Streamlit UI, pointed at a running API.

    Streamlit has no supported in-process entry point, so this shells out to the
    module the same way `streamlit run` does. FINRAG_API is how the pages find
    the API; it is set here so nobody has to export it by hand.
    """
    import os
    import subprocess

    entry = REPO_ROOT_UI / "app.py"
    if not entry.exists():
        print(f"No UI entry script at {entry}.")
        return 1

    env = dict(os.environ)
    env["FINRAG_API"] = api
    print(f"Starting the UI on http://127.0.0.1:{port}, talking to {api}")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(entry),
            "--server.port",
            str(port),
        ],
        env=env,
        check=False,
    )
    return completed.returncode
```

`command_ui` needs the path to `ui/`. Add this beside the other module constants
in cli.py:

```python
# The Streamlit pages live outside the package, next to pyproject.toml.
REPO_ROOT_UI = Path(__file__).resolve().parents[2] / "ui"
```

## 5. cli.py: the dispatch in `main`

```python
    if args.command == "serve":
        return command_serve(args.host, args.port, args.reload)
    if args.command == "ui":
        return command_ui(args.port, args.api)
    return command_pending(args.command)
```

## 6. Optional: `src/finrag/api/__init__.py`

One line, so `from finrag.api import create_app` works and the uvicorn import
string can shorten to `finrag.api:create_app`:

```python
from finrag.api.app import create_app

__all__ = ["create_app"]
```

Left out of R6 because `__init__.py` is an existing file.

## 7. Recommended: `src/finrag/config.py`, two lines

`vertex_model` defaults to `gemini-2.0-flash`, and Vertex now answers
`404 NOT_FOUND` for it in this project. Verified from this repo against the real
API in the dev project:

| model | result |
|---|---|
| `gemini-2.0-flash` | 404, retired |
| `gemini-2.5-flash` | works in `us-central1` |
| `gemini-3.6-flash` | works only with `VERTEX_LOCATION=global` |

Until config.py changes, `llm/vertex.py` reads `VERTEX_MODEL` and falls back to
`gemini-2.5-flash` itself. The tidier fix:

```python
-    vertex_model: str = "gemini-2.0-flash"
+    vertex_model: str = "gemini-2.5-flash"
```

```python
-            vertex_model=os.environ.get("VERTEX_MODEL", "gemini-2.0-flash"),
+            vertex_model=os.environ.get("VERTEX_MODEL", "gemini-2.5-flash"),
```

Then `build_llm` in `llm/vertex.py` can go back to `settings.vertex_model` and
its `model_name()` helper can be deleted.

## 8. Optional: `tests/conftest.py`

R6's tests build their clients with plain helper functions inside the test files
so no shared fixture file changed. If the house rule is fixtures-only, these two
belong in conftest.py and the helpers in `tests/test_api.py` and
`tests/test_generate.py` can call them instead:

```python
@pytest.fixture
def micro_answerer(micro_corpus, policy_path):
    """An answerer over the six-chunk corpus, with the real policy and FakeLLM."""
    from finrag.generate import Answerer
    from finrag.policy.engine import PolicyEngine
    from finrag.retrieval.registry import IndexBundle

    return Answerer(
        bundle=IndexBundle(chunks=list(micro_corpus)),
        policy=PolicyEngine.from_yaml(policy_path),
        llm=FakeLLM(),
    )


@pytest.fixture
def api_client(micro_answerer):
    """A TestClient over the micro corpus. Offline, no artifacts."""
    from fastapi.testclient import TestClient

    from finrag.api.app import create_app

    return TestClient(create_app(micro_answerer))
```

## How it runs once wired

```bash
finrag ingest                      # writes artifacts/
finrag serve                       # http://127.0.0.1:8000, /docs for the schema
finrag ui                          # http://127.0.0.1:8501, three pages
FINRAG_LLM=off finrag serve        # extractive answers, no model, no credentials
FINRAG_LLM=fake finrag serve       # the deterministic FakeLLM
```

`FINRAG_LLM` picks the generation backend: `vertex` (default), `fake`, or `off`.
With no `VERTEX_PROJECT` in the environment the API serves extractive answers and
says so in the response (`extractive: true`), rather than refusing to start.

Verified by hand before handover, with the real 318-chunk corpus:

- `GET /health` -> `{"status":"ok","chunks":318,"modes":["bm25","dense","naive"],"llm":"vertex"}`
- `POST /ask` as `finance`, "what was the Q4 gross margin", mode bm25, k 3 ->
  `"The Q4 gross margin was 64% [1]."`, one citation, audit `82` of `318` chunks
  visible to the role, `236` withheld.
- the same question as `engineering` about HR salaries -> `reason: no_answer`,
  the model refused on general-department context rather than inventing a figure.
