# finrag

ABAC RAG over the FinSolve corpus, built rung by rung with graded evals and
graph retrieval. See `DECISIONS.md` for the build log.

```
uv sync --group dev              # install, including the test group
uv run pytest                    # the whole suite, offline, no credentials
uv run finrag ingest             # corpus -> artifacts/chunks.jsonl
uv run finrag eval --mode naive  # score a mode, write evals/results/*.json
```

`eval` validates the gold set against the built chunks and the policy before
scoring, and exits 1 if the labels have drifted.

On this machine (iCloud-synced Desktop) macOS keeps hiding the editable
install's `.pth` file after a sync, and Python then ignores it, so the
`finrag` script cannot import its package. `pytest` is unaffected. Use
`./dev.sh sync|test|eval`, which repairs the flag every time.
