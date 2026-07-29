# finrag

ABAC RAG over the FinSolve corpus, built rung by rung with graded evals and
graph retrieval. See `DECISIONS.md` for the build log.

```
uv sync --group dev              # install, including the test group
uv run pytest                    # the whole suite, offline, no credentials
uv run finrag ingest             # corpus -> chunks.jsonl, bm25/, dense.npz
uv run finrag eval --mode dense  # score a mode, write evals/results/*.json
```

Modes: `naive`, `bm25`, `dense`, `hybrid`, `hybrid_rerank`.

`hybrid` fuses the BM25 and dense ranked lists with Reciprocal Rank Fusion
(k=60), which needs no extra credential beyond the embeddings. `hybrid_rerank`
then asks Gemini once per query to reorder hybrid's top 20, so an eval run makes
50 real API calls and cannot be replayed offline; on any failure it falls back to
the hybrid order and the report records how often that happened.

`eval` validates the gold set against the built chunks and the policy before
scoring, and exits 1 if the labels have drifted.

Dense retrieval calls Vertex AI text embeddings, so `ingest` needs
`VERTEX_PROJECT` (or `GCP_PROJECT`) and `VERTEX_LOCATION` in a local `.env`,
plus application default credentials. Vectors are cached in
`artifacts/embed_cache.npz` under a sha256 of each text, so a second ingest over
an unchanged corpus and a repeated eval both make zero API calls. Set
`FINRAG_EMBEDDER=fake` to run the whole pipeline offline with deterministic
stand-in vectors; the test suite does that for itself.

On this machine (iCloud-synced Desktop) macOS keeps hiding the editable
install's `.pth` file after a sync, and Python then ignores it, so the
`finrag` script cannot import its package. `pytest` is unaffected. Use
`./dev.sh sync|test|eval`, which repairs the flag every time.
