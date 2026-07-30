# finrag

ABAC RAG over the FinSolve corpus, built rung by rung with graded evals and
graph retrieval. See `DECISIONS.md` for the build log.

```
uv sync --group dev              # install, including the test group
uv run pytest                    # the whole suite, offline, no credentials
uv run finrag ingest             # corpus -> chunks.jsonl, bm25/, dense.npz
uv run finrag graph              # chunks.jsonl -> the :Finrag graph in Neo4j
uv run finrag eval --mode dense  # score a mode, write evals/results/*.json
uv run finrag sweep              # the ABAC leakage gate: every question, role and mode
uv run finrag serve              # the API on http://127.0.0.1:8000, /docs for the schema
uv run finrag ui                 # the Streamlit pages, against a running API
```

Modes: `naive`, `bm25`, `dense`, `hybrid`, `hybrid_rerank`, `graphrag`.
Route real questions to `hybrid_rerank`; see the six-mode table in `DECISIONS.md`.

`hybrid` fuses the BM25 and dense ranked lists with Reciprocal Rank Fusion
(k=60), which needs no extra credential beyond the embeddings. `hybrid_rerank`
then asks Gemini once per query to reorder hybrid's top 20, so an eval run makes
50 real API calls and cannot be replayed offline; on any failure it falls back to
the hybrid order and the report records how often that happened.

`graphrag` walks a Neo4j graph two hops from the query terms, with the role mask
inside the Cypher. It needs the dev container:

```
docker run -d --name finrag-neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/finragdev neo4j:5-community
```

`finrag graph` writes the structural graph, which is read off the chunk metadata
and costs nothing. `finrag graph --llm` adds an entity layer by asking Gemini for
triples, one call per chunk. With the container down, `pytest` skips the graph
tests and `finrag sweep` drops the mode and says so.

`eval` validates the gold set against the built chunks and the policy before
scoring, and exits 1 if the labels have drifted.

`serve` answers only from the retrieved excerpts and cites them by chunk id. With
no `VERTEX_PROJECT`, or with `FINRAG_LLM=off`, answers are extractive quotes and
say so; `FINRAG_LLM=fake` uses the offline stand-in.

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
