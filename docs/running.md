# Running FinRAG

FinRAG needs Python 3.12 and
[`uv`](https://docs.astral.sh/uv/getting-started/installation/).

Run commands from the repository root.

## Run offline

Install the base package, tests, and web app:

```bash
uv sync --group dev --group ui
```

Build chunks and indexes with deterministic fake embeddings:

```bash
FINRAG_EMBEDDER=fake FINRAG_LLM=fake uv run finrag ingest
```

Run the tests:

```bash
uv run pytest -q
```

Run one evaluation that needs no model or embedding service:

```bash
FINRAG_EMBEDDER=fake FINRAG_LLM=fake uv run finrag eval \
  --query-strategy direct \
  --retriever bm25 \
  --reranker none \
  --rung local
```

Each report is written to `evals/results/`. Its name contains the rung, full
pipeline ID, and time.

Run the full offline access sweep without Neo4j:

```bash
FINRAG_EMBEDDER=fake uv run finrag sweep \
  --query-strategies direct hyde react \
  --retrievers naive bm25 dense hybrid \
  --rerankers none llm \
  --rung local
```

The sweep never calls a model. HyDE, ReAct, and model reranking use their
failure paths. Fake embeddings keep dense and hybrid search offline. It leaves
out every retriever that declares a `GRAPH` requirement.

## Run the API and web app

Start the API:

```bash
FINRAG_EMBEDDER=fake FINRAG_LLM=fake uv run finrag serve
```

It listens on `http://127.0.0.1:8000`. Open `/docs` for the generated API page.

In another shell, start the Streamlit app:

```bash
FINRAG_EMBEDDER=fake FINRAG_LLM=fake uv run finrag ui
```

It listens on `http://127.0.0.1:8501`. The Ask page shows one run. The Compare
page changes one strategy part while holding the other two fixed.

The API routes are:

| Route | Use |
|---|---|
| `GET /health` | Process, chunk, strategy, and answer-model status |
| `GET /strategies` | Names, defaults, and service needs |
| `GET /roles` | Roles from the policy |
| `POST /ask` | One question through one pipeline |

Example request:

```bash
curl -s http://127.0.0.1:8000/ask \
  -H 'content-type: application/json' \
  -d '{
    "question": "What was the Q4 gross margin?",
    "role": "finance",
    "query_strategy": "direct",
    "retrieval_strategy": "bm25",
    "reranking_strategy": "none",
    "k": 5
  }'
```

The role comes from the request only for this demo. A real service must get it
from a verified identity.

## Use Vertex AI

Install the Vertex package too:

```bash
uv sync --group dev --group ui --group vertex
```

Set up Google Cloud Application Default Credentials, then add settings like
these to `.env`. Use your own project and model values.

```dotenv
VERTEX_PROJECT=your-project-id
VERTEX_LOCATION=us-central1
VERTEX_MODEL=gemini-2.5-flash
FINRAG_VERTEX_EMBED_MODEL=text-embedding-005
FINRAG_VERTEX_RERANK_MODEL=gemini-2.5-flash
FINRAG_EMBEDDER=vertex
FINRAG_LLM=vertex
```

Shell variables override `.env` values.

In a Claude Code cloud session there is no gcloud login and no `.env`. Put the credential JSON
(a service-account key, or the contents of `~/.config/gcloud/application_default_credentials.json`)
in `GOOGLE_SERVICE_ACCOUNT_JSON` in the environment's settings, with the other variables above
beside it. `.claude/hooks/session-start.sh` writes it to `artifacts/gcp-key.json`, points
`GOOGLE_APPLICATION_CREDENTIALS` at it, installs the groups above and builds the offline indexes.

Rebuild the dense matrix with Vertex embeddings:

```bash
uv run finrag ingest
```

Run a model-backed pipeline:

```bash
uv run finrag eval \
  --query-strategy hyde \
  --retriever hybrid \
  --reranker llm \
  --rung vertex
```

HyDE and ReAct need a query model. Model reranking needs a rerank model. A named
evaluation stops when one is missing. It does not publish a simpler fallback
under the requested name.

The API is different at the final answer step. `FINRAG_LLM=off` keeps the API
running and returns an extract from the top source instead of generated prose.
The response sets `extractive` to `true`.

## Use Neo4j and GraphRAG

Start a local Neo4j 5 container. Choose your own local password:

```bash
docker run -d --name finrag-neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/choose-a-local-password \
  neo4j:5-community
```

Add matching values to `.env`:

```dotenv
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=choose-a-local-password
```

Build the deterministic structural and temporal layers from the current
chunks:

```bash
uv run finrag graph
```

Add the entity layer with the configured Vertex model:

```bash
uv run finrag graph --llm
```

The command always rebuilds the deterministic layers and then adds entity and
claim links. The first run may make one extraction call per chunk. Parsed
triples are cached in the local artifacts folder. The cache key includes the
chunk ID, chunk-text hash, model, prompt version, and triple limit. An unchanged
second build should report cache hits and zero model calls.

The build output prints the model, prompt version, cache hits, calls, failures,
triple count, and edge counts by layer. The cache is ignored by Git.

This graph format uses a `layer` value on every edge. Rebuild once after
updating from a graph without layers. The build already replaces the FinRAG
namespace, so there is no migration command.

The graph retrievers are:

| Name | Entry and walk | Needs |
|---|---|---|
| `graph_keyword` | Keyword entry; two-hop path count across available layers | Neo4j |
| `graph_entity` | Five dense seeds; up to three hops on entity edges | Embedder, Neo4j, and an entity layer |
| `graph_temporal` | Five dense seeds; up to three hops on quarter and metric edges | Embedder and Neo4j |

`graph_entity` stops with a clear instruction to run `finrag graph --llm` when
the entity layer is missing.

Run the controlled comparison after building the live embedding index and the
full graph:

```bash
uv run finrag eval \
  --query-strategy direct \
  --retriever dense \
  --reranker none \
  --rung graph_compare

uv run finrag eval \
  --query-strategy direct \
  --retriever graph_keyword \
  --reranker none \
  --rung graph_compare

uv run finrag eval \
  --query-strategy direct \
  --retriever graph_entity \
  --reranker none \
  --rung graph_compare

uv run finrag eval \
  --query-strategy direct \
  --retriever graph_temporal \
  --reranker none \
  --rung graph_compare
```

Use the same corpus, chunks, embedding model, query strategy, reranker, and `k`
for all four. Fake embeddings may test the flow, but they are not retrieval
quality evidence.

After those evaluations have cached every gold question embedding, run the
full leakage sweep. It uses cached embeddings and forced model fallbacks, so it
does not call a paid model:

```bash
uv run finrag sweep --rung graph_leakage
```

When Neo4j is answering, the default sweep includes all three graph
retrievers. When it is down, the output names each graph strategy that was not
swept. An explicit graph request fails instead of changing the strategy list.

## Other settings

| Variable | Default | Use |
|---|---|---|
| `FINRAG_CHUNKER` | `heading` | Choose `heading` or `table_aware` chunks |
| `FINRAG_CORPUS_DIR` | `data/corpus` | Change the source folder |
| `FINRAG_ARTIFACTS_DIR` | `artifacts` | Change the index folder |
| `FINRAG_GOLD_PATH` | `evals/gold/gold_set.json` | Change the gold set |
| `FINRAG_RESULTS_DIR` | `evals/results` | Change report output |
| `FINRAG_POLICY_PATH` | `policy/policy.yaml` | Change the policy file |

The saved gold section IDs match the default `heading` chunker. Evaluation with
another chunker warns that section labels were not checked for that chunking
scheme.
