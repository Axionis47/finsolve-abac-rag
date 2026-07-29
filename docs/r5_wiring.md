# Wiring R5 (graph RAG on Neo4j)

Everything in `src/finrag/graph/` is finished and tested. Nothing is registered.
This file is the whole integration: three edits, copy-paste.

The graph lives in Neo4j, not in `artifacts/`, so `IndexBundle.graph_path` stays
unused. A `GraphStore()` with no arguments reads `NEO4J_URI`, `NEO4J_USER` and
`NEO4J_PASSWORD` and falls back to the dev container (`bolt://localhost:7687`,
`neo4j` / `finragdev`).

## 1. `src/finrag/retrieval/registry.py` — the mode

Add the import beside the other retriever imports:

```python
from finrag.graph.retriever import GraphRetriever
from finrag.graph.store import GraphStore
```

Add the factory beside `make_hybrid`, and one line in `FACTORIES`:

```python
def make_graphrag(bundle: IndexBundle) -> Retriever:
    """Two-hop traversal over the Neo4j graph `finrag graph` built."""
    return GraphRetriever(bundle.chunks, GraphStore())
```

```python
    "graphrag": make_graphrag,
```

`GraphStore()` opens a driver but no connection, so the other modes pay nothing
for this import. Nothing needs adding to `IndexBundle`, and no mode needs adding
to `MODES_NEEDING_AN_EMBEDDER` or `MODES_NEEDING_AN_LLM`: the traversal calls no
model.

## 2. `src/finrag/cli.py` — the build step

Add the imports:

```python
from finrag.graph.build_structural import build_and_push
from finrag.graph.store import GraphStore
```

Delete the `"graph"` line from `PENDING_COMMANDS`, add the command, and dispatch it:

```python
def command_graph() -> int:
    """Rebuild the :Finrag graph from artifacts/chunks.jsonl. Wipes, then writes."""
    settings = Settings.from_env()
    if not settings.chunks_path.exists():
        print(f"No chunks at {settings.chunks_path}. Run `finrag ingest` first.")
        return 1
    with GraphStore() as store:
        counts = build_and_push(read_chunks(settings.chunks_path), store)
    print(f"Wrote {counts.n_nodes} nodes and {counts.n_edges} relationships to :Finrag")
    return 0
```

```python
    if args.command == "graph":
        return command_graph()
```

## 3. The pipeline order

`finrag graph` reads `artifacts/chunks.jsonl`, so it runs after `finrag ingest`
and must be re-run whenever ingest is. A stale graph is not silently wrong:
`GraphRetriever` drops any chunk id the current `chunks.jsonl` no longer has,
rather than serving text whose policy it can no longer check.

```
finrag ingest      # writes chunks.jsonl, the BM25 index, dense.npz
finrag graph       # wipes :Finrag, writes the structural graph
finrag eval --mode graphrag --rung r5
```

## Optional: the LLM triple layer

The structural build is the floor (344 nodes, 709 relationships on the current
corpus, of which only 6 nodes are entities). The entity layer comes from
`build_llm.py`, which takes any `LLMBackend`, so the Vertex client R3 added drops
straight in. One extra command, or a `--llm` flag on the one above:

```python
from finrag.graph import build_llm, build_structural
from finrag.llm.vertex import build_llm as build_vertex_llm

structural_nodes, structural_edges = build_structural.extract(chunks)
llm_nodes, llm_edges, stats = build_llm.extract(build_vertex_llm(settings), chunks)
counts = store.write_graph(structural_nodes + llm_nodes, structural_edges + llm_edges)
print(f"triples: {stats.n_triples}, fallback rate: {stats.fallback_rate:.2f}")
```

Write both lists in one call, as above: `write_graph` wipes first, so writing them
one after the other would leave only the second.

## The dev container

```
docker run -d --name finrag-neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/finragdev neo4j:5-community
```

The graph tests use the `:FinragTest` namespace and delete only that, so they can
never touch `:Finrag`. They skip with one line when bolt is closed, so `pytest -q`
is green with the container up or down.
