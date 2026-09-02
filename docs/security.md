# Security

FinRAG uses attribute-based access control. A rule looks at the user's role and
the source section's owner, type, and sensitivity. The default answer is deny.

The important choice is timing. The system checks access before ranking or
model use. A prompt that says “do not reveal private data” would be too late.

## The trust path

```text
policy rules
  -> allowed roles stamped on each chunk at ingest
  -> candidate mask before retrieval scoring
  -> final mask before reranking
  -> answer from allowed hits only
  -> evaluation replays each hit against the policy rules
```

The stamped roles make each search fast. The policy replay does not trust that
stamp. It checks returned hits against the YAML rules again.

## Before text ranking

Each retriever starts with the IDs the role may see. BM25 applies that set as a
weight mask before it cuts the list to `k`. Dense and hybrid search also build
their candidate lists from allowed sections.

This order matters. If the code scored everything and removed blocked sections
only at the end, a blocked section could take a rank slot or affect a fused
score. The final visible list would still carry information from hidden text.

The pipeline performs another access and duplicate check after the query
strategy returns. Only that checked list reaches the reranker.

## Inside GraphRAG

A graph can leak through a path even when the final node is removed. A hidden
edge may connect two allowed nodes and change their scores.

`graph_entity` and `graph_temporal` first run dense search over role-visible
chunks only. Blocked chunks cannot become seeds. The Neo4j walk then checks two
facts for every edge: its `layer` must match the selected strategy, and its
source chunk ID must be in the role's visible set. A blocked edge cannot be
walked and cannot add to a score.

`graph_keyword` may walk all layers, but it applies the same source-chunk check
inside every path.

All graph retrievers check a returned chunk ID again before creating a `Hit`.
The shared pipeline then applies its final access and duplicate check. Tests
inspect live Neo4j walks to prove that every traversed edge has the selected
layer and a role-visible source.

## Before model prompts

ReAct sees only extracts returned by a masked retriever. A blocked chunk cannot
enter an agent observation.

The reranker receives the pipeline's checked candidate list. Its prompt cannot
contain blocked text. It may only reorder those candidates.

The answer model receives the final checked hits. Citations are parsed against
the same list, so a model cannot cite a source that was not supplied.

## HR rows

The HR CSV has two layers. Search text contains only name, role, department,
and location. Fields such as email, date of birth, salary, leave, and ratings
stay in the row payload and out of the search index.

Only `hr` and `c_level` may retrieve an HR row. Other roles cannot use the row
as evidence or learn its place in a result list. Five summary chunks support
questions such as headcount or average salary, but they follow the same HR
access rules by default.

## What the evaluation checks

Every evaluation asks its gold questions under their named roles. It sends each
returned hit back through the policy engine and records any failed check as
leakage. The target is always zero.

The leakage sweep is wider. It asks every question as every role across the
selected strategy combinations. It uses model failure paths and fake or cached
embeddings, so it can run without a paid model call. Graph strategies join the
sweep when Neo4j is available; otherwise every strategy that requires `GRAPH`
is named as not swept. A planted leaky retriever in the tests proves that the
sweep fails when a blocked chunk returns.

The saved full sweep checked 42 pipelines, six roles, 12,600 cells, and 117,798
returned hits. It found zero violations. The full report is
[`graph_leakage_sweep_20260901T2234.json`](../evals/results/graph_leakage_sweep_20260901T2234.json).

## Deployment limit

The demo API accepts the role in the request body. This is useful for a role
picker, but it is not authentication. Any client can claim any role.

A deployed service must read the role from a verified identity and ignore a
client-supplied role. The project does not include an identity provider, token
validation, or a full prompt-injection defense. It is a retrieval and agent
design test bed, not a finished security product.
