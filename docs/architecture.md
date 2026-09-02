# Architecture

Retrieval-augmented generation (RAG) finds source text before a model answers.
FinRAG splits retrieval work into three parts so each choice has a clear job.

## Request flow

```text
question and role
  -> query strategy makes one or more searches
  -> retriever ranks sections the role may see
  -> pipeline removes blocked and repeated sections
  -> reranker keeps or changes the order
  -> shared answer step sees the original question and final sections
  -> answer, citations, trace, and access counts
```

The pipeline ID names all three choices:

```text
query__retrieval__reranking
```

For example, `hyde__graph_entity__llm` uses a HyDE search query, the entity
graph retriever, and model reranking. The default is `direct__bm25__none`.

## How the design maps to questions

A strategy should earn its place by helping a known question shape. This table
states the design intent. It does not claim a result. The
[experiments guide](experiments.md#strategy-to-question-evidence-map) shows
what the tests actually found.

| Question group | Retrieval problem | Choices designed to help | Why they may help |
|---|---|---|---|
| T1 direct | The source and question share clear words | Direct query, BM25 | Preserve the useful words and rank exact matches |
| T2 paraphrase | The meaning stays the same while the words change | Dense, hybrid, HyDE, entity graph | Bridge changed wording through embeddings, generated search text, or entity links |
| T3 time | Similar quarters reuse metrics and numbers | BM25 or hybrid, plus a fact-shaped temporal graph | Keep exact time clues and bind each value to its quarter and activity |
| T4 multi-step and conflict | The answer needs several sections or competing sources | Hybrid, ReAct, entity graph, claim-source graph | Search for separate facts and preserve which source made each claim |
| T5 summary | The answer needs broad but complete coverage | Hybrid, document structure, safe summary chunks | Join lexical and semantic coverage with document context |
| T6 access | A blocked source must have no effect on retrieval or generation | Policy checks around every strategy | Security is a gate, not a retrieval-quality feature |

The table gives us a testable reason for each choice. A measured failure can
reject that reason. The current temporal graph is the clearest example: it was
built for T3, but its broad links made T3 retrieval worse.

## Query strategies

The query strategy decides what to search for. It never creates evidence.

| Strategy | Designed use | What it adds | Cost or failure mode |
|---|---|---|---|
| `direct` | Control and clear T1 questions | One search with the original question | Cannot repair weak wording or split a task |
| `hyde` | T2 paraphrases | A short possible answer used only as search text | One model call can introduce misleading words |
| `react` | T4 questions that may need separate searches | Up to four planned searches and an evidence choice | More calls; searches may repeat or miss a needed part |

HyDE means Hypothetical Document Embeddings. Its made-up text is never sent to
the answer step. If its model call fails, it uses the original question.
ReAct means Reason and Act. It cannot write the final answer. It rejects
repeated queries and falls back to found evidence or one direct search when
needed.

Keeping ReAct out of answer writing makes its effect easier to judge. A score
change comes from search and evidence choice, not from a different writer.

## Retrievers

Every retriever follows one contract: return at most `k` allowed sections,
best first.

| Retriever | Designed question fit | Rank signal | Main limit | Service |
|---|---|---|---|---|
| `naive` | T1 sanity check | Shared words | Weak term scoring | None |
| `bm25` | T1 and exact T3 clues | BM25 text score | Changed wording | None |
| `dense` | T2 paraphrases | Embedding similarity | Similar facts can look alike | Embedder |
| `hybrid` | Broad T1-T5 baseline | BM25 and dense ranks joined with Reciprocal Rank Fusion | More work than either search alone | Embedder |
| `graph_keyword` | T1 graph baseline | Keyword entry and distinct graph path count | Paraphrases and broad hubs | Neo4j |
| `graph_entity` | T2 and T4 | Dense seeds plus entity and claim links | Noisy or vague extracted entities | Embedder and Neo4j |
| `graph_temporal` | Intended for T3 | Dense seeds plus quarter and metric links | Current links do not bind a complete fact | Embedder and Neo4j |

Dense and hybrid search use a saved matrix for corpus sections. They embed only
the query at search time. The local embedding cache avoids repeat calls.

## One graph, three edge layers

The graph build replaces the project namespace in Neo4j. Each edge stores an
internal `layer` value and the source chunk that created it.

| Layer | Edges | Built by |
|---|---|---|
| `structural` | document `contains` chunk; department `owned_by` document | `finrag graph` |
| `temporal` | chunk `mentions` quarter; chunk `reports` metric | `finrag graph` |
| `entity` | chunk-to-entity provenance links and extracted subject-relation-object triples | `finrag graph --llm` |

### Graph shapes: built and proposed

Different graph shapes answer different joins. Adding edges is not the goal.
The graph must preserve the relation used by the question.

| Graph shape | Status | Relation it preserves | Target questions | Current judgment |
|---|---|---|---|---|
| Keyword walk over all layers | Built as `graph_keyword` | Words to nearby graph nodes | T1 baseline | Useful failure control; weak retrieval |
| Entity and relation triples | Built as `graph_entity` | Entity to entity through a named relation, with source chunks | T2 and T4 | Improved coverage; keep as an experiment |
| Quarter and metric hubs | Built as `graph_temporal` | Chunk to quarter or metric | T3 | Too broad; it mixed similar reports |
| Fact or event nodes | Proposed | Subject, activity, metric, value, quarter, and source as one fact | T3 | Best next temporal test |
| Claim and provenance nodes | Proposed | Claim, value, source, and explicit disagreement | T4 conflicts | Worth testing if conflict retrieval remains a goal |
| Document and section structure | Built, but not isolated as a retriever | Department, document, and contained section | T5 context | Test separately before claiming a summary gain |
| Topic communities | Not planned | Groups of sections about a broad theme | Large-corpus T5 summaries | Too much machinery for this corpus today |

A fact node would stop a value from floating between unrelated quarters. A
claim node would keep two conflicting values separate while linking both to
the question topic. Neither design is implemented, so the documentation labels
them as proposed tests rather than current features.

`graph_keyword` is the old baseline under a clear name. It finds labels that
share words with the question, walks at most two hops across all layers, and
ranks chunks by distinct path count. It keeps this simple behavior on purpose.

`graph_entity` and `graph_temporal` share the same code. Only the allowed edge
layer changes:

```text
question
  -> dense search picks five role-visible seed chunks
  -> Neo4j walks one selected layer for at most three hops
  -> seeds and connected chunks are merged
  -> scores are sorted, with chunk ID as the final tie-break
```

A seed at rank `r` gets a direct score of `1 / r`. A connected chunk gets this
contribution from each distinct seed:

```text
1 / (seed rank * (shortest path length + 1))
```

Only the shortest path for each seed-to-chunk pair counts. Raw path count does
not. Three hops cover `chunk -> entity -> entity -> chunk` and the shorter
`chunk -> quarter or metric -> chunk` path.

Each graph hit records the layer, seed rank, hop limit, direct score, graph
score, and shortest paths in `Hit.debug`. Evaluation reports also record the
embedding model, layer counts, seed count, hop limit, and score rule.

If `graph_entity` cannot find entity edges, it stops and tells the user to run
`finrag graph --llm`. This is safer than silently using a different layer.

## Entity extraction cache

The entity build asks the existing model for triples. Its local JSONL cache is
keyed by chunk ID, chunk-text hash, extraction model, prompt version, and the
triple limit. A changed chunk, model, prompt, or limit causes a new call. An
unchanged chunk reuses its parsed triples.

The build output records the model, prompt version, cache hits, model calls,
failures, triple count, and edge counts by layer. The cache stays local and is
ignored by Git.

## Reranking contract

| Strategy | Designed use | What it can change | What it cannot fix |
|---|---|---|---|
| `none` | Control and already strong retrieval order | Nothing; it keeps retrieval order | Weak ordering |
| `llm` | Questions where the candidate set contains the answer but ranks it too low | The order of up to 20 allowed sections | Missing evidence or a bad candidate set |

Model reranking may help any group when retrieval has good recall but weak
ordering. It is not a cure for failed retrieval. If the right section is
absent, the reranker cannot add it.

The parser rejects a missing, repeated, invented, or malformed section number.
The pipeline checks the same rule again. If all attempts fail, the old order is
kept. A reranker may change rank, but it may not change the evidence set.

## Shared answer step

Every pipeline ends at the same answer step. It receives the original question
and final real sections. Query text made by HyDE or ReAct is not a source.

With no hits, the system returns `no_answer`. With no answer model, it returns
an extract from the top section and marks it as extractive. With a model, the
prompt requires an answer grounded in the supplied sections. Parsed citations
must point to those sections.

## Services and fallbacks

| Need | Used by |
|---|---|
| Embedder | `dense`, `hybrid`, `graph_entity`, `graph_temporal` |
| Neo4j | all three graph retrievers |
| Query model | `hyde`, `react` |
| Rerank model | `llm` reranking |
| Answer model | generated answers only |

The runtime creates a client only when the selected strategy needs it. A named
evaluation stops if a required service or graph layer is missing. It does not
publish a fallback under the requested name.

The final answer path is more forgiving. The API can return an extract when an
answer model is off. The response marks that choice.

## Framework boundary

FinRAG does not use LangChain or LangGraph. Small Python protocols define query
strategies, retrievers, and rerankers. Registries compose them, the Neo4j driver
runs graph queries, and the model client is called directly.

I chose this boundary so the experiment is visible: a reader can see which
part made a query, added a candidate, changed a rank, or invoked a service. The
cost is more local plumbing and fewer ready-made integrations. A larger product
might choose a framework for those integrations; this project keeps the parts
plain because understanding their effects is the goal.

## Traces

Each request records the full pipeline ID, generated searches, hit counts,
model calls, fallbacks, agent failures, ranked section IDs, citations, and
visible and blocked corpus counts. Trace text explains what the system did. It
is not answer evidence.
